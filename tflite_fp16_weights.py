"""Experimental FLOAT32 constant -> FLOAT16 + DEQUANTIZE graph rewrite.
For the single-subgraph, builtin-op YOLOv8n TFLite FP32 export in this project.
Not a TensorFlow/LiteRT official CLI. Does not export from .pt or train a model.
Method based on the upstream constant FP16 + DEQUANTIZE transformation:
https://github.com/tensorflow/tensorflow/blob/v2.20.0/tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/quantize_weights_portable.cc
Uses ai-edge-litert==2.3.0 schema and interpreter. Run in the existing WSL
export environment. Keep original .tflite and .conversion.json report.
Synthetic convolution and structural checks tested; actual YOLO model and
TFLite 2.20/XNNPACK execution must be checked with the 50-image experiment.
"""
import argparse
import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s


def pack_model(model):
    b = flatbuffers.Builder(1024)
    b.Finish(model.Pack(b), file_identifier=b'TFL3')
    return bytes(b.Output())


def unpack(data):
    if data[4:8] != b'TFL3':
        raise ValueError('Not a TFLite flatbuffer (TFL3)')
    return s.ModelT.InitFromObj(s.Model.GetRootAsModel(data, 0))


def packed_object(obj):
    b = flatbuffers.Builder(256)
    b.Finish(obj.Pack(b))
    return bytes(b.Output())


def opcode(code):
    return max(code.builtinCode, code.deprecatedBuiltinCode)


def io_info(model):
    g = model.subgraphs[0]
    def details(indices):
        return [{'index': int(i), 'type': int(g.tensors[i].type),
                 'shape': [int(v) for v in g.tensors[i].shape],
                 'name': (g.tensors[i].name or b'').decode(errors='replace')}
                for i in indices]
    return {'inputs': details(g.inputs), 'outputs': details(g.outputs)}


def transform(data):
    model = unpack(data)
    if len(model.subgraphs) != 1:
        raise ValueError('Only single-subgraph models supported by this script')
    g = model.subgraphs[0]
    if model.buffers[0].data is not None and len(model.buffers[0].data):
        raise ValueError('Expected empty buffer zero')
    if any(opcode(c) == s.BuiltinOperator.CUSTOM for c in model.operatorCodes):
        raise ValueError('Custom/Flex operators are outside this script scope')
    if any(getattr(b, 'offset', 0) or getattr(b, 'size', 0) for b in model.buffers):
        raise ValueError('External-buffer models not supported')
    if any(t.isVariable or t.sparsity is not None for t in g.tensors):
        raise ValueError('Variable/sparse tensors not supported')
    if any(t.type == s.TensorType.FLOAT16 for t in g.tensors):
        raise ValueError('Source already contains FLOAT16 tensors; refusing repeat conversion')
    before_io = io_info(model)
    if any(t['type'] != s.TensorType.FLOAT32 for k in before_io.values() for t in k):
        raise ValueError('This project expects FLOAT32 model inputs and outputs')
    before_ops = [packed_object(op) for op in g.operators]
    original_tensor_count = len(g.tensors)
    signatures = [packed_object(sig) for sig in (model.signatureDefs or [])]
    producers = {int(i) for op in g.operators for i in op.outputs if i >= 0}
    boundary = set(map(int, g.inputs)) | set(map(int, g.outputs))
    first_use = {}
    for position, op in enumerate(g.operators):
        for idx in op.inputs:
            if idx >= 0:
                first_use.setdefault(int(idx), position)
    candidates = []
    for idx, first in sorted(first_use.items()):
        t = g.tensors[idx]
        b = model.buffers[t.buffer]
        if t.type != s.TensorType.FLOAT32 or b.data is None or len(b.data) == 0:
            continue
        if idx in producers or idx in boundary:
            raise ValueError('Constant also produced by op or exposed at graph boundary')
        raw = bytes(b.data)
        values = np.frombuffer(raw, dtype='<f4')
        if values.size != int(np.prod(t.shape, dtype=np.int64)):
            raise ValueError(f'Constant buffer/shape mismatch at tensor {idx}')
        if not np.isfinite(values).all() or np.any(np.abs(values) > 65504):
            raise ValueError(f'Tensor {idx} has nonfinite/out-of-range values; no silent saturation')
        half = values.astype('<f2')
        candidates.append((idx, first, half, values))
    if not candidates:
        raise ValueError('No eligible FLOAT32 constants found')
    dq_code = s.OperatorCodeT()
    dq_code.builtinCode = s.BuiltinOperator.DEQUANTIZE
    dq_code.deprecatedBuiltinCode = s.BuiltinOperator.DEQUANTIZE
    dq_code.version = 3  # TFLite DEQUANTIZE v3 supports FLOAT16 input.
    code_index = len(model.operatorCodes)
    model.operatorCodes.append(dq_code)
    insertions = {}
    changes = []
    for idx, first, half, values in candidates:
        t = g.tensors[idx]
        old_buffer = int(t.buffer)
        b = s.BufferT()
        b.data = np.frombuffer(half.tobytes(), dtype=np.uint8).copy()
        half_tensor = copy.deepcopy(t)
        half_tensor.name = (t.name or b'constant') + b'_fp16_storage'
        half_tensor.type = s.TensorType.FLOAT16
        half_tensor.buffer = len(model.buffers)
        half_tensor.quantization = None
        model.buffers.append(b)
        half_idx = len(g.tensors)
        g.tensors.append(half_tensor)
        # Original FP32 tensor and all original op connections keep their IDs.
        # It becomes the runtime result of DEQUANTIZE rather than a constant.
        t.buffer = 0
        dq = s.OperatorT()
        dq.opcodeIndex = code_index
        dq.inputs = np.array([half_idx], dtype=np.int32)
        dq.outputs = np.array([idx], dtype=np.int32)
        insertions.setdefault(first, []).append(dq)
        changes.append({'tensor_index': idx, 'name': (t.name or b'').decode(errors='replace'),
                        'fp16_tensor_index': half_idx, 'old_buffer': old_buffer,
                        'elements': int(half.size),
                        'max_abs_rounding_error': float(np.max(np.abs(values-half.astype(np.float32))))})
    original_ops = list(g.operators)
    g.operators = []
    for i, op in enumerate(original_ops):
        g.operators.extend(insertions.get(i, []))
        g.operators.append(op)
    # Remove payload only when no tensor or metadata still references its buffer.
    used = {int(t.buffer) for graph in model.subgraphs for t in graph.tensors}
    used.update(int(md.buffer) for md in (model.metadata or []))
    if model.metadataBuffer is not None:
        used.update(map(int, model.metadataBuffer))
    for change in changes:
        if change['old_buffer'] not in used:
            model.buffers[change['old_buffer']].data = None
    result = pack_model(model)
    verify = unpack(result)
    vg = verify.subgraphs[0]
    assert io_info(verify) == before_io, 'Input/output contract changed'
    assert [packed_object(sig) for sig in (verify.signatureDefs or [])] == signatures
    filtered = [op for op in vg.operators if op.opcodeIndex != code_index]
    assert [packed_object(op) for op in filtered] == before_ops, 'Original operations changed'
    for i in range(original_tensor_count):
        assert vg.tensors[i].type != s.TensorType.FLOAT16, 'Original activation type changed'
    for c in changes:
        assert vg.tensors[c['fp16_tensor_index']].type == s.TensorType.FLOAT16
        assert vg.tensors[c['tensor_index']].type == s.TensorType.FLOAT32
    return result, {'converted_constants': len(changes),
                    'converted_elements': sum(c['elements'] for c in changes),
                    'original_operators': len(before_ops), 'added_dequantize_ops': len(changes),
                    'original_ops_unchanged': True, 'io_unchanged': True,
                    'io': before_io, 'constants': changes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('models/tflite_fp32/yolov8n_fp32.tflite'))
    p.add_argument('--output', type=Path, default=Path('models/tflite_fp16_weights/yolov8n_fp16_weights.tflite'))
    args = p.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        p.error('Choose a new output path; source/existing model will not be overwritten')
    report_path = args.output.with_suffix('.conversion.json')
    if report_path.exists():
        p.error('Conversion report already exists; choose a new output path')
    data = args.input.read_bytes()
    result, report = transform(data)
    # Preserve associated files in an appended ZIP, e.g. Ultralytics metadata.
    if zipfile.is_zipfile(io.BytesIO(data)):
        stream = io.BytesIO(result)
        with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(stream, 'a') as dest:
            for entry in source.infolist():
                dest.writestr(entry, source.read(entry.filename))
        result = stream.getvalue()
    # Structural checks are not sufficient: require a real interpreter allocation
    # and one finite-output invocation before writing the converted artifact.
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    runtime = Interpreter(model_content=result, num_threads=1,
                          experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    runtime.allocate_tensors()
    for detail in runtime.get_input_details():
        runtime.set_tensor(detail['index'], np.zeros(detail['shape'], dtype=np.float32))
    runtime.invoke()
    for detail in runtime.get_output_details():
        if not np.isfinite(runtime.get_tensor(detail['index'])).all():
            raise RuntimeError('Nonfinite output on zero-input smoke test')
    if args.input.read_bytes() != data:
        raise RuntimeError('Source changed during conversion')
    report.update(input=str(args.input), output=str(args.output),
                  input_sha256=hashlib.sha256(data).hexdigest(),
                  output_sha256=hashlib.sha256(result).hexdigest(),
                  input_bytes=len(data), output_bytes=len(result),
                  size_ratio=len(result)/len(data),
                  zero_input_builtin_invoke='passed',
                  precision='FP16 constant storage; original FP32 operation inputs/outputs retained',
                  limitation='Real model mAP and C++ TFLite 2.20/XNNPACK execution still require validation')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f:
        f.write(result)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'constants'}, indent=2))
    print('Report:', report_path)


if __name__ == '__main__':
    main()
