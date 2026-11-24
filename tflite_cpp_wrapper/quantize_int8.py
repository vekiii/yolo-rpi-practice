"""Quantize the verified YOLOv8n TFLite FP32 model using the fixed COCO subset."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np

from prepare_calibration import COUNT, ROOT, TRAIN, VAL, load_manifest, read_images, sha256, verify

SOURCE = Path("models/tflite_fp32/yolov8n_fp32.tflite")
OUTPUT = Path("models/tflite_int8")
VERSIONS = {"ultralytics": "8.4.208", "litert-torch": "0.10.0",
            "ai-edge-litert": "2.3.0", "ai-edge-quantizer": "0.10.1"}
INPUT_SHAPE = (1, 3, 640, 640)
OUTPUT_SHAPE = (1, 84, 8400)


def preprocess(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV cannot decode {path}")
    height, width = image.shape[:2]
    ratio = min(np.float32(640) / np.float32(width), np.float32(640) / np.float32(height))
    w = int(np.rint(np.float32(width) * ratio))
    h = int(np.rint(np.float32(height) * ratio))
    left = round((640 - w) / 2.0 - 0.1)
    top = round((640 - h) / 2.0 - 0.1)
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, 640 - h - top, left, 640 - w - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)


def fill_inputs(records, root, storage):
    count = len(records)
    if not count or len({row["id"] for row in records}) != count or len(storage) != count:
        raise ValueError("Expected one tensor slot per distinct calibration image")
    for index, row in enumerate(records):
        storage[index] = preprocess(root / row["local_path"])
        if (index + 1) % 100 == 0:
            print(f"Prepared {index + 1}/{count} calibration inputs", flush=True)


class CalibrationInputs:
    """Reiterable views over disk-backed input tensors for quantizer 0.10.1."""

    def __init__(self, storage, input_key):
        self.storage = storage
        self.input_key = input_key
        self.pass_counts = []

    def __iter__(self):
        pass_index = len(self.pass_counts)
        self.pass_counts.append(0)
        for tensor in self.storage:
            self.pass_counts[pass_index] += 1
            yield {self.input_key: tensor}
            count = self.pass_counts[pass_index]
            if pass_index == 1 and count % 50 == 0:
                print(f"Calibrated {count}/{len(self.storage)}", flush=True)


def write_json(path, payload, **kwargs):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, **kwargs)
        stream.write("\n")


def io_details(interpreter):
    interpreter.allocate_tensors()
    details = {"inputs": interpreter.get_input_details(), "outputs": interpreter.get_output_details()}
    for key, shape in (("inputs", INPUT_SHAPE), ("outputs", OUTPUT_SHAPE)):
        tensors = details[key]
        if len(tensors) != 1 or tuple(tensors[0]["shape"]) != shape or tensors[0]["dtype"] != np.float32:
            raise ValueError(f"Unexpected {key} contract: {tensors}")
    return {
        key: [
            {"name": t["name"], "index": t["index"], "shape": t["shape"].tolist(),
             "dtype": np.dtype(t["dtype"]).name,
             "quantization": {k: v.tolist() if isinstance(v, np.ndarray) else v
                              for k, v in t["quantization_parameters"].items()}}
            for t in tensors
        ] for key, tensors in details.items()
    }


def inspect_graph(path):
    from ai_edge_litert import schema_py_generated as schema

    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(path.read_bytes(), 0))
    types = {value: name for name, value in vars(schema.TensorType).items() if isinstance(value, int)}
    codes = {value: name for name, value in vars(schema.BuiltinOperator).items() if isinstance(value, int)}
    tensors, ops = Counter(), Counter()
    float_ops, boundaries, convolutions = [], [], []
    for graph_index, graph in enumerate(model.subgraphs):
        tensors.update(types[t.type] for t in graph.tensors)

        def tensor(index):
            if index < 0:
                return None
            t = graph.tensors[index]
            q = t.quantization
            return {"index": int(index), "name": t.name.decode("utf-8") if t.name else "",
                    "type": types[t.type], "scales": 0 if q is None or q.scale is None else len(q.scale),
                    "quantized_dimension": None if q is None or q.scale is None else q.quantizedDimension}

        for index, op in enumerate(graph.operators):
            code = model.operatorCodes[op.opcodeIndex]
            name = codes.get(max(code.builtinCode, code.deprecatedBuiltinCode), "UNKNOWN")
            if name == "CUSTOM" and code.customCode:
                name += ":" + code.customCode.decode("utf-8")
            inputs = [tensor(i) for i in op.inputs] if op.inputs is not None else []
            outputs = [tensor(i) for i in op.outputs] if op.outputs is not None else []
            in_types = tuple(t["type"] if t else "-" for t in inputs)
            out_types = tuple(t["type"] if t else "-" for t in outputs)
            ops[(name, code.version, in_types, out_types)] += 1
            entry = dict(subgraph=graph_index, index=index, operation=name, inputs=inputs, outputs=outputs)
            if name in ("QUANTIZE", "DEQUANTIZE"):
                boundaries.append(entry)
            elif "FLOAT32" in in_types or "FLOAT32" in out_types:
                float_ops.append(entry)
            if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                convolutions.append(dict(entry, int8_weights=in_types[1] == "INT8",
                                         int8_activations=in_types[0] == "INT8" and out_types[0] == "INT8"))
    return {
        "tensor_counts_by_type": dict(tensors),
        "operation_signatures": [dict(operation=name, version=ver, inputs=ins, outputs=outs, count=count)
                                 for (name, ver, ins, outs), count in sorted(ops.items())],
        "convolution_coverage": {
            "total": len(convolutions),
            "int8_weights": sum(op["int8_weights"] for op in convolutions),
            "int8_activations": sum(op["int8_activations"] for op in convolutions),
            "int8_weights_and_activations": sum(op["int8_weights"] and op["int8_activations"]
                                               for op in convolutions),
            "operators": convolutions,
        },
        "operators_with_float32_io_excluding_conversion_boundaries": float_ops,
        "quantize_dequantize_boundaries": boundaries,
    }


def calibrate_qsv(qt, dataset, threads, mode, aggregation):
    if aggregation == "ema":
        return qt.calibrate(dataset, num_threads=threads, mode=mode)

    from ai_edge_quantizer import calibrator, recipe_manager
    from ai_edge_quantizer.utils import qsv_utils

    # Quantizer.calibrate() uses the default merger. The Calibrator factory
    # accepts a custom merger with the same profiler and recipe.
    manager = recipe_manager.RecipeManager()
    manager.load_quantization_recipe(qt.get_quantization_recipe())
    profiler = calibrator.Calibrator(str(SOURCE), num_threads=threads, mode=mode,
                                     qsv_update_func=qsv_utils.min_max_update)
    profiler.calibrate(dataset, manager)
    return profiler.get_model_qsvs()


def run(output, reuse_qsv, threads, aggregation="ema", manifest_path=ROOT / "manifest.json",
        scratch_dir=None):
    from ai_edge_litert.interpreter import Interpreter
    from ai_edge_quantizer import calibrator, qtyping, quantizer, recipe
    from ai_edge_quantizer.utils import calibration_utils

    packages = {name: version(name) for name in VERSIONS}
    if packages != VERSIONS:
        raise ValueError(f"Unexpected installed versions: {packages}. Expected: {VERSIONS}")
    root = manifest_path.parent
    source_hash = sha256(SOURCE)
    contract = json.loads(SOURCE.with_suffix(".contract.json").read_text(encoding="utf-8"))
    if source_hash != contract["model_sha256"]:
        raise ValueError("FP32 model does not match the verified contract")
    manifest_hash = sha256(manifest_path)
    if manifest_path.name != "manifest.json":
        raise ValueError("Expected a calibration manifest named manifest.json")
    if manifest_path.resolve() == (ROOT / "manifest.json").resolve():
        # Preserve the original manifest path in existing QSV provenance.
        manifest_path = ROOT / "manifest.json"
        root = ROOT
        manifest = load_manifest(read_images(TRAIN), read_images(VAL), sha256(TRAIN), root, create=False)
        per_class = False
    else:
        from prepare_calibration_per_class import load_plan

        manifest, _ = load_plan(root)
        per_class = True
    records = manifest["images"]
    count = len(records)

    source = Interpreter(model_path=str(SOURCE), num_threads=threads)
    source_io = io_details(source)
    signatures = source.get_signature_list()
    if len(signatures) != 1:
        raise ValueError(f"Expected one model signature: {signatures}")
    signature, definition = next(iter(signatures.items()))
    if len(definition["inputs"]) != 1:
        raise ValueError(f"Expected one signature input: {definition}")
    input_key = definition["inputs"][0]
    del source

    qt = quantizer.Quantizer(str(SOURCE))
    qt.load_quantization_recipe(recipe.static_wi8_ai8())
    for op in (qtyping.TFLOperationName.INPUT, qtyping.TFLOperationName.OUTPUT):
        qt.update_quantization_recipe(regex=".*", operation_name=op,
                                      algorithm_key=recipe.AlgorithmName.NO_QUANTIZE)
    mode = calibrator.CalibrationMode.CALIBRATION_PROFILER_BASED
    provenance = {
        "source_model": str(SOURCE), "source_sha256": source_hash,
        "manifest": str(manifest_path), "manifest_sha256": manifest_hash,
        "image_ids": [row["id"] for row in records], "seed": manifest["seed"],
        "packages": packages, "opencv": cv2.__version__, "numpy": np.__version__,
        "recipe": "recipe.static_wi8_ai8() + NO_QUANTIZE for INPUT and OUTPUT",
        "mode": mode.name, "threads": threads, "signature": signature, "signature_input": input_key,
    }
    if not per_class and aggregation == "global_minmax":
        ema_report = json.loads((OUTPUT / "inspection.json").read_text(encoding="utf-8"))
        if ema_report["provenance"] != provenance or ema_report["num_samples_calibrated"] != COUNT:
            raise ValueError("Calibration inputs or settings differ from the saved EMA experiment")
    if aggregation == "global_minmax":
        provenance["qsv_aggregation"] = "global_minmax"
        provenance["qsv_update_function"] = "ai_edge_quantizer.utils.qsv_utils.min_max_update"
    if per_class:
        reference_dir = OUTPUT if aggregation == "ema" else OUTPUT.with_name("tflite_int8_global_minmax")
        reference = json.loads((reference_dir / "inspection.json").read_bytes())["provenance"]
        data_keys = {"manifest", "manifest_sha256", "image_ids"}
        if {k: v for k, v in reference.items() if k not in data_keys} != {
                k: v for k, v in provenance.items() if k not in data_keys}:
            raise ValueError("Quantization settings differ from the corresponding random-500 reference")
        provenance["calibration_schema"] = manifest["schema_version"]
        provenance["qsv_aggregation"] = aggregation
        if aggregation == "ema":
            provenance["ema_smoothing_factor"] = 0.95
    qsv_path = output / "calibration_qsv.json"
    if reuse_qsv:
        qsvs, saved = calibration_utils.load_calibration_results(str(reuse_qsv))
        if saved.get("provenance") != provenance or saved.get("num_samples_calibrated") != count or not qsvs:
            raise ValueError("Saved calibration belongs to a different source, manifest or configuration")
        metadata = saved
    verify(root, manifest)
    if not reuse_qsv:
        scratch_parent = scratch_dir or output.parent
        scratch_parent.mkdir(parents=True, exist_ok=True)
        required = count * int(np.prod(INPUT_SHAPE)) * np.dtype(np.float32).itemsize
        if shutil.disk_usage(scratch_parent).free < required:
            raise ValueError(f"Need at least {required / 2**30:.2f} GiB free at {scratch_parent}; "
                             "use --scratch-dir on a volume with sufficient space")
    output.mkdir(parents=True, exist_ok=False)
    if not reuse_qsv:
        # The installed calibrator materializes list(dataset) to count its work,
        # then iterates again to invoke. The first pass holds views, not RAM copies.
        with tempfile.TemporaryDirectory(prefix="calibration_inputs_", dir=scratch_dir or output) as scratch:
            storage = np.memmap(Path(scratch) / "inputs.dat", dtype=np.float32, mode="w+",
                                shape=(count, *INPUT_SHAPE))
            fill_inputs(records, root, storage)
            storage.flush()
            inputs = CalibrationInputs(storage, input_key)
            qsvs = calibrate_qsv(qt, {signature: inputs}, threads, mode, aggregation)
            if inputs.pass_counts != [count, count] or not qsvs:
                raise RuntimeError(f"Unexpected calibration dataset consumption: {inputs.pass_counts}")
            metadata = dict(num_samples_calibrated=inputs.pass_counts[1], provenance=provenance)
            del inputs, storage
    if sha256(SOURCE) != source_hash or sha256(manifest_path) != manifest_hash:
        raise RuntimeError("The FP32 source or manifest changed during calibration")
    write_json(qsv_path, dict(model_qsvs=qsvs, metadata=metadata), cls=calibration_utils.NumpyEncoder)

    result = qt.quantize(calibration_result=qsvs)
    model = output / "yolov8n_int8.tflite"
    result.export_model(str(model), overwrite=False)
    write_json(output / "recipe.json", result.recipe)
    report = dict(provenance=provenance, num_samples_calibrated=metadata["num_samples_calibrated"],
                  reused_qsv=str(reuse_qsv) if reuse_qsv else None, source_io=source_io,
                  model=str(model), model_bytes=model.stat().st_size, sha256=sha256(model),
                  qsv_sha256=sha256(qsv_path), graph=inspect_graph(model))
    try:
        interpreter = Interpreter(model_path=str(model), num_threads=threads)
        report["io"] = io_details(interpreter)
        interpreter.set_tensor(report["io"]["inputs"][0]["index"],
                               preprocess(root / records[0]["local_path"]))
        interpreter.invoke()
        returned = interpreter.get_tensor(report["io"]["outputs"][0]["index"])
        if not np.isfinite(returned).all():
            raise ValueError("Model produced non-finite output")
        report["python_execution"] = {"status": "passed", "image_id": records[0]["id"]}
    except Exception as error:
        report["python_execution"] = {"status": "failed", "error": str(error)}
        raise
    finally:
        write_json(output / "inspection.json", report)
    print(f"Created {model} ({model.stat().st_size} bytes)", flush=True)


def main(default_output=OUTPUT, aggregation="ema"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output, help="New directory; must not exist")
    parser.add_argument("--reuse-qsv", type=Path, help="Reuse a matching calibration_qsv.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    parser.add_argument("--scratch-dir", type=Path, help="Directory for temporary disk-backed tensors")
    parser.add_argument("--threads", type=int, default=16, help="Calibration and Python check threads")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Expected a positive thread count")
    if args.manifest.resolve() != (ROOT / "manifest.json").resolve() and args.output_dir == default_output:
        parser.error("Use --output-dir to give the new manifest a separate output directory")
    run(args.output_dir, args.reuse_qsv, args.threads, aggregation, args.manifest, args.scratch_dir)


if __name__ == "__main__":
    main()
