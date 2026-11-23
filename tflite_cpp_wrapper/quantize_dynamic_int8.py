"""Export and inspect a dynamic INT8 TFLite model from the original FP32 export."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import shlex
import sys

import numpy as np

from quantize_int8 import INPUT_SHAPE, OUTPUT_SHAPE, SOURCE, inspect_graph, io_details, sha256, write_json

OUTPUT_DIR = Path("models/tflite_dynamic_int8")
PACKAGES = {"ai-edge-quantizer": "0.10.1", "ai-edge-litert": "2.3.0"}
RECIPE_NAME = "dynamic_wi8_afp32"


def checked_source_io(interpreter, contract):
    details = io_details(interpreter)
    for direction in ("inputs", "outputs"):
        expected = contract[direction]
        actual = details[direction]
        if len(expected) != len(actual):
            raise ValueError(f"{direction} count differs from FP32 contract")
        for original, tensor in zip(expected, actual):
            if (original["name"] != tensor["name"] or original["shape"] != tensor["shape"]
                    or original["dtype"] not in ("float32", "<class 'numpy.float32'>")
                    or original["quantization"] != [0.0, 0]
                    or tensor["quantization"]["scales"] or tensor["quantization"]["zero_points"]):
                raise ValueError(f"{direction} differ from FP32 contract: {tensor}")
    return details


def operation_counts(graph):
    counts = Counter()
    for item in graph["operation_signatures"]:
        counts[item["operation"]] += item["count"]
    return dict(sorted(counts.items()))


def convolution_parameters(path, graph, schema):
    """Read full filter parameters and all serialized FP32 operands in the graph."""
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(path.read_bytes(), 0))
    filters = []
    for op in graph["convolution_coverage"]["operators"]:
        activation, weight, *other_inputs = op["inputs"]
        node = model.subgraphs[op["subgraph"]].operators[op["index"]]
        tensor = model.subgraphs[op["subgraph"]].tensors[weight["index"]]
        quant = tensor.quantization
        filters.append({
            "subgraph": op["subgraph"], "operator_index": op["index"], "operation": op["operation"],
            # FlatBuffer operators have no separate layer name; use the original filter name.
            "name": weight["name"], "type": weight["type"],
            "operator_version": model.operatorCodes[node.opcodeIndex].version,
            "shape": tensor.shape.tolist(),
            "activation_input_type": activation["type"], "activation_output_type": op["outputs"][0]["type"],
            "bias_type": other_inputs[0]["type"] if other_inputs and other_inputs[0] else None,
            "scales": [] if quant is None or quant.scale is None else quant.scale.tolist(),
            "zero_points": [] if quant is None or quant.zeroPoint is None else quant.zeroPoint.tolist(),
            "quantized_dimension": weight["quantized_dimension"],
        })

    fp32_operands = []
    codes = {value: key for key, value in vars(schema.BuiltinOperator).items() if isinstance(value, int)}
    conv_codes = {schema.BuiltinOperator.CONV_2D, schema.BuiltinOperator.DEPTHWISE_CONV_2D}
    for graph_index, subgraph in enumerate(model.subgraphs):
        for index, op in enumerate(subgraph.operators):
            opcode = model.operatorCodes[op.opcodeIndex]
            code = max(opcode.builtinCode, opcode.deprecatedBuiltinCode)
            if op.inputs is None:
                continue
            for slot, i in enumerate(op.inputs):
                if i < 0:
                    continue
                tensor = subgraph.tensors[i]
                data = model.buffers[tensor.buffer].data
                if tensor.type != schema.TensorType.FLOAT32 or data is None or not len(data):
                    continue
                if code in conv_codes and slot == 2:
                    reason = "The dynamic recipe keeps convolution bias FP32."
                elif code in conv_codes and slot == 1:
                    reason = "FP32 convolution filter remains; inspect the quantizer diagnostics."
                else:
                    reason = "Operand is outside the dynamic recipe's supported operator set for this graph."
                fp32_operands.append(dict(subgraph=graph_index, operator_index=index,
                                          operation=codes[code], input_slot=slot,
                                          name=tensor.name.decode("utf-8"), reason=reason))
    return filters, fp32_operands


def run(source_path, contract_path, output):
    from ai_edge_litert import interpreter, schema_py_generated
    from ai_edge_quantizer import quantizer, recipe

    versions = {package: version(package) for package in PACKAGES}
    if versions != PACKAGES:
        raise ValueError(f"Expected {PACKAGES}, found {versions}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing FP32 model: {source_path}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output directory must be new: {output}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    source_hash = sha256(source_path)
    if source_hash != contract["model_sha256"]:
        raise ValueError("Source SHA-256 differs from the original FP32 contract")

    source = interpreter.Interpreter(model_path=str(source_path), num_threads=4)
    original_io = checked_source_io(source, contract)
    del source
    original_graph = inspect_graph(source_path)
    if original_graph["convolution_coverage"]["total"] == 0:
        raise ValueError("FP32 graph contains no convolutions to quantize")
    original_filters, _ = convolution_parameters(source_path, original_graph, schema_py_generated)

    qt = quantizer.Quantizer(str(source_path))
    qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
    if qt.need_calibration:
        raise RuntimeError("Selected recipe unexpectedly requires calibration")

    output.mkdir(parents=True, exist_ok=False)
    model = output / "yolov8n_dynamic_int8.tflite"
    metadata = {
        "recipe": RECIPE_NAME, "calibration_used": False, "packages": versions,
        "module_paths": {m.__name__: m.__file__ for m in (recipe, quantizer, interpreter, schema_py_generated)},
        "python": sys.version, "python_executable": sys.executable,
        "command": shlex.join([sys.executable, *sys.argv]), "working_directory": str(Path.cwd()),
        "source": {"path": str(source_path), "sha256": source_hash, "bytes": source_path.stat().st_size},
        "contract": {"path": str(contract_path), "sha256": sha256(contract_path)},
    }
    write_json(output / "recipe.json", qt.get_quantization_recipe())
    report = {"source": {"io": original_io, "operation_counts": operation_counts(original_graph),
                         "operation_signatures": original_graph["operation_signatures"],
                         "convolution_filters": original_filters},
              "layout": {"input": "NCHW", "output": "batch, features, anchors"}}
    try:
        result = qt.quantize()
        if result.recipe != qt.get_quantization_recipe():
            raise RuntimeError("Exported recipe differs from the selected dynamic recipe")
        result.export_model(str(model), overwrite=False)
        metadata["output"] = {"path": str(model), "sha256": sha256(model), "bytes": model.stat().st_size}
        if sha256(source_path) != source_hash:
            raise RuntimeError("FP32 source changed during conversion")

        graph = inspect_graph(model)
        filters, fp32 = convolution_parameters(model, graph, schema_py_generated)
        counts = operation_counts(graph)
        report["output"] = {
            "operation_counts": counts, "operation_signatures": graph["operation_signatures"],
            "tensor_counts": graph["tensor_counts_by_type"], "convolution_filters": filters,
            "convolutions_by_filter_type": dict(Counter(f["type"] for f in filters)),
            "fp32_constant_operands": fp32, "quantize_dequantize_boundaries": graph["quantize_dequantize_boundaries"],
        }
        if not any(f["type"] == "INT8" for f in filters) or len(filters) != len(original_filters):
            raise RuntimeError("Unexpected convolution coverage in the dynamic graph")
        if counts.get("DEQUANTIZE", 0) or any(
            (f["activation_input_type"], f["activation_output_type"]) != ("FLOAT32", "FLOAT32") for f in filters
        ):
            raise RuntimeError("Graph does not have the expected dynamic convolution interfaces")

        # Test the standalone built-in resolver separately from the Windows delegate path.
        builtin = interpreter.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        runtime = interpreter.Interpreter(model_path=str(model), num_threads=4, experimental_op_resolver_type=builtin)
        report["output"]["io"] = checked_source_io(runtime, contract)
        sample = np.random.default_rng(42).random(INPUT_SHAPE, dtype=np.float32)
        runtime.set_tensor(report["output"]["io"]["inputs"][0]["index"], sample)
        runtime.invoke()
        output_tensor = runtime.get_tensor(report["output"]["io"]["outputs"][0]["index"])
        if output_tensor.shape != OUTPUT_SHAPE or not np.isfinite(output_tensor).all():
            raise ValueError("Unexpected output shape or non-finite output")
        report["python_execution"] = {"status": "passed", "resolver": builtin.name,
                                      "synthetic_input_seed": 42, "threads": 4,
                                      "output_min": float(output_tensor.min()), "output_max": float(output_tensor.max())}
        metadata["status"] = "passed"
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "metadata.json", metadata)
        write_json(output / "inspection.json", report)
    print(f"Exported {model} ({metadata['output']['bytes']} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--contract", type=Path, help="Defaults to the FP32 contract next to the source model")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    try:
        run(args.source, args.contract or args.source.with_suffix(".contract.json"), args.output_dir)
    except Exception as error:
        parser.exit(1, f"Dynamic INT8 export failed: {error}\n")


if __name__ == "__main__":
    main()
