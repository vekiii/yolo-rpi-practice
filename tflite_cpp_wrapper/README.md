# YOLOv8n TFLite C++ FP32 CPU experiment

Extract this folder into D:\yolo_test\tflite_cpp_wrapper. Existing dataset, weights, Python baseline and result files remain unchanged. All Windows commands below run from D:\yolo_test in the MSVC x64 developer PowerShell. This is a source project, not a prebuilt Windows executable.

## 1. Export: separate Linux environment (WSL Ubuntu or Colab CPU)

Current Ultralytics LiteRT export is supported on Linux x86_64/macOS, not native Windows. The resulting .tflite runs on Windows. Do not install the exporter into your working Windows .venv. An existing usable WSL/Ubuntu installation is assumed below; if unavailable, first set it up or use Colab CPU and transfer the original .pt plus export script there.

WSL Ubuntu shell:

```bash
cd /mnt/d/yolo_test
python3 -m venv ~/venvs/yolo-tflite-export
source ~/venvs/yolo-tflite-export/bin/activate
python -m pip install --upgrade pip
python -m pip install -r tflite_cpp_wrapper/requirements-export.txt
python tflite_cpp_wrapper/export_fp32.py
```

If venv is unavailable on Ubuntu: `sudo apt install python3-venv` first. Use a Python version supported by current exporter dependencies; Python 3.11/3.12 is a practical starting point. WSL accesses Windows project files directly. Colab users upload the original yolov8n.pt and export_fp32.py, install requirements, run the script, then download the entire models/tflite_fp32 output directory.

Expected outputs:
- models/tflite_fp32/yolov8n_fp32.tflite
- models/tflite_fp32/yolov8n_fp32.contract.json (tensor shapes, hashes, exporter version)
- models/tflite_fp32/yolov8n_fp32.environment.txt (exact package versions)

Inspect the contract: FP32 input/output, input [1,3,640,640] or [1,640,640,3], raw output [1,84,8400] or [1,8400,84]. The wrapper rejects other layouts/dtypes instead of guessing. The provided export uses normalized box coordinates. The C++ command therefore explicitly says `normalized`; a model from a different export route may require `pixels`. Never select this by whichever gives a nicer mAP.

Current export dependencies evolve; this requirements file is a version range, NOT a lock. Preserve the saved environment and model hashes before comparing runs. Do not upgrade packages mid-experiment. The C++ runtime is pinned separately to TensorFlow 2.20.0. If the model needs operators newer than this runtime, inference fails explicitly; align exporter/runtime before proceeding (no Flex/custom ops fallback is silently enabled).

## 2. Build the official TFLite Interpreter C++ API on Windows

```powershell
cmake -S .\tflite_cpp_wrapper -B .\tflite_cpp_wrapper\build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build .\tflite_cpp_wrapper\build --parallel 4
```

First build fetches TensorFlow v2.20.0, OpenCV 4.11.0 (core/imgproc/imgcodecs), nlohmann/json 3.11.3 and TFLite transitive dependencies. Requires internet and several GB disk space; expect a lengthy build. GPU/NNAPI are disabled. CMake 4 compatibility policy minimum 3.5 is set for older dependency files. If a dependency still fails, retain the FIRST error and its preceding context; do not delete source/model/dataset folders.

Expected binary with this NMake configuration: tflite_cpp_wrapper/build/bin/yolo_tflite.exe.

## 3. Prepare and evaluate 50 images

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\coco_tools.py prepare
.\tflite_cpp_wrapper\build\bin\yolo_tflite.exe .\models\tflite_fp32\yolov8n_fp32.tflite .\val2017 .\results\manifest_50.json .\results\tflite_fp32_50 4 normalized xnnpack
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\coco_tools.py evaluate --predictions results/tflite_fp32_50_predictions.json
```

Compare against Python's 50-image mAP50-95 0.4104 and mAP50 0.5524. Similar aggregate metrics support correct conversion/decoding but do not prove bitwise identical predictions. If metrics collapse, check contract, layouts, coordinate units, class mapping and preprocessing BEFORE full evaluation.

Explicit XNNPACK CPU delegate, four threads, no forced FP16. Unsupported nodes can fall back to built-in TFLite CPU kernels; this is not proof all nodes are delegated. Use backend `builtin` instead of `xnnpack` for an explicitly separate CPU-kernel diagnostic run and choose another output prefix. Report backend with all results.

## 4. Full 5,000-image run after correctness check

```powershell
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\coco_tools.py prepare --limit 0 --manifest results/manifest_5000.json
.\tflite_cpp_wrapper\build\bin\yolo_tflite.exe .\models\tflite_fp32\yolov8n_fp32.tflite .\val2017 .\results\manifest_5000.json .\results\tflite_fp32_5000 4 normalized xnnpack
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\coco_tools.py evaluate --manifest results/manifest_5000.json --predictions results/tflite_fp32_5000_predictions.json --output results/tflite_fp32_5000_metrics.json
```

Reference Python/PyTorch full-set mAP50-95: 0.3657. Repeat latency measurements under fixed power/thermal/background-load conditions before drawing speed conclusions.

## Output and measurement contract

- PREFIX_predictions.json: COCO detection results (overwritten on same prefix).
- PREFIX_timing.json: model bytes, runtime version, backend, threads, layouts, coordinate units, per-image timings and mean timings.
- Metrics JSON from coco_tools.py; limit=0 selects the entire validation set.
- Five warmup invocations, excluded. One image at a time, 640x640 square letterbox, RGB /255, padding 114, OpenCV linear resize.
- Best class per anchor, confidence >=.001, class-aware NMS at .7 IoU, cap 100 detections, clip boxes after NMS. Rounding uses nearbyint (nearest-even under default rounding).
- Total time includes disk load/decode, preparation, Invoke, postprocessing; excludes logging, JSON serialization and COCO evaluation. Interpreter/model loading and delegate setup excluded. Total timing boundaries differ slightly from Python predict calls; this is an implementation+runtime comparison, not a pure language comparison.
- Flat output assumed decoded xywh + 80 probabilities; no second sigmoid, anchor decoding or embedded NMS.
- This version implements FP32 I/O only. It does NOT produce FP16/INT8 models or claim support for quantized I/O. Add quantization only after the FP32 path is validated.
- Current Ultralytics LiteRT export does not separately export FP16 weights. That later experiment needs a separately chosen conversion path; GPU FP16 is not a substitute for the intended CPU experiment.
- C++ source is portable, but RPi ARM32 compilation, dependency support and model execution still need verification. Windows build artifacts cannot be copied as executables to RPi. No RPi setup commands are included yet.
- Add `models/` to root .gitignore if excluding generated model files; folder-local .gitignore excludes build artifacts. Keep sources, contracts/environment records and final summaries under version control as appropriate.

## Validation status

Python syntax/CLI and synthetic COCO evaluation checked where dependencies are available. C++ build and real model execution have not been verified on Windows in the author's environment. Follow stages in order and report configuration/build errors before proceeding. Do not treat this package as a tested binary distribution.

Sources:
https://developers.google.com/edge/litert/build/cmake
https://docs.ultralytics.com/integrations/litert/
https://docs.ultralytics.com/reference/utils/export/litert/
