# YOLOv8n NCNN FP32 CPU wrapper

Extract this directory into D:\yolo_test\ncnn_fp32_wrapper. Run every command below from D:\yolo_test in the VS developer PowerShell already activated for MSVC x64.

## Configure and build (first run downloads source dependencies)

```powershell
cmake -S .\ncnn_fp32_wrapper -B .\ncnn_fp32_wrapper\build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build .\ncnn_fp32_wrapper\build --parallel 4
```

Dependencies are built from source: NCNN tag 20250503, OpenCV 4.11.0 (core/imgproc/imgcodecs), nlohmann/json 3.11.3. Internet access is required. No Python environment is used by the C++ executable. The first build can take substantial time and disk space. Subsequent builds reuse the dependencies. If export from a newer NCNN toolchain uses unsupported operators, loading will fail explicitly: match the runtime version to the exporter rather than accepting incorrect results.

## Prepare exactly the same 50 images as the Python baseline

```powershell
.\.venv\Scripts\python.exe .\ncnn_fp32_wrapper\coco_tools.py prepare
```

## Run C++ inference

```powershell
.\ncnn_fp32_wrapper\build\yolo_ncnn.exe .\yolov8n_ncnn_model .\val2017 .\results\manifest_50.json .\results\ncnn_fp32_50 4
```

Arguments: model directory, image directory, manifest, output prefix, thread count. Outputs: PREFIX_predictions.json and PREFIX_timing.json. Both files are overwritten if the command is repeated with the same prefix. The original Python evaluation files are not modified.

## Evaluate predictions

```powershell
.\.venv\Scripts\python.exe .\ncnn_fp32_wrapper\coco_tools.py evaluate --predictions .\results\ncnn_fp32_50_predictions.json
```

Compare mAP against evaluation_50.json (not against the 5,000-image result yet). Inference should report out0 dims=2, with w/h equal to 8400/84 or 84/8400. Any other layout causes a clear error instead of silent decoding.

## Full evaluation, only after the 50-image check

```powershell
.\.venv\Scripts\python.exe .\ncnn_fp32_wrapper\coco_tools.py prepare --limit 0 --manifest results/manifest_5000.json
.\ncnn_fp32_wrapper\build\yolo_ncnn.exe .\yolov8n_ncnn_model .\val2017 .\results\manifest_5000.json .\results\ncnn_fp32_5000 4
.\.venv\Scripts\python.exe .\ncnn_fp32_wrapper\coco_tools.py evaluate --manifest results/manifest_5000.json --predictions results/ncnn_fp32_5000_predictions.json --output results/ncnn_fp32_5000_metrics.json
```

## Measurement and matching details

- CPU only; FP16 packed/storage/arithmetic, BF16 storage, INT8 inference disabled.
- FP32 model required. No training or quantization is performed.
- RGB, normalization 1/255, fixed square 640x640 letterbox, padding value 114, OpenCV linear resize. Rounding uses nearbyint (nearest-even under the default rounding mode), matching Python round for these values.
- Model output is decoded as pixel-space center-x/center-y/width/height plus 80 probabilities. The export already contains sigmoid and box decoding; do not apply either again. Output shape is checked but semantic correctness still needs the 50-image comparison.
- Single best class per anchor, confidence >= 0.001, class-aware NMS IoU > 0.7 suppressed, maximum 100 final detections. Border boxes are clipped after NMS, as in the baseline. Minor differences can arise for tied scores and resize/numerical implementations.
- Deterministic ascending COCO image IDs; noncontiguous category IDs mapped by class names.
- Five warmup passes excluded. Stage times include extractor creation in inference. Total includes image load, preprocessing, inference, postprocessing; excludes console output, JSON assembly/serialization and COCO evaluation.
- Native C++ total and Python predict-call time are approximate application-level comparisons: they are not instruction-identical measurement boundaries. Do not call any gain a pure language effect; runtime changes too.
- Four inference threads. Pin power conditions and record compiler/build/version details for final runs. Repeat tests; this first pass is a correctness check.
- No image rendering or RAM profiling is included in timed runs.

Add ncnn_fp32_wrapper/build/ to .gitignore. Keep source files tracked.

## Validation performed by the author

Python syntax and argument parser checked. A synthetic COCO fixture can verify the helper independently. The C++ build and actual exported model have not been run in the author's environment; verify on Windows before treating measurements as final. Do not assume matching mAP until compared.
