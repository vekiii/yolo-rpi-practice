$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $model = 'models/tflite_int8_global_minmax/yolov8n_int8.tflite'
    $manifest = 'results/eval_50/manifest_50.json'
    $emaTiming = 'results/eval_50/tflite_int8_50_timing.json'
    $prefix = 'results/eval_50/tflite_int8_global_minmax_50'
    $wrapper = 'tflite_cpp_wrapper/build/bin/yolo_tflite.exe'
    $python = '.venv/Scripts/python.exe'
    foreach ($path in @($model, $manifest, $emaTiming, $wrapper, $python)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Missing file: $path"
        }
    }
    $images = @( (Get-Content -Raw $manifest | ConvertFrom-Json).images.id )
    $baseline = Get-Content -Raw $emaTiming | ConvertFrom-Json
    $emaImages = @( $baseline.per_image.image_id )
    if ($images.Count -ne 50 -or ($images -join ',') -ne ($emaImages -join ',')) {
        throw 'The manifest must contain the same 50 image IDs in the same order as the EMA evaluation.'
    }
    foreach ($suffix in @('predictions', 'timing', 'metrics')) {
        if (Test-Path -LiteralPath "${prefix}_$suffix.json") {
            throw "Output already exists: ${prefix}_$suffix.json"
        }
    }
    & $wrapper $model 'val2017' $manifest $prefix 4 normalized xnnpack
    if ($LASTEXITCODE -ne 0) { throw "C++ inference failed with exit code $LASTEXITCODE" }
    & $python 'tflite_cpp_wrapper/coco_tools.py' evaluate --manifest $manifest --predictions "${prefix}_predictions.json" --output "${prefix}_metrics.json"
    if ($LASTEXITCODE -ne 0) { throw "COCO evaluation failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
