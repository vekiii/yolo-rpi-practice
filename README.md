# YOLOv8n eksperimenti

## Rezultati

| Skup | Lokalni artefakti | Izveštaj |
|---|---|---|
| Evaluacija na 50 slika | `results/eval_50/` | [Rezultati evaluacije](docs/results/results.md) |
| Evaluacija na 5000 slika | `results/eval_5000/` | [Rezultati evaluacije](docs/results/results.md) |
| 5000 ponavljanja iste slike | `results/single_image_5000/` | [Benchmark jedne slike](docs/results/single_image_results.md) |

## Modeli

- PyTorch checkpoint je u `models/yolov8n.pt`.
- TFLite modeli su u `models/tflite_fp32/`, `models/tflite_fp16_weights/` i `models/tflite_int8/`.
- `models/tflite_fp32/yolov8n.tflite` je sačuvana kopija originalnog FP32 izvoza. SHA-256 poklapa se sa `yolov8n_fp32.tflite`.

## Uputstva

- [Windows build i TFLite FP32](docs/guides/tflite_wrapper.md)
- [Aktiviranje MSVC okruženja](docs/guides/aktiviranje_cpp.md)
- [Raspberry Pi build i evaluacija](docs/guides/rpi_tflite_build.md)
- [Konverzija FP32 u FP16 zapis težina](docs/guides/tflite_fp32_to_fp16.md)
- [INT8 kvantizacija](docs/guides/tflite_int8.md) i [priprema kalibracionog skupa](calibration/README.md)
- [Dokumentacija C++ wrappera](tflite_cpp_wrapper/README.md) i [evidencija rada](docs/evidance.md)

Komande iz uputstava koriste putanje relativne prema korenu projekta, osim kada je naveden drugi radni direktorijum.
