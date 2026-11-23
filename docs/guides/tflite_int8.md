# Statička INT8 kvantizacija YOLOv8n TFLite

Polazni model je `models/tflite_fp32/yolov8n_fp32.tflite`. Skripta proverava njegov SHA-256 prema postojećem FP32 ugovoru i čita ime signature i ime njenog ulaza iz modela. Windows wrapper nastavlja da prima FP32 tenzor `[1,3,640,640]` i da čita FP32 izlaz `[1,84,8400]` sa normalizovanim koordinatama.

## WSL kalibracija i izvoz

```bash
cd /mnt/d/yolo_test
source ~/venvs/yolo-tflite-export-py312/bin/activate
python -m unittest discover -s tflite_cpp_wrapper -p 'test_quantize_int8.py'
python -u tflite_cpp_wrapper/quantize_int8.py
```

Koriste se postojeći Ultralytics 8.4.208, litert-torch 0.10.0, ai-edge-litert 2.3.0 i ai-edge-quantizer 0.10.1. Skripta uzima svih 500 stavki manifesta po njegovom redosledu, proverava izbor, presek sa val2017, dekodiranje, dimenzije i hash vrednosti. Preprocessing prati `prepare()` iz wrappera: FP32 računanje odnosa, zaokruživanje na parno, linearni resize, letterbox sa vrednošću 114, BGR u RGB, deljenje sa 255 i NCHW. Privremeni ulazni tenzori koriste oko 2,46 GB prostora na disku. Kvantizator 0.10.1 prvo enumerira ulaze da izbroji posao, zatim prolazi kroz njih radi kalibracije. Prvi prolaz dobija poglede u memorijski mapiranu datoteku.

Direktni poziv prati instalirani Ultralytics `quantize=8`: `static_wi8_ai8()`, algoritam `min_max_uniform_quantize`, `CALIBRATION_PROFILER_BASED` sa 16 niti i `NO_QUANTIZE` za `INPUT` i `OUTPUT`. Profiler meri minimum i maksimum aktivacija po tenzoru za svaku sliku. Kalibrator spaja opsege metodom **EMA min–max sa faktorom 0,95**. Prva izmerena vrednost inicijalizuje opseg, a svaka naredna ažurira minimum i maksimum zasebno kao `0.95 * prethodna + 0.05 * nova`. Recept traži simetrične INT8 težine po kanalu i asimetrične INT8 aktivacije po tenzoru za podržane operacije. Instalirani API nudi i režim `CALIBRATION_PRESERVE_ALL_TENSORS`. Popis stvarnih tipova po operatorima i granice konverzije zapisani su u `inspection.json`. Pokrivenost konvolucija opisuje tipove tenzora u grafu. Preciznost pojedinačnih delova detekcione glave i izvršavanje pojedinačnih kernela preko XNNPACK-a zahtevaju zasebnu analizu.

Izlazi u `models/tflite_int8/` su `yolov8n_int8.tflite`, `calibration_qsv.json`, `recipe.json` i `inspection.json`. QSV se čuva u formatu koji podržava `load_calibration_results()`. Za ponovno kvantizovanje sa istim modelom, manifestom i postavkama koristi novi izlazni direktorijum.

```bash
python tflite_cpp_wrapper/quantize_int8.py \
  --reuse-qsv models/tflite_int8/calibration_qsv.json \
  --output-dir models/tflite_int8_reuse
```

## Windows provera na 50 slika

Iz PowerShell-a u `D:\yolo_test`, sa postojećim Windows izvršnim fajlom. Ako već imaš rezultate pod ovim prefiksom, izaberi drugi prefiks i odgovarajuća imena za COCO evaluaciju.

```powershell
.\tflite_cpp_wrapper\build\bin\yolo_tflite.exe .\models\tflite_int8\yolov8n_int8.tflite .\val2017 .\results\eval_50\manifest_50.json .\results\eval_50\tflite_int8_50 4 normalized xnnpack
if ($LASTEXITCODE -ne 0) { throw "TFLite C++ izvršavanje nije uspelo" }
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\coco_tools.py evaluate --manifest .\results\eval_50\manifest_50.json --predictions .\results\eval_50\tflite_int8_50_predictions.json --output .\results\eval_50\tflite_int8_50_metrics.json
```

## Dobijeni model i rezultati

Kalibrator je obradio svih 500 slika. Na prvih pet slika upoređen je WSL Python preprocessing sa funkcijom `prepare()` kompajliranom uz Windows OpenCV 4.11.0. Tenzori se poklapaju vrednost po vrednost. Model je prošao Python alokaciju i izvršavanje sa konačnim izlaznim vrednostima.

Ponovni izvoz iz sačuvanog QSV-a proizveo je identične bajtove modela bez nove kalibracije. SHA-256 izvornog FP32 modela i manifesta ostali su isti.

Model ima 3.483.792 bajta, odnosno 72,87% manje od FP32 modela. SHA-256 je `760da7cfe6da9f4ff46caf9e4ddfe04f242d621c7f11da3273004a51c572d107`. Graf sadrži 356 INT8, 90 INT32 i dva FP32 tenzora. Svih 64 konvolucija imaju INT8 težine i INT8 ulaznu i izlaznu aktivaciju. Kod 63 konvolucije težine imaju više skala, a kod jedne jednu skalu. Aktivacije svih konvolucija imaju po jednu skalu. Graf ima 32 `QUANTIZE` i jedan `DEQUANTIZE` operator. FP32 tip se pojavljuje samo na ulaznoj i izlaznoj granici. Detaljan pregled je u [inspection.json](../../models/tflite_int8/inspection.json).

Windows TFLite C++ 2.20.0 sa XNNPACK-om i četiri niti obradio je istih 50 slika istim redosledom kao sačuvana FP32 referenca.

| Metrika | FP32 | INT8 (EMA min–max, 0,95), FP32 ulaz/izlaz | Promena |
|---|---:|---:|---:|
| mAP50–95 | 0,410426 | 0,339151 | −7,13 procentnih poena |
| mAP50 | 0,552408 | 0,522214 | −3,02 procentna poena |
| mAP75 | 0,453771 | 0,381892 | −7,19 procentnih poena |
| Prosečan inference, ms/slika | 85,432 | 70,550 | −17,42% |
| Prosečna ukupna obrada, ms/slika | 95,221 | 84,636 | −11,12% |

Tabela poredi dva sačuvana prolaza. Ukupno vreme uključuje učitavanje slike. Pad tačnosti je zabeležen uz isti recept i bez posebnih izuzetaka za detekcionu glavu. Izvori su [COCO metrike](../../results/eval_50/tflite_int8_50_metrics.json), [vremena](../../results/eval_50/tflite_int8_50_timing.json), [predikcije](../../results/eval_50/tflite_int8_50_predictions.json), [Windows log](../../results/eval_50/tflite_int8_50_windows.log) i [log konverzije](../../results/eval_50/tflite_int8_conversion.log).
