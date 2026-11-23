# YOLOv8n TFLite sa dinamičkom INT8 kvantizacijom

Polazni model je originalni FP32 FlatBuffer od 12.841.242 bajta. Novi model ima **3.470.736 bajtova**, odnosno **72,97% manje**. Njegov SHA-256 je `72bce29e70214d0ab36f6bdf0501cc85131a0fcf02eb4dd2cea0832e0cc70e8d`. Skripta proverava izvor prema ugovoru i potvrđuje da mu se SHA-256 nije promenio tokom izvoza. [Model](yolov8n_dynamic_int8.tflite) · [Metapodaci](metadata.json) · [FP32 ugovor](../tflite_fp32/yolov8n_fp32.contract.json)

## Recept iz instaliranog paketa

`ai-edge-quantizer` 0.10.1 poziva `dynamic_wi8c_afp32()` iz `recipe.dynamic_wi8_afp32()`. Učitani recept koristi `min_max_uniform_quantize`, simetrične težine od 8 bita i granularnost `CHANNELWISE`. Minimum i maksimum računaju se iz samih težina. Recept postavlja `compute_precision=INTEGER`, `explicit_dequantize=false`, `skip_checks=false` i `min_weight_elements=0`. Aktivaciona konfiguracija nije zadata. `Quantizer.quantize()` se poziva bez kalibracionih statistika ili slika. [Primenjeni recept](recipe.json)

Lokalna politika podržava `CONV_2D`, `CONV_2D_TRANSPOSE`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, `BATCH_MATMUL` i `EMBEDDING_LOOKUP` za ovaj recept. Od njih je u izvornom YOLO grafu prisutan `CONV_2D`. Materijalizacija može preskočiti filter koji već ima kvantizacione parametre ili je ispod zadatog minimuma elemenata. Sa pragom nula filteri ovog modela nisu isključeni po veličini.

## Rezultat inspekcije

- Svih **64 od 64** `CONV_2D` operacija ima INT8 filtere sa nultim zero point-om i kvantizacionom dimenzijom 0. Kod 63 filtera postoji više skala, a kod jednog postoji jedna skala jer filter ima samo jedan izlazni kanal. Pregled grafa sadrži indeks operacije, naziv i oblik filtera i sve vrednosti skala i zero point-a.
- Sve konvolucije imaju FP32 ulazne i izlazne aktivacije. Preostalih 58 FP32 konvolucionih konstanti su bias vrednosti koje dinamički recept ostavlja u FP32. Još dve FP32 konstante ulaze u `MUL`, koji je van skupa podržanih operacija za ovaj recept.
- Oba grafa imaju 256 operacija sa istim brojem operacija po tipu i istim verzijama. `CONV_2D` je verzija 1, a `RESIZE_NEAREST_NEIGHBOR` verzija 3. Novi FlatBuffer nema eksplicitne `QUANTIZE` ili `DEQUANTIZE` operacije. Imena konvolucija u izveštaju predstavljena su nazivima njihovih filtera.
- Ulaz `serving_default_args_0` je FP32 i NCHW `[1,3,640,640]`. Izlaz `serving_default_output_0_output` je FP32 `[1,84,8400]`, sa redosledom batch, features, anchors. Oba tenzora su bez parametara kvantizacije i ugovor je očuvan. [Pregled izvornog i novog grafa](inspection.json)

## Provera izvršavanja

LiteRT 2.3.0 je uspešno alocirao tenzore i izvršio jednu inferencu sa sintetičkim ulazom i ugrađenim operatorima, bez podrazumevanih delegata. Izlaz je očekivanog oblika i sve vrednosti su konačne. Lokalni izvor TFLite 2.20.0 za ugrađeni `CONV_2D` sadrži hibridnu putanju koja kvantizuje FP32 aktivacije tokom izvršavanja kada su filteri INT8.

Postojeći Windows wrapper sa TFLite 2.20.0 i četiri niti prošao je evaluaciju na istih 50 slika, istim redosledom kao FP32. Log potvrđuje kreiranje XNNPACK delegata. Lokalna konfiguracija kompilacije ima `XNNPACK_DELEGATE_USE_LATEST_OPS`, a lokalni XNNPACK delegate sadrži putanju za dinamički kvantizovan `CONV_2D`. Broj delegiranih konvolucija i izbor pojedinačnih kernela nisu profilisani u ovom prolazu. [Windows log](../../results/eval_50/tflite_dynamic_int8_50_windows.log)

| Metrika na 50 slika | Sačuvani FP32 prolaz | Dinamički INT8 |
|---|---:|---:|
| mAP50–95 | 0,410426 | 0,409629 |
| mAP50 | 0,552408 | 0,555860 |
| mAP75 | 0,453771 | 0,478441 |
| Prosečan inference, ms/slika | 85,431748 | 79,165062 |
| Prosečna ukupna obrada, ms/slika | 95,221360 | 89,663456 |

Ovo su pojedinačni sačuvani prolazi. Ukupno vreme uključuje učitavanje slike. [COCO metrike](../../results/eval_50/tflite_dynamic_int8_50_metrics.json) · [Vremena](../../results/eval_50/tflite_dynamic_int8_50_timing.json) · [Predikcije](../../results/eval_50/tflite_dynamic_int8_50_predictions.json)

## Reprodukcija

U WSL-u pokreni izvoz u novi direktorijum. Za prvi izvoz koristi `models/tflite_dynamic_int8`, a za ponavljanje drugi direktorijum.

```bash
cd /mnt/d/yolo_test
~/venvs/yolo-tflite-export-py312/bin/python tflite_cpp_wrapper/quantize_dynamic_int8.py \
  --source models/tflite_fp32/yolov8n_fp32.tflite \
  --output-dir models/tflite_dynamic_int8
```

U Windows PowerShell-u iz `D:\yolo_test` koristi slobodan prefiks pri ponavljanju. Ako si odabrao drugi direktorijum za izvoz, prilagodi putanju modela.

```powershell
$prefix = '.\results\eval_50\tflite_dynamic_int8_50'
foreach ($suffix in 'predictions', 'timing', 'metrics') {
    if (Test-Path -LiteralPath "${prefix}_$suffix.json") { throw 'Odaberi slobodan prefiks' }
}
.\tflite_cpp_wrapper\build\bin\yolo_tflite.exe .\models\tflite_dynamic_int8\yolov8n_dynamic_int8.tflite .\val2017 .\results\eval_50\manifest_50.json $prefix 4 normalized xnnpack
if ($LASTEXITCODE -ne 0) { throw 'C++ inference failed' }
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\coco_tools.py evaluate --manifest .\results\eval_50\manifest_50.json --predictions "${prefix}_predictions.json" --output "${prefix}_metrics.json"
```
