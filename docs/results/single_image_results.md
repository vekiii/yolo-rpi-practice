# Single-image TFLite benchmark rezultati

Zapis testova u kojima jedna fiksirana slika prolazi ponovljeni preprocessing, inference i postprocessing. Izvori su `results/single_image_5000/*_benchmark.json`. Jedinice su milisekunde. Standardna devijacija je populaciona (`N` divisor), prema metapodatku svakog izveštaja.

## Runovi

| Run | Model | Slika (ID) | Seed | Warmup | Merenja | Niti | Backend | Model bajtova |
|---|---|---|---:|---:|---:|---:|---|---:|
| Raspberry Pi 4, FP32 | `models/tflite_fp32/yolov8n_fp32.tflite` | `000000239274.jpg` (239274) | 1 (slika eksplicitna) | 50 | 5000 | 4 | XNNPACK | 12,841,242 |
| Raspberry Pi 4, FP16 težine | `models/tflite_fp16_weights/yolov8n_fp16_weights.tflite` | `000000239274.jpg` (239274) | 1 (slika eksplicitna) | 50 | 5000 | 4 | XNNPACK | 6,529,518 |
| Raspberry Pi 4, INT8 (EMA min–max, 0,95) | `models/tflite_int8/yolov8n_int8.tflite` | `000000239274.jpg` (239274) | 1 (slika eksplicitna) | 50 | 5000 | 4 | XNNPACK | 3,483,792 |
| Raspberry Pi 4, INT8 (globalni min–max) | `models/tflite_int8_global_minmax/yolov8n_int8.tflite` | `000000239274.jpg` (239274) | 1 (slika eksplicitna) | 50 | 5000 | 4 | XNNPACK | 3,481,104 |
| Raspberry Pi 4, dinamički INT8 | `models/tflite_dynamic_int8/yolov8n_dynamic_int8.tflite` | `000000239274.jpg` (239274) | 1 (slika eksplicitna) | 50 | 5000 | 4 | XNNPACK | 3,470,736 |

Svih pet prolaza koristi TFLite C++ Interpreter API 2.20.0, ulaz `[1,3,640,640]`, izlaz `[1,84,8400]`, normalizovane bbox koordinate i istu, eksplicitno izabranu sliku (`seed_used: null`).

## Vreme po fazi

Svaka ćelija prikazuje **mean / median / std / min / max**, u ms, iz 5000 pojedinačnih ponavljanja.

| Run | Faza | Mean | Median | Std | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| RPi 4 FP32 | Preprocess | 8.248191 | 8.236568 | 0.099575 | 7.972167 | 10.315499 |
| RPi 4 FP32 | Inference | 430.723977 | 430.216647 | 6.375332 | 425.817636 | 599.354855 |
| RPi 4 FP32 | Postprocess | 5.543852 | 5.536072 | 0.037456 | 5.509448 | 6.996048 |
| RPi 4 FP32 | Total | 444.516117 | 444.008578 | 6.422278 | 439.376402 | 614.112082 |
| RPi 4 FP16 težine | Preprocess | 8.261316 | 8.253783 | 0.087750 | 8.072735 | 10.200438 |
| RPi 4 FP16 težine | Inference | 431.628796 | 431.541464 | 1.149257 | 428.554222 | 457.177253 |
| RPi 4 FP16 težine | Postprocess | 5.635392 | 5.627041 | 0.046575 | 5.605857 | 7.150537 |
| RPi 4 FP16 težine | Total | 445.525602 | 445.429846 | 1.155754 | 442.379644 | 471.107063 |
| RPi 4 INT8 (EMA 0,95) | Preprocess | 8.641906 | 8.639721 | 0.097965 | 8.364193 | 10.626330 |
| RPi 4 INT8 (EMA 0,95) | Inference | 212.892130 | 212.783853 | 0.886190 | 211.779126 | 251.733493 |
| RPi 4 INT8 (EMA 0,95) | Postprocess | 5.509734 | 5.503521 | 0.036840 | 5.482345 | 6.944142 |
| RPi 4 INT8 (EMA 0,95) | Total | 227.043868 | 226.931026 | 0.900880 | 225.828917 | 265.900760 |
| RPi 4 INT8 (globalni min–max) | Preprocess | 8.529434 | 8.518598 | 0.088119 | 8.326334 | 10.001376 |
| RPi 4 INT8 (globalni min–max) | Inference | 213.368764 | 213.259810 | 0.604115 | 212.399802 | 222.348144 |
| RPi 4 INT8 (globalni min–max) | Postprocess | 5.520822 | 5.514551 | 0.048767 | 5.494033 | 7.505046 |
| RPi 4 INT8 (globalni min–max) | Total | 227.419118 | 227.309620 | 0.618634 | 226.428530 | 236.515272 |
| RPi 4 dinamički INT8 | Preprocess | 8.459267 | 8.452068 | 0.086896 | 8.242576 | 10.320722 |
| RPi 4 dinamički INT8 | Inference | 378.679182 | 378.467351 | 2.752529 | 377.184408 | 498.043481 |
| RPi 4 dinamički INT8 | Postprocess | 5.547970 | 5.541558 | 0.021290 | 5.514026 | 5.886279 |
| RPi 4 dinamički INT8 | Total | 392.686518 | 392.476355 | 2.763232 | 391.158348 | 512.554938 |

`Total` statistika izračunata je iz svakog ponavljanja `total_ms`; nije dobijena sabiranjem statistika faza. Total obuhvata preprocessing, `Invoke()` i postprocessing. Učitavanje slike i modela, interpreter/delegate inicijalizacija, konzolni ispis i JSON zapisivanje nisu uključeni.

## Poređenje

Za ovu sliku i ova dva pojedinačna prolaza, FP16 težina model imao je inference mean **0.904820 ms (0.21%) viši**, a total mean **1.009485 ms (0.23%) viši** od FP32. Model FP16 težina je približno 49.15% manji po bajtovima. Ovo poređenje samo opisuje ove runove; nije ponovljeni skup nezavisnih eksperimenata i ne pokazuje da je razlika performansi posledica težina.

U odnosu na FP32, INT8 je u ovom prolazu imao **50,57% kraće prosečno vreme inference** i **48,92% kraće ukupno vreme** po ponavljanju. U benchmark JSON fajlovima polje `precision` dopunjeno je prema lokalnim modelima čije se putanje i veličine poklapaju sa zapisima. Sva tri modela imaju FP32 ulaz i izlaz, a zapis težina je FP32, FP16 ili INT8. Tipove INT8 tenzora detaljnije beleži [pregled grafa](../../models/tflite_int8/inspection.json).

Globalni min–max je u ovom prolazu imao **0,48 ms (0,22%) dužu prosečnu inferencu** i **0,38 ms (0,17%) duže ukupno vreme** od EMA 0,95. Obe INT8 varijante imaju FP32 ulaz i izlaz.

Dinamički INT8 imao je **12,08% kraće prosečno vreme inference** i **11,66% kraće ukupno vreme** od FP32 u ovim prolazima.

## Maksimalni RSS

Dinamički INT8 je dostigao **61,97 MiB** maksimalnog RSS-a. Vrednost obuhvata ceo proces wrappera, uključujući inicijalizaciju i zagrevanje. [Izveštaj o memoriji](../../results/single_image_5000/rpi_bench_dynamic_int8_img239274_memory.json)

## Definicije faza

- **Preprocess:** resize/letterbox i padding do 640×640, BGR→RGB, normalizacija `/255` i upis u ulazni tenzor.
- **Inference:** poziv TFLite `Invoke()`.
- **Postprocess:** dekodiranje modelskog izlaza, izbor klase, confidence prag, sortiranje, class-aware NMS i clipping kutija na originalnu sliku.
- **Total:** monotono vreme od početka preprocessinga do kraja postprocessinga za jedno ponavljanje.

## Izvorni fajlovi i provere

- FP32: [rpi_bench_fp32_img239274_benchmark.json](../../results/single_image_5000/rpi_bench_fp32_img239274_benchmark.json), izmenjen 16.11.2026. 12:46:25.
- FP16 težine: [rpi_bench_fp16_img239274_benchmark.json](../../results/single_image_5000/rpi_bench_fp16_img239274_benchmark.json), izmenjen 16.11.2026. 13:27:44.
- INT8 EMA: [rpi_bench_int8_img239274_benchmark.json](../../results/single_image_5000/rpi_bench_int8_img239274_benchmark.json), izmenjen 21.11.2026. 13:13:14.
- INT8 globalni min–max: [rpi_bench_int8_global_minmax_img239274_benchmark.json](../../results/single_image_5000/rpi_bench_int8_global_minmax_img239274_benchmark.json).
- Dinamički INT8: [rpi_bench_dynamic_int8_img239274_benchmark.json](../../results/single_image_5000/rpi_bench_dynamic_int8_img239274_benchmark.json).

Svaki fajl ima 5000 uzoraka. Statistike za INT8 proverene su prema `per_iteration` podacima. Datumi su vremena poslednje izmene kopiranih JSON fajlova. Ovaj benchmark meri vreme obrade iste slike.
