# Single-image TFLite benchmark rezultati

Zapis testova u kojima jedna fiksirana slika prolazi ponovljeni preprocessing, inference i postprocessing. Izvori su `results/*_benchmark.json`. Jedinice su milisekunde. Standardna devijacija je populaciona (`N` divisor), prema metapodatku svakog izveštaja.

## Runovi

| Run | Model | Slika (ID) | Seed | Warmup | Merenja | Niti | Backend | Model bajtova |
|---|---|---|---:|---:|---:|---:|---|---:|
| Raspberry Pi 4, FP32 | `models/tflite_fp32/yolov8n_fp32.tflite` | `000000239274.jpg` (239274) | 1 (slika eksplicitna) | 50 | 5000 | 4 | XNNPACK | 12,841,242 |
| Raspberry Pi 4, FP16 težine | `models/tflite_fp16_weights/yolov8n_fp16_weights.tflite` | `000000239274.jpg` (239274) | 1 (slika eksplicitna) | 50 | 5000 | 4 | XNNPACK | 6,529,518 |

Oba runa koriste TFLite C++ Interpreter API 2.20.0, ulaz `[1,3,640,640]`, izlaz `[1,84,8400]`, normalizovane bbox koordinate i istu sliku. `seed_used: null` ukazuje da slika je izabrana eksplicitno; `seed: 1` je samo zabeležena seed vrednost, ne uzrok izbora slike.

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

`Total` statistika izračunata je iz svakog ponavljanja `total_ms`; nije dobijena sabiranjem statistika faza. Total obuhvata preprocessing, `Invoke()` i postprocessing. Učitavanje slike i modela, interpreter/delegate inicijalizacija, konzolni ispis i JSON zapisivanje nisu uključeni.

## Poređenje

Za ovu sliku i ova dva pojedinačna prolaza, FP16 težina model imao je inference mean **0.904820 ms (0.21%) viši**, a total mean **1.009485 ms (0.23%) viši** od FP32. Model FP16 težina je približno 49.15% manji po bajtovima. Ovo poređenje samo opisuje ove runove; nije ponovljeni skup nezavisnih eksperimenata i ne pokazuje da je razlika performansi posledica težina.

## Definicije faza

- **Preprocess:** resize/letterbox i padding do 640×640, BGR→RGB, normalizacija `/255` i upis u ulazni tenzor.
- **Inference:** poziv TFLite `Invoke()`.
- **Postprocess:** dekodiranje modelskog izlaza, izbor klase, confidence prag, sortiranje, class-aware NMS i clipping kutija na originalnu sliku.
- **Total:** monotono vreme od početka preprocessinga do kraja postprocessinga za jedno ponavljanje.

## Izvorni fajlovi i provere

- FP32: [rpi_bench_fp32_img239274_benchmark.json](results/rpi_bench_fp32_img239274_benchmark.json), izmenjen 16.11.2026. 12:46:25.
- FP16 težine: [rpi_bench_fp16_img239274_benchmark.json](results/rpi_bench_fp16_img239274_benchmark.json), izmenjen 16.11.2026. 13:27:44.

Provereno je da oba fajla imaju po 5000 uzoraka i da mean, median, populacioni std, min i max za svaku fazu odgovaraju `per_iteration` podacima. Datumi su vremena poslednje izmene kopiranih JSON fajlova; nisu nužno tačan početak benchmarka. Nema COCO mAP evaluacije u ovom benchmark režimu.
