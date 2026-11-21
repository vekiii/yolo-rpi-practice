# Rezultati testova

Ažurirano 21.11.2026. iz JSON fajlova. Statistike koriste pojedinačna vremena za preprocess, inference i postprocess. Standardna devijacija je uzoračka (n−1), u ms.

## Pregled po redosledu sačuvanih rezultata

Datumi i vremena označavaju lokalno prikazano vreme poslednje izmene izveštaja, a ne početak testa. Kod prenetih Raspberry Pi fajlova mogu označavati i vreme kopiranja; JSON ne čuva datum izvršavanja. AP vrednosti su na skali 0–1; vremena su proseci u ms po slici. Testovi 1–6 su raniji laptop rezultati, dok su testovi 7–8 označeni prefiksom `rpi_`.

| Test | Datum | Slika | Niti | mAP50–95 | mAP50 | mAP75 | Inference | Predict / ukupno |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1. Python/PyTorch | 04.11.2026. 12:47:42 | 50 | 8 | 0.410425851 | 0.552407994 | 0.453770563 | 107.942174 | 130.867954 |
| 2. Python/PyTorch FP32 baseline | 04.11.2026. 14:36:26 | 5000 | 8 | 0.365719332 | 0.513690330 | 0.397506553 | 124.917075 | 152.878522 |
| 5. C++/TFLite FP32 + XNNPACK | 11.11.2026. 18:09:14 | 50 | 4 | 0.410425851 | 0.552407994 | 0.453770563 | 85.431748 | 95.221360 |
| 6. C++/TFLite FP32 + XNNPACK | 11.11.2026. 18:23:31 | 5000 | 4 | 0.365711977 | 0.513711181 | 0.397511274 | 115.516409 | 131.939457 |
| 7. RPi C++/TFLite FP32 + XNNPACK | 12.11.2026. 14:40:47 | 50 | 4 | 0.410425851 | 0.552407994 | 0.453770563 | 420.947902 | 442.650930 |
| 8. RPi C++/TFLite FP32 + XNNPACK | 12.11.2026. 15:46:53 | 5000 | 4 | 0.365712541 | 0.513711214 | 0.397511281 | 425.772376 | 447.709495 |
| 9. Laptop C++/TFLite FP16 težine + XNNPACK | 12.11.2026. 16:55:42 | 50 | 4 | 0.409113742 | 0.551952254 | 0.453345430 | 74.695602 | 83.842106 |
| 10. Laptop C++/TFLite FP16 težine + XNNPACK | 13.11.2026. 14:41:03 | 5000 | 4 | 0.365739231 | 0.513939614 | 0.397913472 | 102.745131 | 116.238675 |
| 11. RPi C++/TFLite FP16 težine + XNNPACK | 13.11.2026. 15:28:05 | 5000 | 4 | 0.365741991 | 0.513939253 | 0.397913689 | 440.392685 | 463.385106 |
| 12. Laptop C++/TFLite INT8 + XNNPACK | 19.11.2026. 12:33:58 | 5000 | 4 | 0.281072752 | 0.454539481 | 0.308678337 | 116.931647 | 142.448698 |

Zajednički parametri: YOLOv8n, CPU, COCO val2017, deterministički izbor po rastućem ID-u, 640×640, batch 1, `conf=0.001`, NMS IoU 0.7, najviše 100 detekcija, pet warmup prolaza isključenih iz merenja. Provereno je da Python i TFLite koriste iste ID-eve istim redosledom za odgovarajuću veličinu testa. TFLite testovi su završeni: postoje i timing i COCO metrics izveštaji.

NCNN izvori i model uklonjeni su u commitu `a6cd793`; dostupni su u Git istoriji. NCNN timing i metrics fajlovi i dalje postoje, dok pripadajući predictions fajlovi trenutno nisu prisutni.

## 1. Python/PyTorch — 50 slika

Izvor: [evaluation_50.json](results/evaluation_50.json); detekcije: [evaluation_50_predictions.json](results/evaluation_50_predictions.json).

| Faza / statistika | ms/slika |
|---|---:|
| Pretprocesiranje | 3.241362 |
| Inference | 107.942174 |
| Postprocesiranje | 2.736224 |
| Prosečan `model.predict` poziv | 130.867954 |
| Medijana `model.predict` poziva | 137.092250 |

Rezultat eksplicitno beleži `half=false`. Ovo je provera na prvih 50 slika, čiji AP ne treba direktno porediti sa AP na svih 5000 slika.

## 2. Python/PyTorch — 5000 slika

Izvor: [evaluation_fp32_5000.json](results/evaluation_fp32_5000.json); detekcije: [evaluation_fp32_5000_predictions.json](results/evaluation_fp32_5000_predictions.json).

| Faza / statistika | ms/slika |
|---|---:|
| Pretprocesiranje | 3.997781 |
| Inference | 124.917075 |
| Postprocesiranje | 3.191073 |
| Prosečan `model.predict` poziv | 152.878522 |
| Medijana `model.predict` poziva | 148.112600 |

Polje `half` nije prisutno u ovom izveštaju. Naziv fajla označava FP32 baseline; korišćen je `yolov8n.pt` na CPU-u.

Oba Python izveštaja beleže: Python 3.12.10, Windows-10-10.0.19045-SP0, Torch 2.15.0+cpu, Ultralytics 8.4.202 i pycocotools 2.0.11. Ovo su vrednosti iz izveštaja, ne naknadno proverene verzije okruženja.

## 3. C++/NCNN FP32 — 50 slika

Izvori: [timing](results/ncnn_fp32_50_timing.json), [COCO metrike](results/ncnn_fp32_50_metrics.json), [manifest](results/manifest_50.json). Raniji `ncnn_fp32_50_predictions.json` trenutno nije prisutan.

| Faza | Prosek, ms/slika |
|---|---:|
| Učitavanje slike | 25.100156 |
| Pretprocesiranje | 6.586240 |
| Inference | 122.462036 |
| Postprocesiranje | 2.099906 |
| Ukupna obrada | 156.249160 |

Timing sadrži 50 zapisa. mAP50–95, mAP50 i mAP75 jednaki su Python testu na 50 slika na preciznosti sačuvanoj u JSON-u. Prosečan inference i ukupna obrada sporiji su od odgovarajućeg Python testa.

## 4. C++/NCNN FP32 — 5000 slika

Izvori: [timing](results/ncnn_fp32_5000_timing.json), [COCO metrike](results/ncnn_fp32_5000_metrics.json), [manifest](results/manifest_5000.json). Raniji `ncnn_fp32_5000_predictions.json` trenutno nije prisutan.

| Faza | Prosek, ms/slika |
|---|---:|
| Učitavanje slike | 24.913897 |
| Pretprocesiranje | 6.194501 |
| Inference | 117.466344 |
| Postprocesiranje | 1.860334 |
| Ukupna obrada | 150.435963 |

Timing sadrži 5000 zapisa. Proseci su u `means.inference_ms`, `means.total_ms` i ostalim poljima objekta `means` (početak objekta je na liniji 20092 izvornog fajla).

Zbir sačuvanih `total_ms` iznosi 752179.8155 ms, odnosno oko 12 min 32.18 s. To je zbir merenih obrada slika, a ne ukupno trajanje procesa sa warmup-om, ispisom, JSON eksportom i COCO evaluacijom.

## 5. C++/TFLite FP32 + XNNPACK — 50 slika

Izvori: [timing](results/tflite_fp32_50_timing.json), [COCO metrike](results/tflite_fp32_50_metrics.json), [detekcije](results/tflite_fp32_50_predictions.json).

| Faza | Prosek, ms/slika |
|---|---:|
| Učitavanje slike | 5.668390 |
| Pretprocesiranje | 3.089692 |
| Inference (`Invoke`) | 85.431748 |
| Postprocesiranje | 1.031012 |
| Ukupna obrada | 95.221360 |

Timing sadrži 50 zapisa; proseci su provereni prema `per_image`. Tri glavne AP metrike jednake su Python rezultatu na 50 slika na sačuvanoj preciznosti. To podržava ispravnost konverzije i dekodiranja, ali ne dokazuje identičnost svih detekcija. Zbir merenih ukupnih obrada iznosi 4761.068 ms.

## 6. C++/TFLite FP32 + XNNPACK — 5000 slika

Izvori: [timing](results/tflite_fp32_5000_timing.json), [COCO metrike](results/tflite_fp32_5000_metrics.json), [detekcije](results/tflite_fp32_5000_predictions.json).

| Faza | Prosek, ms/slika |
|---|---:|
| Učitavanje slike | 10.410996 |
| Pretprocesiranje | 4.343535 |
| Inference (`Invoke`) | 115.516409 |
| Postprocesiranje | 1.667835 |
| Ukupna obrada | 131.939457 |

Timing sadrži 5000 zapisa; proseci su provereni prema `per_image`. Zbir `total_ms` je 659697.287 ms, odnosno oko 10 min 59.70 s. To nije ukupno trajanje procesa sa inicijalizacijom, warmup-om, ispisom, eksportom i evaluacijom.

Pune AP vrednosti: mAP50–95 **0.36571197672614925**, mAP50 **0.5137111814279052**, mAP75 **0.3975112740264031**.

Oba laptop TFLite testa beleže:

- Runtime: `TFLite C++ Interpreter API`, verzija `2.20.0`; backend `xnnpack`, četiri niti.
- Preciznost: `FP32 baseline; no FP16 forcing`; pet warmup prolaza.
- Model: `models/tflite_fp32/yolov8n_fp32.tflite`, 12841242 bajta.
- Ulaz `[1,3,640,640]`, izlaz `[1,84,8400]`, koordinate `normalized`.

[Ugovor modela](models/tflite_fp32/yolov8n_fp32.contract.json) potvrđuje FP32 I/O i beleži izvoz kroz Ultralytics 8.4.208 u WSL2 Linux okruženju. [Environment zapis](models/tflite_fp32/yolov8n_fp32.environment.txt) čuva pakete izvoza; to nije okruženje C++ benchmarka. Trenutni model i izvorne težine provereni su prema SHA-256 vrednostima ugovora. Timing ne beleži hash u trenutku testa.

## 7. Raspberry Pi C++/TFLite FP32 + XNNPACK — 50 slika

Izvori: [timing](results/rpi_tflite_fp32_50_timing.json), [COCO metrike](results/rpi_tflite_fp32_50_metrics.json), [detekcije](results/rpi_tflite_fp32_50_predictions.json).

| Faza | Prosek, ms/slika |
|---|---:|
| Učitavanje slike | 7.703337 |
| Pretprocesiranje | 8.457287 |
| Inference (`Invoke`) | 420.947902 |
| Postprocesiranje | 5.538167 |
| Ukupna obrada | 442.650930 |

Timing sadrži 50 zapisa, a predictions fajl 3932 detekcije. Provereni su proseci svih faza, redosled ID-eva prema Python referenci i pripadnost svih detekcija izabranim slikama. Sve COCO metrike iz `all_coco_stats` jednake su laptop TFLite rezultatu na 50 slika na sačuvanoj preciznosti.

Zbir merenih ukupnih obrada iznosi **22132.54649 ms**, odnosno oko **22.13 s**. Recipročna vrednost prosečnog ukupnog vremena je **2.26 slike/s**; za sam inference **2.38 slike/s**. Ove vrednosti ne uključuju faze van timing granica.

## 8. Raspberry Pi C++/TFLite FP32 + XNNPACK — 5000 slika

Izvori: [timing](results/rpi_tflite_fp32_5000_timing.json), [COCO metrike](results/rpi_tflite_fp32_5000_metrics.json), [detekcije](results/rpi_tflite_fp32_5000_predictions.json).

| Faza | Prosek, ms/slika |
|---|---:|
| Učitavanje slike | 8.137549 |
| Pretprocesiranje | 8.312392 |
| Inference (`Invoke`) | 425.772376 |
| Postprocesiranje | 5.482990 |
| Ukupna obrada | 447.709495 |

Timing sadrži 5000 zapisa, a predictions fajl 376148 detekcija. Provereni su proseci svih faza, redosled ID-eva prema Python referenci i pripadnost svih detekcija izabranim slikama.

Pune AP vrednosti: mAP50–95 **0.36571254064182906**, mAP50 **0.5137112137860647**, mAP75 **0.3975112811256434**.

Zbir `total_ms` je **2238547.473779 ms**, odnosno oko **37 min 18.55 s**. To je zbir obrada slika, a ne ukupno trajanje procesa. Recipročna vrednost prosečnog ukupnog vremena daje **2.23 slike/s**, a samog inference vremena **2.35 slike/s**.

Oba RPi izveštaja beleže TFLite C++ Interpreter API `2.20.0`, backend `xnnpack`, četiri niti, batch 1, pet warmup prolaza i `FP32 baseline; no FP16 forcing`. Model je `models/tflite_fp32/yolov8n_fp32.tflite`, veličine 12841242 bajta, sa ulazom `[1,3,640,640]`, izlazom `[1,84,8400]` i normalizovanim koordinatama. Parametri detekcije odgovaraju laptop testovima.

Oznaka Raspberry Pi preuzeta je iz naziva novih fajlova. JSON ne navodi tačan model uređaja, CPU/OS, ARM arhitekturu, kompajler, hlađenje, napajanje ili frekvenciju procesora. Ista putanja i veličina modela ne potvrđuju identične bajtove modela na uređaju, jer runtime izveštaj ne sadrži hash.

## 9. Laptop C++/TFLite — FP16 težine, 50 slika

Evaluacija je završena: postoje [timing](results/tflite_fp16_weights_50_timing.json), [COCO metrike](results/tflite_fp16_weights_50_metrics.json) i [detekcije](results/tflite_fp16_weights_50_predictions.json). Timing i predictions fajlovi poslednji put su izmenjeni 12.11.2026. u 16:55:30, a metrics u 16:55:42. To su vremena izmene lokalnih fajlova, ne zapis početka testa.

| Faza | Prosek, ms/slika |
|---|---:|
| Učitavanje slike | 5.165222 |
| Pretprocesiranje | 3.034116 |
| Inference (`Invoke`) | 74.695602 |
| Postprocesiranje | 0.946752 |
| Ukupna obrada | 83.842106 |

Pune AP vrednosti: mAP50–95 **0.4091137421567885**, mAP50 **0.5519522535280055**, mAP75 **0.4533454299030062**.

Timing sadrži 50 zapisa. Provereni su proseci prema `per_image` i identičnost manifesta sa laptop FP32 testom na 50 slika. Zbir merenih `total_ms` iznosi **4192.1053 ms**, odnosno oko **4.19 s**, bez inicijalizacije, warmup-a, ispisa, JSON izvoza i COCO evaluacije.

Konfiguracija: TFLite C++ Interpreter API `2.20.0`, `xnnpack`, četiri niti, pet warmup prolaza, batch 1, 640×640, `conf=0.001`, NMS IoU 0.7 i `max_det=100`. Ulaz `[1,3,640,640]` i izlaz `[1,84,8400]` ostaju FP32; koordinate su `normalized`.

Model: [yolov8n_fp16_weights.tflite](models/tflite_fp16_weights/yolov8n_fp16_weights.tflite), **6529518 bajta**. [Zapis konverzije](models/tflite_fp16_weights/yolov8n_fp16_weights.conversion.json) beleži 132 konvertovane konstante, 132 dodata `DEQUANTIZE` operatora i zadržane originalne operatore i FP32 ulaze/izlaze operacija. Proveren SHA-256 trenutnog modela je `5f42268131b1052e3c1687000b2595e38175ab0b7c6f5d6eee17449076268e57`, identičan zapisu konverzije.

**Ovo je test FP16 zapisa težina/konstanti, a ne potvrda FP16 računanja mreže.** Timing polje `precision` i dalje ima generički tekst `FP32 baseline; no FP16 forcing`; samo po sebi ne opisuje zapis težina. Preciznost modela zato je dokumentovana uz zapis konverzije. Nema profilisanja kojim bi bila potvrđena preciznost svakog XNNPACK kernela.

### Poređenje sa laptop FP32 testom na istih 50 slika

| Metrika | FP32 model | FP16 težine |
|---|---:|---:|
| Veličina modela, bajta | 12841242 | 6529518 |
| mAP50–95 | 0.410425851 | 0.409113742 |
| mAP50 | 0.552407994 | 0.551952254 |
| mAP75 | 0.453770563 | 0.453345430 |
| Inference, ms/slika | 85.431748 | 74.695602 |
| Ukupno, ms/slika | 95.221360 | 83.842106 |

Model je približno **49.15% manji**, a prosečno inference vreme je **12.57% manje** u ovom prolazu. mAP50–95 je manji za približno **0.001312109**, odnosno **0.131211 procentnih poena**. Ovo su pojedinačni testovi na 50 slika; nisu dokaz ponovljivog ubrzanja. Evaluacija na 5000 slika nalazi se u odeljku 10.

| Dodatna COCO metrika | FP16 težine, 50 slika |
|---|---:|
| AP small | 0.250367 |
| AP medium | 0.465726 |
| AP large | 0.661281 |
| AR maxDets=1 | 0.321008 |
| AR maxDets=10 | 0.516671 |
| AR maxDets=100 | 0.528771 |
| AR small | 0.329366 |
| AR medium | 0.596435 |
| AR large | 0.722569 |

## 10. Laptop C++/TFLite — FP16 težine, 5000 slika

Izveštaji: [timing](results/tflite_fp16_weights_5000_timing.json), [COCO metrike](results/tflite_fp16_weights_5000_metrics.json) i [detekcije](results/tflite_fp16_weights_5000_predictions.json). Timing i predictions su izmenjeni 13.11.2026. u 14:24:35, metrics u 14:41:03. To su vremena izmene fajlova, ne početka testa.

| Faza | Prosek, ms/slika |
|---|---:|
| Učitavanje slike | 7.929429 |
| Pretprocesiranje | 4.064151 |
| Inference (`Invoke`) | 102.745131 |
| Postprocesiranje | 1.499472 |
| Ukupna obrada | 116.238675 |

COCO rezultati: mAP50–95 **0.36573923091675437**, mAP50 **0.5139396135822089**, mAP75 **0.39791347238869496**. Timing sadrži 5000 zapisa. Zbir `total_ms` je **581193.3727 ms** (oko **9 min 41.19 s**), bez inicijalizacije, warmup-a, ispisa, JSON izvoza i evaluacije.

Konfiguracija: TFLite C++ Interpreter API `2.20.0`, XNNPACK, četiri niti, batch 1 i pet warmup prolaza. Model je FP16 težina/konstanti, uz FP32 ulaze/izlaze operacija; FP16 računanje nije forsirano.

### Poređenje sa laptop TFLite FP32 modelom na istih 5000 slika

| Metrika | FP32 model | FP16 težine |
|---|---:|---:|
| mAP50–95 | 0.365711977 | 0.365739231 |
| mAP50 | 0.513711181 | 0.513939614 |
| mAP75 | 0.397511274 | 0.397913472 |
| Inference, ms/slika | 115.516409 | 102.745131 |
| Ukupno, ms/slika | 131.939457 | 116.238675 |

U ovom prolazu FP16 težine daju oko **11.06% niže inference vreme** i **11.90% niže ukupno vreme**. mAP50–95 je viši za **0.000027254190605**, odnosno **0.002725 procentnih poena**. Ovo su pojedinačni prolazi i ne daju procenu varijabilnosti.

## 11. Raspberry Pi C++/TFLite — FP16 težine, 5000 slika

Izveštaji: [timing](results/rpi_tflite_fp16_weights_5000_timing.json), [COCO metrike](results/rpi_tflite_fp16_weights_5000_metrics.json) i [detekcije](results/rpi_tflite_fp16_weights_5000_predictions.json). Timing i predictions su izmenjeni 13.11.2026. oko 15:27, metrics u 15:28:05. Vremena fajlova nisu nužno vreme početka testa.

| Faza | Prosek, ms/slika |
|---|---:|
| Učitavanje slike | 9.262373 |
| Pretprocesiranje | 8.202460 |
| Inference (`Invoke`) | 440.392685 |
| Postprocesiranje | 5.523155 |
| Ukupna obrada | 463.385106 |

COCO rezultati: mAP50–95 **0.3657419911988745**, mAP50 **0.5139392526488519**, mAP75 **0.3979136889107991**. Timing ima 5000 zapisa, predictions fajl 376186 detekcija, a proseci su provereni prema pojedinačnim zapisima. Redosled slika poklapa se sa RPi FP32 testom, a sve detekcije pripadaju tom skupu. Zbir `total_ms` je **2316925.529645 ms**, oko **38 min 36.93 s**; prosečno ukupno vreme odgovara oko **2.16 slike/s**.

Model ima FP16 težine/konstante, dok ulaz i izlaz mreže ostaju FP32; FP16 računanje nije forsirano. Timing beleži TFLite C++ Interpreter API `2.20.0`, XNNPACK, četiri niti, batch 1 i pet warmup prolaza. JSON ne navodi tačan Raspberry Pi model ili sistemske uslove.

### Poređenje sa RPi TFLite FP32 testom na 5000 slika

| Metrika | FP32 težine | FP16 težine |
|---|---:|---:|
| mAP50–95 | 0.365712541 | 0.365741991 |
| mAP50 | 0.513711214 | 0.513939253 |
| mAP75 | 0.397511281 | 0.397913689 |
| Inference, ms/slika | 425.772376 | 440.392685 |
| Ukupno, ms/slika | 447.709495 | 463.385106 |

FP16 težina prolaz ima 3.43% više inference vremena i 3.50% više ukupnog vremena od RPi FP32 prolaza. mAP50–95 je veći za 0.000029450557045 (0.002945 procentnih poena). Pojedinačni prolazi ne utvrđuju uzrok razlike u vremenu.

## 12. Laptop C++/TFLite INT8 — 5000 slika

TFLite C++ 2.20.0, XNNPACK, četiri niti i pet warmup prolaza. Model ima FP32 ulaz i izlaz, a preciznost kernela nije profilisana. Timing sadrži 5000 zapisa sa istim ID-jevima i redosledom kao laptop FP32 prolaz. [Timing](results/tflite_int8_5000_timing.json) · [COCO metrike](results/tflite_int8_5000_metrics.json) · [Predikcije](results/tflite_int8_5000_predictions.json)

| Faza | Prosek, ms/slika |
|---|---:|
| Učitavanje | 19.306740 |
| Pretprocesiranje | 5.040568 |
| Inference | 116.931647 |
| Postprocesiranje | 1.168834 |
| Ukupno | 142.448698 |

| COCO metrika | Sve veličine | Male | Srednje | Velike |
|---|---:|---:|---:|---:|
| AP50–95 | 0.281073 | 0.106658 | 0.294870 | 0.442378 |
| AR, maxDets=100 | 0.455592 | 0.227372 | 0.503621 | 0.633470 |

mAP50 je 0.454539, mAP75 je 0.308678, AR@1 je 0.254274, a AR@10 je 0.416644.

U odnosu na laptop FP32 prolaz na istim slikama, mAP50–95 je niži za **8,46 procentnih poena**, a prosečno inference vreme je **1,23% duže** (116,93 prema 115,52 ms/slika). Obe vrednosti vremena potiču iz pojedinačnih sačuvanih prolaza.

## Poređenje laptop i Raspberry Pi TFLite rezultata na 5000 slika

| Metrika | Laptop FP32 | Laptop FP16 težine | RPi FP32 | RPi FP16 težine |
|---|---:|---:|---:|---:|
| CPU niti | 4 | 4 | 4 | 4 |
| mAP50–95 | 0.365711977 | 0.365739231 | 0.365712541 | 0.365741991 |
| mAP50 | 0.513711181 | 0.513939614 | 0.513711214 | 0.513939253 |
| mAP75 | 0.397511274 | 0.397913472 | 0.397511281 | 0.397913689 |
| Inference, ms/slika | 115.516409 | 102.745131 | 425.772376 | 440.392685 |
| Ukupno, ms/slika | 131.939457 | 116.238675 | 447.709495 | 463.385106 |
| Slike/s iz prosečnog ukupnog vremena | 7.58 | 8.60 | 2.23 | 2.16 |

Sa FP32 težinama, RPi inference vreme je **3.69 puta veće**, a ukupno vreme obrade **3.39 puta veće** od laptop prolaza. Sa FP16 težinama, RPi inference je **4.29 puta veći**, a ukupno vreme **3.98 puta veće** od laptop prolaza. mAP rezultati su vrlo bliski, ali nisu bitno identični. Poređenje menja hardver i uslove izvršavanja.

## Dodatne COCO metrike

Za Python su ove vrednosti preuzete iz tekstualnog `coco_summary` i zato imaju samo tri decimale. TFLite vrednosti su radi prikaza zaokružene na isti broj decimala; pune vrednosti su u `all_coco_stats`.

| Metrika | Python 50 | Python 5000 | NCNN 50 | NCNN 5000 | TFLite 50 | TFLite 5000 |
|---|---:|---:|---:|---:|---:|---:|
| AP small | 0.252 | 0.173 | 0.252 | 0.173 | 0.252 | 0.173 |
| AP medium | 0.469 | 0.404 | 0.469 | 0.404 | 0.469 | 0.404 |
| AP large | 0.663 | 0.518 | 0.663 | 0.518 | 0.663 | 0.518 |
| AR maxDets=1 | 0.322 | 0.303 | 0.322 | 0.303 | 0.322 | 0.303 |
| AR maxDets=10 | 0.518 | 0.492 | 0.518 | 0.492 | 0.518 | 0.492 |
| AR maxDets=100 | 0.531 | 0.531 | 0.531 | 0.531 | 0.531 | 0.531 |
| AR small | 0.331 | 0.289 | 0.331 | 0.289 | 0.331 | 0.289 |
| AR medium | 0.600 | 0.592 | 0.600 | 0.591 | 0.600 | 0.591 |
| AR large | 0.729 | 0.703 | 0.729 | 0.703 | 0.729 | 0.703 |

Nove RPi metrike su takođe zaokružene na tri decimale; pune vrednosti su u odgovarajućim `all_coco_stats` nizovima.

| Metrika | RPi TFLite 50 | RPi TFLite 5000 |
|---|---:|---:|
| AP small | 0.252 | 0.173 |
| AP medium | 0.469 | 0.404 |
| AP large | 0.663 | 0.518 |
| AR maxDets=1 | 0.322 | 0.303 |
| AR maxDets=10 | 0.518 | 0.492 |
| AR maxDets=100 | 0.531 | 0.531 |
| AR small | 0.331 | 0.289 |
| AR medium | 0.600 | 0.591 |
| AR large | 0.729 | 0.703 |

## Značenje vremena i ograničenja

## Statistika vremena po slici

Vrednosti su izračunate iz svakog run-a i prikazane u ms. `Mean` je aritmetička sredina, `std` uzoračka standardna devijacija (`n−1`), a min/max su najniža i najviša vrednost. U tabeli su date samo preprocess, inference i postprocess faze; nema učitavanja ni ukupnog vremena.

| Run | Faza | Mean | Median | Std | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| Laptop FP32, 5000 | Preprocess | 4.343535 | 4.224400 | 1.375500 | 1.897700 | 39.564400 |
| Laptop FP32, 5000 | Inference | 115.516409 | 113.528300 | 19.704939 | 69.402700 | 336.510400 |
| Laptop FP32, 5000 | Postprocess | 1.667835 | 1.674600 | 0.549350 | 0.605200 | 18.871200 |
| Laptop FP16 weights, 5000 | Preprocess | 4.064151 | 3.863350 | 1.444324 | 1.727200 | 53.140100 |
| Laptop FP16 weights, 5000 | Inference | 102.745131 | 97.100350 | 19.221173 | 61.334400 | 385.967000 |
| Laptop FP16 weights, 5000 | Postprocess | 1.499472 | 1.440100 | 0.321640 | 0.547700 | 7.622800 |
| Laptop INT8, 5000 | Preprocess | 5.040568 | 4.324000 | 2.608298 | 2.304600 | 62.926600 |
| Laptop INT8, 5000 | Inference | 116.931647 | 106.684300 | 31.519264 | 81.277300 | 685.982000 |
| Laptop INT8, 5000 | Postprocess | 1.168834 | 0.980600 | 0.780780 | 0.579300 | 33.888600 |
| RPi FP32, 5000 | Preprocess | 8.312392 | 8.147997 | 0.531358 | 6.929957 | 10.797741 |
| RPi FP32, 5000 | Inference | 425.772376 | 425.561561 | 3.001331 | 420.287201 | 525.867355 |
| RPi FP32, 5000 | Postprocess | 5.482990 | 5.452058 | 0.152634 | 5.272367 | 7.219903 |
| RPi FP16 weights, 5000 | Preprocess | 8.202460 | 8.023194 | 0.557028 | 6.840084 | 14.296861 |
| RPi FP16 weights, 5000 | Inference | 440.392685 | 440.229800 | 1.744963 | 433.021123 | 462.672541 |
| RPi FP16 weights, 5000 | Postprocess | 5.523155 | 5.494257 | 0.144905 | 5.318276 | 6.721144 |

### Šta znače faze i koje vreme je prikazano

**Preprocess** priprema ulaznu sliku za mrežu: letterbox/resize na 640×640, padding, BGR u RGB i normalizacija podelom sa 255, uz upis u ulazni tenzor. **Inference** je izvršavanje modela (`Invoke()` u TFLite wrapperu). **Postprocess** obrađuje izlaz mreže: bira klase i skorove, filtrira pragom pouzdanosti, radi class-aware NMS i vraća/ograničava kutije na dimenzije originalne slike.

Statistika iznad zasebno računa sve tri faze iz odgovarajućih `per_image` uzoraka. Dakle inference statistika koristi samo `inference_ms`; preprocess i postprocess su prikazani odvojeno. Ne koristi `total_ms`, niti meša tri faze u jednu vrednost. Python baseline nema sačuvana pojedinačna `preprocess_ms` i `postprocess_ms` polja u istom formatu, pa izostavljen je iz fazne tabele.

- Python: `timing_ms.mean_by_stage` sadrži Ultralytics vremena faza. `mean_predict_call` obuhvata učitavanje slike i bibliotečki overhead; ne obuhvata izvoz JSON-a, COCO evaluaciju i ispis napretka. Zbir vremena faza zato nije jednak vremenu celog poziva.
- C++: `means.total_ms` obuhvata učitavanje, pretprocesiranje, inference i postprocesiranje. NCNN inference obuhvata i kreiranje extractora, dok TFLite inference meri `Invoke()` već pripremljenog interpretera. JSON sklapanje/serijalizacija, ispis i COCO evaluacija nisu uključeni. Ukupno se meri zasebno od pojedinačnih faza.
- TFLite: učitavanje modela, kreiranje interpretera/delegata i alokacija tenzora nisu uključeni. XNNPACK je eksplicitno uključen, ali nema izveštaja koji potvrđuje da su svi operatori delegirani. Nema sačuvanog testa sa `builtin` backendom.
- 
- Za prvi pojedinačni test postoji sačuvana slika, ali nema numeričkog izveštaja. Vizualizacija 50 slika takođe nema zaseban benchmark. Laptop INT8 evaluacija na 50 slika je u [tflite_int8.md](tflite_int8.md), a prolaz na 5000 slika u odeljku 12. Nema RAM merenja.
- Pri ažuriranju ponovo su provereni svi C++ timing JSON fajlovi: broj zapisa, redosled ID-eva i proseci prema `per_image`. Za nove RPi predictions fajlove proverena je pripadnost ID-eva manifestu. COCO metrike su preuzete iz sačuvanih metrics JSON fajlova; COCO evaluacija nije ponovo izvršavana.
- Nema sačuvanih nezavisnih ponavljanja iz kojih bi se procenila varijabilnost proseka. Vrednosti su dokumentacija postojećih prolaza.


