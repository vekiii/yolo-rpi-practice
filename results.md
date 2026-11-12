# Rezultati testova

Sastavljeno 09.11.2026, ažurirano 12.11.2026. iz sačuvanih lokalnih JSON fajlova, uključujući nove Raspberry Pi TFLite FP32 rezultate. Razvojni koraci i ranije poreklo dokaza: [evidance.md](evidance.md). Aktivni runtime je TFLite C++ sa XNNPACK backendom; NCNN rezultati ostaju istorijska referenca. Testovi nisu ponovo pokretani za potrebe dokumentovanja.

## Pregled po redosledu sačuvanih rezultata

Datumi i vremena označavaju lokalno prikazano vreme poslednje izmene izveštaja, a ne početak testa. Kod prenetih Raspberry Pi fajlova mogu označavati i vreme kopiranja; JSON ne čuva datum izvršavanja. AP vrednosti su na skali 0–1; vremena su proseci u ms po slici. Testovi 1–6 su raniji laptop rezultati, dok su testovi 7–8 označeni prefiksom `rpi_`.

| Test | Datum | Slika | Niti | mAP50–95 | mAP50 | mAP75 | Inference | Predict / ukupno |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1. Python/PyTorch | 04.11.2026. 12:47:42 | 50 | 8 | 0.410425851 | 0.552407994 | 0.453770563 | 107.942174 | 130.867954 |
| 2. Python/PyTorch FP32 baseline | 04.11.2026. 14:36:26 | 5000 | 8 | 0.365719332 | 0.513690330 | 0.397506553 | 124.917075 | 152.878522 |
| 3. C++/NCNN FP32 | 08.11.2026. 20:47:47 | 50 | 4 | 0.410425851 | 0.552407994 | 0.453770563 | 122.462036 | 156.249160 |
| 4. C++/NCNN FP32 | 08.11.2026. 21:10:03 | 5000 | 4 | 0.365711804 | 0.513711233 | 0.397510626 | 117.466344 | 150.435963 |
| 5. C++/TFLite FP32 + XNNPACK | 11.11.2026. 18:09:14 | 50 | 4 | 0.410425851 | 0.552407994 | 0.453770563 | 85.431748 | 95.221360 |
| 6. C++/TFLite FP32 + XNNPACK | 11.11.2026. 18:23:31 | 5000 | 4 | 0.365711977 | 0.513711181 | 0.397511274 | 115.516409 | 131.939457 |
| 7. RPi C++/TFLite FP32 + XNNPACK | 12.11.2026. 14:40:47 | 50 | 4 | 0.410425851 | 0.552407994 | 0.453770563 | 420.947902 | 442.650930 |
| 8. RPi C++/TFLite FP32 + XNNPACK | 12.11.2026. 15:46:53 | 5000 | 4 | 0.365712541 | 0.513711214 | 0.397511281 | 425.772376 | 447.709495 |

Zajednički parametri: YOLOv8n, CPU, COCO val2017, deterministički izbor po rastućem ID-u, 640×640, batch 1, `conf=0.001`, NMS IoU 0.7, najviše 100 detekcija, pet warmup prolaza isključenih iz merenja. Provereno je da Python, NCNN i TFLite koriste iste ID-eve istim redosledom za odgovarajuću veličinu testa. TFLite testovi su završeni: postoje i timing i COCO metrics izveštaji.

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

Timing sadrži 50 zapisa; proseci su provereni prema `per_image`. Tri glavne AP metrike jednake su Python i NCNN rezultatu na 50 slika na sačuvanoj preciznosti. To podržava ispravnost konverzije i dekodiranja, ali ne dokazuje identičnost svih detekcija. Zbir merenih ukupnih obrada iznosi 4761.068 ms.

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

## Poređenje laptop i Raspberry Pi TFLite rezultata na 5000 slika

| Metrika | Laptop TFLite + XNNPACK | RPi TFLite + XNNPACK |
|---|---:|---:|
| CPU niti | 4 | 4 |
| mAP50–95 | 0.36571197672614925 | 0.36571254064182906 |
| mAP50 | 0.5137111814279052 | 0.5137112137860647 |
| mAP75 | 0.3975112740264031 | 0.3975112811256434 |
| Učitavanje, ms/slika | 10.410996 | 8.137549 |
| Pretprocesiranje, ms/slika | 4.343535 | 8.312392 |
| Inference, ms/slika | 115.516409 | 425.772376 |
| Postprocesiranje, ms/slika | 1.667835 | 5.482990 |
| Ukupno, ms/slika | 131.939457 | 447.709495 |
| Slike/s iz prosečnog ukupnog vremena | 7.58 | 2.23 |

RPi inference vreme je **3.69 puta veće**, a ukupno vreme obrade **3.39 puta veće** od laptop rezultata u ovim sačuvanim prolazima. Razlika mAP50–95 je samo **+0.000000563915680** (oko **+0.0000564 procentna poena**), što pokazuje vrlo blisku agregatnu tačnost. To nije dokaz identičnih pojedinačnih predikcija. Poređenje menja hardver i uslove izvršavanja; ne izoluje efekte runtime-a ili arhitekture procesora.

## Poređenje laptop TFLite, NCNN i Python rezultata na 5000 slika

| Metrika | Python/PyTorch | C++/NCNN (istorijski) | C++/TFLite + XNNPACK |
|---|---:|---:|---:|
| CPU niti | 8 | 4 | 4 |
| mAP50–95 | 0.365719332 | 0.365711804 | 0.365711977 |
| mAP50 | 0.513690330 | 0.513711233 | 0.513711181 |
| mAP75 | 0.397506553 | 0.397510626 | 0.397511274 |
| Inference, ms/slika | 124.917075 | 117.466344 | 115.516409 |
| Predict / ukupno, ms/slika | 152.878522 | 150.435963 | 131.939457 |
| Učitavanje, ms/slika | Nije izdvojeno | 24.913897 | 10.410996 |

TFLite inference ima približno **7.53% manje prosečno vreme od Python baseline-a** i **1.66% manje od NCNN-a** u ovim prolazima. Razlika mAP50–95 prema Python-u iznosi približno −0.00000735552918125, odnosno −0.000736 procentnih poena.

Razlika ukupne obrade ne potiče samo od inference-a: učitavanje je 10.41 ms u TFLite testu naspram 24.91 ms u NCNN testu. Uzrok te razlike nije zabeležen; bez kontrole uslova ne može se pripisati runtime-u. TFLite inference na 5000 slika je 115.52 ms, a na 50 slika 85.43 ms; ova dva prolaza nisu nezavisna ponavljanja istog testa i ne daju procenu stabilnosti performansi.

## Istorijsko poređenje NCNN-a sa Python-om na 5000 istih slika

| Metrika | Python/PyTorch | C++/NCNN | Razlika C++ − Python |
|---|---:|---:|---:|
| mAP50–95 | 0.3657193322553305 | 0.3657118042028542 | −0.0000075280524763 |
| mAP50 | 0.5136903303753626 | 0.5137112326245359 | +0.0000209022491733 |
| mAP75 | 0.3975065525507988 | 0.3975106255931308 | +0.0000040730423320 |
| Inference, ms/slika | 124.917075 | 117.466344 | −7.450731 |
| Predict / ukupno, ms/slika | 152.878522 | 150.435963 | −2.442559 |

NCNN inference ima približno 5.96% manje prosečno vreme, a ukupna obrada približno 1.60% manje vreme u odnosu na Python `predict` poziv u ovim sačuvanim prolazima. Tačnost je vrlo bliska: razlika mAP50–95 iznosi oko −0.000753 procentna poena.

Ovo poređenje opisuje konkretna merenja sa različitim runtime-ovima i brojem niti (Python 8, NCNN 4). Ne izoluje efekat programskog jezika i ne potvrđuje ponovljivo ubrzanje.

## Dodatne COCO metrike

Za Python su ove vrednosti preuzete iz tekstualnog `coco_summary` i zato imaju samo tri decimale. NCNN i TFLite vrednosti su radi prikaza zaokružene na isti broj decimala; pune vrednosti su u `all_coco_stats`.

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

- Python: `timing_ms.mean_by_stage` sadrži Ultralytics vremena faza. `mean_predict_call` obuhvata učitavanje slike i bibliotečki overhead; ne obuhvata izvoz JSON-a, COCO evaluaciju i ispis napretka. Zbir vremena faza zato nije jednak vremenu celog poziva.
- C++: `means.total_ms` obuhvata učitavanje, pretprocesiranje, inference i postprocesiranje. NCNN inference obuhvata i kreiranje extractora, dok TFLite inference meri `Invoke()` već pripremljenog interpretera. JSON sklapanje/serijalizacija, ispis i COCO evaluacija nisu uključeni. Ukupno se meri zasebno od pojedinačnih faza.
- TFLite: učitavanje modela, kreiranje interpretera/delegata i alokacija tenzora nisu uključeni. XNNPACK je eksplicitno uključen, ali nema izveštaja koji potvrđuje da su svi operatori delegirani. Nema sačuvanog testa sa `builtin` backendom.
- Oba NCNN testa beleže NCNN tag `20250503`, `precision="FP32 weights and compute"`, četiri niti i pet warmup prolaza.
- Za prvi pojedinačni test postoji sačuvana slika, ali nema numeričkog izveštaja. Vizualizacija 50 slika takođe nema zaseban benchmark. Do 12.11.2026. pronađeno je osam numeričkih testova navedenih u pregledu, uključujući dva RPi FP32 testa; nisu pronađeni FP16/INT8 rezultati ili RAM merenja.
- Pri ažuriranju ponovo su provereni svi C++ timing JSON fajlovi: broj zapisa, redosled ID-eva i proseci prema `per_image`. Za nove RPi predictions fajlove proverena je pripadnost ID-eva manifestu. COCO metrike su preuzete iz sačuvanih metrics JSON fajlova; COCO evaluacija nije ponovo izvršavana.
- Nema sačuvanih nezavisnih ponavljanja iz kojih bi se procenila varijabilnost proseka. Vrednosti su dokumentacija postojećih prolaza.
