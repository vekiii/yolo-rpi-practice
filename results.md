# Rezultati testova

Sastavljeno 09.11.2026. iz sačuvanih lokalnih JSON fajlova. Razvojni koraci i poreklo dokaza: [evidance.md](evidance.md). Testovi nisu ponovo pokretani za potrebe dokumentovanja.

## Pregled po redosledu sačuvanih rezultata

Datumi i vremena označavaju poslednju izmenu izveštaja, a ne početak testa. AP vrednosti su na skali 0–1; vremena su proseci u ms po slici.

| Test | Datum | Slika | Niti | mAP50–95 | mAP50 | mAP75 | Inference | Predict / ukupno |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1. Python/PyTorch | 04.11.2026. 12:47:42 | 50 | 8 | 0.410425851 | 0.552407994 | 0.453770563 | 107.942174 | 130.867954 |
| 2. Python/PyTorch FP32 baseline | 04.11.2026. 14:36:26 | 5000 | 8 | 0.365719332 | 0.513690330 | 0.397506553 | 124.917075 | 152.878522 |
| 3. C++/NCNN FP32 | 08.11.2026. 20:47:47 | 50 | 4 | 0.410425851 | 0.552407994 | 0.453770563 | 122.462036 | 156.249160 |
| 4. C++/NCNN FP32 | 08.11.2026. 21:10:03 | 5000 | 4 | 0.365711804 | 0.513711233 | 0.397510626 | 117.466344 | 150.435963 |

Zajednički parametri: YOLOv8n, CPU, COCO val2017, deterministički izbor po rastućem ID-u, 640×640, batch 1, `conf=0.001`, NMS IoU 0.7, najviše 100 detekcija, pet warmup prolaza isključenih iz merenja. Provereno je da Python i NCNN koriste iste ID-eve istim redosledom za odgovarajuću veličinu testa.

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

Izvori: [timing](results/ncnn_fp32_50_timing.json), [COCO metrike](results/ncnn_fp32_50_metrics.json), [detekcije](results/ncnn_fp32_50_predictions.json), [manifest](results/manifest_50.json).

| Faza | Prosek, ms/slika |
|---|---:|
| Učitavanje slike | 25.100156 |
| Pretprocesiranje | 6.586240 |
| Inference | 122.462036 |
| Postprocesiranje | 2.099906 |
| Ukupna obrada | 156.249160 |

Timing sadrži 50 zapisa. mAP50–95, mAP50 i mAP75 jednaki su Python testu na 50 slika na preciznosti sačuvanoj u JSON-u. Prosečan inference i ukupna obrada sporiji su od odgovarajućeg Python testa.

## 4. C++/NCNN FP32 — 5000 slika

Izvori: [timing](results/ncnn_fp32_5000_timing.json), [COCO metrike](results/ncnn_fp32_5000_metrics.json), [detekcije](results/ncnn_fp32_5000_predictions.json), [manifest](results/manifest_5000.json).

| Faza | Prosek, ms/slika |
|---|---:|
| Učitavanje slike | 24.913897 |
| Pretprocesiranje | 6.194501 |
| Inference | 117.466344 |
| Postprocesiranje | 1.860334 |
| Ukupna obrada | 150.435963 |

Timing sadrži 5000 zapisa. Proseci su u `means.inference_ms`, `means.total_ms` i ostalim poljima objekta `means` (početak objekta je na liniji 20092 izvornog fajla).

Zbir sačuvanih `total_ms` iznosi 752179.8155 ms, odnosno oko 12 min 32.18 s. To je zbir merenih obrada slika, a ne ukupno trajanje procesa sa warmup-om, ispisom, JSON eksportom i COCO evaluacijom.

## Poređenje na 5000 istih slika

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

Za Python su ove vrednosti preuzete iz tekstualnog `coco_summary` i zato imaju samo tri decimale. NCNN vrednosti su radi prikaza zaokružene na isti broj decimala; pune vrednosti su u `all_coco_stats`.

| Metrika | Python 50 | Python 5000 | NCNN 50 | NCNN 5000 |
|---|---:|---:|---:|---:|
| AP small | 0.252 | 0.173 | 0.252 | 0.173 |
| AP medium | 0.469 | 0.404 | 0.469 | 0.404 |
| AP large | 0.663 | 0.518 | 0.663 | 0.518 |
| AR maxDets=1 | 0.322 | 0.303 | 0.322 | 0.303 |
| AR maxDets=10 | 0.518 | 0.492 | 0.518 | 0.492 |
| AR maxDets=100 | 0.531 | 0.531 | 0.531 | 0.531 |
| AR small | 0.331 | 0.289 | 0.331 | 0.289 |
| AR medium | 0.600 | 0.592 | 0.600 | 0.591 |
| AR large | 0.729 | 0.703 | 0.729 | 0.703 |

## Značenje vremena i ograničenja

- Python: `timing_ms.mean_by_stage` sadrži Ultralytics vremena faza. `mean_predict_call` obuhvata učitavanje slike i bibliotečki overhead; ne obuhvata izvoz JSON-a, COCO evaluaciju i ispis napretka. Zbir vremena faza zato nije jednak vremenu celog poziva.
- C++: `means.total_ms` obuhvata učitavanje, pretprocesiranje, inference i postprocesiranje. Inference obuhvata i kreiranje NCNN extractora. JSON sklapanje/serijalizacija, ispis i COCO evaluacija nisu uključeni. Ukupno se meri zasebno od pojedinačnih faza.
- Oba NCNN testa beleže NCNN tag `20250503`, `precision="FP32 weights and compute"`, četiri niti i pet warmup prolaza.
- Za prvi pojedinačni test postoji sačuvana slika, ali nema numeričkog izveštaja. Vizualizacija 50 slika takođe nema zaseban benchmark. Nisu pronađeni drugi sačuvani numerički testovi, FP16/INT8 rezultati, RAM merenja ili Raspberry Pi rezultati.
- Nema sačuvanih nezavisnih ponavljanja iz kojih bi se procenila varijabilnost proseka. Vrednosti su dokumentacija postojećih prolaza.
