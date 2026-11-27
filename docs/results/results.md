# Raspberry Pi: evaluacija na 5000 slika

Svih šest prolaza koristi YOLOv8n, iste COCO val2017 slike istim redosledom, TFLite C++ 2.20.0, XNNPACK i četiri niti. Ulaz je 640×640 uz batch 1, prag pouzdanosti 0,001, NMS IoU 0,7 i najviše 100 detekcija po slici.

## Poređenje modela

| Metrika | FP32 | FP16 težine | INT8, EMA min–max (0,95) | INT8, EMA min–max (0,95), 189 slika po klasi | INT8, globalni min–max | Dinamički INT8 |
|---|---:|---:|---:|---:|---:|---:|
| mAP50–95 | 0.365712541 | 0.365741991 | 0.281063989 | 0.280016293 | 0.287795150 | 0.362365098 |
| mAP50 | 0.513711214 | 0.513939253 | 0.454534217 | 0.456046117 | 0.459993896 | 0.512821305 |
| mAP75 | 0.397511281 | 0.397913689 | 0.308673822 | 0.304985808 | 0.316572124 | 0.395935278 |
| Učitavanje, ms/slika | 8.137549 | 9.262373 | 9.640214 | 9.243673 | 9.833755 | 9.253076 |
| Preprocess, ms/slika | 8.312392 | 8.202460 | 8.246468 | 8.194952 | 8.358723 | 8.152348 |
| Inference, ms/slika | 425.772376 | 440.392685 | 210.095403 | 211.496281 | 213.196596 | 382.009705 |
| Postprocess, ms/slika | 5.482990 | 5.523155 | 5.443570 | 5.474946 | 5.525058 | 5.487267 |
| Ukupno, ms/slika | 447.709495 | 463.385106 | 233.429724 | 234.413728 | 236.918113 | 404.906291 |
| Slike/s iz prosečnog ukupnog vremena | 2.23 | 2.16 | 4.28 | 4.27 | 4.22 | 2.47 |

Na istih 5000 slika globalni min–max ima **0,67 procentnih poena viši mAP50–95** od EMA varijante, uz **1,48% duže prosečno vreme inference**. U odnosu na FP32, EMA INT8 ima **50,66% kraće prosečno vreme inference** i **8,46 procentnih poena niži mAP50–95**. Vremena su iz pojedinačnih sačuvanih prolaza.

Dinamički INT8 je u ovom prolazu imao **10,28% kraće prosečno vreme inference** od FP32, uz **0,33 procentna poena niži mAP50–95**.

**Novi kalibracioni skup (189 slika po klasi, 14.283 jedinstvene slike):** EMA INT8 postiže mAP50–95 0,280016, što je **0,09 procentnih poena niže** od EMA kalibrisanog na 500 nasumičnih slika (0,281064) i **8,57 procentnih poena niže** od FP32. Prosečno vreme inference je 211,50 ms, odnosno 0,67% više nego kod EMA sa 500 slika. Ovo je jedan prolaz po varijanti, pa razlika u mAP od 0,09 pp ne pokazuje da je neki od dva kalibraciona skupa bolji. Globalni min–max sa novim skupom još nije rađen.

## Dodatne COCO metrike

| Metrika | FP32 | FP16 težine | INT8, EMA min–max (0,95) | INT8, EMA min–max (0,95), 189 slika po klasi | INT8, globalni min–max | Dinamički INT8 |
|---|---:|---:|---:|---:|---:|---:|
| AP small | 0.173344 | 0.173503 | 0.106638 | 0.109835 | 0.111708 | 0.174526 |
| AP medium | 0.403669 | 0.403404 | 0.294872 | 0.299685 | 0.303871 | 0.399218 |
| AP large | 0.518428 | 0.518614 | 0.442368 | 0.438796 | 0.438322 | 0.516434 |
| AR maxDets=1 | 0.303053 | 0.302694 | 0.254274 | 0.251453 | 0.254534 | 0.303165 |
| AR maxDets=10 | 0.492428 | 0.492285 | 0.416631 | 0.418265 | 0.415194 | 0.491443 |
| AR maxDets=100 | 0.530585 | 0.530298 | 0.455594 | 0.456711 | 0.452559 | 0.529373 |
| AR small | 0.288927 | 0.288991 | 0.227309 | 0.225000 | 0.225156 | 0.295077 |
| AR medium | 0.591390 | 0.590679 | 0.503641 | 0.509338 | 0.497058 | 0.586219 |
| AR large | 0.703091 | 0.703210 | 0.633530 | 0.635597 | 0.627949 | 0.707196 |

## Statistika vremena po slici

Sve vrednosti su u ms, iz 5000 pojedinačnih timing zapisa po modelu. Standardna devijacija je uzoračka (n−1).

| Model | Faza | Mean | Median | Std | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | Preprocess | 8.312392 | 8.147996 | 0.531358 | 6.929957 | 10.797741 |
| FP32 | Inference | 425.772376 | 425.561561 | 3.001331 | 420.287201 | 525.867355 |
| FP32 | Postprocess | 5.482990 | 5.452058 | 0.152634 | 5.272367 | 7.219903 |
| FP16 težine | Preprocess | 8.202460 | 8.023194 | 0.557028 | 6.840084 | 14.296861 |
| FP16 težine | Inference | 440.392685 | 440.229800 | 1.744963 | 433.021123 | 462.672541 |
| FP16 težine | Postprocess | 5.523155 | 5.494257 | 0.144905 | 5.318276 | 6.721144 |
| INT8, EMA min–max (0,95) | Preprocess | 8.246468 | 8.087449 | 0.521639 | 6.931173 | 14.314424 |
| INT8, EMA min–max (0,95) | Inference | 210.095403 | 209.903901 | 0.941081 | 208.651681 | 234.099782 |
| INT8, EMA min–max (0,95) | Postprocess | 5.443570 | 5.419756 | 0.115944 | 5.272063 | 6.286271 |
| INT8, EMA (189 slika po klasi) | Preprocess | 8.194952 | 8.034282 | 0.525206 | 6.802643 | 12.840251 |
| INT8, EMA (189 slika po klasi) | Inference | 211.496281 | 211.232154 | 1.280201 | 210.042407 | 254.506857 |
| INT8, EMA (189 slika po klasi) | Postprocess | 5.474946 | 5.452336 | 0.115656 | 5.316128 | 6.405466 |
| INT8, globalni min–max | Preprocess | 8.358723 | 8.187354 | 0.533911 | 6.994361 | 10.661327 |
| INT8, globalni min–max | Inference | 213.196596 | 212.675070 | 5.217607 | 211.876820 | 334.183883 |
| INT8, globalni min–max | Postprocess | 5.525058 | 5.501118 | 0.114690 | 5.355421 | 7.216392 |
| Dinamički INT8 | Preprocess | 8.152348 | 7.989830 | 0.520300 | 6.805408 | 12.683914 |
| Dinamički INT8 | Inference | 382.009705 | 381.796583 | 3.787483 | 377.199037 | 517.321130 |
| Dinamički INT8 | Postprocess | 5.487267 | 5.458101 | 0.146536 | 5.280479 | 6.535549 |

## Metodologija i izvori

Preprocess priprema ulazni tenzor, inference meri samo TFLite `Invoke()`, a postprocess uključuje dekodiranje izlaza, filtriranje i NMS. Ukupno vreme uključuje učitavanje slike. Zapisivanje rezultata i COCO evaluacija su van merenja. Timing izveštaji beleže pet internih warmup poziva.

Sve INT8 varijante imaju FP32 ulaz i izlaz. EMA koristi faktor 0,95, a globalni min–max zadržava najmanji minimum i najveći maksimum po tenzoru preko istih 500 kalibracionih slika. Kolona sa 189 slika po klasi koristi [novi kalibracioni skup](../../calibration/README.md) i isti EMA recept. Dinamički model koristi INT8 težine i FP32 aktivacije na granicama operacija u grafu, uz izvoz bez kalibracije. Preciznost izvršavanja pojedinačnih kernela nije profilisana. [Statička kalibracija](../guides/tflite_int8.md) · [Dinamički model](../../models/tflite_dynamic_int8/README.md)

- FP32 · [vremena](../../results/eval_5000/rpi_tflite_fp32_5000_timing.json) · [COCO metrike](../../results/eval_5000/rpi_tflite_fp32_5000_metrics.json) · [predikcije](../../results/eval_5000/rpi_tflite_fp32_5000_predictions.json)
- FP16 težine · [vremena](../../results/eval_5000/rpi_tflite_fp16_weights_5000_timing.json) · [COCO metrike](../../results/eval_5000/rpi_tflite_fp16_weights_5000_metrics.json) · [predikcije](../../results/eval_5000/rpi_tflite_fp16_weights_5000_predictions.json)
- INT8, EMA min–max (0,95) · [vremena](../../results/eval_5000/rpi_tflite_int8_5000_timing.json) · [COCO metrike](../../results/eval_5000/rpi_tflite_int8_5000_metrics.json) · [predikcije](../../results/eval_5000/rpi_tflite_int8_5000_predictions.json)
- INT8, globalni min–max · [vremena](../../results/eval_5000/rpi_tflite_int8_global_minmax_5000_timing.json) · [COCO metrike](../../results/eval_5000/rpi_tflite_int8_global_minmax_5000_metrics.json) · [predikcije](../../results/eval_5000/rpi_tflite_int8_global_minmax_5000_predictions.json)
- Dinamički INT8 · [vremena](../../results/eval_5000/rpi_tflite_dynamic_int8_5000_timing.json) · [COCO metrike](../../results/eval_5000/rpi_tflite_dynamic_int8_5000_metrics.json) · [predikcije](../../results/eval_5000/rpi_tflite_dynamic_int8_5000_predictions.json)
- INT8, EMA min–max (0,95), 189 slika po klasi · [vremena](../../results/eval_5000/rpi_tflite_int8_189_per_class_ema_5000_timing.json) · [COCO metrike](../../results/eval_5000/rpi_tflite_int8_189_per_class_ema_5000_metrics.json) · [predikcije](../../results/eval_5000/rpi_tflite_int8_189_per_class_ema_5000_predictions.json)
