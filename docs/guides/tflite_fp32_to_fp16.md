# YOLOv8n — postojeći TFLite FP32 u FP16 zapis težina

Ovo uputstvo dokumentuje uspešnu opciju 1: transformaciju već proverenog `.tflite` modela, bez ponovnog izvoza iz `.pt`, ONNX-a ili SavedModel-a. Obuhvata konverziju u WSL-u i proveru postojećim Windows C++ wrapper-om na istih 50 COCO slika.

## 1. Šta se menja

| Deo | Posle konverzije |
|---|---|
| Podržane konstantne FP32 vrednosti, uključujući težine i bias | FP16 zapis u fajlu |
| Ulaz / izlaz | FP32 |
| Originalne operacije i njihove veze | Očuvane |
| Dodatne operacije | `DEQUANTIZE`, FP16 konstante → FP32 |
| Cilj izvršavanja | Postojeći CPU TFLite/XNNPACK wrapper, bez forsiranja FP16 aritmetike |

Precizan naziv eksperimenta: **FP16 zapis konstanti/težina uz FP32 računanje**. Nije model sa FP16 aktivacijama i kompletnim FP16 izvršavanjem. Graf dobija dodatne operacije, iako se originalne operacije ne menjaju.

`tflite_fp16_weights.py` je projektna eksperimentalna skripta, ne zvanična TensorFlow/LiteRT CLI komanda. Koristi FlatBuffers šemu i transformaciju zasnovanu na obrascu FP16 konstante + DEQUANTIZE. Skripta pretvara sve podržane konstantne FP32 tenzore koji se koriste kao ulazi operacija; ne samo filtere konvolucija. Ne obavlja trening ni kalibraciju.

## 2. Fajlovi i okruženja

| Stavka | Putanja / okruženje |
|---|---|
| Windows projekat | `D:\yolo_test` |
| Isti projekat u WSL-u | `/mnt/d/yolo_test` |
| WSL Python okruženje | `~/venvs/yolo-tflite-export-py312` |
| Skripta | `D:\yolo_test\tflite_fp16_weights.py` |
| Izvorni model | `models/tflite_fp32/yolov8n_fp32.tflite` |
| Novi model | `models/tflite_fp16_weights/yolov8n_fp16_weights.tflite` |
| Izveštaj | `models/tflite_fp16_weights/yolov8n_fp16_weights.conversion.json` |
| Windows C++ program | `tflite_cpp_wrapper/build/bin/yolo_tflite.exe` |

Novi model namerno ostaje u `models/tflite_fp16_weights`, ne u root folderu. Izvorni FP32 model ostaje neizmenjen. Windows `.venv` koristi se za COCO evaluaciju, a postojeći WSL venv za konverziju. Nije potrebno praviti novo okruženje niti ponovo instalirati ceo izvoznik.

## 3. Aktiviranje WSL okruženja i provera zavisnosti

**Ubuntu/WSL terminal na laptopu:**

```bash
source ~/venvs/yolo-tflite-export-py312/bin/activate
cd /mnt/d/yolo_test
python --version
```

```bash
python - <<'PY'
from importlib.metadata import version
import numpy
import flatbuffers
from ai_edge_litert import schema_py_generated
from ai_edge_litert.interpreter import Interpreter
for package in ['ai-edge-litert', 'numpy', 'flatbuffers']:
    print(package, version(package))
print('Zavisnosti su dostupne')
PY
```

U potvrđenom export okruženju koristi se `ai-edge-litert==2.3.0`. NumPy i FlatBuffers su već dostupne zavisnosti. Zadrži postojeće verzije ako provera prođe; nema potrebe za nadogradnjom usred eksperimenta.

Proveri ulazne fajlove:

```bash
ls -lh tflite_fp16_weights.py models/tflite_fp32/yolov8n_fp32.tflite
```

## 4. Konverzija

Za prvo izvršavanje, dok izlaz još ne postoji:

```bash
python tflite_fp16_weights.py
```

Ista komanda sa eksplicitnim putanjama:

```bash
python tflite_fp16_weights.py \
  --input models/tflite_fp32/yolov8n_fp32.tflite \
  --output models/tflite_fp16_weights/yolov8n_fp16_weights.tflite
```

Ove dve komande su alternative: ne pokreći obe uzastopno. Skripta odbija prepisivanje modela ili postojećeg izveštaja. Pošto je tvoja konverzija već završena, za nastavak eksperimenta preskoči ovaj korak i koristi postojeći FP16 model.

Ako želiš zaseban ponovljeni pokušaj bez brisanja prethodnog rezultata:

```bash
python tflite_fp16_weights.py \
  --input models/tflite_fp32/yolov8n_fp32.tflite \
  --output models/tflite_fp16_weights/yolov8n_fp16_weights_repeat.tflite
```

To pravi novi model i novi izveštaj. Za njegov test koristi odgovarajuću novu putanju, dok glavni eksperiment nastavlja sa originalnim uspešno napravljenim FP16 fajlom.

## 5. Šta skripta proverava

Skripta proverava da model ima jedan podgraf, očekivane FP32 ulaze/izlaze i podržane konstante. Odbija custom/Flex operacije, spoljne bafere, promenljive/sparse tenzore, već prisutne FP16 tenzore i konstante sa nefinite vrednostima ili vrednostima izvan konačnog FP16 opsega.

Pri transformaciji:

1. Konstantne FP32 vrednosti zaokružuje u FP16.
2. Dodaje FP16 tenzore i `DEQUANTIZE` v3 operacije.
3. Zadržava originalne ID-jeve tenzora i veze originalnih operacija.
4. Uklanja neiskorišćene stare FP32 payload-e, uz proveru deljenih bafera.
5. Proverava da originalne operacije, signature definicije i ulaz/izlaz nisu izmenjeni.
6. Čuva pridružene ZIP fajlove/metapodatke ako postoje.
7. Alocira i izvršava novi model nad nultim ulazom kroz Python LiteRT builtin interpreter, bez podrazumevanih delegata, i proverava konačne izlazne vrednosti.
8. Zapisuje novi fajl i JSON izveštaj sa hash vrednostima i veličinama.

Nulti ulaz je provera izvršivosti, ne merenje tačnosti i ne potvrda rada XNNPACK-a. To se proverava sledećim C++/COCO testom.

## 6. Provera izveštaja i potvrđeni ishod konverzije

**WSL:**

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('models/tflite_fp16_weights/yolov8n_fp16_weights.conversion.json')
r = json.loads(p.read_text(encoding='utf-8'))
for key in ['converted_constants', 'original_operators', 'added_dequantize_ops',
            'original_ops_unchanged', 'io_unchanged', 'zero_input_builtin_invoke',
            'input_bytes', 'output_bytes', 'size_ratio']:
    print(key, ':', r[key])
print('Ulaz/izlaz:', r['io'])
PY
```

Potvrđeno u završenoj konverziji:

| Stavka | Rezultat |
|---|---:|
| Konvertovane konstante | 132 |
| Originalne operacije | 256 |
| Dodate DEQUANTIZE operacije | 132 |
| Ukupan broj operacija posle transformacije | 388 |
| Originalne operacije očuvane | `true` |
| Ulaz/izlaz očuvan | `true` |
| Provera nultim ulazom | `passed` |
| FP32 fajl | 12.841.242 bajta |
| FP16 fajl | 6.529.518 bajtova |

FP16 fajl ima oko 6,53 MB (decimalno), odnosno oko 6,23 MiB. Veličina je oko 50,85% originala: smanjenje oko 49,15%. To nije garancija upola manje potrošnje RAM-a.

Ulaz ostaje `[1,3,640,640]`, izlaz `[1,84,8400]`, FP32. U izveštaju numerička vrednost tipa `0` odgovara FLOAT32 u korišćenoj TFLite šemi. Bbox koordinate ostaju normalizovane, pa wrapper koristi argument `normalized`.

## 7. Windows C++ inferenca na istih 50 slika

**Windows PowerShell:**

```powershell
cd D:\yolo_test
.\tflite_cpp_wrapper\build\bin\yolo_tflite.exe .\models\tflite_fp16_weights\yolov8n_fp16_weights.tflite .\val2017 .\results\eval_50\manifest_50.json .\results\eval_50\tflite_fp16_weights_50 4 normalized xnnpack
```

Koristi se već napravljeni wrapper, bez ponovne kompilacije. Parametri ostaju isti kao za FP32: CPU, četiri niti, XNNPACK, 640×640, batch 1, confidence 0,001, NMS IoU 0,7, maksimalno 100 detekcija i pet zagrevajućih prolaza izuzetih iz merenja.

Nastaju:

- `results/eval_50/tflite_fp16_weights_50_predictions.json`;
- `results/eval_50/tflite_fp16_weights_50_timing.json`.

Ponovno korišćenje istog prefiksa prepisuje prethodne rezultate inferencije; za novi prolaz promeni prefiks, npr. `tflite_fp16_weights_50_run2`.

## 8. COCO evaluacija na laptopu

**Windows PowerShell:**

```powershell
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\coco_tools.py evaluate --manifest results/eval_50/manifest_50.json --predictions results/eval_50/tflite_fp16_weights_50_predictions.json --output results/eval_50/tflite_fp16_weights_50_metrics.json
```

Anotacije su u `annotations/instances_val2017.json`; skripta podrazumevano koristi tu putanju. Izbor slika je postojeći manifest, isti kao u FP32 proveri.

## 9. Potvrđeni rezultat na 50 slika

| Metrika | Rezultat |
|---|---:|
| mAP50–95 | 0,409114 |
| Inferenca | 74,70 ms/slika |
| Ukupna obrada | 83,84 ms/slika |
| Veličina modela | 6,53 MB |

Prethodna FP32 provera na istih 50 slika imala je mAP50–95 približno 0,4104. U odnosu na tu zaokruženu vrednost, odstupanje je približno −0,001286, odnosno −0,1286 procentnih poena.

Ovo je uspešna provera na 50 slika, ne konačan mAP za ceo dataset. Iz vremena se bez kontrolisanih ponavljanja ne zaključuje da je razlika izazvana samo FP16 zapisom: važni su stanje laptopa, napajanje, temperatura, pozadinski procesi i optimizacije runtime-a. Graf zadržava FP32 operacione ulaze/izlaze, a ne forsira nativnu FP16 aritmetiku.

## 10. Sledeće: puna evaluacija na 5.000 slika

Ove komande su pripremljen nastavak, ne već potvrđeni rezultat:

```powershell
.\tflite_cpp_wrapper\build\bin\yolo_tflite.exe .\models\tflite_fp16_weights\yolov8n_fp16_weights.tflite .\val2017 .\results\eval_5000\manifest_5000.json .\results\eval_5000\tflite_fp16_weights_5000 4 normalized xnnpack
```

```powershell
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\coco_tools.py evaluate --manifest results/eval_5000/manifest_5000.json --predictions results/eval_5000/tflite_fp16_weights_5000_predictions.json --output results/eval_5000/tflite_fp16_weights_5000_metrics.json
```

Sačuvaj originalni FP32 model, FP16 model, conversion JSON (uključujući hash vrednosti), korišćenu Python skriptu i predictions/timing/metrics fajlove. FP32 laptop referenca na svih 5.000 slika je mAP50–95 `0,365712`; poređenje pune FP16 tačnosti radi se tek kada puna evaluacija bude završena.

## 11. Nastavak u novom terminalu

Za konverziju: otvori WSL, aktiviraj postojeći `yolo-tflite-export-py312` venv i pređi u `/mnt/d/yolo_test`. Za C++ inferencu: Windows PowerShell iz `D:\yolo_test`, bez potrebe za MSVC aktivacijom ako se program samo izvršava. Za COCO evaluaciju koristi Windows `.venv\Scripts\python.exe`.

Za već napravljeni FP16 model ne ponavlja se konverzija, ne pokreće se ponovni `.pt` izvoz i ne menja se postojeći C++ build.
