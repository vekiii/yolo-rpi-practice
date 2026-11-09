# Evidencija rada: YOLOv8n, PyTorch i C++/NCNN FP32

Dokument sastavljen 09.11.2026. na osnovu lokalnih projektnih fajlova, sačuvanih rezultata i Git istorije zaključno sa commitom `9d3e3af`. Numerički rezultati su u [results.md](results.md).

## Poreklo dokaza

- Commitovi potvrđuju razvoj koda, a ne datum izvršavanja testa.
- Redosled testova rekonstruisan je iz vremena poslednje izmene rezultatskih fajlova. JSON fajlovi nemaju zaseban datum izvršavanja; vreme izmene nije pouzdan zapis početka testa.
- `results/`, `runs/`, `.venv/`, `*.pt` i `*.docx` su ignorisani kroz [.gitignore](.gitignore). Rezultati i slike postoje lokalno, ali nisu sačuvani u pregledanim commitovima.
- Komande ispod rekonstruisane su iz skripti, README-a, parametara rezultata i commit poruka; osim komande izvoza iz commit poruke, nisu potvrđene sačuvanom istorijom terminala.

## 1. Inicijalizacija — 04.11.2026.

Commit `f288cac`, 12:18:16 +01:00 (`init`), dodaje početni `.gitignore`.

Ulazni resursi postoje lokalno: `yolov8n.pt`, direktorijum `val2017/` i `annotations/instances_val2017.json`. Nema sačuvane evidencije o njihovom preuzimanju niti posebnog treninga modela.

## 2. Prva inferencija i Python evaluacija — 04.11.2026.

Commit `633c807`, 13:48:10 +01:00 (`init evaluations`), dodaje [test_inference.py](test_inference.py) i [evaluate_coco.py](evaluate_coco.py).

`test_inference.py` bira prvi JPG po imenu i pokreće YOLOv8n na CPU-u sa `imgsz=640`, uz čuvanje vizuelnog izlaza. Lokalni dokaz je [prva obrađena slika](runs/detect/first_test/000000000139.jpg). Za ovaj pojedinačni test nije pronađen sačuvan numerički izveštaj o vremenu ili tačnosti.

Evaluaciona skripta bira slike po rastućem COCO ID-u, mapira 80 klasa na COCO kategorije, isključuje warmup iz merenja i računa COCO bbox metrike samo nad izabranim slikama.

Sačuvana su dva testa:

1. 50 slika: [evaluation_50.json](results/evaluation_50.json), izmenjen 04.11.2026. u 12:47:42; pripadajući `evaluation_50_predictions.json`.
2. 5000 slika: [evaluation_fp32_5000.json](results/evaluation_fp32_5000.json), izmenjen 04.11.2026. u 14:36:26; pripadajući `evaluation_fp32_5000_predictions.json`.

Vreme prvog rezultata prethodi commitu skripte; kod je mogao biti izvršen pre commitovanja, ali tačna revizija korišćena u testu nije zabeležena u rezultatu.

Komande koje odgovaraju sačuvanim parametrima:

```powershell
.\.venv\Scripts\python.exe .\test_inference.py
.\.venv\Scripts\python.exe .\evaluate_coco.py --limit 50 --threads 8 --output results/evaluation_50.json
.\.venv\Scripts\python.exe .\evaluate_coco.py --limit 0 --threads 8 --output results/evaluation_fp32_5000.json
```

## 3. Vizuelna provera — 04.11.2026.

Commit `540f90b`, 14:39:12 +01:00 (`visualization script`), dodaje [visualization.py](visualization.py) i uklanja eksplicitni `half=False` iz evaluacione skripte.

Vizualizacija koristi ID-eve prvih 50 evaluiranih slika, ali prag `conf=0.25`, dok evaluacija koristi `conf=0.001`. Zato nacrtani prikazi nisu isti skup detekcija kao evaluacioni JSON.

Lokalno postoji 50 JPG prikaza u `runs/detect/results/visualization_50/`, npr. [prikaz slike 139](runs/detect/results/visualization_50/000000000139.jpg). Skripta u poruci navodi `results/visualization_50`, ali u tom direktorijumu nisu pronađeni JPG fajlovi. Vizualizacija nije zaseban benchmark.

## 4. Izvoz modela u NCNN FP32 — 08.11.2026.

Commit `24599d5`, 18:52:02 +01:00, sadrži komandu izvoza:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='ncnn', imgsz=640, batch=1, device='cpu', quantize=32)"
```

Dokazi: [metadata.yaml](yolov8n_ncnn_model/metadata.yaml), [model.ncnn.param](yolov8n_ncnn_model/model.ncnn.param), `model.ncnn.bin` i [model_ncnn.py](yolov8n_ncnn_model/model_ncnn.py). Metadata beleži izvoz 08.11.2026. u 18:49:40 +01:00, Ultralytics 8.4.202, batch 1 i ulaz 640×640. Izvoz nije zaseban test performansi.

## 5. C++ wrapper i priprema build-a — 08.11.2026.

Commit `8737150`, 19:33:31 +01:00 (`add c++ wrapper`), dodaje [main.cpp](ncnn_fp32_wrapper/main.cpp), [coco_tools.py](ncnn_fp32_wrapper/coco_tools.py), [CMakeLists.txt](ncnn_fp32_wrapper/CMakeLists.txt), [README](ncnn_fp32_wrapper/README.md) i [aktiviranje_cpp.md](aktiviranje_cpp.md).

Wrapper koristi CPU, C++17, NCNN `20250503`, OpenCV `4.11.0` i nlohmann/json `3.11.3`. Isključuje Vulkan, FP16, BF16 i INT8 izvršavanje. Obrada uključuje RGB, normalizaciju 1/255, letterbox 640×640 sa padding vrednošću 114, dekodiranje izlaza i NMS po klasi. Podešavanja su `conf=0.001`, `nms_iou=0.7`, `max_det=100`.

```powershell
cmake -S .\ncnn_fp32_wrapper -B .\ncnn_fp32_wrapper\build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build .\ncnn_fp32_wrapper\build --parallel 4
```

Commit `9d3e3af`, 21:27:14 +01:00 (`ncnn FP32 eval`), podešava MSVC runtime i dodatne include putanje i dopunjuje `.gitignore`. Sam commit ne sadrži numeričke rezultate, niti dokazuje da su raniji testovi izvršeni baš ovom revizijom build konfiguracije.

## 6. C++/NCNN provera na 50 slika — 08.11.2026.

```powershell
.\.venv\Scripts\python.exe .\ncnn_fp32_wrapper\coco_tools.py prepare
.\ncnn_fp32_wrapper\build\yolo_ncnn.exe .\yolov8n_ncnn_model .\val2017 .\results\manifest_50.json .\results\ncnn_fp32_50 4
.\.venv\Scripts\python.exe .\ncnn_fp32_wrapper\coco_tools.py evaluate --predictions results/ncnn_fp32_50_predictions.json
```

Dokazi: [manifest_50.json](results/manifest_50.json), [ncnn_fp32_50_timing.json](results/ncnn_fp32_50_timing.json), `ncnn_fp32_50_predictions.json` i [ncnn_fp32_50_metrics.json](results/ncnn_fp32_50_metrics.json). Timing je izmenjen u 20:47:08, metrike u 20:47:47.

Provereno je da se svih 50 ID-eva i njihov redosled u NCNN manifestu poklapaju sa Python rezultatom. Timing ima 50 zapisa. Tri glavne AP metrike jednake su Python rezultatu na sačuvanoj preciznosti; to samo po sebi ne dokazuje identičnost svih detekcija.

## 7. C++/NCNN evaluacija na 5000 slika — 08.11.2026.

```powershell
.\.venv\Scripts\python.exe .\ncnn_fp32_wrapper\coco_tools.py prepare --limit 0 --manifest results/manifest_5000.json
.\ncnn_fp32_wrapper\build\yolo_ncnn.exe .\yolov8n_ncnn_model .\val2017 .\results\manifest_5000.json .\results\ncnn_fp32_5000 4
.\.venv\Scripts\python.exe .\ncnn_fp32_wrapper\coco_tools.py evaluate --manifest results/manifest_5000.json --predictions results/ncnn_fp32_5000_predictions.json --output results/ncnn_fp32_5000_metrics.json
```

Dokazi: [manifest_5000.json](results/manifest_5000.json), [ncnn_fp32_5000_timing.json](results/ncnn_fp32_5000_timing.json), `ncnn_fp32_5000_predictions.json` i [ncnn_fp32_5000_metrics.json](results/ncnn_fp32_5000_metrics.json). Timing je izmenjen u 21:05:28, metrike u 21:10:03.

Provereno je poklapanje svih 5000 ID-eva i redosleda sa Python testom i prisustvo 5000 timing zapisa. Proseci se nalaze u objektu `means`, posle objekta `manifest`; `per_image` čuva pojedinačna vremena.

## Granice zaključivanja

Python rezultati beleže 8 niti; NCNN rezultati 4 niti. Menjaju se runtime i implementacija obrade. Nema dokaza o ponovljenim nezavisnim benchmark prolazima, kontroli napajanja, RAM profilu ili testiranju na Raspberry Pi uređaju. Sačuvani Python environment navodi Windows 10, Python 3.12.10 i Intel64 Family 6 Model 154 Stepping 4; tačan komercijalni naziv CPU-a nije naveden.

`notes.docx` sadrži pitanja o batch-u, rezoluciji, metrikama, granicama merenja, FP16, CMake-u i MSVC-u. Pitanja nisu dokaz izvršenih eksperimenata. README napomena da autor nije izvršio C++ build opisuje stanje pri pisanju uputstva; lokalni JSON fajlovi zasebno dokumentuju kasnije rezultate.
