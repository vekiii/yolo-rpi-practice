# Evidencija rada: YOLOv8n, PyTorch, NCNN i TFLite FP32

Dokument sastavljen 09.11.2026, ažuriran 11.11.2026. na osnovu lokalnih fajlova, rezultata i Git istorije zaključno sa commitom `054e586`. Numerički rezultati su u [results.md](results.md). Aktivni C++ runtime sada je TFLite sa XNNPACK backendom; NCNN deo ostaje istorijska evidencija prethodne faze.

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

Istorijski dokazi u commitu `24599d5`: `yolov8n_ncnn_model/metadata.yaml`, `model.ncnn.param`, `model.ncnn.bin` i `model_ncnn.py`. Ovi fajlovi su kasnije uklonjeni iz radnog stabla. Metadata beleži izvoz 08.11.2026. u 18:49:40 +01:00, Ultralytics 8.4.202, batch 1 i ulaz 640×640. Izvoz nije zaseban test performansi.

## 5. C++ wrapper i priprema build-a — 08.11.2026.

Commit `8737150`, 19:33:31 +01:00 (`add c++ wrapper`), dodaje `ncnn_fp32_wrapper/main.cpp`, `coco_tools.py`, `CMakeLists.txt`, `README.md` i [aktiviranje_cpp.md](aktiviranje_cpp.md). NCNN izvori su kasnije uklonjeni; dostupni su u Git istoriji, npr. `git show 9d3e3af:ncnn_fp32_wrapper/main.cpp`. NCNN komande u ovom dokumentu opisuju prethodnu fazu i ne rade nad trenutnim stablom bez vraćanja izvora i modela.

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

## 8. Dokumentovanje i prelazak sa NCNN-a na TFLite — 09–10.11.2026.

Commit `a2896ac`, 09.11.2026. u 11:37:11 +01:00, dodaje `evidance.md` i `results.md`. Commit `a6cd793`, 10.11.2026. u 13:51:03 +01:00 (`delete ncnn runtime files`), uklanja NCNN wrapper i izvezeni NCNN model. Ranije metrike i timing fajlovi ostaju lokalno; NCNN predictions fajlovi za 50 i 5000 slika trenutno nisu prisutni.

Commit `239e1c5`, 10.11.2026. u 13:56:20 +01:00 (`tflite wrapper add`), dodaje [C++ wrapper](tflite_cpp_wrapper/main.cpp), [izvoz](tflite_cpp_wrapper/export_fp32.py), [COCO pomoćnu skriptu](tflite_cpp_wrapper/coco_tools.py), [CMake konfiguraciju](tflite_cpp_wrapper/CMakeLists.txt), [uputstvo](tflite_cpp_wrapper/README.md) i [zapis početne validacije](tflite_cpp_wrapper/VALIDATION.md).

Novi wrapper koristi TFLite C++ Interpreter API, sa eksplicitnim XNNPACK CPU delegatom ili opcionim `builtin` backendom. Sačuvani testovi koriste `xnnpack`; nije zabeležen poseban `builtin` test. Nije potvrđeno da su svi operatori delegirani XNNPACK-u. FP16 nije forsiran, a wrapper zahteva FP32 ulaz i izlaz. GPU i NNAPI su isključeni u CMake-u.

## 9. Izvoz TFLite FP32 modela i build — 11.11.2026.

Commit `dce9036`, 15:09:07 +01:00 (`tflite 32FP model add`), dodaje model i prateće zapise. Izvozna skripta poziva `model.export(format='litert', imgsz=640, batch=1, device='cpu', quantize=32)`.

Dokazi: [model](models/tflite_fp32/yolov8n_fp32.tflite), [ugovor modela](models/tflite_fp32/yolov8n_fp32.contract.json) i [paketi izvoznog okruženja](models/tflite_fp32/yolov8n_fp32.environment.txt). Ugovor beleži WSL2 Linux x86_64, Python 3.12.16 i Ultralytics 8.4.208. Ovo je izvozno okruženje, ne dokaz da je C++ benchmark izvršen u WSL-u.

- Ulaz: FP32 `[1,3,640,640]` (NCHW).
- Izlaz: FP32 `[1,84,8400]`, dekodirani xywh i 80 verovatnoća; koordinate `normalized`.
- Veličina modela: 12841242 bajta.
- SHA-256 modela: `c2cbb871b2690f9aaf1a3c500e651c0560016a25f7685bcaa781ff4eac80b50b`.
- SHA-256 izvornih težina: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.

Pri ažuriranju dokumentacije provereno je da trenutni model i kopija `yolov8n.tflite` odgovaraju hash-u iz ugovora, kao i da `yolov8n.pt` odgovara hash-u izvornih težina. Timing fajlovi ne čuvaju hash modela, pa je ovo provera trenutnih artefakata, a ne nezavisna potvrda hash-a u trenutku izvršavanja.

Commit `054e586`, 18:09:05 +01:00 (`edit configuration`), podešava dinamički MSVC runtime, TensorFlow verzijske definicije i include putanje. CMake vezuje TensorFlow/TFLite `v2.20.0`, OpenCV `4.11.0` i nlohmann/json `3.11.3`.

Komande predviđene projektom (izvoz u odvojenom Linux okruženju, build u Windows MSVC developer PowerShell-u):

```bash
python tflite_cpp_wrapper/export_fp32.py
```

```powershell
cmake -S .\tflite_cpp_wrapper -B .\tflite_cpp_wrapper\build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build .\tflite_cpp_wrapper\build --parallel 4
```

## 10. TFLite FP32 provera na 50 slika — 11.11.2026.

```powershell
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\coco_tools.py prepare
.\tflite_cpp_wrapper\build\bin\yolo_tflite.exe .\models\tflite_fp32\yolov8n_fp32.tflite .\val2017 .\results\manifest_50.json .\results\tflite_fp32_50 4 normalized xnnpack
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\coco_tools.py evaluate --predictions results/tflite_fp32_50_predictions.json
```

Dokazi: [timing](results/tflite_fp32_50_timing.json), [detekcije](results/tflite_fp32_50_predictions.json) i [metrike](results/tflite_fp32_50_metrics.json). Timing je izmenjen u 18:08:01, metrike u 18:09:14. Vreme timing fajla prethodi poslednjem CMake commitu; tačna revizija izvršnog fajla nije zabeležena.

Provereno je 50 timing zapisa, poklapanje redosleda ID-eva sa Python testom i tačnost sačuvanih proseka prema pojedinačnim vremenima. Tri glavne AP metrike jednake su Python i NCNN testu na 50 slika na sačuvanoj preciznosti.

## 11. TFLite FP32 evaluacija na 5000 slika — 11.11.2026.

```powershell
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\coco_tools.py prepare --limit 0 --manifest results/manifest_5000.json
.\tflite_cpp_wrapper\build\bin\yolo_tflite.exe .\models\tflite_fp32\yolov8n_fp32.tflite .\val2017 .\results\manifest_5000.json .\results\tflite_fp32_5000 4 normalized xnnpack
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\coco_tools.py evaluate --manifest results/manifest_5000.json --predictions results/tflite_fp32_5000_predictions.json --output results/tflite_fp32_5000_metrics.json
```

Dokazi: [timing](results/tflite_fp32_5000_timing.json), [detekcije](results/tflite_fp32_5000_predictions.json) i [metrike](results/tflite_fp32_5000_metrics.json). Timing je izmenjen u 18:21:24, metrike u 18:23:31. Prisustvo oba izveštaja dokumentuje završen prolaz i evaluaciju.

Provereno je 5000 timing zapisa, redosled ID-eva identičan Python testu i svi proseci iz `means` prema `per_image`. Oba TFLite testa beleže runtime `2.20.0`, `xnnpack`, četiri niti, pet warmup prolaza, batch 1, 640×640, `conf=0.001`, IoU 0.7 i najviše 100 detekcija. RGB /255, letterbox i NMS implementirani su u wrapperu; mrežni graf izvršava `interpreter->Invoke()`.

## Granice zaključivanja

Python rezultati beleže 8 niti; NCNN i TFLite rezultati 4 niti. Menjaju se runtime, implementacija obrade i verzija izvoznog alata. Nema dokaza o ponovljenim nezavisnim benchmark prolazima, kontroli napajanja, RAM profilu ili testiranju na Raspberry Pi uređaju. Sačuvani Python environment navodi Windows 10, Python 3.12.10 i Intel64 Family 6 Model 154 Stepping 4; tačan komercijalni naziv CPU-a nije naveden. C++ timing fajlovi ne beleže potpun hardverski i kompajlerski profil.

`notes.docx` je u prethodnom pregledu sadržao pitanja o batch-u, rezoluciji, metrikama, granicama merenja, FP16, CMake-u i MSVC-u. Pitanja nisu dokaz izvršenih eksperimenata. README/VALIDATION napomene o neizvršenom C++ buildu opisuju stanje pri pripremi wrappera; lokalni JSON fajlovi zasebno dokumentuju kasnije rezultate. Pri ovom ažuriranju nisu ponovo pokretani izvoz, build ili benchmark.
