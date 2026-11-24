# COCO train2017 kalibracioni skup

Komande pokreni iz `D:\yolo_test` u PowerShell-u. Skripta koristi postojeći Python 3.11+ i Pillow iz projektnog okruženja.

```powershell
# Samo izaberi slike i sačuvaj manifest, bez mreže.
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\prepare_calibration.py select

# Preuzmi izabrane slike ili nastavi prekinuto preuzimanje istom komandom.
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\prepare_calibration.py prepare

# Proveri kompletan skup i napravi pregled bez mreže.
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\prepare_calibration.py verify

# Proveri izbor i nastavak na malom lokalnom primeru.
.\.venv\Scripts\python.exe -m unittest discover -s tflite_cpp_wrapper -p test_prepare_calibration.py
```

Čuvaj i ponovo koristi [manifest](coco_train2017_500_seed42/manifest.json). Redosled je rezultat `random.Random(42).sample` nad zapisima `images` sortiranim po numeričkom ID-u. Preuzimanje koristi originalni `coco_url` iz anotacija, uz COCO adrese oblika `http://images.cocodataset.org/train2017/<file_name>` kad to polje nedostaje. Fajlovi se upisuju privremeno i prihvataju posle dekodiranja i provere dimenzija. `sha256: null` označava da hash još nije zabeležen. Istu komandu `prepare` pokreni ponovo za nastavak. Podrazumevano koristi šest radnika, tri pokušaja po slici i mrežni timeout od 20 sekundi. Brojeve možeš da podesiš sa `--jobs`, `--attempts` i `--timeout`.

Posle uspešne provere, `preview_first50.png` u direktorijumu skupa prikazuje prvih 50 stavki u redosledu manifesta, uz nazive i ID-eve. Umanjenje važi samo za pregled. Originalni JPEG bajtovi ostaju u `images/`. Provera prijavljuje grupe sa istim SHA-256, uz zadržavanje svih izabranih stavki.

Plan izvoza je statička INT8 post-trening kvantizacija sa `quantize=8`, uz isti manifest za sva uporediva ispitivanja. Evaluacija koristi odvojeni `val2017`. Planirano izvozno okruženje koristi Ultralytics 8.4.208, litert-torch 0.10.0, ai-edge-litert 2.3.0 i ai-edge-quantizer 0.10.1. U ovom izvozniku FP32 ulaz i izlaz su eksplicitno očuvani. Preciznost detekcione glave treba utvrditi pregledom konačnog grafa.

## Stratifikovan izbor 189 slika po klasi

Kategorija **89 (`hair drier`) ima 189 različitih slika** sa validnom anotacijom. Taj broj je cilj za svaku od 80 kategorija.

Izbor razmatra 80 kategorija po stvarnom COCO ID-ju. Anotacija mora imati `iscrowd=0` i konačne pozitivne širinu, visinu i area. Sortirani kandidati svake klase uzorkuju se zajedničkim lokalnim `random.Random(42)`. Plan ima 15.120 izbora pre deduplikacije. Sortirani unikatni ID-jevi se zatim promešaju drugim `random.Random(42)`. Manifest čuva baš taj redosled. Izveštaji JSON, CSV i SVG razlikuju ciljane izbore, stvarnu pokrivenost slika i stvaran broj validnih instanci. Pozadinske klase i slike sa više objekata mogu dati neujednačenu raspodelu. Poređenje sa random-500 istovremeno menja veličinu i sastav skupa, pa se efekat balansiranja ne može izdvojiti iz tog poređenja.

Iz PowerShell-a u `D:\yolo_test`, redom nakon uspeha svakog koraka:

```powershell
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\prepare_calibration_per_class.py select
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\prepare_calibration_per_class.py download
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\prepare_calibration_per_class.py verify
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\prepare_calibration_per_class.py preview
```

Posle uspešnog izbora, `class_distribution.json`, `class_distribution.csv` i `class_distribution.svg` su u `calibration/coco_train2017_189_per_class_seed42/`. Ista komanda `download` nastavlja preuzimanje. Prvo proverava postojeći fajl u novom skupu, zatim traži kopiju u random-500 skupu ili lokalnom `train2017/`. Tek zatim pokušava mrežni URL. Verifikacija ponovo izračunava izbor i redosled iz izvornog JSON-a, proverava odvojenost od val2017 i traži ispravnu datoteku sa hash vrednošću za svaki ID. Komanda `preview` posle provere generiše `preview_first50.png` za ručni pregled.

Posle uspešne verifikacije i ručnog pregleda, pokreni dve kalibracije iz WSL-a:

```bash
cd /mnt/d/yolo_test
python_bin=~/venvs/yolo-tflite-export-py312/bin/python
manifest=calibration/coco_train2017_189_per_class_seed42/manifest.json
"$python_bin" tflite_cpp_wrapper/quantize_int8.py --manifest "$manifest" --output-dir models/tflite_int8_189_per_class_ema
"$python_bin" tflite_cpp_wrapper/quantize_int8_global_minmax.py --manifest "$manifest" --output-dir models/tflite_int8_189_per_class_global_minmax
```

Obe komande porede postavke sa odgovarajućim random-500 prolazom, koriste TFLite profiler sa 16 niti i iste FP32 ulazne i izlazne tipove. Preprocessing ostaje OpenCV linearni resize i letterbox 640×640 uz isto zaokruživanje i padding 114, RGB, normalizaciju sa 255 i NCHW FP32. Broj jedinstvenih stavki u manifestu određuje broj uzoraka za oba prolaza kroz podatke. Privremeni memorijski mapirani fajl na disku traži oko 4,69 MiB po slici. Ako na izlaznom disku nema dovoljno prostora, `--scratch-dir` može pokazivati na drugi disk. QSV se ponovo koristi samo ako se poklapaju hash modela i manifesta, redosled uzoraka, metoda i ostale postavke.

U PowerShell-u iz `D:\yolo_test` pokreni evaluaciju svake varijante na istom postojećem manifestu od 50 slika, sa četiri niti i XNNPACK-om:

```powershell
foreach ($variant in 'ema', 'global_minmax') {
    $model = ".\models\tflite_int8_189_per_class_${variant}\yolov8n_int8.tflite"
    $prefix = ".\results\eval_50\tflite_int8_189_per_class_${variant}_50"
    $manifest = '.\results\eval_50\manifest_50.json'
    foreach ($suffix in 'predictions', 'timing', 'metrics') {
        if (Test-Path -LiteralPath "${prefix}_$suffix.json") { throw "Izlaz već postoji: $prefix" }
    }
    .\tflite_cpp_wrapper\build\bin\yolo_tflite.exe $model .\val2017 $manifest $prefix 4 normalized xnnpack
    if ($LASTEXITCODE -ne 0) { throw 'Inferenca nije uspela' }
    .\.venv\Scripts\python.exe .\tflite_cpp_wrapper\coco_tools.py evaluate --manifest $manifest --predictions "${prefix}_predictions.json" --output "${prefix}_metrics.json"
    if ($LASTEXITCODE -ne 0) { throw 'COCO evaluacija nije uspela' }
}
```
