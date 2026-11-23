# RPi 4 — TFLite C++ wrapper: priprema, konfiguracija i build preko SSH-a

Uputstvo beleži postupak projekta YOLO na RPI-ju, sa konačnim ispravkama potrebnim za uspešan ARM32 build i test na 50 slika. Ako izvršni fajl već postoji i radi, ne treba ponavljati konfiguraciju i kompilaciju: pređi na odeljak 8.

## 1. Potvrđeno okruženje

| Stavka | Vrednost |
|---|---|
| Uređaj | Raspberry Pi 4, Cortex-A72, oko 8 GB RAM-a |
| Sistem | Raspbian GNU/Linux 13, trixie |
| Korisničko okruženje | 32-bitno, `armhf` |
| Kernel | `aarch64` |
| Kompajler | GCC/G++ 14.2, cilj `arm-linux-gnueabihf` |
| CMake | 3.31.6 |
| Projekat | `/home/veki/yolo_test` |
| Build | `tflite_cpp_wrapper/build-rpi` |
| Runtime | TFLite Interpreter C++ API, TensorFlow izvori v2.20.0 |
| Backend | XNNPACK CPU, četiri niti za inferencu |
| Ostale biblioteke | OpenCV 4.11.0, nlohmann/json 3.11.3 |

`uname -m` opisuje kernel, a ne nužno bitnost izvršnog fajla. U ovom slučaju 64-bitni kernel i 32-bitni kompajler zahtevaju pažljivo zadat ARM32 cilj za zavisnosti.

## 2. Povezivanje i osnovni alati

**Windows PowerShell na laptopu:**

```powershell
ssh veki@veki.local
```

Ako ime nije dostupno na konkretnoj mreži, koristi poznatu trenutnu IP adresu RPi-ja umesto `veki.local`.

**RPi, SSH terminal:**

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config python3 ca-certificates curl unzip protobuf-compiler file

cat /etc/os-release
getconf LONG_BIT
uname -m
dpkg --print-architecture
gcc -dumpmachine
cmake --version
protoc --version
free -h
df -h ~ /tmp
mkdir -p ~/yolo_test
```

U potvrđenom okruženju: `LONG_BIT=32`, `armhf`, `arm-linux-gnueabihf`, približno 7,3 GiB dostupnog RAM-a i 49 GB slobodnog prostora. Ne menjati swap samo radi ponavljanja ovog postupka; postojeći sistem je imao 2 GiB swap-a.

`protobuf-compiler` je važan: bez njega je build kasnije stao sa `protoc: not found`. U ovom objedinjenom postupku instalira se unapred.

## 3. Prenos projekta sa laptopa

RPi-ju trebaju izvori wrapper-a, TFLite model, manifest i slike. Ne prenose se Windows build, `.exe`, Windows `.venv` niti MSVC biblioteke. COCO anotacije ostaju na laptopu za računanje mAP-a.

Ako je paket već prenet i raspakovan, preskoči ovaj odeljak. Ponovno raspakivanje prepisuje istoimene izvore, pa ga ne radi preko naknadno izmenjenih fajlova bez provere.

**Windows PowerShell, iz `D:\yolo_test`:**

```powershell
cd D:\yolo_test
```

```powershell
@'
import json
import tarfile
from pathlib import Path

manifest = json.loads(Path('results/eval_50/manifest_50.json').read_text(encoding='utf-8'))

def exclude_build(info):
    if any(p in {'build', 'build-rpi', '__pycache__', '.git'} for p in Path(info.name).parts):
        return None
    return info

with tarfile.open('rpi_test_50.tar.gz', 'w:gz') as archive:
    archive.add('tflite_cpp_wrapper', filter=exclude_build)
    archive.add('models/tflite_fp32')
    archive.add('results/eval_50/manifest_50.json')
    for image in manifest['images']:
        archive.add(Path('val2017') / image['file_name'])
print('Paket je napravljen; broj slika:', len(manifest['images']))
'@ | .\.venv\Scripts\python.exe -
```

```powershell
scp .\rpi_test_50.tar.gz veki@veki.local:~/yolo_test/
```

**RPi, SSH:**

```bash
cd ~/yolo_test
tar -xzf rpi_test_50.tar.gz
ls tflite_cpp_wrapper
ls models/tflite_fp32
find val2017 -type f -name '*.jpg' | wc -l
```

Za početni paket očekuje se 50 slika. Ako je kasnije prenet ceo dataset, broj je 5.000.

Koristi se postojeći, uspešno prilagođeni projektni `CMakeLists.txt`, opisan i u `tflite_wrapper.md`: TensorFlow v2.20.0, XNNPACK ON, GPU/NNAPI/external delegate OFF, OpenCV samo core/imgproc/imgcodecs, C++17. Windows postavke su u `if(MSVC)` i ne primenjuju se na GCC. Konačna putanja programa je `build-rpi/bin/yolo_tflite`.

## 4. Priprema host alata flatc — samo ako već ne postoji

Build je imao dve faze: prvo je lokalno napravljen `flatc`; zatim je konačna konfiguracija zadala ARM32 cilj i eksplicitno koristila taj alat. Ne treba nasumično menjati arhitekturu ili brisati build folder između faza.

Prvo proveri:

```bash
cd ~/yolo_test
./tflite_cpp_wrapper/build-rpi/flatbuffers-flatc/bin/flatc --version
```

Ako radi, pređi na odeljak 5.

Ako je ovo nov build, pripremi početnu native konfiguraciju bez `CMAKE_SYSTEM_NAME` i `CMAKE_SYSTEM_PROCESSOR`:

```bash
cmake -S tflite_cpp_wrapper -B tflite_cpp_wrapper/build-rpi \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_C_FLAGS="-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard -mfp16-format=ieee" \
  -DCMAKE_CXX_FLAGS="-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard -mfp16-format=ieee"
```

Ako je u folderu ostao keš prvog, neuspešnog pokušaja sa `CMAKE_SYSTEM_*`, ovu početnu komandu pokreni sa `cmake --fresh` umesto `cmake`. Nemoj dodavati `--fresh` pri svakom običnom nastavku builda.

U istorijskom postupku `flatc` je napravljen tokom početne kompilacije. Za izdvojeno ponavljanje samo tog dela može se izgraditi odgovarajući CMake cilj:

```bash
cmake --build tflite_cpp_wrapper/build-rpi --target flatbuffers-flatc --parallel 2
./tflite_cpp_wrapper/build-rpi/flatbuffers-flatc/bin/flatc --version
```

Izdvojeni poziv cilja je skraćena reprodukcija host-alat faze; konačni uspešan build koristio je već postojeći `flatc` na gornjoj putanji. Ne prelazi na sledeći korak dok ta provera ne prođe. Ako cilj nije generisan, sačuvaj konfiguracionu grešku i proveri trenutnu strukturu builda umesto biranja drugog sistemskog flatc-a nepoznate verzije.

## 5. Konačna konfiguracija za ARM32

Ovo je ključna konfiguracija koja je rešila pogrešan izbor `aarch64-isa.c` u cpuinfo zavisnosti. Eksplicitni ARM32 cilj sprečava da se izbor izvora osloni na 64-bitni kernel; `TFLITE_HOST_TOOLS_DIR` rešava zahtev za host `flatc` alatom.

**RPi, SSH:**

```bash
cd ~/yolo_test

cmake --fresh -S tflite_cpp_wrapper -B tflite_cpp_wrapper/build-rpi \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=armv7 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_C_FLAGS="-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard -mfp16-format=ieee" \
  -DCMAKE_CXX_FLAGS="-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard -mfp16-format=ieee" \
  -DTFLITE_HOST_TOOLS_DIR="$HOME/yolo_test/tflite_cpp_wrapper/build-rpi/flatbuffers-flatc/bin"
```

`--fresh` osvežava CMake keš i CMakeFiles, ne uklanja ceo build folder sa preuzetim izvorima i napravljenim host alatom. Ovde je nameran pri promeni konfiguracije; ne koristi ga za običan nastavak kompilacije.

Sačekaj `Configuring done`, `Generating done` i završetak bez fatalne greške. `-mfp16-format=ieee` je opcija za podršku tipu u zavisnostima, ne zahtev za FP16 izvršavanje modela.

## 6. Kompilacija

```bash
cmake --build tflite_cpp_wrapper/build-rpi --parallel 2
```

Prva kompilacija traje dugo. Ako se prekine zbog greške ili SSH prekida, proveri da li proces još radi pre pokretanja novog. Posle ispravke nastavi istom komandom; ne briši build-rpi.

| Simptom | Rešenje korišćeno u postupku |
|---|---|
| Traži se host flatc pri ARM konfiguraciji | Pripremi lokalni flatc i zadaj `TFLITE_HOST_TOOLS_DIR` |
| cpuinfo bira `aarch64-isa.c` za armhf kompajler | Konačna konfiguracija sa `CMAKE_SYSTEM_PROCESSOR=armv7` |
| `protoc: not found` | `sudo apt install -y protobuf-compiler`, pa nastavak builda |
| Napomene GCC-a o ABI promeni iz GCC 7.1 | Razlikovati napomene/upozorenja od prve stvarne fatalne greške |

Ne primenjuj automatski Windows MSVC ispravke zaglavlja na GCC. Ako izvor već radi na RPi-ju, zadrži ga i postojeći build.

## 7. Provera programa

```bash
file tflite_cpp_wrapper/build-rpi/bin/yolo_tflite
./tflite_cpp_wrapper/build-rpi/bin/yolo_tflite
```

Potvrđen rezultat: `ELF 32-bit LSB executable, ARM, EABI5`, interpreter `/lib/ld-linux-armhf.so.3`. Program bez argumenata ispisuje:

```text
Usage: yolo_tflite MODEL.tflite IMAGES MANIFEST OUTPUT_PREFIX THREADS normalized|pixels [xnnpack|builtin]
```

Poziv bez argumenata prikazuje pomoć i može vratiti nenulti izlazni kod; to samo po sebi nije neuspešan build.

## 8. Provera inferencije na jednoj, zatim na 50 slika

**RPi, SSH:**

```bash
cd ~/yolo_test
python3 - <<'PY'
import json
from pathlib import Path
m = json.loads(Path('results/eval_50/manifest_50.json').read_text())
m['images'] = m['images'][:1]
Path('results/eval_50/manifest_1.json').write_text(json.dumps(m, indent=2))
PY
```

```bash
./tflite_cpp_wrapper/build-rpi/bin/yolo_tflite \
  models/tflite_fp32/yolov8n_fp32.tflite \
  val2017 results/eval_50/manifest_1.json results/eval_50/rpi_tflite_fp32_1 \
  4 normalized xnnpack
```

Ako prođe:

```bash
./tflite_cpp_wrapper/build-rpi/bin/yolo_tflite \
  models/tflite_fp32/yolov8n_fp32.tflite \
  val2017 results/eval_50/manifest_50.json results/eval_50/rpi_tflite_fp32_50 \
  4 normalized xnnpack
```

Potvrđeni oblici: ulaz `[1,3,640,640]`, izlaz `[1,84,8400]`. Pet zagrevajućih prolaza izuzeto je iz vremena. Podešavanja: batch 1, 640×640, četiri niti, normalizovane bbox koordinate, XNNPACK CPU.

**Windows PowerShell, iz `D:\yolo_test`:**

```powershell
scp veki@veki.local:~/yolo_test/results/eval_50/rpi_tflite_fp32_50_predictions.json .\results\eval_50\
scp veki@veki.local:~/yolo_test/results/eval_50/rpi_tflite_fp32_50_timing.json .\results\eval_50\
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\coco_tools.py evaluate --manifest results/eval_50/manifest_50.json --predictions results/eval_50/rpi_tflite_fp32_50_predictions.json --output results/eval_50/rpi_tflite_fp32_50_metrics.json
```

Potvrđena evaluacija na RPi rezultatima: mAP50–95 `0,410`, mAP50 `0,552`, AP75 `0,454`, do prikazane preciznosti jednako laptopu. COCO anotacije i Python evaluacija ostaju na laptopu.

## 9. Puna FP32 inferenca na 5.000 slika

Ako ceo dataset već postoji na RPi-ju, ne prenosi ga ponovo.

**Windows PowerShell:**

```powershell
cd D:\yolo_test
tar -cf val2017.tar val2017
scp .\val2017.tar veki@veki.local:~/yolo_test/
ssh veki@veki.local "mkdir -p ~/yolo_test/results/eval_5000"
scp .\results\eval_5000\manifest_5000.json veki@veki.local:~/yolo_test/results/eval_5000/
```

**RPi:**

```bash
cd ~/yolo_test
tar -xf val2017.tar
find val2017 -type f -name '*.jpg' | wc -l

./tflite_cpp_wrapper/build-rpi/bin/yolo_tflite \
  models/tflite_fp32/yolov8n_fp32.tflite \
  val2017 results/eval_5000/manifest_5000.json results/eval_5000/rpi_tflite_fp32_5000 \
  4 normalized xnnpack
```

Očekuje se 5.000 slika. Raspakivanje prepisuje prvih 50 istim fajlovima. Drži običnu SSH vezu otvorenom tokom rada; prekid veze može prekinuti proces. Zadrži iste uslove napajanja/hlađenja i bez drugih zahtevnih zadataka.

**Windows PowerShell:**

```powershell
scp veki@veki.local:~/yolo_test/results/eval_5000/rpi_tflite_fp32_5000_predictions.json .\results\eval_5000\
scp veki@veki.local:~/yolo_test/results/eval_5000/rpi_tflite_fp32_5000_timing.json .\results\eval_5000\
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\coco_tools.py evaluate --manifest results/eval_5000/manifest_5000.json --predictions results/eval_5000/rpi_tflite_fp32_5000_predictions.json --output results/eval_5000/rpi_tflite_fp32_5000_metrics.json
```

Ovo su komande pune evaluacije, ne tvrdnja da je njen konačni rezultat već zabeležen u ovom uputstvu. Sačuvaj predictions, timing i metrics fajlove. Za ponovljena merenja promeni izlazni prefiks da prethodni prolaz ne bude prepisan.

## 10. Nastavak rada

Za ponovnu inferencu: SSH → `cd ~/yolo_test` → odgovarajuća komanda iz odeljka 8 ili 9. Nije potrebno ponavljati instalaciju, izvoz ili build. Za izmene C++ koda koristi postojeći build-rpi i `cmake --build ... --parallel 2`.

Sačuvaj izvore, konačni CMake keš/podešavanja i rezultate. Windows i RPi imaju različite build foldere i izvršne fajlove, ali koriste isti `.tflite` model.
