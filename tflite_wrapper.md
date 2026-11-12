# YOLOv8n — TFLite C++ wrapper: kompletno uputstvo

Ovo uputstvo objedinjuje konačne korake korišćene za uspešan izvoz, Windows build i COCO evaluaciju. Ne sadrži neuspešne pokušaje. Podrazumeva raspakovan paket `tflite_cpp_wrapper`, postojeći `yolov8n.pt`, COCO slike i anotacije u projektu `D:\yolo_test`.

## 1. Okruženja i uloge

| Okruženje | Namena |
|---|---|
| Windows, postojeća Python 3.12.10 `.venv` | COCO evaluacija rezultata |
| WSL Ubuntu x86-64, zaseban Python 3.12 venv | Izvoz `.pt` u `.tflite` |
| Windows PowerShell sa aktiviranim MSVC x64 alatima | CMake konfiguracija, kompilacija i C++ inferenca |

WSL je Linux okruženje za izabrani LiteRT izvoznik. Nije potreban prilikom izvršavanja Windows C++ aplikacije. Windows putanja `D:\yolo_test` i WSL putanja `/mnt/d/yolo_test` označavaju iste fajlove.

Korišćene verzije: Ultralytics 8.4.208, torch 2.14.1+cpu, litert-torch 0.10.0, ai-edge-litert 2.3.0, TensorFlow C++ izvori v2.20.0, OpenCV 4.11.0 i nlohmann/json 3.11.3. WSL venv je u završenom eksperimentu koristio Python 3.12.16. Komanda `uv python install 3.12` bira raspoloživu patch verziju 3.12.

## 2. Preduslovi i fajlovi

U Windowsu su potrebni Git, CMake i Visual Studio Build Tools sa C++ x64 kompajlerom, NMake i Windows SDK-om. U WSL-u su potrebni pristup internetu i `curl`.

U projektu treba da postoje:

- `yolov8n.pt` — početni checkpoint;
- `tflite_cpp_wrapper/` — CMakeLists.txt, main.cpp, export_fp32.py, requirements-export.txt i coco_tools.py;
- `val2017/` i `annotations/` — COCO dataset;
- `results/manifest_50.json` i `results/manifest_5000.json` — isti manifesti kao u prethodnom eksperimentu;
- `.venv/` — postojeće Windows Python okruženje za evaluaciju.

Obezbediti prostor i na C: i na D:. WSL venv i privremeni fajlovi nisu automatski na D: samo zato što se projekat nalazi na D:.

### Virtuelna memorija korišćena za Windows build

Uspešan postupak uključivao je postojeći sistemski paging file na C: i dodatni na D:. Ove veličine su praktična postavka korišćena na laptopu sa 16 GB RAM-a, a ne univerzalni minimum za build.

1. Win + R → `SystemPropertiesAdvanced`.
2. Performance → Settings → Advanced → Virtual memory → Change.
3. Isključiti automatsko upravljanje za sve diskove radi pojedinačnog podešavanja.
4. C: → System managed size → Set.
5. D: → Custom size: Initial `16384` MB, Maximum `32768` MB → Set.
6. Potvrditi i restartovati Windows. D: mora imati prostor za paging file.

## 3. WSL: Python 3.12 okruženje

Sledeće komande izvršavati u Ubuntu/WSL terminalu.

Instalirati `uv` ako već nije instaliran:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Instalirati Python i napraviti zasebno export okruženje. Postojeći venv ne treba ponovo praviti.

```bash
uv python install 3.12
uv venv --python 3.12 --seed ~/venvs/yolo-tflite-export-py312
source ~/venvs/yolo-tflite-export-py312/bin/activate
cd /mnt/d/yolo_test
python --version
python -m pip --version
```

Sistemski Python Ubuntu-a i Windows `.venv` ostaju odvojeni od ovog okruženja.

## 4. CPU PyTorch i ograničenja zavisnosti

Instalirati kompatibilni CPU PyTorch direktno iz CPU repozitorijuma. Torchvision se razrešava uz zadatu verziju torch-a.

```bash
python -m pip install --no-cache-dir "torch==2.14.1+cpu" torchvision --index-url https://download.pytorch.org/whl/cpu
```

Sačuvati tačne instalirane verzije kao ograničenja i primeniti ih i na pip i na uv:

```bash
python -m pip freeze | awk '/^(torch|torchvision)==/' > ~/venvs/yolo-tflite-export-py312/cpu-constraints.txt
export PIP_CONSTRAINT="$HOME/venvs/yolo-tflite-export-py312/cpu-constraints.txt"
export UV_CONSTRAINT="$PIP_CONSTRAINT"
export PIP_NO_CACHE_DIR=1
```

Ove promenljive važe u trenutnom terminalu. Posle otvaranja novog terminala ponovo aktivirati venv i zadati sva tri `export` reda pre instalacije ili izvoza.

Provera:

```bash
python -c "import torch, torchvision; print('Torch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA build:', torch.version.cuda); print('CUDA dostupna:', torch.cuda.is_available())"
```

Očekivano: torch `2.14.1+cpu`, CUDA build `None`, CUDA dostupna `False`.

## 5. Ultralytics i biblioteke za izvoz

Za ponavljanje korišćene verzije instalirati:

```bash
python -m pip install "ultralytics==8.4.208"
```

Priloženi `requirements-export.txt` traži `ultralytics>=8.4.83`; ovde je namerno fiksirana verzija koja je korišćena u uspešnom izvozu.

Proveriti plan dodatne instalacije:

```bash
uv pip install --dry-run "litert-torch==0.10.0" "ai-edge-litert==2.3.0"
```

Plan treba da zadrži CPU torch bez dodavanja CUDA biblioteka. Zatim instalirati:

```bash
uv pip install --no-cache "litert-torch==0.10.0" "ai-edge-litert==2.3.0"
python -m pip check
```

Očekivano: `No broken requirements found.`

```bash
python -c "import torch; import litert_torch; print('Torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('LiteRT import OK')"
```

Očekivano: `2.14.1+cpu`, `None` i `LiteRT import OK`. U uspešnom postupku uv je po potrebi koristio kopiranje umesto hardlinkova; to ne sprečava instalaciju.

## 6. Izvoz FP32 TFLite modela

U istom WSL terminalu, iz `/mnt/d/yolo_test`:

```bash
python tflite_cpp_wrapper/export_fp32.py
```

Skripta koristi:

```python
model.export(format='litert', imgsz=640, batch=1, device='cpu', quantize=32)
```

Izvoz je potvrđen porukom `LiteRT: export success`. Skripta zatim ispisuje metapodatke i priprema model za wrapper.

| Stavka | Potvrđeni rezultat |
|---|---|
| Ulaz | `[1, 3, 640, 640]`, float32, NCHW |
| Izlaz | `[1, 84, 8400]`, float32 |
| Bbox jedinice | `normalized` |
| Veličina | približno 12,2 MB |

Windows fajlovi:

- `D:\yolo_test\yolov8n.tflite` — izlaz izvoznika;
- `D:\yolo_test\models\tflite_fp32\yolov8n_fp32.tflite` — kopija za wrapper;
- prateći fajlovi koje skripta napravi u `models/tflite_fp32/` — sačuvati radi evidencije izvoza.

Nema dodatnog kopiranja iz WSL-a jer je projekat na `/mnt/d/`.

U originalnom `.pt` checkpoint-u proverene su FP16 težine na disku; izvezeni model ovde ima FP32 zapis. To ne vraća ranije izgubljenu preciznost težina. Za rezultate razlikovati zapis težina, runtime i preciznost izvršavanja.

## 7. Windows: aktiviranje MSVC alata

Sledeće komande izvršavati u Windows PowerShell-u. Python `.venv` nije potreban za kompilaciju.

```powershell
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
& "$vsPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64
cd D:\yolo_test
cl
nmake /?
```

`cl` i `nmake` treba da prikažu Microsoft informacije. Konfiguraciju i build nastaviti u ovom istom terminalu. Ovo može biti i PowerShell terminal unutar VS Code-a. Aktivacija ne prelazi automatski u drugi već otvoren terminal.

## 8. Konačni CMakeLists.txt

U `D:\yolo_test\tflite_cpp_wrapper\CMakeLists.txt` koristiti sledeći objedinjeni sadržaj, sa svim postavkama koje su korišćene za uspešan build:

```cmake
cmake_minimum_required(VERSION 3.24)
project(yolo_tflite LANGUAGES C CXX)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    set(BUILD_WITH_STATIC_CRT OFF CACHE BOOL
        "Use dynamic MSVC runtime for OpenCV" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING
    "Compatibility for dependency CMake files")
set(TFLITE_ENABLE_XNNPACK ON CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_NNAPI OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_EXTERNAL_DELEGATE OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(tensorflow
    GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
    GIT_TAG v2.20.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR tensorflow/lite)
FetchContent_MakeAvailable(tensorflow)

target_compile_definitions(tensorflow-lite PUBLIC
    TF_MAJOR_VERSION=2
    TF_MINOR_VERSION=20
    TF_PATCH_VERSION=0
    "TF_VERSION_SUFFIX=\"\"")

set(BUILD_LIST core,imgproc,imgcodecs CACHE STRING "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
set(WITH_FFMPEG OFF CACHE BOOL "" FORCE)
set(WITH_OPENCL OFF CACHE BOOL "" FORCE)
set(WITH_IPP OFF CACHE BOOL "" FORCE)
set(WITH_ITT OFF CACHE BOOL "" FORCE)
set(WITH_PROTOBUF OFF CACHE BOOL "" FORCE)
set(WITH_WEBP OFF CACHE BOOL "" FORCE)
set(WITH_OPENEXR OFF CACHE BOOL "" FORCE)
set(WITH_AVIF OFF CACHE BOOL "" FORCE)
FetchContent_Declare(opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG 4.11.0 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(opencv)
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(json)

add_executable(yolo_tflite main.cpp)
target_include_directories(yolo_tflite PRIVATE
    ${opencv_SOURCE_DIR}/modules/core/include
    ${opencv_SOURCE_DIR}/modules/imgproc/include
    ${opencv_SOURCE_DIR}/modules/imgcodecs/include
    ${opencv_BINARY_DIR}
    ${CMAKE_BINARY_DIR})
target_link_libraries(yolo_tflite PRIVATE
    tensorflow-lite opencv_core opencv_imgproc opencv_imgcodecs
    nlohmann_json::nlohmann_json)
set_target_properties(yolo_tflite PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
```

`BUILD_SHARED_LIBS OFF` bira statičke biblioteke projekta, a `/MD` bira dinamički Microsoft C/C++ runtime. To su različite postavke i ova kombinacija je namerna.

## 9. Konfiguracija i priprema izvora

U aktiviranom MSVC PowerShell-u:

```powershell
cmake -S .\tflite_cpp_wrapper -B .\tflite_cpp_wrapper\build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
```

Sačekati `Configuring done`, `Generating done` i završetak bez fatalne greške. Prva konfiguracija preuzima biblioteke.

Proveriti da je FP16 zavisnost pripremljena:

```powershell
Test-Path .\tflite_cpp_wrapper\build\FP16-source\include\fp16.h
```

Očekivano je `True`. Ako izvor još nije pripremljen, konačni postupak za njegovu pripremu je:

```powershell
cmake --build .\tflite_cpp_wrapper\build\FP16-download
```

Zatim ponoviti proveru i glavnu CMake konfiguraciju. FP16 biblioteka je zavisnost build-a i ne određuje preciznost našeg modela.

### Najava klase Tensor za korišćeni MSVC build

Pre glavne kompilacije otvoriti:

```text
D:\yolo_test\tflite_cpp_wrapper\build\_deps\tensorflow-src\tensorflow\lite\core\model_building.h
```

Odmah posle sledeće linije dodati `class Tensor;`:

```cpp
using Quantization = std::variant<NoQuantization, AffineQuantization>;

class Tensor;
```

Ostaviti postojeći `friend class Tensor;` u klasi Buffer. Ova eksplicitna najava korišćena je u uspešnom Windows build-u. Izmena je u preuzetoj zavisnosti: pri novom preuzimanju izvora proveriti da je ponovo prisutna. Postojeći build folder ne brisati prilikom običnog nastavka kompilacije.

## 10. Kompilacija i provera izvršnog fajla

```powershell
cmake --build .\tflite_cpp_wrapper\build --parallel 2
```

Korišćena je zadata paralelnost 2; konkretni generator NMake određuje kako se poslovi izvršavaju. Kompilacija runtime-a i zavisnosti može potrajati.

Po završetku:

```powershell
Test-Path .\tflite_cpp_wrapper\build\bin\yolo_tflite.exe
.\tflite_cpp_wrapper\build\bin\yolo_tflite.exe
```

Očekivano: `True` i prikaz upotrebe:

```text
Usage: yolo_tflite MODEL.tflite IMAGES MANIFEST OUTPUT_PREFIX THREADS normalized|pixels [xnnpack|builtin]
```

## 11. Inferenca i COCO evaluacija na 50 slika

Koristiti postojeći manifest da izbor slika ostane isti kao u PyTorch eksperimentu.

```powershell
.\tflite_cpp_wrapper\build\bin\yolo_tflite.exe .\models\tflite_fp32\yolov8n_fp32.tflite .\val2017 .\results\manifest_50.json .\results\tflite_fp32_50 4 normalized xnnpack
```

Poslednji argumenti: četiri niti, normalizovane bbox koordinate i XNNPACK CPU backend.

Evaluaciju pokrenuti Windows Python-om iz postojećeg projekta:

```powershell
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\coco_tools.py evaluate --manifest .\results\manifest_50.json --predictions .\results\tflite_fp32_50_predictions.json --output .\results\tflite_fp32_50_metrics.json
```

Potvrđeni zaokruženi rezultati: mAP50–95 `0.410`, mAP50 `0.552`, AP75 `0.454`. Prethodna PyTorch referenca za mAP50–95 bila je `0.4104`.

## 12. Inferenca i evaluacija na svih 5.000 slika

```powershell
.\tflite_cpp_wrapper\build\bin\yolo_tflite.exe .\models\tflite_fp32\yolov8n_fp32.tflite .\val2017 .\results\manifest_5000.json .\results\tflite_fp32_5000 4 normalized xnnpack
```

Po završetku:

```powershell
.\.venv\Scripts\python.exe .\tflite_cpp_wrapper\coco_tools.py evaluate --manifest .\results\manifest_5000.json --predictions .\results\tflite_fp32_5000_predictions.json --output .\results\tflite_fp32_5000_metrics.json
```

Sačuvati predictions, metrics i timing fajlove u `results/`, model i metapodatke izvoza. Pri ponavljanju promeniti OUTPUT_PREFIX i metrics izlaz ako treba sačuvati prethodni prolaz.

## 13. Potvrđeni rezultati eksperimenta

| Metrika na 5.000 slika | Python/PyTorch referenca | C++/TFLite + XNNPACK |
|---|---:|---:|
| mAP50–95 | približno 0,3657 | 0,365712 |
| Inferenca po slici | 124,92 ms | 115,52 ms |
| Ukupno po slici | 152,88 ms | 131,94 ms |
| FPS iz ukupnog vremena | 6,54 | 7,58 |

Podešavanja: isti laptop, CPU, COCO val2017, 640×640, batch 1 i četiri niti. FP32 TFLite zapis i FP32 izvršavanje predstavljaju referencu ovog wrapper-a. Izmereno vreme pripada konkretnoj implementaciji i runtime-u, pa se razlika ne pripisuje samo jeziku C++.

## 14. Nastavak rada u novom terminalu

Za ponovnu inferencu nije potrebno ponavljati izvoz ili build. Otvoriti Windows PowerShell, preći u `D:\yolo_test` i koristiti komande iz odeljaka 11–12.

Za novi build ponovo aktivirati MSVC alate iz odeljka 7. Za novi izvoz otvoriti WSL, aktivirati export venv i ponovo postaviti PIP_CONSTRAINT, UV_CONSTRAINT i PIP_NO_CACHE_DIR iz odeljka 4, pa pokrenuti export skriptu.

Ovo uputstvo dokumentuje potvrđen Windows laptop postupak. Raspberry Pi zahteva zasebnu ARM kompilaciju; Windows `.exe` se ne prenosi kao izvršni fajl za RPi.
