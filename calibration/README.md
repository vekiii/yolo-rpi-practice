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
