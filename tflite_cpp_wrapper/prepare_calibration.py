"""Prepare the fixed COCO train2017 calibration subset from the project root."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import random
import shutil
import sys
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

from PIL import Image, ImageDraw

TRAIN = Path("annotations/instances_train2017.json")
VAL = Path("annotations/instances_val2017.json")
ROOT = Path("calibration/coco_train2017_500_seed42")
COUNT = 500
SEED = 42
METHOD = "random.Random(42).sample on images sorted by numeric id, without replacement; retain sample order"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_images(path):
    return json.loads(path.read_bytes())["images"]


def sample(images):
    return random.Random(SEED).sample(sorted(images, key=lambda row: int(row["id"])), COUNT)


def source_record(row):
    name = row["file_name"]
    if not name.isascii() or not name.endswith(".jpg") or not name[:-4].isdigit():
        raise ValueError(f"Unexpected COCO filename: {name!r}")
    url = row.get("coco_url") or f"http://images.cocodataset.org/train2017/{name}"
    parsed = urlsplit(url)
    if (parsed.scheme not in ("http", "https") or parsed.netloc != "images.cocodataset.org"
            or parsed.path != f"/train2017/{name}" or parsed.query or parsed.fragment):
        raise ValueError(f"Unexpected train2017 URL: {url!r}")
    return dict(id=int(row["id"]), file_name=name, coco_url=row.get("coco_url"),
                url=url, local_path=f"images/{name}",
                width=row["width"], height=row["height"])


def save_manifest(path, manifest):
    temp = path.with_suffix(".json.part")
    temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_manifest(train_images, val_images, source_hash, root, create):
    path = root / "manifest.json"
    by_id = {int(row["id"]): row for row in train_images}
    val_ids = {int(row["id"]) for row in val_images}
    if len(by_id) != len(train_images):
        raise ValueError("Duplicate ID in source train2017 images list")
    expected_selection = sample(train_images)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    elif create:
        manifest = dict(seed=SEED, selection=METHOD, source=str(TRAIN.as_posix()),
                        source_sha256=source_hash, image_count=COUNT,
                        images=[dict(source_record(row), sha256=None)
                                for row in expected_selection])
    else:
        raise ValueError(f"Manifest missing: {path}")

    if (manifest["source_sha256"] != source_hash or manifest["seed"] != SEED
            or manifest["selection"] != METHOD or manifest["image_count"] != COUNT):
        raise ValueError("Manifest does not match the expected source or selection settings")
    records = manifest["images"]
    ids = {row["id"] for row in records}
    names = {row["file_name"] for row in records}
    if len(records) != COUNT or len(ids) != COUNT or len(names) != COUNT:
        raise ValueError("Expected 500 distinct IDs and filenames")
    if not ids <= by_id.keys() or ids & val_ids:
        raise ValueError("Selected IDs must belong to train2017 and be disjoint from val2017")
    if [row["id"] for row in records] != [int(row["id"]) for row in expected_selection]:
        raise ValueError("Manifest IDs or their order differ from the expected seed-42 selection")
    for row in records:
        expected = source_record(by_id[row["id"]])
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError(f"Manifest metadata differs from train2017 for ID {row['id']}")
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        save_manifest(path, manifest)
    return manifest


def check_image(path, row):
    with Image.open(path) as image:
        if image.format != "JPEG" or image.size != (row["width"], row["height"]):
            raise ValueError(f"Incorrect format or dimensions: {path}")
        image.load()
    digest = sha256(path)
    if row["sha256"] is not None and digest != row["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return digest


def acquire(root, row, attempts, timeout):
    target = root / row["local_path"]
    if target.is_file():
        try:
            return check_image(target, row)
        except (OSError, ValueError):
            pass
    temp = target.with_suffix(".jpg.part")
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(row["url"], timeout=timeout) as response, temp.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            digest = check_image(temp, row)
            temp.replace(target)
            return digest
        except (OSError, ValueError, HTTPException) as error:
            if attempt == attempts:
                raise RuntimeError(f"{row['id']} {row['url']}: {error}") from error
            time.sleep(2 ** (attempt - 1))
        finally:
            temp.unlink(missing_ok=True)


def download(root, manifest, jobs, attempts, timeout):
    (root / "images").mkdir(exist_ok=True)
    failures = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        pending = {pool.submit(acquire, root, row, attempts, timeout): row
                   for row in manifest["images"]}
        for count, future in enumerate(as_completed(pending), 1):
            row = pending[future]
            try:
                digest = future.result()
            except RuntimeError as error:
                failures.append(str(error))
                print(error, file=sys.stderr, flush=True)
            else:
                if row["sha256"] != digest:
                    row["sha256"] = digest
                    save_manifest(root / "manifest.json", manifest)
            if count % 25 == 0 or count == COUNT:
                print(f"Checked {count}/{COUNT}, failures: {len(failures)}", flush=True)
    if failures:
        raise ValueError(f"Incomplete calibration set: {len(failures)} download failures")


def verify(root, manifest):
    groups = defaultdict(list)
    errors = []
    for row in manifest["images"]:
        try:
            if row["sha256"] is None:
                raise ValueError("No saved SHA-256")
            groups[check_image(root / row["local_path"], row)].append(row["id"])
        except (OSError, ValueError) as error:
            errors.append(f"{row['id']}: {error}")
    if errors:
        raise ValueError(f"Verification failed for {len(errors)}/{COUNT} images. "
                         + " | ".join(errors[:5]))
    for digest, ids in groups.items():
        if len(ids) > 1:
            print(f"Identical content: {digest} IDs={ids}", flush=True)
    print(f"Verified {COUNT} JPEGs with expected dimensions and hashes. "
          f"Duplicate-content groups: {sum(len(ids) > 1 for ids in groups.values())}", flush=True)


def preview(root, manifest):
    cell_w, cell_h = 180, 146
    sheet = Image.new("RGB", (5 * cell_w, 10 * cell_h), "#202020")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(manifest["images"][:50]):
        x, y = index % 5 * cell_w, index // 5 * cell_h
        with Image.open(root / row["local_path"]) as original:
            thumb = original.convert("RGB")
            thumb.thumbnail((cell_w - 8, cell_h - 28), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 2))
        draw.text((x + 5, y + cell_h - 24),
                  f"{row['file_name']}  ID {row['id']}", fill="white")
    temp = root / "preview_first50.png.part"
    sheet.save(temp, format="PNG")
    temp.replace(root / "preview_first50.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "prepare", "verify"))
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 16 or not 1 <= args.attempts <= 5 or not 0 < args.timeout <= 300:
        parser.error("Expected jobs 1–16, attempts 1–5 and timeout >0 and <=300 seconds")
    try:
        manifest = load_manifest(read_images(TRAIN), read_images(VAL), sha256(TRAIN),
                                 ROOT, create=args.action != "verify")
        print(f"Validated selection: {ROOT / 'manifest.json'}", flush=True)
        if args.action == "prepare":
            download(ROOT, manifest, args.jobs, args.attempts, args.timeout)
        if args.action != "select":
            verify(ROOT, manifest)
            preview(ROOT, manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
