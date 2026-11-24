"""Plan and prepare a deterministic, deduplicated COCO train2017 calibration set."""

import argparse
from collections import Counter
import csv
from html import escape
import json
import math
from pathlib import Path
import random

import prepare_calibration as base

ROOT = Path("calibration/coco_train2017_189_per_class_seed42")
SCHEMA = "coco_train2017_per_class/v1"
PER_CLASS = 189
SEED = 42
METHOD = ("Sort categories and candidate image IDs numerically; sample each category with a shared "
          "random.Random(42); sort union of IDs and shuffle with a separate random.Random(42).")


def valid(ann):
    if ann.get("iscrowd") != 0:
        return False
    bbox = ann.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    return all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0
               for v in (bbox[2], bbox[3], ann.get("area")))


def build_plan(coco, val_ids, source_hash, per_class=PER_CLASS, category_count=80):
    images = {row["id"]: row for row in coco["images"]}
    cats = {row["id"]: row["name"] for row in coco["categories"]}
    if len(images) != len(coco["images"]):
        raise ValueError("Duplicate image ID in train2017")
    if len(cats) != category_count or len(coco["categories"]) != category_count:
        raise ValueError(f"Expected {category_count} distinct COCO categories")
    candidates = {cid: set() for cid in cats}
    for ann in coco["annotations"]:
        if valid(ann):
            iid, cid = ann["image_id"], ann["category_id"]
            if iid not in images or cid not in cats:
                raise ValueError(f"Annotation references unknown image {iid} or category {cid}")
            candidates[cid].add(iid)
    shortages = [f"{cid} ({cats[cid]}): {len(candidates[cid])}"
                 for cid in sorted(cats) if len(candidates[cid]) < per_class]
    if shortages:
        raise ValueError(f"Need {per_class} distinct images with valid annotations per category. "
                         + "Below threshold: " + "; ".join(shortages))
    rng = random.Random(SEED)
    selections = [dict(category_id=cid, name=cats[cid],
                       image_ids=rng.sample(sorted(candidates[cid]), per_class))
                  for cid in sorted(cats)]
    union = {iid for category in selections for iid in category["image_ids"]}
    overlap = union & val_ids
    if overlap:
        raise ValueError(f"Selection overlaps val2017: {sorted(overlap)[:10]}")
    order = sorted(union)
    random.Random(SEED).shuffle(order)
    records = [dict(base.source_record(images[iid]), sha256=None) for iid in order]
    if len({row["file_name"] for row in records}) != len(records):
        raise ValueError("Selected training images have duplicate filenames")

    instances = Counter(ann["category_id"] for ann in coco["annotations"]
                        if ann["image_id"] in union and valid(ann))
    rows = [dict(category_id=cid, name=cats[cid], candidate_images=len(candidates[cid]),
                 sampled_images=per_class, actual_images=len(candidates[cid] & union),
                 valid_instances=instances[cid]) for cid in sorted(cats)]
    manifest = dict(schema_version=SCHEMA, seed=SEED, selection=METHOD,
                    filters={"iscrowd": 0, "bbox_width": ">0 and finite",
                             "bbox_height": ">0 and finite", "area": ">0 and finite"},
                    source=base.TRAIN.as_posix(), source_sha256=source_hash,
                    per_class=per_class, selections_before_dedup=category_count * per_class,
                    image_count=len(records), categories=selections, images=records)
    return manifest, rows


def load_plan(root, *, create=False):
    path = root / "manifest.json"
    if not path.exists() and not create:
        raise FileNotFoundError(f"Run select first: {path}")
    coco = json.loads(base.TRAIN.read_bytes())
    expected, rows = build_plan(coco, {row["id"] for row in base.read_images(base.VAL)}, base.sha256(base.TRAIN))
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        base.save_manifest(path, expected)
        return expected, rows
    manifest = json.loads(path.read_bytes())
    comparable = dict(manifest, images=[dict(row, sha256=None) for row in manifest["images"]])
    if expected != comparable:
        raise ValueError("Manifest differs from the reconstructed seed-42 selection, order or metadata")
    return manifest, rows


def write_reports(root, manifest, rows):
    counts = {"selections_before_dedup": manifest["selections_before_dedup"],
              "unique_images": manifest["image_count"], "classes": rows}
    (root / "class_distribution.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    with (root / "class_distribution.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    height = 70 + 22 * len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1110" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="13">',
        '<text x="12" y="23" font-size="17">Actual COCO class coverage after deduplication</text>',
        '<text x="12" y="48">ID and category</text>',
        '<text x="190" y="48">Images containing class</text>',
        '<text x="650" y="48">Valid instances</text>',
    ]
    max_images = max(row["actual_images"] for row in rows)
    max_instances = max(row["valid_instances"] for row in rows)
    for index, row in enumerate(rows):
        y = 58 + 22 * index
        parts.append(f'<text x="12" y="{y+12}">{row["category_id"]} {escape(row["name"])}</text>')
        for x, field, maximum, color in ((190, "actual_images", max_images, "#136e9a"),
                                         (650, "valid_instances", max_instances, "#a05019")):
            value = row[field]
            width = 385 * value / maximum
            parts.append(f'<rect x="{x}" y="{y}" width="{width:.2f}" height="14" fill="{color}"/>')
            parts.append(f'<text x="{x+width+5:.2f}" y="{y+12}">{value}</text>')
    parts.append("</g></svg>")
    (root / "class_distribution.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def local_sources():
    old_path = base.ROOT / "manifest.json"
    hashes = {}
    if old_path.is_file():
        hashes = {row["file_name"]: row["sha256"]
                  for row in json.loads(old_path.read_bytes())["images"]}
    return ((base.ROOT / "images", hashes), (Path("train2017"), {}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "download", "verify", "preview"))
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 16 or not 1 <= args.attempts <= 5 or not 0 < args.timeout <= 300:
        parser.error("Expected jobs 1–16, attempts 1–5 and timeout >0 and <=300 seconds")
    try:
        manifest, rows = load_plan(ROOT, create=args.action == "select")
        print(f'{manifest["selections_before_dedup"]} selections, {manifest["image_count"]} unique images')
        if args.action == "select":
            write_reports(ROOT, manifest, rows)
        elif args.action == "download":
            base.download(ROOT, manifest, args.jobs, args.attempts, args.timeout,
                          local_sources=local_sources(), checkpoint=100)
        else:
            base.verify(ROOT, manifest)
            if args.action == "preview":
                base.preview(ROOT, manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
