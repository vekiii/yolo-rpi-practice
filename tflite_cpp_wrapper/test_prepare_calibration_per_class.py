import copy
from functools import partial
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import prepare_calibration as base
import prepare_calibration_per_class as stratified


def annotation(image, category, **overrides):
    return dict({"image_id": image, "category_id": category, "iscrowd": 0,
                 "bbox": [0, 0, 2, 2], "area": 4}, **overrides)


class StratifiedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.coco = {
            "images": [dict(id=i, file_name=f"{i:012d}.jpg", width=4, height=3)
                       for i in (40, 30, 20, 10)],
            "categories": [{"id": 89, "name": "dryer"}, {"id": 3, "name": "car"}],
            "annotations": [annotation(10, 3), annotation(10, 3), annotation(20, 3),
                            annotation(10, 89), annotation(30, 89),
                            annotation(40, 89, iscrowd=1), annotation(40, 3, area=0),
                            annotation(40, 3, bbox=[0, 0, -1, 2])],
        }

    def plan(self, val_ids=frozenset(), n=2):
        return stratified.build_plan(self.coco, val_ids, "fixture", per_class=n, category_count=2)

    def test_seeded_selection_deduplication_and_distribution(self):
        state = random.getstate()
        manifest, rows = self.plan()
        self.assertEqual(state, random.getstate())
        self.assertEqual([x["category_id"] for x in manifest["categories"]], [3, 89])
        rng = random.Random(42)
        self.assertEqual([x["image_ids"] for x in manifest["categories"]],
                         [rng.sample([10, 20], 2), rng.sample([10, 30], 2)])
        order = [10, 20, 30]
        random.Random(42).shuffle(order)
        self.assertEqual([x["id"] for x in manifest["images"]], order)
        self.assertEqual((manifest["selections_before_dedup"], manifest["image_count"]), (4, 3))
        self.assertEqual([x["valid_instances"] for x in rows], [3, 2])
        self.assertTrue(all(x["candidate_images"] == x["actual_images"] == 2 for x in rows))
        for key in self.coco:
            self.coco[key].reverse()
        self.assertEqual(self.plan(), (manifest, rows))

    def test_shortage_and_val_overlap(self):
        with self.assertRaisesRegex(ValueError, r"89 \(dryer\): 2"):
            self.plan(n=3)
        with self.assertRaisesRegex(ValueError, "overlaps val2017"):
            self.plan(val_ids={10})

    def test_manifest_verifier_rejects_changes(self):
        expected, rows = self.plan()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            train = path.with_name("train.json")
            val = path.with_name("val.json")
            train.write_text(json.dumps(self.coco), encoding="utf-8")
            val.write_text('{"images": []}', encoding="utf-8")
            base.save_manifest(path, expected)
            small_plan = partial(stratified.build_plan, per_class=2, category_count=2)
            with patch.object(base, "TRAIN", train), patch.object(base, "VAL", val), \
                 patch.object(base, "sha256", return_value="fixture"), \
                 patch.object(stratified, "build_plan", side_effect=small_plan):
                expected["source"] = train.as_posix()
                base.save_manifest(path, expected)
                self.assertEqual(stratified.load_plan(path.parent), (expected, rows))
                for field in ("images", "categories"):
                    changed = copy.deepcopy(expected)
                    changed[field].reverse()
                    base.save_manifest(path, changed)
                    with self.assertRaisesRegex(ValueError, "reconstructed"):
                        stratified.load_plan(path.parent)

    def test_local_copy_reused_without_network(self):
        row = dict(base.source_record(self.coco["images"][0]), sha256=None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            source = root / row["file_name"]
            Image.new("RGB", (4, 3), "red").save(source)
            original = source.read_bytes()
            with patch.object(base, "urlopen", side_effect=AssertionError("network used")):
                digest = base.acquire(root, row, 1, 1, [(root, {source.name: base.sha256(source)})])
            self.assertEqual(digest, base.sha256(source))
            self.assertEqual((root / row["local_path"]).read_bytes(), original)
            self.assertEqual(source.read_bytes(), original)

    def test_legacy_random_500_selector(self):
        records = [{"id": i} for i in range(600, 0, -1)]
        self.assertEqual([x["id"] for x in base.sample(records)],
                         random.Random(42).sample(list(range(1, 601)), 500))


if __name__ == "__main__":
    unittest.main()
