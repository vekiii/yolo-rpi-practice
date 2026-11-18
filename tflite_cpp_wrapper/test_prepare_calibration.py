import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import prepare_calibration as calibration


class CalibrationTests(unittest.TestCase):
    def test_selection_and_existing_manifest_order(self):
        rows = [dict(id=i, file_name=f"{i:012d}.jpg", width=4, height=3)
                for i in range(501)]
        expected = [row["id"] for row in calibration.sample(rows)]
        self.assertEqual(len(set(expected)), 500)
        self.assertEqual(expected, [row["id"] for row in calibration.sample(rows[::-1])])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = calibration.load_manifest(rows, [], "fixture", root, create=True)
            self.assertEqual(expected, [row["id"] for row in manifest["images"]])
            path = root / "manifest.json"
            original = path.read_bytes()
            self.assertEqual(manifest, calibration.load_manifest(
                rows, [], "fixture", root, create=True))
            self.assertEqual(original, path.read_bytes())
            with self.assertRaisesRegex(ValueError, "disjoint"):
                calibration.load_manifest(rows, [{"id": expected[0]}], "fixture", root, create=False)
            selected = manifest["images"]
            unused = next(row for row in rows if row["id"] not in expected)
            for changed in ([selected[1], selected[0], *selected[2:]],
                            [dict(calibration.source_record(unused), sha256=None), *selected[1:]]):
                with self.subTest(first_id=changed[0]["id"]):
                    calibration.save_manifest(path, dict(manifest, images=changed))
                    modified = path.read_bytes()
                    with self.assertRaisesRegex(ValueError, "expected seed-42 selection"):
                        calibration.load_manifest(rows, [], "fixture", root, create=True)
                    self.assertEqual(modified, path.read_bytes())

    def test_resume_skips_valid_image_and_repairs_corrupt_one(self):
        buffer = io.BytesIO()
        Image.new("RGB", (4, 3), "red").save(buffer, format="JPEG")
        jpeg = buffer.getvalue()
        row = dict(calibration.source_record(
            dict(id=1, file_name="000000000001.jpg", width=4, height=3)), sha256=None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / row["local_path"]
            target.parent.mkdir()
            temp = target.with_suffix(".jpg.part")
            temp.write_bytes(b"interrupted")
            with patch.object(calibration, "urlopen", side_effect=lambda *a, **k: io.BytesIO(jpeg)) as fetch:
                row["sha256"] = calibration.acquire(root, row, attempts=1, timeout=1)
                self.assertEqual(calibration.acquire(root, row, 1, 1), row["sha256"])
                self.assertEqual(fetch.call_count, 1)
                target.write_bytes(b"corrupt")
                self.assertEqual(calibration.acquire(root, row, 1, 1), row["sha256"])
                self.assertEqual(fetch.call_count, 2)
                self.assertEqual(target.read_bytes(), jpeg)
                self.assertFalse(temp.exists())
            target.unlink()
            with patch.object(calibration, "urlopen", side_effect=OSError("unavailable")) as fetch:
                with patch.object(calibration.time, "sleep"), self.assertRaisesRegex(RuntimeError, "unavailable"):
                    calibration.acquire(root, row, attempts=2, timeout=1)
                self.assertEqual(fetch.call_count, 2)
                self.assertFalse(target.exists())
                self.assertFalse(temp.exists())


if __name__ == "__main__":
    unittest.main()
