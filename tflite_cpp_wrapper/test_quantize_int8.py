import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import quantize_int8 as quant


class QuantizeInt8Tests(unittest.TestCase):
    def test_cpp_rounding_letterbox_and_rgb_fp32(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "asymmetric.png"
            # Ratio 1/2 and height 641 give 320.5, rounded to even (320).
            pixels = np.full((641, 1280, 3), (3, 37, 241), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image), pixels))
            tensor = quant.preprocess(image)
            self.assertEqual(tensor.shape, quant.INPUT_SHAPE)
            self.assertEqual(tensor.dtype, np.float32)
            self.assertTrue(tensor.flags.c_contiguous)
            values = np.array([241, 37, 3], dtype=np.float32) / np.float32(255)
            expected = np.full(quant.INPUT_SHAPE, np.float32(114) / np.float32(255))
            expected[0, :, 160:480, :] = values[:, None, None]
            np.testing.assert_array_equal(tensor, expected)

            # The odd amount of vertical padding goes to the bottom.
            image = Path(directory) / "odd_padding.png"
            self.assertTrue(cv2.imwrite(str(image), pixels[:638]))
            tensor = quant.preprocess(image)
            expected.fill(np.float32(114) / np.float32(255))
            expected[0, :, 160:479, :] = values[:, None, None]
            np.testing.assert_array_equal(tensor, expected)

    def test_all_manifest_items_in_original_order_on_both_passes(self):
        records = json.loads((quant.ROOT / "manifest.json").read_text())["images"]
        path_to_id = {quant.ROOT / row["local_path"]: row["id"] for row in records}
        storage = np.empty((quant.COUNT, 1), dtype=np.float32)
        # Use tiny tensors here to exercise the data plumbing only.
        with patch.object(quant, "preprocess", side_effect=lambda path: [path_to_id[path]]) as preprocess:
            quant.fill_inputs(records, quant.ROOT, storage)
        self.assertEqual([call.args[0] for call in preprocess.call_args_list],
                         [quant.ROOT / row["local_path"] for row in records])
        expected = [row["id"] for row in records]
        inputs = quant.CalibrationInputs(storage, "signature_input")
        for _ in range(2):
            self.assertEqual([int(item["signature_input"][0]) for item in inputs], expected)
        self.assertEqual(inputs.pass_counts, [500, 500])


if __name__ == "__main__":
    unittest.main()
