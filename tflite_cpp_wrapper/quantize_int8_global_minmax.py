"""Repeat INT8 profiler calibration with global min/max aggregation."""

from pathlib import Path

from quantize_int8 import main


if __name__ == "__main__":
    main(default_output=Path("models/tflite_int8_global_minmax"), aggregation="global_minmax")
