"""Export original YOLOv8n weights to a raw FP32 TFLite model on Linux x86_64/macOS."""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, default=Path('yolov8n.pt'))
    p.add_argument('--output', type=Path, default=Path('models/tflite_fp32/yolov8n_fp32.tflite'))
    args = p.parse_args()
    if platform.system() == 'Windows':
        p.error('Current Ultralytics LiteRT exporter requires Linux x86_64 or macOS. Use WSL Ubuntu or Colab CPU. C++ inference itself runs on Windows.')
    if not args.weights.is_file():
        p.error('Original weights file not found; supply the existing yolov8n.pt')
    if args.output.exists():
        p.error('Output already exists; choose a new output path to avoid overwriting results')
    from ultralytics import YOLO
    import ultralytics
    model = YOLO(str(args.weights))
    if model.task != 'detect' or len(model.names) != 80:
        p.error('Expected the original COCO YOLOv8n detection model')
    exported = Path(model.export(format='litert', imgsz=640, batch=1, device='cpu', quantize=32))
    if not exported.is_file() or exported.suffix != '.tflite':
        raise RuntimeError(f'Unexpected export result: {exported}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != args.output.resolve():
        shutil.copy2(exported, args.output)
    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=str(args.output))
    interpreter.allocate_tensors()
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    def detail(t):
        return {'name': t['name'], 'shape': t['shape'].tolist(), 'dtype': str(t['dtype']),
                'quantization': list(t['quantization'])}
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError('Expected one input and one raw output; inspect the exported graph')
    contract = {'source_weights': str(args.weights), 'weights_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                'model_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                'ultralytics_version': ultralytics.__version__, 'python': sys.version,
                'platform': platform.platform(), 'inputs': [detail(t) for t in inputs],
                'outputs': [detail(t) for t in outputs], 'names': model.names,
                'box_units': 'normalized',
                'note': 'Current Ultralytics LiteRT export normalizes YOLOv8 xywh by input size. Do not use this contract for arbitrary third-party models.'}
    args.output.with_suffix('.contract.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    frozen = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    args.output.with_suffix('.environment.txt').write_text(frozen, encoding='utf-8')
    print(json.dumps(contract, indent=2))
    print(f'Copy all files from {args.output.parent} to the Windows project models/tflite_fp32 directory.')


if __name__ == '__main__':
    main()
