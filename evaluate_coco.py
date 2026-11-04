"""COCO detection smoke evaluation; run from the yolo_test project directory."""
import argparse
import contextlib
import io
import json
import platform
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=Path('val2017'))
    parser.add_argument('--annotations', type=Path, default=Path('annotations/instances_val2017.json'))
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--limit', type=int, default=50, help='0 selects the entire validation set')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--output', type=Path, default=Path('results/evaluation_50.json'))
    args = parser.parse_args()
    if args.limit < 0 or args.threads < 1 or args.warmup < 0:
        parser.error('limit/warmup must be nonnegative and threads must be positive')
    if not args.images.is_dir() or not args.annotations.is_file():
        parser.error('Image directory or annotation JSON not found; check paths')

    import torch
    import ultralytics
    from ultralytics import YOLO
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from importlib.metadata import version

    torch.set_num_threads(args.threads)
    coco = COCO(str(args.annotations))
    image_ids = sorted(coco.getImgIds())
    if args.limit:
        image_ids = image_ids[:args.limit]
    if not image_ids:
        raise RuntimeError('No images selected')
    paths = [args.images / coco.imgs[i]['file_name'] for i in image_ids]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} selected images missing, e.g. {missing[0]}')

    model = YOLO(args.model)
    categories = {c['name']: c['id'] for c in coco.loadCats(coco.getCatIds())}
    # COCO category IDs contain gaps: a model class index is NOT category_id.
    names = model.names
    if set(names.values()) != set(categories):
        raise ValueError('Model classes must match the 80 COCO detection classes')
    class_to_category = {int(k): categories[v] for k, v in names.items()}
    predict_options = dict(device='cpu', imgsz=args.imgsz, rect=False,
                           half=False, conf=0.001, iou=0.7, max_det=100,
                           verbose=False, save=False)
    print(f'Warming up: {args.warmup} passes (excluded from timing)')
    for _ in range(args.warmup):
        model.predict(source=str(paths[0]), **predict_options)

    predictions = []
    wall_times = []
    stage_times = {'preprocess': [], 'inference': [], 'postprocess': []}
    for number, (image_id, path) in enumerate(zip(image_ids, paths), 1):
        start = time.perf_counter()
        result = model.predict(source=str(path), **predict_options)[0]
        wall_times.append((time.perf_counter() - start) * 1000)
        for stage in stage_times:
            stage_times[stage].append(float(result.speed[stage]))
        boxes = result.boxes
        if boxes is not None:
            for xyxy, score, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                x1, y1, x2, y2 = xyxy
                predictions.append(dict(image_id=image_id,
                                         category_id=class_to_category[int(cls)],
                                         bbox=[x1, y1, x2-x1, y2-y1], score=score))
        print(f'[{number}/{len(paths)}] {path.name}: {wall_times[-1]:.1f} ms, {len(boxes) if boxes is not None else 0} detections')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output.with_name(args.output.stem + '_predictions.json')
    prediction_path.write_text(json.dumps(predictions), encoding='utf-8')
    if predictions:
        detected = coco.loadRes(predictions)
    else:
        detected = COCO()
        detected.dataset = {'images': [coco.imgs[i] for i in image_ids],
                            'categories': coco.loadCats(coco.getCatIds()), 'annotations': []}
        detected.createIndex()
    evaluator = COCOeval(coco, detected, 'bbox')
    # Essential: evaluate ONLY the selected images, not all 5,000.
    evaluator.params.imgIds = image_ids
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    print(capture.getvalue())
    stats = evaluator.stats
    summary = {
        'model': args.model, 'device': 'cpu', 'half': False,
        'image_count': len(image_ids), 'image_ids': image_ids,
        'selection': 'ascending COCO image ID; first N',
        'images': str(args.images.resolve()), 'annotations': str(args.annotations.resolve()),
        'imgsz': args.imgsz, 'rect': False, 'batch': 1,
        'conf': 0.001, 'nms_iou': 0.7, 'max_det': 100,
        'threads': torch.get_num_threads(), 'warmup_passes': args.warmup,
        'map50_95': float(stats[0]), 'map50': float(stats[1]), 'map75': float(stats[2]),
        'timing_ms': {
            'mean_by_stage': {k: statistics.mean(v) for k, v in stage_times.items()},
            'mean_predict_call': statistics.mean(wall_times),
            'median_predict_call': statistics.median(wall_times),
            'predict_call_per_image': wall_times,
        },
        'timing_note': 'Predict call includes image loading and library overhead; excludes JSON export, COCO evaluation and progress printing. Stage times are Ultralytics timings. This is a single-pass smoke test.',
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'processor': platform.processor(), 'torch': torch.__version__,
                        'ultralytics': ultralytics.__version__, 'pycocotools': version('pycocotools')},
        'coco_summary': capture.getvalue(),
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'mAP50-95: {stats[0]:.4f}; mAP50: {stats[1]:.4f}')
    print(f'Mean inference: {statistics.mean(stage_times["inference"]):.2f} ms/image')
    print(f'Mean predict call: {statistics.mean(wall_times):.2f} ms/image')
    print(f'Results: {args.output.resolve()}')
    print(f'Predictions: {prediction_path.resolve()}')


if __name__ == '__main__':
    main()
