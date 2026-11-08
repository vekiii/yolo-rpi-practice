"""Prepare a deterministic image manifest or evaluate C++ COCO predictions."""
import argparse
import json
from pathlib import Path

COCO_NAMES = ('person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,'
              'fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,'
              'elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,'
              'skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,'
              'surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,'
              'apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,'
              'potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,'
              'cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,'
              'scissors,teddy bear,hair drier,toothbrush').split(',')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'evaluate'])
    parser.add_argument('--annotations', type=Path, default=Path('annotations/instances_val2017.json'))
    parser.add_argument('--manifest', type=Path, default=Path('results/manifest_50.json'))
    parser.add_argument('--limit', type=int, default=50, help='0 selects all images')
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/ncnn_fp32_50_metrics.json'))
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('limit must be nonnegative')
    from pycocotools.coco import COCO
    coco = COCO(str(args.annotations))
    if args.action == 'prepare':
        ids = sorted(coco.getImgIds())
        if args.limit:
            ids = ids[:args.limit]
        categories = {c['name']: c['id'] for c in coco.loadCats(coco.getCatIds())}
        manifest = {'images': [{'id': i, 'file_name': coco.imgs[i]['file_name']} for i in ids],
                    'category_ids': [categories[n] for n in COCO_NAMES]}
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'Manifest: {args.manifest} ({len(ids)} images)')
        return
    if args.predictions is None:
        parser.error('evaluate requires --predictions')
    from pycocotools.cocoeval import COCOeval
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    ids = [i['id'] for i in manifest['images']]
    predictions = json.loads(args.predictions.read_text(encoding='utf-8'))
    if any(p['image_id'] not in set(ids) for p in predictions):
        raise ValueError('Predictions contain image IDs outside manifest')
    if predictions:
        detected = coco.loadRes(predictions)
    else:
        detected = COCO()
        detected.dataset = {'images': [coco.imgs[i] for i in ids],
                            'categories': list(coco.cats.values()), 'annotations': []}
        detected.createIndex()
    evaluator = COCOeval(coco, detected, 'bbox')
    evaluator.params.imgIds = ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    result = {'image_count': len(ids), 'map50_95': float(evaluator.stats[0]),
              'map50': float(evaluator.stats[1]), 'map75': float(evaluator.stats[2]),
              'all_coco_stats': evaluator.stats.tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Metrics: {args.output}')


if __name__ == '__main__':
    main()
