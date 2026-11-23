import json
from pathlib import Path
from ultralytics import YOLO

summary = json.loads(
    Path("results/eval_50/evaluation_50.json").read_text(encoding="utf-8")
)
annotations = json.loads(
    Path("annotations/instances_val2017.json").read_text(encoding="utf-8")
)

filenames = {
    image["id"]: image["file_name"]
    for image in annotations["images"]
}

model = YOLO("models/yolov8n.pt")

for image_id in summary["image_ids"]:
    image = Path("val2017") / filenames[image_id]
    model.predict(
        source=str(image),
        device="cpu",
        imgsz=640,
        rect=False,
        conf=0.25,
        iou=0.7,
        max_det=100,
        save=True,
        project=str(Path("results/eval_50").resolve()),
        name="visualization_50",
        exist_ok=True,
    )

print("Slike su sacuvane u results/eval_50/visualization_50")
