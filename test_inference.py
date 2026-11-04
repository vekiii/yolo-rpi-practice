from pathlib import Path

from ultralytics import YOLO


def main():
    project_dir = Path(__file__).resolve().parent
    images_dir = project_dir / "val2017"

    image_path = next(iter(sorted(images_dir.glob("*.jpg"))), None)
    if image_path is None:
        raise FileNotFoundError(
            f"Nema JPG slika u folderu: {images_dir}"
        )

    model = YOLO("yolov8n.pt")

    model.predict(
        source=str(image_path),
        device="cpu",
        imgsz=640,
        save=True,
        project=str(project_dir / "runs" / "detect"),
        name="first_test",
        exist_ok=True,
    )

    print(f"Ulazna slika: {image_path.name}")
    print(f"Rezultat: {project_dir / 'runs' / 'detect' / 'first_test'}")


if __name__ == "__main__":
    main()