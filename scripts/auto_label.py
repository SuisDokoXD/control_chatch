"""Generate YOLO predictions and copy suspicious images/labels for review."""

import argparse
import csv
import os
import shutil
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

PROJECT_DIR = Path(__file__).resolve().parents[1]
YOLO_SETTINGS = PROJECT_DIR / "reports" / "ultralytics"
YOLO_SETTINGS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_SETTINGS))

from ultralytics import YOLO


def image_quality(path: Path) -> tuple[float, float]:
    with Image.open(path).convert("L") as image:
        stats = ImageStat.Stat(image)
        brightness = stats.mean[0]
        sharpness = ImageStat.Stat(image.filter(ImageFilter.FIND_EDGES)).var[0]
    return brightness, sharpness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "dataset" / "auto_labels")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--review-conf", type=float, default=0.65)
    parser.add_argument("--iou", type=float, default=0.60)
    args = parser.parse_args()

    output = args.output.resolve()
    labels_dir = output / "labels"
    review_images = output / "suspicious" / "images"
    review_labels = output / "suspicious" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    review_images.mkdir(parents=True, exist_ok=True)
    review_labels.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    rows: list[dict[str, object]] = []
    image_paths = [p for p in args.source.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    for result in model.predict(source=[str(p) for p in image_paths], conf=args.conf, iou=args.iou, device="cpu", verbose=False):
        image_path = Path(result.path)
        boxes = result.boxes
        confidences = [float(value) for value in boxes.conf.cpu().tolist()] if boxes is not None else []
        classes = [int(value) for value in boxes.cls.cpu().tolist()] if boxes is not None else []
        xyxy = boxes.xyxy.cpu().tolist() if boxes is not None else []
        label_path = labels_dir / f"{image_path.stem}.txt"
        with label_path.open("w", encoding="utf-8") as handle:
            for cls, confidence, box in zip(classes, confidences, xyxy):
                x1, y1, x2, y2 = box
                width, height = result.orig_shape[1], result.orig_shape[0]
                xc = ((x1 + x2) / 2) / width
                yc = ((y1 + y2) / 2) / height
                bw = (x2 - x1) / width
                bh = (y2 - y1) / height
                # Keep standard 5-field YOLO labels; confidence is recorded in the CSV report.
                handle.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        reasons: list[str] = []
        if not confidences:
            reasons.append("no_detections")
        if confidences and min(confidences) < args.review_conf:
            reasons.append("low_confidence")
        if len(xyxy) > 1:
            for index, first in enumerate(xyxy):
                for second in xyxy[index + 1 :]:
                    ix1, iy1 = max(first[0], second[0]), max(first[1], second[1])
                    ix2, iy2 = min(first[2], second[2]), min(first[3], second[3])
                    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    area1 = max(1, first[2] - first[0]) * max(1, first[3] - first[1])
                    area2 = max(1, second[2] - second[0]) * max(1, second[3] - second[1])
                    if intersection / (area1 + area2 - intersection) > args.iou:
                        reasons.append("overlapping_boxes")
                        break
                if "overlapping_boxes" in reasons:
                    break

        try:
            brightness, _ = image_quality(image_path)
            if brightness < 25 or brightness > 235:
                reasons.append("extreme_brightness")
        except Exception:
            reasons.append("quality_check_failed")

        if reasons:
            shutil.copy2(image_path, review_images / image_path.name)
            shutil.copy2(label_path, review_labels / label_path.name)
        rows.append({"image": str(image_path), "detections": len(confidences), "min_confidence": min(confidences) if confidences else "", "reasons": ";".join(sorted(set(reasons)))})

    with (output / "suspicious_report.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "detections", "min_confidence", "reasons"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
