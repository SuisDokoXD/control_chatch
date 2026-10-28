"""用一对一 IoU 匹配评估 OpenCV 形状识别的漏检、误检和分类准确率。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from 程序 import opencv_shape_vision as vision

ROOT = Path(__file__).resolve().parent.parent


def match_one_to_one(predictions, truths, iou_min: float):
    pairs = []
    for pi, pred in enumerate(predictions):
        for gi, (truth, gt_box) in enumerate(truths):
            scaled_gt = tuple(v * vision.DEFAULTS["scale"] for v in gt_box)
            iou = vision.iou_box(scaled_gt, pred.bbox)
            if iou >= iou_min:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches = []
    for iou, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((iou, pi, gi))
    return matches, used_pred, used_gt


def evaluate(image_dir: Path, label_dir: Path, iou_min: float) -> dict:
    per_image = []
    total_gt = total_pred = matched = correct = rejected = wrong = 0
    for label_path in sorted(label_dir.glob("*.txt")):
        if label_path.name == "classes.txt":
            continue
        image_path = image_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue
        image = vision.cv2.imread(str(image_path))
        truths = vision.load_yolo_boxes(image_path, label_path)
        _, predictions = vision.analyse_image(image)
        pairs, used_pred, used_gt = match_one_to_one(predictions, truths, iou_min)
        cls_correct = 0
        cls_rejected = 0
        for _, pi, gi in pairs:
            if predictions[pi].label == vision.UNKNOWN:
                cls_rejected += 1
            elif predictions[pi].label == truths[gi][0]:
                cls_correct += 1
        cls_wrong = len(pairs) - cls_correct - cls_rejected
        total_gt += len(truths)
        total_pred += len(predictions)
        matched += len(pairs)
        correct += cls_correct
        rejected += cls_rejected
        wrong += cls_wrong
        per_image.append({
            "image": image_path.name,
            "gt": len(truths),
            "pred": len(predictions),
            "matched": len(pairs),
            "missed": len(truths) - len(used_gt),
            "extra": len(predictions) - len(used_pred),
            "classification_correct": cls_correct,
            "classification_wrong": cls_wrong,
            "classification_rejected": cls_rejected,
            "pairs": [
                {"iou": round(iou, 3), "truth": truths[gi][0], "pred": predictions[pi].label}
                for iou, pi, gi in pairs
            ],
        })
    committed = correct + wrong
    result = {
        "iou_min": iou_min,
        "images": per_image,
        "summary": {
            "images": len(per_image),
            "gt": total_gt,
            "pred": total_pred,
            "matched": matched,
            "missed": total_gt - matched,
            "extra": total_pred - matched,
            "detection_recall": matched / total_gt if total_gt else 0.0,
            "detection_precision": matched / total_pred if total_pred else 0.0,
            "classification_correct": correct,
            "classification_wrong": wrong,
            "classification_rejected": rejected,
            # 已判定（未拒判）的块里判对的比例：真正决定"抓错"的概率
            "accuracy_on_committed": correct / committed if committed else 0.0,
            "rejection_rate": rejected / matched if matched else 0.0,
            "classification_accuracy_on_matched": correct / matched if matched else 0.0,
            "end_to_end_correct_over_gt": correct / total_gt if total_gt else 0.0,
        },
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="一对一 IoU 评估 OpenCV 形状识别")
    parser.add_argument("--images", type=Path, default=vision.RAW_IMAGES)
    parser.add_argument("--labels", type=Path, default=vision.RAW_LABELS)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    images = args.images if args.images.is_absolute() else ROOT / args.images
    labels = args.labels if args.labels.is_absolute() else ROOT / args.labels
    result = evaluate(images, labels, args.iou)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    if args.json:
        path = args.json if args.json.is_absolute() else ROOT / args.json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"详细结果已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
