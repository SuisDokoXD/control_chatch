"""扫描白平衡，找出让白托盘接近中性灰（a≈0, b≈0）的值。

用法：
    python scripts/tune_camera_white_balance.py --index 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "captures" / "camera_pair"


def settle(cap, frames: int = 8):
    frame = None
    for _ in range(frames):
        ok, candidate = cap.read()
        if ok and candidate is not None:
            frame = candidate
        time.sleep(0.04)
    return frame


def neutral_stats(frame: np.ndarray) -> dict:
    """只看"亮且低色度"的像素（也就是白托盘），统计它们的 a/b 偏离。"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0
    sel = L > 140
    if sel.sum() < 1000:
        sel = L > np.percentile(L, 60)
    return {
        "a": float(np.median(a[sel])),
        "b": float(np.median(b[sel])),
        "L": float(np.median(L[sel])),
        "cast": float(np.hypot(np.median(a[sel]), np.median(b[sel]))),
    }


def probe(index: int, output: Path) -> dict:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"编号 {index}: 打不开")
        return {}
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592.0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944.0)
    cap.set(cv2.CAP_PROP_FPS, 15.0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7.0)
    time.sleep(0.6)

    print(f"\n=== 编号 {index} ===")
    rows = []

    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    time.sleep(1.5)
    frame = settle(cap, 12)
    if frame is not None:
        s = neutral_stats(frame)
        rows.append(("自动", s))
        print(f"  自动白平衡   a {s['a']:+6.1f}  b {s['b']:+6.1f}  L {s['L']:5.0f}  偏色 {s['cast']:5.1f}")
        cv2.imwrite(str(output / f"wb_{index}_auto.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    for value in (2800, 3200, 3600, 4000, 4600, 5000, 5400, 5800, 6500):
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(value))
        time.sleep(0.4)
        frame = settle(cap, 6)
        if frame is None:
            continue
        s = neutral_stats(frame)
        rows.append((value, s))
        flag = "  ← 中性" if s["cast"] < 3 else ("  ← 较接近" if s["cast"] < 8 else "")
        print(f"  WB {value:>5}K   a {s['a']:+6.1f}  b {s['b']:+6.1f}  L {s['L']:5.0f}  "
              f"偏色 {s['cast']:5.1f}{flag}")

    cap.release()
    return {"index": index, "rows": [(str(k), v) for k, v in rows]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="扫描白平衡")
    parser.add_argument("--index", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    report = [probe(i, out) for i in args.index]
    (out / "white_balance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细写入 {out / 'white_balance_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
