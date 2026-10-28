"""扫描相机曝光/白平衡，找出让白托盘亮度合适、又不过曝的设置。

用法：
    python scripts/tune_camera_exposure.py --index 2
    python scripts/tune_camera_exposure.py --index 1 --index 2
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


def settle(cap, frames: int = 8) -> np.ndarray | None:
    frame = None
    for _ in range(frames):
        ok, candidate = cap.read()
        if ok and candidate is not None:
            frame = candidate
        time.sleep(0.04)
    return frame


def stats(frame: np.ndarray) -> dict:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    return {
        "median": float(np.median(L)),
        "p05": float(np.percentile(L, 5)),
        "p95": float(np.percentile(L, 95)),
        "saturated": float((L > 250).mean() * 100),
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
    time.sleep(0.5)

    print(f"\n=== 编号 {index} ===")
    print(f"曝光属性范围: min={cap.get(cv2.CAP_PROP_EXPOSURE)} 当前={cap.get(cv2.CAP_PROP_EXPOSURE)}")
    rows = []

    # 先试自动曝光
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    time.sleep(1.5)
    frame = settle(cap, 12)
    if frame is not None:
        s = stats(frame)
        rows.append(("自动", s))
        print(f"  自动曝光            L中位 {s['median']:6.1f}  5% {s['p05']:6.1f}  95% {s['p95']:6.1f}  过曝 {s['saturated']:5.2f}%")
        cv2.imwrite(str(output / f"exp_{index}_auto.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # 手动扫
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    for value in (-11, -9, -8, -7, -6, -5, -4, -3, -2, -1):
        ok = cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
        time.sleep(0.4)
        frame = settle(cap, 6)
        if frame is None:
            continue
        s = stats(frame)
        rows.append((value, s))
        flag = ""
        if s["saturated"] > 2:
            flag = "  ← 过曝"
        elif 150 <= s["median"] <= 230 and s["saturated"] < 1:
            flag = "  ← 可用"
        print(f"  曝光 {value:>4} (set={ok})    L中位 {s['median']:6.1f}  5% {s['p05']:6.1f}  "
              f"95% {s['p95']:6.1f}  过曝 {s['saturated']:5.2f}%{flag}")

    cap.release()
    return {"index": index, "rows": [(str(k), v) for k, v in rows]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="扫描相机曝光")
    parser.add_argument("--index", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    report = [probe(i, out) for i in args.index]
    (out / "exposure_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细写入 {out / 'exposure_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
