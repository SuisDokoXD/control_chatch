"""测量两台相机的同步质量：帧间隔稳定性、两帧到达时间差及其抖动。

关键概念：
- `grab()`/`retrieve()` 的时间差只反映**主机侧读取**的先后，不等于两台相机
  真正曝光时刻的差。
- 两台 UVC 相机各自自由运行，曝光时刻的相位差是**任意**的，上界是一个帧周期
  （15fps → 66.7ms）。所以真正要量的是这个差的**分布和抖动**，而不是平均值
  （平均值里混着两台的固有延迟，测不出来）。
- 抖动小 → 可以靠"固定偏移"补偿；抖动大 → 必须硬件触发。

用法：
    python scripts/measure_camera_sync.py
    python scripts/measure_camera_sync.py --pairs 200
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from 程序 import camera_config  # noqa: E402

OUT_DIR = ROOT / "captures" / "camera_pair"


def open_camera(resolved: dict):
    fourcc = str(resolved["fourcc"])[:4].ljust(4, " ")
    cap = cv2.VideoCapture(int(resolved["index"]), cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(resolved["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(resolved["height"]))
    if resolved.get("fps"):
        cap.set(cv2.CAP_PROP_FPS, float(resolved["fps"]))
    if resolved.get("auto_exposure") is False and resolved.get("exposure") is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(resolved["exposure"]))
    if resolved.get("auto_white_balance") is False and resolved.get("white_balance") is not None:
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(resolved["white_balance"]))
    else:
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    for _ in range(15):
        cap.read()
    return cap


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="测量双相机同步质量")
    parser.add_argument("--pairs", type=int, default=150)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    settings = camera_config.load_settings()
    roles = camera_config.load_roles()
    v = camera_config.resolve(settings, "vertical", roles.get("vertical", {}).get("index"))
    s = camera_config.resolve(settings, "side_a", roles.get("side_a", {}).get("index"))
    if v.get("index") is None or s.get("index") is None:
        print("缺少相机编号，请先配置 配置/camera_roles.json", file=sys.stderr)
        return 2
    print(f"垂直 {camera_config.describe(v)}")
    print(f"侧视 {camera_config.describe(s)}")

    v_cap, s_cap = open_camera(v), open_camera(s)
    if v_cap is None or s_cap is None:
        print("打不开相机", file=sys.stderr)
        return 1

    frame_period_ms = None
    print(f"\n配置里请求的帧率：{v.get('fps') or '未设置'}（这两台相机的 CAP_PROP_FPS 被驱动忽略，实际帧率见下）")
    print(f"连续抓 {args.pairs} 对，每对先各自 grab 再 retrieve …\n")

    v_times: list[float] = []
    s_times: list[float] = []
    v_pos: list[float] = []
    s_pos: list[float] = []
    try:
        for _ in range(args.pairs):
            ok_v = v_cap.grab()
            ok_s = s_cap.grab()
            t0 = time.perf_counter()  # 这一对开始读取的时刻
            if ok_v:
                v_cap.retrieve()
            t_v = time.perf_counter()
            if ok_s:
                s_cap.retrieve()
            t_s = time.perf_counter()
            if ok_v and ok_s:
                v_times.append(t_v)
                s_times.append(t_s)
                v_pos.append(float(v_cap.get(cv2.CAP_PROP_POS_MSEC)))
                s_pos.append(float(s_cap.get(cv2.CAP_PROP_POS_MSEC)))
            time.sleep(0.005)
    finally:
        v_cap.release()
        s_cap.release()

    if len(v_times) < 10:
        print("有效帧对太少", file=sys.stderr)
        return 1

    skew = [(a - b) * 1000.0 for a, b in zip(v_times, s_times)]
    v_interval = [(b - a) * 1000.0 for a, b in zip(v_times, v_times[1:])]
    s_interval = [(b - a) * 1000.0 for a, b in zip(s_times, s_times[1:])]
    frame_period_ms = statistics.median(v_interval)
    fps_actual = 1000.0 / frame_period_ms
    print(f"实测帧周期 中位 {frame_period_ms:.1f} ms（≈{fps_actual:.1f} fps）")

    print(f"有效帧对 {len(v_times)}")
    print(f"垂直帧间隔  中位 {statistics.median(v_interval):6.1f} ms  "
          f"抖动(std) {statistics.pstdev(v_interval):5.1f} ms")
    print(f"侧视帧间隔  中位 {statistics.median(s_interval):6.1f} ms  "
          f"抖动(std) {statistics.pstdev(s_interval):5.1f} ms")
    print()
    print("两帧到达时间差（垂直 − 侧视）：")
    print(f"  中位 {statistics.median(skew):+7.1f} ms  "
          f"均值 {statistics.fmean(skew):+7.1f} ms  std {statistics.pstdev(skew):5.1f} ms")
    print(f"  范围 [{min(skew):+.1f}, {max(skew):+.1f}] ms  "
          f"5%~95% [{percentile(skew, 5):+.1f}, {percentile(skew, 95):+.1f}] ms")

    jitter = statistics.pstdev(skew)
    spread_5_95 = percentile(skew, 95) - percentile(skew, 5)
    print()
    if jitter < 2.0 and spread_5_95 < 6.0:
        verdict = "抖动很小：可以靠固定偏移补偿，接近硬件同步效果"
    elif jitter < 8.0 and spread_5_95 < 25.0:
        verdict = "抖动中等：静态场景可用，运动物体需要硬件触发"
    else:
        verdict = "抖动很大：必须硬件触发，软件补偿不可靠"
    print(f"结论：{verdict}")

    pos_ok = len(v_pos) > 0 and (max(v_pos) - min(v_pos)) > 0
    print(f"\n驱动帧时间戳 CAP_PROP_POS_MSEC：{'可用' if pos_ok else '不可用'}")
    if pos_ok:
        v_span = (v_pos[-1] - v_pos[0]) if v_pos[-1] > v_pos[0] else float("nan")
        print(f"  垂直 时间戳跨度 {v_span:.0f} ms（实际经过 {1000 * (v_times[-1] - v_times[0]):.0f} ms）")

    report = {
        "frame_period_ms": round(frame_period_ms, 2),
        "pairs": len(v_times),
        "vertical_interval_median_ms": round(statistics.median(v_interval), 2),
        "side_interval_median_ms": round(statistics.median(s_interval), 2),
        "skew_median_ms": round(statistics.median(skew), 2),
        "skew_std_ms": round(jitter, 2),
        "skew_min_ms": round(min(skew), 2),
        "skew_max_ms": round(max(skew), 2),
        "skew_p5_ms": round(percentile(skew, 5), 2),
        "skew_p95_ms": round(percentile(skew, 95), 2),
        "verdict": verdict,
        "pos_msec_available": bool(pos_ok),
    }
    (out / "sync_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n报告写入 {out / 'sync_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
