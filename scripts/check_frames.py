"""数一数采到的帧：每台相机各有多少张达标（角点 ≥ min-corners）。

用法：
    python scripts/check_frames.py
    python scripts/check_frames.py --min-corners 12 --frames captures/calibration/frames

标定脚本的门槛：单张角点 ≥12（calibrate_stereo.py 的 --min-corners 默认值），
每台至少 3 张才能标，建议 10~15 张；外参要求"两台都达标"≥6 组，建议 15~20 组。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BOARD = ROOT / "captures" / "calibration" / "charuco_screen_244x170mm.json"
DEFAULT_FRAMES = ROOT / "captures" / "calibration" / "frames"
ROLES = (("vertical", "垂直(编号1)"), ("side", "侧视(编号2)"))


def load_board(path: Path):
    params = json.loads(Path(path).read_text(encoding="utf-8"))
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, params["dictionary"]))
    board = cv2.aruco.CharucoBoard(tuple(params["squares"]), params["square_mm"],
                                   params["marker_mm"], dictionary)
    return board, cv2.aruco.CharucoDetector(board), params


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="统计采集帧的达标情况")
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--list", action="store_true", help="逐帧列出角点数")
    args = parser.parse_args(argv)

    board_path = args.board if args.board.is_absolute() else ROOT / args.board
    frames_dir = args.frames if args.frames.is_absolute() else ROOT / args.frames
    if not frames_dir.exists():
        print(f"没有帧目录 {frames_dir}", file=sys.stderr)
        return 2

    board, detector, params = load_board(board_path)
    stems = sorted({p.name.split("_vertical")[0] for p in frames_dir.glob("*_vertical.jpg")})
    if not stems:
        print(f"{frames_dir} 里还没有 *_vertical.jpg", file=sys.stderr)
        return 2

    print(f"标定板 {params.get('label')}  棋盘 {params['squares'][0]}×{params['squares'][1]}"
          f"（内角点上限 {len(board.getChessboardCorners())}）")
    print(f"门槛：单张角点 ≥ {args.min_corners}\n")

    ok = {key: 0 for key, _ in ROLES}
    both = 0
    counts: dict[str, list[int]] = {key: [] for key, _ in ROLES}
    for stem in stems:
        line = {}
        for key, _label in ROLES:
            path = frames_dir / f"{stem}_{key}.jpg"
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None
            if gray is None:
                line[key] = None
                continue
            corners, ids, _, _ = detector.detectBoard(gray)
            line[key] = 0 if ids is None else len(ids)
            counts[key].append(line[key])
            if line[key] >= args.min_corners:
                ok[key] += 1
        if all((line.get(k) or 0) >= args.min_corners for k, _ in ROLES):
            both += 1
        if args.list:
            marks = "  ".join(
                f"{label} {line[k] if line[k] is not None else '缺失'}" for k, label in ROLES)
            flag = "✓" if all((line.get(k) or 0) >= args.min_corners for k, _ in ROLES) else " "
            print(f"  {flag} {stem:<12} {marks}")

    print(f"共 {len(stems)} 组")
    for key, label in ROLES:
        vals = counts[key]
        med = sorted(vals)[len(vals) // 2] if vals else 0
        print(f"  {label:<12} 达标 {ok[key]:>2} 张 / 共 {len(vals):>2} 张   角点中位 {med:>2}  "
              f"范围 {min(vals) if vals else 0}~{max(vals) if vals else 0}")
    print(f"  两台都达标 {both} 组（外参需要 ≥6，建议 15~20）")

    print("\n结论：")
    for key, label in ROLES:
        if ok[key] < 3:
            print(f"  {label}：{ok[key]} 张，不够标定（至少 3 张），继续采")
        elif ok[key] < 10:
            print(f"  {label}：{ok[key]} 张，能标但偏少，建议补到 10~15 张")
        else:
            print(f"  {label}：{ok[key]} 张，够用了")
    if both < 6:
        print(f"  外参：两台都达标只有 {both} 组，还差 {6 - both} 组（后面 --mode both 平放采）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
