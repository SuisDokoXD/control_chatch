"""双相机采图：同时抓两台相机，实时显示 ChArUco 角点质量，按空格存一帧组。

用法：
    # 第一步 内参：板子拿在手上，举到某台相机前倾斜 10~15 个姿态
    python scripts/capture_stereo_frames.py --mode vertical
    python scripts/capture_stereo_frames.py --mode side
    # 第二步 外参：板子平放托盘，位置/转角散开，采 15~20 组
    python scripts/capture_stereo_frames.py --mode both

操作：
    空格  存一帧组（--mode both 要求两台都看到且静止；单相机模式只要求那台看到）
    d     丢弃上一帧组（采错了删掉）
    q/ESC 退出

要点：
- 两台相机各自自由运行（软件同步差 40ms 级），所以 --mode both 采图时板子必须静止。
  本工具用"相邻帧差异"判断静止，不静止就不让存。单相机模式默认不做静止检查
  （板子拿在手上不可能静止），要强制就加 --require-still。
- 单相机模式仍然把两台图像成对存下（另一台可能是空的），这样 calibrate_stereo.py
  读帧的方式不用改。
- 曝光默认取 配置/camera_settings.json；现场亮度不同时用 --exposure -6 临时覆盖。
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
sys.path.insert(0, str(ROOT))

from 程序 import camera_config  # noqa: E402
from 程序 import overlay_text  # noqa: E402

DEFAULT_BOARD = ROOT / "captures" / "calibration" / "charuco_screen_244x170mm.json"
OUT_DIR = ROOT / "captures" / "calibration" / "frames"


def load_board(path: Path):
    params = json.loads(Path(path).read_text(encoding="utf-8"))
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, params["dictionary"]))
    board = cv2.aruco.CharucoBoard(tuple(params["squares"]), params["square_mm"],
                                   params["marker_mm"], dictionary)
    return board, cv2.aruco.CharucoDetector(board), params


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
    for _ in range(12):
        cap.read()
    return cap


def detect(detector, frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _, _ = detector.detectBoard(gray)
    n = 0 if corners is None else len(corners)
    return corners, ids, n


def motion(prev: np.ndarray | None, cur: np.ndarray) -> float:
    """相邻帧的差异（0~255 尺度的平均绝对差），用来判断场景是否静止。"""
    if prev is None:
        return 999.0
    small_a = cv2.resize(prev, (320, 240), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(cur, (320, 240), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(small_a, small_b)))


def next_index(out: Path) -> int:
    """扫描已有 pair_NNN_*.jpg，返回下一个可用序号（避免覆盖之前采的帧）。"""
    nums = []
    for p in out.glob("pair_*_vertical.jpg"):
        parts = p.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            nums.append(int(parts[1]))
    return max(nums) + 1 if nums else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="双相机采图")
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--min-corners", type=int, default=12,
                        help="每台相机至少要看到这么多角点才允许存")
    parser.add_argument("--still-threshold", type=float, default=1.5,
                        help="相邻帧平均绝对差低于此值算静止")
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--mode", choices=("both", "any", "vertical", "side"), default="both",
                        help="both=两台都要看到（外参用）；any/vertical/side=只要求指定相机看到"
                             "（内参用，板子拿在手上倾斜时用），两台图像仍成对存下")
    parser.add_argument("--exposure", type=float, default=None,
                        help="临时覆盖曝光（不改配置文件）")
    parser.add_argument("--auto-exposure", action="store_true", help="临时改成自动曝光")
    parser.add_argument("--require-still", action="store_true",
                        help="单相机模式也要求静止（默认不要求：板子拿在手上不可能静止）")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="跑多少秒后自动退出（0=一直跑，靠 q/ESC 退出）")
    args = parser.parse_args(argv)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    start_idx = next_index(out)
    if start_idx > 1:
        print(f"{out.relative_to(ROOT)} 里已有 {start_idx - 1} 组，本次从 pair_{start_idx:03d} 开始（不覆盖）",
              flush=True)
    board_path = args.board if args.board.is_absolute() else ROOT / args.board
    if not board_path.exists():
        print(f"找不到标定板参数 {board_path}", file=sys.stderr)
        return 2
    _, detector, board_params = load_board(board_path)
    print(f"标定板 {board_params['label']}：{board_params['squares'][0]}×{board_params['squares'][1]} 格，"
          f"格边长 {board_params['square_mm']}mm")

    settings = camera_config.load_settings()
    roles = camera_config.load_roles()
    v = camera_config.resolve(settings, "vertical", roles.get("vertical", {}).get("index"))
    s = camera_config.resolve(settings, "side_a", roles.get("side_a", {}).get("index"))
    for _resolved in (v, s):
        if args.auto_exposure:
            _resolved["auto_exposure"] = True
        elif args.exposure is not None:
            _resolved["auto_exposure"] = False
            _resolved["exposure"] = args.exposure
    print(f"垂直 {camera_config.describe(v)}")
    print(f"侧视 {camera_config.describe(s)}")
    v_cap, s_cap = open_camera(v), open_camera(s)
    if v_cap is None or s_cap is None:
        print("打不开相机", file=sys.stderr)
        return 1

    saved: list[tuple[Path, Path]] = []
    prev_v = prev_s = None
    window = f"stereo capture [{args.mode}] (space=save  d=undo last  q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1400, 560)
    t0 = time.time()
    try:
        while True:
            if args.seconds and time.time() - t0 > args.seconds:
                print(f"到 {args.seconds}s，自动退出", flush=True)
                break
            ok_v, frame_v = v_cap.read()
            ok_s, frame_s = s_cap.read()
            if not (ok_v and ok_s):
                print("读帧失败", file=sys.stderr)
                break
            corners_v, ids_v, n_v = detect(detector, frame_v)
            corners_s, ids_s, n_s = detect(detector, frame_s)
            mv, ms = motion(prev_v, frame_v), motion(prev_s, frame_s)
            need_still = args.mode == "both" or args.require_still
            still = mv < args.still_threshold and ms < args.still_threshold
            seen_v = n_v >= args.min_corners
            seen_s = n_s >= args.min_corners
            if args.mode == "both":
                enough = seen_v and seen_s
            elif args.mode == "any":
                enough = seen_v or seen_s
            elif args.mode == "vertical":
                enough = seen_v
            else:
                enough = seen_s
            ready = enough and (still or not need_still)

            tiles = []
            for name, frame, corners, ids, n, m in (("vertical", frame_v, corners_v, ids_v, n_v, mv),
                                                    ("side", frame_s, corners_s, ids_s, n_s, ms)):
                tile = cv2.resize(frame, (700, int(700 * frame.shape[0] / frame.shape[1])),
                                  interpolation=cv2.INTER_AREA)
                if corners is not None and len(corners):
                    cv2.aruco.drawDetectedCornersCharuco(tile, corners, ids, (0, 0, 255))
                color = (0, 200, 0) if n >= args.min_corners else (0, 165, 255)
                tile = overlay_text.draw_text(
                    tile, f"{name}  角点 {n}  帧差 {m:.2f}", (8, 6), 26, color)
                tiles.append(tile)
            h = min(t.shape[0] for t in tiles)
            view = np.hstack([t[:h] for t in tiles])
            if ready:
                status = "可以存（空格）"
            elif not enough:
                status = f"角点不足（垂直 {n_v} / 侧视 {n_s}，需要 {args.min_corners}，模式 {args.mode}）"
            else:
                status = "板子在动，等静止"
            view = overlay_text.draw_text(
                view, f"已存 {len(saved)} 组   {status}", (10, h - 44), 26,
                (0, 200, 0) if ready else (0, 165, 255))
            cv2.imshow(window, view)

            prev_v, prev_s = frame_v, frame_s
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and ready:
                idx = start_idx + len(saved)
                pv = out / f"pair_{idx:03d}_vertical.jpg"
                ps = out / f"pair_{idx:03d}_side.jpg"
                cv2.imwrite(str(pv), frame_v, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(ps), frame_s, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved.append((pv, ps))
                print(f"存第 {idx} 组  角点 垂直 {n_v} / 侧视 {n_s}  帧差 {mv:.2f}/{ms:.2f}")
            elif key == ord("d") and saved:
                pv, ps = saved.pop()
                pv.unlink(missing_ok=True)
                ps.unlink(missing_ok=True)
                print(f"已丢弃，剩 {len(saved)} 组")
    finally:
        v_cap.release()
        s_cap.release()
        cv2.destroyAllWindows()

    print(f"\n共存 {len(saved)} 组到 {out}（模式 {args.mode}）")
    if args.mode == "both":
        if len(saved) < 12:
            print("外参建议 15~20 组：板子平放托盘，位置和转角都要散开")
    elif len(saved) < 12:
        print("内参建议每台 10~15 个不同倾角（只平放会导致焦距不可观测）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
