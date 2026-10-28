"""无窗口排查：两台相机各抓若干帧，判断 ChArUco 标定板在不在画面里。

用法：
    python scripts/check_board_visible.py
    python scripts/check_board_visible.py --frames 20
    python scripts/check_board_visible.py --board captures/calibration/charuco_screen_244x170mm.json

产出（默认 captures/calibration/）：
    check_<role>.jpg          该相机角点最多的一帧（已画角点）
    check_<role>_preview.jpg  1200px 宽的预览
    check_view.html           两台相机并排看

退出码：0 = 两台都看到板子；1 = 至少一台看不到；2 = 参数/相机问题。

判定口径：ArUco marker 是硬证据（棋盘格角点可能因为反光/裁切而退化，
但只要有 marker 被识别出来，就说明板子在视野里）。
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from 程序 import camera_config  # noqa: E402

DEFAULT_BOARD = ROOT / "captures" / "calibration" / "charuco_screen_244x170mm.json"
DEFAULT_OUT = ROOT / "captures" / "calibration"
ROLES = ("vertical", "side_a")
ROLE_LABEL = {"vertical": "垂直相机", "side_a": "侧视相机"}


def load_board(path: Path):
    params = json.loads(Path(path).read_text(encoding="utf-8"))
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, params["dictionary"]))
    board = cv2.aruco.CharucoBoard(tuple(params["squares"]), params["square_mm"],
                                   params["marker_mm"], dictionary)
    return board, params


def make_detectors(board):
    """默认参数 + 一组放宽的兜底参数（反光/远距离/低对比时用）。"""
    detectors = [("默认", cv2.aruco.CharucoDetector(board))]
    relaxed = cv2.aruco.DetectorParameters()
    relaxed.adaptiveThreshWinSizeMin = 3
    relaxed.adaptiveThreshWinSizeMax = 53
    relaxed.adaptiveThreshWinSizeStep = 6
    relaxed.minMarkerPerimeterRate = 0.01
    relaxed.maxMarkerPerimeterRate = 6.0
    relaxed.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    relaxed.adaptiveThreshConstant = 5.0
    det = cv2.aruco.CharucoDetector(board)
    det.setDetectorParameters(relaxed)
    detectors.append(("放宽", det))
    return detectors


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


def frame_stats(gray: np.ndarray) -> dict:
    bright = (gray > 235).astype(np.uint8)
    dark = (gray < 12).astype(np.uint8)
    n = gray.size
    return {
        "mean": float(gray.mean()),
        "overexposed_pct": float(bright.sum() / n * 100),
        "dark_pct": float(dark.sum() / n * 100),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "bright_area_pct": float(bright.sum() / n * 100),
    }


def grab_best(cap, detectors, frames: int):
    """抓 frames 帧，返回角点最多的那一帧和它的检测结果。"""
    best = None
    stats_acc: list[dict] = []
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stats_acc.append(frame_stats(gray))
        for tag, det in detectors:
            corners, ids, marker_corners, marker_ids = det.detectBoard(gray)
            n_ch = 0 if ids is None else len(ids)
            n_mk = 0 if marker_ids is None else len(marker_ids)
            score = n_ch * 1000 + n_mk
            if best is None or score > best["score"]:
                best = {
                    "score": score, "frame": frame.copy(), "params": tag,
                    "corners": corners, "ids": ids,
                    "n_charuco": n_ch, "n_markers": n_mk,
                }
            if n_ch > 0 and n_mk >= 4:
                break
    agg = {}
    if stats_acc:
        for key in stats_acc[0]:
            agg[key] = float(np.mean([s[key] for s in stats_acc]))
    return best, agg, len(stats_acc)


def draw(frame, corners, ids):
    vis = frame.copy()
    if corners is not None and len(corners):
        cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids, (0, 0, 255))
    return vis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="排查标定板在不在两台相机画面里")
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--frames", type=int, default=15, help="每台相机抓多少帧取最优")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--exposure", type=float, default=None,
                        help="临时覆盖曝光（不改配置文件）；灯没开时用 -3 左右先看画面")
    parser.add_argument("--auto-exposure", action="store_true",
                        help="临时改成自动曝光（不改配置文件）")
    args = parser.parse_args(argv)

    board_path = args.board if args.board.is_absolute() else ROOT / args.board
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if not board_path.exists():
        print(f"找不到标定板参数 {board_path}", file=sys.stderr)
        return 2

    board, params = load_board(board_path)
    detectors = make_detectors(board)
    print(f"标定板 {params.get('label')}：{params['squares'][0]}×{params['squares'][1]} 格，"
          f"格边长 {params['square_mm']}mm，字典 {params['dictionary']}")

    settings = camera_config.load_settings()
    roles = camera_config.load_roles()

    results: dict[str, dict] = {}
    caps = {}
    try:
        for role in ROLES:
            resolved = camera_config.resolve(settings, role, roles.get(role, {}).get("index"))
            if args.auto_exposure:
                resolved["auto_exposure"] = True
            elif args.exposure is not None:
                resolved["auto_exposure"] = False
                resolved["exposure"] = args.exposure
            print(f"\n[{ROLE_LABEL[role]}] {camera_config.describe(resolved)}")
            cap = open_camera(resolved)
            if cap is None:
                print("  打不开相机", file=sys.stderr)
                results[role] = {"error": "打不开相机"}
                continue
            caps[role] = cap
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            best, stats, n_read = grab_best(cap, detectors, args.frames)
            if best is None:
                print("  读不到帧", file=sys.stderr)
                results[role] = {"error": "读不到帧"}
                continue
            vis = draw(best["frame"], best["corners"], best["ids"])
            full = out / f"check_{role}.jpg"
            preview = out / f"check_{role}_preview.jpg"
            cv2.imwrite(str(full), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
            ph = 1200 * vis.shape[0] // vis.shape[1]
            cv2.imwrite(str(preview), cv2.resize(vis, (1200, ph), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            results[role] = {
                "image": full.name, "preview": preview.name, "frame_px": f"{w}x{h}",
                "frames_read": n_read, "params": best["params"],
                "markers": best["n_markers"], "charuco": best["n_charuco"],
                **{k: round(v, 2) for k, v in stats.items()},
            }
            print(f"  实际分辨率 {w}x{h}，抓了 {n_read} 帧")
            print(f"  最优帧（{best['params']}参数）：marker {best['n_markers']} 个，"
                  f"棋盘角点 {best['n_charuco']} 个")
            print(f"  亮度 mean {stats['mean']:.1f}  过曝 {stats['overexposed_pct']:.2f}%  "
                  f"全黑 {stats['dark_pct']:.2f}%  清晰度 {stats['sharpness']:.0f}")
            print(f"  已存 {full.relative_to(ROOT)}")
    finally:
        for cap in caps.values():
            cap.release()

    html_path = out / "check_view.html"
    blocks = []
    for role in ROLES:
        r = results.get(role, {})
        title = f"{ROLE_LABEL[role]}（编号 {roles.get(role, {}).get('index')}）"
        if r.get("preview"):
            info = (f"marker {r['markers']} / 角点 {r['charuco']} / "
                    f"{r['frame_px']} / 亮度 {r['mean']} / 清晰度 {r['sharpness']:.0f}")
            blocks.append(f"<h1>{html.escape(title)}<span>{html.escape(info)}</span></h1>"
                          f"<img src=\"{html.escape(r['preview'])}\">")
        else:
            blocks.append(f"<h1>{html.escape(title)}<span>{html.escape(r.get('error', '无结果'))}</span></h1>")
    html_path.write_text(
        "<!doctype html>\n<html lang=\"zh\">\n<head><meta charset=\"utf-8\">"
        "<title>平板检测排查</title>\n<style>\n"
        " body { margin:0; background:#14171a; color:#e8eaed; font:14px/1.6 system-ui,\"Microsoft YaHei\",sans-serif; }\n"
        " h1 { font-size:16px; margin:12px 16px 4px; }\n"
        " h1 span { font-weight:400; color:#9aa4ad; margin-left:10px; font-size:13px; }\n"
        " img { width:min(86vw,1250px); display:block; margin:0 16px 14px; border:1px solid #2c3238; border-radius:6px; }\n"
        "</style></head>\n<body>\n" + "\n".join(blocks) + "\n</body></html>\n",
        encoding="utf-8")

    print("\n==== 结论 ====")
    ok_all = True
    for role in ROLES:
        r = results.get(role, {})
        if r.get("error"):
            print(f"{ROLE_LABEL[role]}：{r['error']}")
            ok_all = False
        elif r["markers"] >= 4:
            print(f"{ROLE_LABEL[role]}：看到板子（marker {r['markers']}，角点 {r['charuco']}）")
        else:
            print(f"{ROLE_LABEL[role]}：看不到板子（marker {r['markers']}，角点 {r['charuco']}）")
            ok_all = False
    print(f"排查图：{html_path.relative_to(ROOT)}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
