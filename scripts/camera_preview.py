"""双相机实时预览：把两台相机的画面 + ChArUco 检出结果持续写到 captures/preview/。

用法（注意：必须用不受火绒摄像头拦截影响的解释器，见下面注释）：
    python scripts/camera_preview.py
    python scripts/camera_preview.py --exposure -3          # 临时换曝光看画面
    python scripts/camera_preview.py --seconds 30           # 跑 30 秒后退出
    python scripts/camera_preview.py --preview-width 1100
    python scripts/camera_preview.py --window                # 开桌面窗口看画面（q/Esc 退出，s 存快照）
    python scripts/camera_preview.py --snapshot              # 抓一帧存图后退出

产出（默认 captures/preview/）：
    preview.html              打开这个看（每 0.4s 自动换图）
    preview_vertical.jpg      垂直相机当前帧（已画角点 + 统计）
    preview_side_a.jpg        侧视相机当前帧（已画角点 + 统计）

为什么不能直接用 .venv：
    本机火绒（hrdevmon 驱动）拦掉了 D:\\code\\src\\ 下解释器的摄像头访问，
    .venv\\Scripts\\python.exe 打不开任何相机。用系统 Python + 项目 site-packages：
        $env:PYTHONPATH = "D:\\code\\src\\sight\\.venv\\Lib\\site-packages"
        & "C:\\Users\\123\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" scripts\\camera_preview.py
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
DEFAULT_OUT = ROOT / "captures" / "preview"
ROLES = (("vertical", "垂直相机"), ("side_a", "侧视相机"))


def load_board(path: Path):
    params = json.loads(Path(path).read_text(encoding="utf-8"))
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, params["dictionary"]))
    board = cv2.aruco.CharucoBoard(tuple(params["squares"]), params["square_mm"],
                                   params["marker_mm"], dictionary)
    return board, params


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
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    if resolved.get("auto_white_balance") is False and resolved.get("white_balance") is not None:
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(resolved["white_balance"]))
    else:
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    for _ in range(15):
        cap.read()
    return cap


def render(frame, detector, name: str, params: dict, preview_w: int, extra: str = ""):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
    n_mk = 0 if marker_ids is None else len(marker_ids)
    n_ch = 0 if ids is None else len(ids)
    vis = frame.copy()
    if marker_corners is not None and len(marker_corners):
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids, (0, 255, 0))
    if corners is not None and len(corners):
        cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids, (0, 0, 255))
    mean = float(gray.mean())
    good = n_mk >= 4
    color = (0, 220, 0) if good else (0, 165, 255)
    scale = preview_w / vis.shape[1]
    vis = cv2.resize(vis, (preview_w, int(vis.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    tsize = max(18, int(vis.shape[1] / 42))
    vis = overlay_text.draw_text(
        vis, f"{name}  marker {n_mk}/{params['marker_count']}  charuco {n_ch}  L {mean:.0f}{extra}",
        (10, 6), tsize, color)
    vis = overlay_text.draw_text(vis, time.strftime("%H:%M:%S"),
                                 (10, vis.shape[0] - tsize - 8), max(14, tsize - 8),
                                 (200, 200, 200), shadow=False)
    return vis, n_mk, n_ch, mean


def write_html(path: Path, board_label: str):
    path.write_text(
        "<!doctype html>\n<html lang=\"zh\">\n<head><meta charset=\"utf-8\">"
        f"<title>相机实时预览 · {board_label}</title>\n<style>\n"
        " body { margin:0; background:#0f1214; color:#e8eaed; font:13px/1.5 system-ui,\"Microsoft YaHei\",sans-serif; }\n"
        " header { padding:10px 14px 4px; color:#9aa4ad; }\n"
        " header b { color:#e8eaed; font-size:14px; }\n"
        " .pane { margin:6px 14px 14px; }\n"
        " img { width:100%; max-width:1500px; display:block; border:1px solid #2c3238; border-radius:6px; background:#000; }\n"
        "</style></head>\n<body>\n"
        f"<header><b>双相机实时预览</b> · 标定板 {board_label} · 绿框=ArUco marker，红点=ChArUco 角点</header>\n"
        "<div class=\"pane\"><img data-live src=\"preview_vertical.jpg\"></div>\n"
        "<div class=\"pane\"><img data-live src=\"preview_side_a.jpg\"></div>\n"
        "<script>\n"
        "setInterval(() => {\n"
        "  document.querySelectorAll('img[data-live]').forEach(im => {\n"
        "    const base = im.getAttribute('src').split('?')[0];\n"
        "    im.src = base + '?t=' + Date.now();\n"
        "  });\n"
        "}, 400);\n"
        "</script>\n</body></html>\n",
        encoding="utf-8")


def _stack(tiles: list[np.ndarray], width: int) -> np.ndarray:
    rows = []
    for i, tile in enumerate(tiles):
        if i:
            rows.append(np.full((6, width, 3), 40, np.uint8))
        h = int(tile.shape[0] * width / tile.shape[1])
        rows.append(cv2.resize(tile, (width, h), interpolation=cv2.INTER_AREA))
    return np.vstack(rows)


def save_snapshot(shots: dict[str, np.ndarray], out: Path, width: int) -> Path:
    path = out / "snapshot.jpg"
    cv2.imwrite(str(path), _stack([shots[role] for role, _ in ROLES], width),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def save_boosted(shots: dict[str, np.ndarray], out: Path, width: int,
                 gamma: float = 2.2) -> Path:
    """gamma 提亮版：把暗部拉起来，方便看清托盘/环境（标定不用这张）。"""
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], np.uint8)
    path = out / "snapshot_boost.jpg"
    cv2.imwrite(str(path), _stack([cv2.LUT(shots[role], lut) for role, _ in ROLES], width),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="双相机实时预览")
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--exposure", type=float, default=None, help="临时覆盖曝光（不改配置文件）")
    parser.add_argument("--auto-exposure", action="store_true", help="临时改成自动曝光")
    parser.add_argument("--preview-width", type=int, default=1100)
    parser.add_argument("--seconds", type=float, default=0.0, help="跑多久后退出，0=一直跑")
    parser.add_argument("--period", type=float, default=0.4, help="两帧之间至少间隔多少秒")
    parser.add_argument("--snapshot", action="store_true",
                        help="抓一帧存成图片（另存 gamma 提亮版）后退出")
    parser.add_argument("--snapshot-width", type=int, default=1400,
                        help="快照合成图宽度")
    parser.add_argument("--window", action="store_true",
                        help="开一个桌面窗口看画面（q/Esc 退出，s 存快照）")
    parser.add_argument("--window-width", type=int, default=850,
                        help="窗口里每路画面的宽度")
    args = parser.parse_args(argv)

    board_path = args.board if args.board.is_absolute() else ROOT / args.board
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if not board_path.exists():
        print(f"找不到标定板参数 {board_path}", file=sys.stderr)
        return 2

    board, params = load_board(board_path)
    params["marker_count"] = len(board.getIds().flatten())
    detector = cv2.aruco.CharucoDetector(board)

    settings = camera_config.load_settings()
    roles = camera_config.load_roles()
    caps: dict[str, object] = {}
    resolved_by_role: dict[str, dict] = {}
    labels: dict[str, str] = {}
    for role, label in ROLES:
        resolved = camera_config.resolve(settings, role, roles.get(role, {}).get("index"))
        if args.auto_exposure:
            resolved["auto_exposure"] = True
        elif args.exposure is not None:
            resolved["auto_exposure"] = False
            resolved["exposure"] = args.exposure
        labels[role] = label
        resolved_by_role[role] = resolved
        print(f"[{label}] {camera_config.describe(resolved)}", flush=True)
        cap = open_camera(resolved)
        if cap is None:
            print(f"[{label}] 打不开相机", file=sys.stderr, flush=True)
            return 1
        caps[role] = cap

    if args.snapshot:
        shots: dict[str, np.ndarray] = {}
        for role, label in ROLES:
            frame = None
            for _ in range(10):
                ok, frame = caps[role].read()
                if ok and frame is not None:
                    break
            if frame is None:
                print(f"[{label}] 读不到帧", file=sys.stderr, flush=True)
                return 1
            vis, n_mk, n_ch, mean = render(frame, detector, label, params, frame.shape[1])
            cv2.imwrite(str(out / f"snapshot_{role}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
            shots[role] = vis
            print(f"[{label}] marker {n_mk}/{params['marker_count']} charuco {n_ch} L {mean:.0f}",
                  flush=True)
        p1 = save_snapshot(shots, out, args.snapshot_width)
        p2 = save_boosted(shots, out, args.snapshot_width)
        for cap in caps.values():
            cap.release()
        print(f"快照   {p1.relative_to(ROOT)}", flush=True)
        print(f"提亮版 {p2.relative_to(ROOT)}", flush=True)
        return 0

    if args.window:
        title = "sight 相机预览  q/Esc=退出  s=存快照  ]=亮一档  [=暗一档"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, args.window_width * len(ROLES) + 8,
                         int(args.window_width * 3 / 4) * len(ROLES) + 8)
        shots: dict[str, np.ndarray] = {}
        try:
            while True:
                tiles = []
                for role, label in ROLES:
                    ok, frame = caps[role].read()
                    if not ok or frame is None:
                        continue
                    r = resolved_by_role[role]
                    exp_txt = "自动" if r.get("auto_exposure") else f"{float(r['exposure']):+.0f}"
                    vis, n_mk, n_ch, mean = render(frame, detector, label, params,
                                                   args.window_width, extra=f"  曝光 {exp_txt}")
                    shots[role] = vis
                    tiles.append(vis)
                if tiles:
                    h = min(t.shape[0] for t in tiles)
                    cv2.imshow(title, np.hstack([t[:h] for t in tiles]))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s") and shots:
                    p1 = save_snapshot(shots, out, args.snapshot_width)
                    p2 = save_boosted(shots, out, args.snapshot_width)
                    print(f"已存 {p1.name} / {p2.name}", flush=True)
                if key in (ord("]"), ord("[")):
                    step = 1.0 if key == ord("]") else -1.0
                    for role, role_label in ROLES:
                        r = resolved_by_role[role]
                        if r.get("auto_exposure") or r.get("exposure") is None:
                            continue
                        new_exp = min(0.0, max(-13.0, float(r["exposure"]) + step))
                        if new_exp == float(r["exposure"]):
                            continue
                        r["exposure"] = new_exp
                        caps[role].release()
                        reopened = open_camera(r)
                        if reopened is None:
                            print(f"[{role_label}] 换曝光后打不开相机", file=sys.stderr, flush=True)
                            return 1
                        caps[role] = reopened
                    print("曝光 → " + "  ".join(
                        f"{lbl} {resolved_by_role[rr]['exposure']:+.0f}" for rr, lbl in ROLES),
                        flush=True)
        finally:
            for cap in caps.values():
                cap.release()
            cv2.destroyAllWindows()
        return 0

    write_html(out / "preview.html", params.get("label", "?"))
    print(f"预览页：{(out / 'preview.html').relative_to(ROOT)}（每 {args.period}s 刷新一次）", flush=True)

    t0 = time.time()
    last = 0.0
    try:
        while True:
            if args.seconds and time.time() - t0 > args.seconds:
                break
            for role, label in ROLES:
                ok, frame = caps[role].read()
                if not ok or frame is None:
                    continue
                vis, n_mk, n_ch, mean = render(frame, detector, label, params, args.preview_width)
                cv2.imwrite(str(out / f"preview_{role}.jpg"), vis,
                            [cv2.IMWRITE_JPEG_QUALITY, 82])
                print(f"{time.strftime('%H:%M:%S')} {label}: marker {n_mk} charuco {n_ch} L {mean:.0f}",
                      flush=True)
            dt = args.period - (time.time() - last)
            if dt > 0:
                time.sleep(dt)
            last = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        for cap in caps.values():
            cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
