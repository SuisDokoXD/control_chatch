"""在实时画面上点 4 个点，标出"斜坡与外沿接触的那个矩形"（托盘外沿）。

操作：
    左键    加一个点（最多 4 个，建议顺序 左上→右上→右下→左下，程序会自动重排）
    r       清空重来
    s       保存：写进 配置/calibration_stereo.json 的 tray.outer_* 并算出斜坡水平宽度
    q/ESC   退出

画面参照：红=平底边界（四角黑点）、蓝=当前 ROI（外扩 8%）、灰线=托盘毫米网格（每 20mm）。

用法（必须用不受火绒拦截的解释器）：
    $env:PYTHONPATH = "D:\\code\\src\\sight\\.venv\\Lib\\site-packages"
    & "C:\\Users\\123\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" scripts\\mark_tray_outer.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from 程序 import camera_config  # noqa: E402
from 程序 import overlay_text  # noqa: E402
from calibrate_tray_plane import find_dots, open_camera, order_points, pick_four  # noqa: E402

JSON_PATH = ROOT / "配置" / "calibration_stereo.json"
WINDOW = "托盘外沿标注  L=加点  r=清空  s=保存  q=退出"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="标注托盘外沿矩形")
    parser.add_argument("--exposure", type=float, default=-4.0)
    parser.add_argument("--width", type=int, default=1500, help="窗口显示宽度")
    parser.add_argument("--role", default="vertical")
    args = parser.parse_args(argv)

    cal = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    tray = cal["tray"]
    Hm = np.array(tray["homography_px_to_mm"], np.float64)
    inv = np.linalg.inv(Hm)
    W, H = tray["size_mm"]

    settings = camera_config.load_settings()
    roles = camera_config.load_roles()
    resolved = camera_config.resolve(settings, args.role, roles.get(args.role, {}).get("index"))
    resolved["auto_exposure"] = False
    resolved["exposure"] = args.exposure
    print(f"[{args.role}] {camera_config.describe(resolved)}", flush=True)
    cap = open_camera(resolved)
    if cap is None:
        print("打不开相机", file=sys.stderr)
        return 1

    points: list[tuple[float, float]] = []      # 原图坐标
    scale = args.width / 2592.0                 # 先给初值，避免回调早于第一帧

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))
            print(f"  点 {len(points)}: ({x / scale:.1f}, {y / scale:.1f})", flush=True)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, args.width, int(args.width * 0.75))
    cv2.setMouseCallback(WINDOW, on_mouse)

    def to_mm(pts: np.ndarray) -> np.ndarray:
        p = np.hstack([pts, np.ones((len(pts), 1))]) @ Hm.T
        return p[:, :2] / p[:, 2:3]

    saved = False
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        h, w = frame.shape[:2]
        scale = args.width / w
        vis = frame.copy()

        # 参照：毫米网格
        for mm in np.arange(0, W + 1e-6, 20.0):
            seg = cv2.perspectiveTransform(np.float32([[[mm, 0], [mm, H]]]), inv).reshape(-1, 2)
            cv2.line(vis, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), (110, 110, 110), 1)
        for mm in np.arange(0, H + 1e-6, 20.0):
            seg = cv2.perspectiveTransform(np.float32([[[0, mm], [W, mm]]]), inv).reshape(-1, 2)
            cv2.line(vis, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), (110, 110, 110), 1)

        # 参照：平底边界（黑点）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        picked = pick_four(find_dots(gray), gray.shape)
        if picked is not None:
            quad = order_points(np.array([picked[k]["center"] for k in ("tl", "tr", "br", "bl")],
                                         np.float32))
            cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 0, 255), 5)
            # 当前 ROI（外扩 8%）
            c = quad.mean(0)
            roi = np.clip(c + (quad - c) * 1.08, [0, 0], [w - 1, h - 1])
            cv2.polylines(vis, [roi.astype(np.int32)], True, (255, 200, 0), 3)

        # 已点的点
        for i, p in enumerate(points):
            cv2.circle(vis, (int(p[0]), int(p[1])), 16, (0, 255, 255), 5)
            vis = overlay_text.draw_text(vis, str(i + 1), (int(p[0]) + 20, int(p[1]) - 56), 56,
                                         (0, 255, 255))
        if len(points) >= 2:
            cv2.polylines(vis, [np.array(points, np.int32)], len(points) == 4, (0, 255, 255), 4)

        hint = (f"红=平底边界  蓝=当前ROI(8%)  灰=20mm网格   |   已点 {len(points)}/4 个"
                f"   左键加点  r清空  s保存  q退出")
        vis = overlay_text.draw_text(vis, hint, (20, 20), 44, (255, 255, 255))

        vis = cv2.resize(vis, (args.width, int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imshow(WINDOW, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            points.clear()
            print("已清空", flush=True)
        if key == ord("s"):
            if len(points) != 4:
                print("要先点满 4 个点", flush=True)
                continue
            pts = order_points(np.array(points, np.float32))
            mm = to_mm(pts)
            inner = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64)
            x_min, y_min = mm.min(0)
            x_max, y_max = mm.max(0)
            slope = {"top_mm": float(0 - y_min), "bottom_mm": float(y_max - H),
                     "left_mm": float(0 - x_min), "right_mm": float(x_max - W)}
            tray["outer_corners_px"] = {k: pts[i].tolist() for i, k in
                                        enumerate(("TL", "TR", "BR", "BL"))}
            tray["outer_size_mm"] = [float(x_max - x_min), float(y_max - y_min)]
            tray["slope_width_mm"] = slope
            tray["outer_note"] = "外沿矩形由人工在实时画面上点出；slope_width_mm = 外沿到平底边界各边的水平距离"
            JSON_PATH.write_text(json.dumps(cal, ensure_ascii=False, indent=1), encoding="utf-8")
            print("\n外沿四角(px): " + "  ".join(f"{k}({p[0]:.1f},{p[1]:.1f})"
                                                 for k, p in zip(("TL", "TR", "BR", "BL"), pts)))
            print(f"外沿尺寸 {x_max - x_min:.1f} × {y_max - y_min:.1f} mm"
                  f"（平底 {W:.1f} × {H:.1f} mm）")
            print(f"斜坡水平宽度: 上 {slope['top_mm']:.1f}  下 {slope['bottom_mm']:.1f}  "
                  f"左 {slope['left_mm']:.1f}  右 {slope['right_mm']:.1f} mm")
            print(f"已写入 {JSON_PATH.relative_to(ROOT)} 的 tray.outer_* / slope_width_mm\n", flush=True)
            saved = True

    cap.release()
    cv2.destroyAllWindows()
    print("退出。" + ("已保存。" if saved else "（未保存）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
