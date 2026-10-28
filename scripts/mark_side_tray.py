"""在侧视相机（或任意相机）的实时画面上点 4 个点，标出托盘的四个黑点。

画面里先用标定结果**投影**出托盘网格（每 20mm）+ 平底边界，作为参照；
你点的 4 个点会和这个投影对比，差值就是"标定预测 vs 实际"的一致性。

操作：
    左键    加一个点（最多 4 个，顺序随意，程序自动重排成 左上→右上→右下→左下）
    r       清空重来
    s       保存：算这台相机的 px→mm 单应，写进 配置/calibration_stereo.json 的 tray.<role>_*
    q/ESC   退出

用法：
    $env:PYTHONPATH = "D:\\code\\src\\sight\\.venv\\Lib\\site-packages"
    & "C:\\Users\\123\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" scripts\\mark_side_tray.py
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
from calibrate_tray_plane import open_camera, order_points  # noqa: E402

JSON_PATH = ROOT / "配置" / "calibration_stereo.json"


def tray_plane_pose(cal: dict):
    """托盘平面在垂直相机系下的位姿（4 角黑点 + 垂直内参）。"""
    tray = cal["tray"]
    W, H = tray["size_mm"]
    K = np.array(cal["intrinsics"]["vertical"]["mtx"], np.float64)
    D = np.array(cal["intrinsics"]["vertical"]["dist"], np.float64)
    pts = np.array([tray["corners_px"][k] for k in ("TL", "TR", "BR", "BL")], np.float32)
    obj = np.float32([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]])
    ok, rvec, tvec = cv2.solvePnP(obj, pts.reshape(-1, 1, 2), K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("托盘平面位姿求解失败")
    Rt, _ = cv2.Rodrigues(rvec)
    return Rt, tvec, W, H


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="标注侧视相机里的托盘黑点")
    parser.add_argument("--role", default="side_a")
    parser.add_argument("--exposure", type=float, default=-4.0)
    parser.add_argument("--width", type=int, default=1500)
    args = parser.parse_args(argv)

    cal = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    Rt, tvec, W, H = tray_plane_pose(cal)
    Ks = np.array(cal["intrinsics"][args.role]["mtx"], np.float64)
    Ds = np.array(cal["intrinsics"][args.role]["dist"], np.float64)
    R = np.array(cal["extrinsics"]["R"], np.float64)
    T = np.array(cal["extrinsics"]["T"], np.float64).reshape(3, 1)

    def project(p_mm: np.ndarray) -> np.ndarray:
        """托盘毫米点 (N,3) → 本相机像素 (N,2)（走内参+外参）。"""
        x_v = (Rt @ np.asarray(p_mm, np.float64).T + tvec).T
        x_s = (R @ x_v.T + T).T
        px, _ = cv2.projectPoints(x_s, np.zeros(3), np.zeros(3), Ks, Ds)
        return px.reshape(-1, 2)

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

    WINDOW = f"{args.role} 托盘黑点标注  L=加点  r=清空  s=保存  q=退出"
    points: list[tuple[float, float]] = []
    scale = args.width / 2592.0

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))
            print(f"  点 {len(points)}: ({x / scale:.1f}, {y / scale:.1f})", flush=True)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, args.width, int(args.width * 0.75))
    cv2.setMouseCallback(WINDOW, on_mouse)

    # 标定预测的四角（作为参照/比对基准）
    pred = project(np.float32([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]]))
    outer = cal["tray"].get("outer_corners_px")
    print("标定预测的托盘四角（投影到本相机）：")
    for k, p in zip(("TL", "TR", "BR", "BL"), pred):
        print(f"  {k} ({p[0]:7.1f},{p[1]:7.1f})")

    saved = False
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        h, w = frame.shape[:2]
        scale = args.width / w
        vis = frame.copy()

        # 参照：投影出来的毫米网格（每 20mm）+ 平底边界
        for mm in np.arange(0, W + 1e-6, 20.0):
            seg = project(np.float32([[mm, 0, 0], [mm, H, 0]]))
            cv2.line(vis, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), (110, 110, 110), 1)
        for mm in np.arange(0, H + 1e-6, 20.0):
            seg = project(np.float32([[0, mm, 0], [W, mm, 0]]))
            cv2.line(vis, tuple(seg[0].astype(int)), tuple(seg[1].astype(int)), (110, 110, 110), 1)
        cv2.polylines(vis, [pred.astype(np.int32)], True, (0, 0, 255), 4)
        for k, p in zip(("TL", "TR", "BR", "BL"), pred):
            cv2.circle(vis, tuple(p.astype(int)), 12, (0, 0, 255), 3)
            vis = overlay_text.draw_text(vis, k, (int(p[0]) + 14, int(p[1]) - 46), 40, (0, 0, 255))
        if outer:
            q = np.array([outer[k] for k in ("TL", "TR", "BR", "BL")], np.float32)
            cv2.polylines(vis, [q.astype(np.int32)], True, (255, 120, 0), 3)

        # 用户点的点
        for i, p in enumerate(points):
            cv2.circle(vis, (int(p[0]), int(p[1])), 15, (0, 255, 255), 5)
            vis = overlay_text.draw_text(vis, str(i + 1), (int(p[0]) + 18, int(p[1]) - 52), 52,
                                         (0, 255, 255))
        if len(points) >= 2:
            cv2.polylines(vis, [np.array(points, np.int32)], len(points) == 4, (0, 255, 255), 4)

        vis = overlay_text.draw_text(
            vis, f"红=标定预测的托盘四角/网格(每20mm)  黄=你点的点 {len(points)}/4"
                 f"   左键加点  r清空  s保存  q退出", (20, 20), 42, (255, 255, 255))
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
            dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
            Hm = cv2.getPerspectiveTransform(pts, dst)
            inv = np.linalg.inv(Hm)
            print("\n你点的四角 vs 标定预测（差越小说明标定越准）：")
            res = []
            for k, a, b in zip(("TL", "TR", "BR", "BL"), pts, pred):
                d = float(np.linalg.norm(a - b))
                res.append(d)
                print(f"  {k} 点的 ({a[0]:7.1f},{a[1]:7.1f})  预测 ({b[0]:7.1f},{b[1]:7.1f})  差 {d:6.1f}px")
            print(f"  残差 中位 {np.median(res):.1f}px  最大 {max(res):.1f}px")
            tray = cal["tray"]
            tray[f"{args.role}_dots_px"] = {k: pts[i].tolist() for i, k in
                                            enumerate(("TL", "TR", "BR", "BL"))}
            tray[f"{args.role}_homography_px_to_mm"] = Hm.tolist()
            tray[f"{args.role}_homography_mm_to_px"] = inv.tolist()
            tray[f"{args.role}_note"] = ("由人工在实时画面上点出四个黑点得到；"
                                         "可直接把本相机像素换算成托盘毫米（平底平面）")
            JSON_PATH.write_text(json.dumps(cal, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"已写入 {JSON_PATH.relative_to(ROOT)} 的 tray.{args.role}_*", flush=True)
            saved = True

    cap.release()
    cv2.destroyAllWindows()
    print("退出。" + ("已保存。" if saved else "（未保存）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
