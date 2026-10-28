"""实时跑"纯 OpenCV 形状识别"（程序/opencv_shape_vision.py）在指定相机上。

窗口里画：
    - 识别到的物块（轮廓 + 类别 + 置信度 + 抓取优先级 #1/#2…）
    - 每个物块的托盘毫米坐标（用标定结果换算：单应 px→mm）
    - 托盘工作区 ROI（黄线）
按键：
    q/ESC 退出      s 存一张标注图      space 暂停/继续
用法：
    $env:PYTHONPATH = "D:\\code\\src\\sight\\.venv\\Lib\\site-packages"
    & "C:\\Users\\123\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" scripts\\live_shape_vision.py
    ... --role side_a        # 换侧视相机
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "程序"))

import opencv_shape_vision as sv  # noqa: E402
from 程序 import camera_config  # noqa: E402
from 程序 import overlay_text  # noqa: E402
from calibrate_tray_plane import open_camera  # noqa: E402

JSON_PATH = ROOT / "配置" / "calibration_stereo.json"


def plane_homography(cal: dict, role: str) -> np.ndarray:
    """托盘平面(mm) → 该相机像素 的单应（由内参+外参解析得到）。"""
    tray = cal["tray"]
    W, H = tray["size_mm"]
    Kv = np.array(cal["intrinsics"]["vertical"]["mtx"], np.float64)
    Dv = np.array(cal["intrinsics"]["vertical"]["dist"], np.float64)
    pts = np.array([tray["corners_px"][k] for k in ("TL", "TR", "BR", "BL")], np.float32)
    obj = np.float32([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]])
    ok, rvec, tvec = cv2.solvePnP(obj, pts.reshape(-1, 1, 2), Kv, Dv, flags=cv2.SOLVEPNP_ITERATIVE)
    Rt, _ = cv2.Rodrigues(rvec)
    if role == "vertical":
        return np.array(tray["homography_mm_to_px"], np.float64)
    R = np.array(cal["extrinsics"]["R"], np.float64)
    T = np.array(cal["extrinsics"]["T"], np.float64).reshape(3, 1)
    K = np.array(cal["intrinsics"][role]["mtx"], np.float64)
    r1 = (R @ Rt[:, 0]).reshape(3, 1)
    r2 = (R @ Rt[:, 1]).reshape(3, 1)
    t = R @ tvec + T
    return K @ np.hstack([r1, r2, t])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="实时纯 OpenCV 形状识别")
    parser.add_argument("--role", default="vertical")
    parser.add_argument("--exposure", type=float, default=-4.0)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--seconds", type=float, default=0.0, help="跑多久自动退出（0=一直跑）")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="覆盖识别阈值，如 --set min_radius=12")
    args = parser.parse_args(argv)

    cal = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    Hmm = plane_homography(cal, args.role)
    Hinv = np.linalg.inv(Hmm)
    W, H = cal["tray"]["size_mm"]

    def px_to_mm(px: float, py: float) -> tuple[float, float]:
        v = Hinv @ np.array([px, py, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    # 自检：把托盘四角投回去，应该正好落在标定存的角点上
    corners_mm = np.float32([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]]).T
    proj = (Hmm @ corners_mm).T
    proj = proj[:, :2] / proj[:, 2:3]
    if args.role == "vertical":
        ref = np.array([cal["tray"]["corners_px"][k] for k in ("TL", "TR", "BR", "BL")], np.float64)
        err = float(np.abs(proj - ref).max())
        print(f"单自检：托盘四角投影 vs 标定角点，最大差 {err:.2f}px")
    else:
        print(f"单应自检：托盘四角投影到 {args.role} = {np.round(proj, 1).tolist()}")

    cfg = dict(sv.DEFAULTS)
    for item in args.set:
        k, _, v = item.partition("=")
        cfg[k.strip()] = float(v)
        print(f"覆盖阈值 {k.strip()} = {v}")

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

    WINDOW = f"纯OpenCV实时识别 [{args.role}]  q=退出 s=存图 space=暂停"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, args.width, int(args.width * 0.75))

    scale = sv.DEFAULTS["scale"]
    paused = False
    last_report = 0.0
    t0 = time.time()
    while True:
        if args.seconds and time.time() - t0 > args.seconds:
            break
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            small, objs = sv.analyse_image(frame, cfg)
            vis = sv.annotate(small, objs)
            # 托盘 ROI
            roi = sv.tray_roi_mask(small, cfg)
            if roi is not None and roi.any():
                cnts, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, (0, 220, 255), 2)
            # 每个物块：毫米坐标（把 small 坐标换算回原图再进单应）
            lines = []
            for obj in objs:
                cx, cy = obj.center
                mx, my = px_to_mm(cx / scale, cy / scale)
                inside = 0 <= mx <= W and 0 <= my <= H
                vis = overlay_text.draw_text(
                    vis, f"({mx:.0f},{my:.0f})mm" + ("" if inside else " 越界"),
                    (int(obj.bbox[0]), int(obj.bbox[1]) + int(obj.bbox[3]) + 2), 22,
                    (0, 255, 0) if inside else (0, 0, 255), shadow=True)
                lines.append(f"{obj.label} {obj.confidence:.2f} @ ({mx:.0f},{my:.0f})mm"
                             + ("" if inside else " 越界"))
            vis = overlay_text.draw_text(
                vis, f"{args.role}  物块 {len(objs)} 个   {time.strftime('%H:%M:%S')}",
                (10, 6), 26, (255, 255, 255))
            h, w = vis.shape[:2]
            disp = cv2.resize(vis, (args.width, int(h * args.width / w)),
                              interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW, disp)
            if time.time() - last_report > 1.0:
                last_report = time.time()
                print(f"{time.strftime('%H:%M:%S')} 物块 {len(objs)} 个 | " + " | ".join(lines),
                      flush=True)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            out = ROOT / "captures" / "preview" / f"live_shape_{args.role}_{int(time.time())}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"已存 {out.relative_to(ROOT)}", flush=True)
        if key == ord(" "):
            paused = not paused
            print("暂停" if paused else "继续", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
