"""托盘平底平面标定：自动找 4 个黑点，求"像素 → 毫米"单应，出网格验收图。

用法（必须用不受火绒拦截的解释器，见 README/项目知识）：
    python scripts/calibrate_tray_plane.py --width-mm 174.2 --height-mm 159
    python scripts/calibrate_tray_plane.py --width-mm 174.2 --height-mm 159 --image captures/xxx.jpg

做的事：
1. 用垂直相机（配置/camera_roles.json 的 vertical）抓一帧（或读 --image 静态图）
2. 自动检出平底四角的 4 个黑点（暗色圆斑），按 左上/右上/右下/左下 排序
3. 用黑点像素坐标 + 平底实际尺寸求单应 H：像素 → 托盘毫米（原点在左上角，x 向右，y 向下）
4. 存验收图 captures/calibration/tray_check.jpg：画黑点、平底边界、每 10mm 一条网格线
5. 结果写进 配置/calibration_stereo.json 的 "tray" 段（保留已有的内参/外参）

判定标准：验收图里网格的**外框**要和你托盘的"平底→斜坡"折线重合；若整体偏了就是黑点配错。
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

from 程序 import camera_config  # noqa: E402
from 程序 import overlay_text  # noqa: E402

OUT_JSON = ROOT / "配置" / "calibration_stereo.json"
OUT_IMAGE = ROOT / "captures" / "calibration" / "tray_check.jpg"
GRID_MM = 10.0


def open_camera(resolved: dict):
    fourcc = str(resolved["fourcc"])[:4].ljust(4, " ")
    cap = cv2.VideoCapture(int(resolved["index"]), cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(resolved["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(resolved["height"]))
    if resolved.get("auto_exposure") is False and resolved.get("exposure") is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(resolved["exposure"]))
    for _ in range(15):
        cap.read()
    return cap


def find_dots(gray: np.ndarray, min_area: int = 200, max_area: int = 20000,
              ksize: int = 61, circ_min: float = 0.5,
              aspect: tuple[float, float] = (0.5, 2.0)):
    """找"比周围更暗的小圆斑"：black-hat（抗不均匀光照）→ 阈值 → 圆度/面积筛。

    为什么不用全局阈值：托盘画面光照很不均（一侧可能大片偏暗），Otsu 会把整块暗区
    当成一个连通域，四个角点根本出不来。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    blackhat = cv2.morphologyEx(cv2.GaussianBlur(gray, (5, 5), 0), cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    cands = []
    h, w = gray.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (min_area <= area <= max_area):
            continue
        if not (aspect[0] < bw / bh < aspect[1]):   # 太扁的不要
            continue
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue                            # 贴边的可能是阴影/边框
        cnts, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        per = cv2.arcLength(cnts[0], True)
        circ = 4 * np.pi * area / (per * per) if per > 0 else 0
        if circ < circ_min:                     # 圆度（斜视时圆斑会变扁，可放宽）
            continue
        cands.append(dict(center=cents[i], area=area, circ=circ,
                          bbox=(int(x), int(y), int(bw), int(bh))))
    return cands


def pick_four(cands, shape):
    """从候选里挑出最像"四个角"的 4 个（每个象限一个，优先圆度×面积）。"""
    h, w = shape
    quads: dict[str, list] = {"tl": [], "tr": [], "br": [], "bl": []}
    for c in cands:
        cx, cy = c["center"]
        key = ("t" if cy < h / 2 else "b") + ("l" if cx < w / 2 else "r")
        quads[key].append(c)
    picked = {}
    for key, items in quads.items():
        if not items:
            return None
        items.sort(key=lambda c: c["circ"] * min(c["area"], 20000), reverse=True)
        picked[key] = items[0]
    return picked


def order_points(pts: np.ndarray) -> np.ndarray:
    """左上/右上/右下/左下（用 x+y、x-y 的极值，比按象限更稳）。"""
    s = pts.sum(1)
    d = pts[:, 0] - pts[:, 1]
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]], np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="托盘平底平面标定")
    parser.add_argument("--width-mm", type=float, required=True, help="平底长（mm）")
    parser.add_argument("--height-mm", type=float, required=True, help="平底宽（mm）")
    parser.add_argument("--exposure", type=float, default=None,
                        help="临时覆盖曝光（托盘比平板暗得多，通常要 -4~-5）")
    parser.add_argument("--auto-exposure", action="store_true", help="临时改成自动曝光")
    parser.add_argument("--image", type=Path, help="用静态图代替抓帧")
    parser.add_argument("--role", default="vertical", help="用哪台相机（默认 vertical）")
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-image", type=Path, default=OUT_IMAGE)
    parser.add_argument("--no-save", action="store_true", help="只看结果，不写 json")
    args = parser.parse_args(argv)

    if args.image:
        img = cv2.imread(str(args.image if args.image.is_absolute() else ROOT / args.image))
        if img is None:
            print(f"读不出图像 {args.image}", file=sys.stderr)
            return 2
    else:
        settings = camera_config.load_settings()
        roles = camera_config.load_roles()
        resolved = camera_config.resolve(settings, args.role,
                                        roles.get(args.role, {}).get("index"))
        if args.auto_exposure:
            resolved["auto_exposure"] = True
        elif args.exposure is not None:
            resolved["auto_exposure"] = False
            resolved["exposure"] = args.exposure
        print(f"[{args.role}] {camera_config.describe(resolved)}", flush=True)
        cap = open_camera(resolved)
        if cap is None:
            print("打不开相机", file=sys.stderr)
            return 1
        img = None
        for _ in range(30):
            ok, frame = cap.read()
            if ok and frame is not None:
                img = frame
        cap.release()
        if img is None:
            print("读不到帧", file=sys.stderr)
            return 1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cands = find_dots(gray)
    print(f"图像 {w}x{h}，暗色圆斑候选 {len(cands)} 个")
    for c in sorted(cands, key=lambda c: -c["area"])[:8]:
        print(f"  中心 ({c['center'][0]:7.1f},{c['center'][1]:7.1f})  面积 {c['area']:6.0f}  "
              f"圆度 {c['circ']:.2f}")
    picked = pick_four(cands, gray.shape)
    if picked is None:
        print("没凑齐四个象限的圆斑——托盘四角黑点可能被裁掉/太暗，改用 --image 或手动点角",
              file=sys.stderr)
        return 1

    pts = np.array([picked[k]["center"] for k in ("tl", "tr", "br", "bl")], np.float32)
    pts = order_points(pts)
    W, H = args.width_mm, args.height_mm
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    Hmat = cv2.getPerspectiveTransform(pts, dst)
    inv = np.linalg.inv(Hmat)

    print("\n黑点像素坐标（左上/右上/右下/左下）：")
    for name, p in zip(("左上", "右上", "右下", "左下"), pts):
        print(f"  {name} ({p[0]:7.1f},{p[1]:7.1f})")
    px_per_mm_x = np.linalg.norm(pts[1] - pts[0]) / W
    px_per_mm_y = np.linalg.norm(pts[3] - pts[0]) / H
    print(f"边缘尺度 ≈ {px_per_mm_x:.2f} px/mm（横向，对应 {W}mm）  "
          f"{px_per_mm_y:.2f} px/mm（纵向，对应 {H}mm）")
    print(f"两个方向尺度差 {100 * abs(px_per_mm_x - px_per_mm_y) / ((px_per_mm_x + px_per_mm_y) / 2):.1f}%"
          f"  （俯视应 <3%；差很多说明长宽方向配错、或黑点认错）")
    ring = cv2.GaussianBlur(gray, (0, 0), 25)
    print("黑点对比度（中心 vs 周围，差得越多越可靠）：")
    for name, p in zip(("左上", "右上", "右下", "左下"), pts):
        cx, cy = int(p[0]), int(p[1])
        c = gray[max(0, cy - 3):cy + 4, max(0, cx - 3):cx + 4].mean()
        print(f"   {name} 中心 {c:5.1f}  周围 {ring[cy, cx]:5.1f}  对比 {ring[cy, cx] - c:5.1f}")

    # 验收图：网格 + 边界 + 黑点
    vis = img.copy()
    for mm in np.arange(0, W + 1e-6, GRID_MM):
        a = cv2.perspectiveTransform(np.float32([[[mm, 0], [mm, H]]]), inv).reshape(-1, 2)
        cv2.line(vis, tuple(a[0].astype(int)), tuple(a[1].astype(int)), (0, 200, 255), 2)
    for mm in np.arange(0, H + 1e-6, GRID_MM):
        a = cv2.perspectiveTransform(np.float32([[[0, mm], [W, mm]]]), inv).reshape(-1, 2)
        cv2.line(vis, tuple(a[0].astype(int)), tuple(a[1].astype(int)), (0, 200, 255), 2)
    quad = cv2.perspectiveTransform(dst.reshape(-1, 1, 2), inv).reshape(-1, 2)
    cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 0, 255), 4)
    for name, p in zip(("TL", "TR", "BR", "BL"), pts):
        cv2.circle(vis, tuple(p.astype(int)), 12, (0, 255, 0), 3)
        vis = overlay_text.draw_text(vis, name, (int(p[0]) + 16, int(p[1]) - 40), 40, (0, 255, 0))
    vis = overlay_text.draw_text(
        vis, f"平底 {W}×{H}mm  每格 {GRID_MM:.0f}mm  外框应贴合'平底→斜坡'折线",
        (20, 20), 44, (0, 0, 255))
    out_img = args.out_image if args.out_image.is_absolute() else ROOT / args.out_image
    out_img.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\n验收图 {out_img.relative_to(ROOT)}  —— 网格外框应贴合托盘折线；不贴合就是黑点配错")

    if not args.no_save:
        out_json = args.out_json if args.out_json.is_absolute() else ROOT / args.out_json
        data = {}
        if out_json.exists():
            try:
                data = json.loads(out_json.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data["tray"] = {
            "camera": args.role,
            "image_size": [w, h],
            "size_mm": [W, H],
            "origin": "平底左上角；x 向右、y 向下（mm）",
            "corners_px": {"TL": pts[0].tolist(), "TR": pts[1].tolist(),
                           "BR": pts[2].tolist(), "BL": pts[3].tolist()},
            "homography_px_to_mm": Hmat.tolist(),
            "homography_mm_to_px": inv.tolist(),
            "px_per_mm": [float(px_per_mm_x), float(px_per_mm_y)],
            "note": "只在平底平面上成立；斜坡上的点需要按高度单独补偿",
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"结果写入 {out_json.relative_to(ROOT)} 的 tray 段")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
