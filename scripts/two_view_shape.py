"""Calibrated two-view regular-solid analysis with offline replay and grasp diagnostics.

Foreground subtraction requires a matching empty-tray reference and stable lighting.
The floor homography provides association seeds only: elevated silhouettes are not
floor footprints. Heuristic profile fits are diagnostic and never executable targets.
See 文档/双视角识别与抓取接口.md for frames, replay and pending hardware checks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from 程序 import camera_config  # noqa: E402
from 程序 import overlay_text  # noqa: E402
from calibrate_tray_plane import open_camera  # noqa: E402
from two_view_model_fit import fit_two_view  # noqa: E402

JSON_PATH = ROOT / "配置" / "calibration_stereo.json"
REF_DIR = ROOT / "captures" / "calibration"
ROLES = (("vertical", "垂直"), ("side_a", "侧视"))
# 侧面轮廓的三种模板：t=高度比例，w=该高度处轮廓宽度 / 底面宽度
PROFILE_MODELS = {
    "rect": lambda t: 1.0,                              # 立方体/圆柱/棱柱
    "circle": lambda t: float(2.0 * np.sqrt(max(t * (1.0 - t), 0.0))),  # 球体落在托盘上的截面
    "triangle": lambda t: max(0.0, 1.0 - t),            # 圆锥
}


# --------------------------------------------------------------------------- #
# 几何：托盘平面 → 相机像素
# --------------------------------------------------------------------------- #
def tray_pose(cal: dict):
    """托盘平面在垂直相机系下的位姿（由四角黑点 + 垂直内参 PnP 求出）+ 托盘尺寸。

    有了它 + 外参 R/T，就能把"托盘毫米点"投影到任意一台相机（含畸变）。
    """
    tray = cal["tray"]
    W, H = tray["size_mm"]
    Kv = np.array(cal["intrinsics"]["vertical"]["mtx"], np.float64)
    Dv = np.array(cal["intrinsics"]["vertical"]["dist"], np.float64)
    pts = np.array([tray["corners_px"][k] for k in ("TL", "TR", "BR", "BL")], np.float32)
    obj = np.float32([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]])
    ok, rvec, tvec = cv2.solvePnP(obj, pts.reshape(-1, 1, 2), Kv, Dv,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise SystemExit("托盘平面位姿求解失败")
    Rt, _ = cv2.Rodrigues(rvec)
    R = np.array(cal["extrinsics"]["R"], np.float64)
    T = np.array(cal["extrinsics"]["T"], np.float64).reshape(3, 1)
    # PnP 解出的 +z 不一定"朝上"：相机在托盘系里的方向 ≈ -Rt^T·tvec，据此定"上"是 +z 还是 -z。
    cam_in_tray = (-Rt.T @ tvec).ravel()
    up = 1.0 if float(cam_in_tray[2]) > 0 else -1.0
    return Rt, tvec, R, T, W, H, up

def side_project(cal: dict, Rt, tvec, R, T, p_mm: np.ndarray) -> np.ndarray:
    """托盘毫米点 (N,3) → 侧视相机像素（带畸变，逐点精确）。"""
    K = np.array(cal["intrinsics"]["side_a"]["mtx"], np.float64)
    D = np.array(cal["intrinsics"]["side_a"]["dist"], np.float64)
    xv = (Rt @ np.asarray(p_mm, np.float64).T + tvec).T
    xs = (R @ xv.T + T).T
    return cv2.projectPoints(xs, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)


# --------------------------------------------------------------------------- #
# 采集与分割
# --------------------------------------------------------------------------- #
def capture_reference(exposure: float, frames: int = 20) -> tuple[dict, dict]:
    """Median-stack each camera and return (reference, last_frame) per role."""
    references, lasts = {}, {}
    settings = camera_config.load_settings()
    roles = camera_config.load_roles()
    for role, label in ROLES:
        resolved = camera_config.resolve(settings, role, roles.get(role, {}).get("index"))
        resolved["auto_exposure"] = False
        resolved["exposure"] = exposure
        cap = open_camera(resolved)
        if cap is None:
            raise SystemExit(f"[{label}] 打不开相机")
        stack = []
        for _ in range(frames + 15):
            ok, frame = cap.read()
            if ok and frame is not None:
                stack.append(frame)
        cap.release()
        if not stack:
            raise SystemExit(f"[{label}] 读不到画面")
        references[role] = np.median(np.stack(stack[-frames:], 0), axis=0).astype(np.uint8)
        lasts[role] = stack[-1]
    return references, lasts


def grab_pair(exposure: float, frames: int = 8) -> dict[str, np.ndarray]:
    out = {}
    settings = camera_config.load_settings()
    roles = camera_config.load_roles()
    for role, label in ROLES:
        r = camera_config.resolve(settings, role, roles.get(role, {}).get("index"))
        r["auto_exposure"] = False
        r["exposure"] = exposure
        cap = open_camera(r)
        if cap is None:
            raise SystemExit(f"[{label}] 打不开相机")
        frame = None
        for _ in range(frames + 15):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        cap.release()
        out[role] = frame
    return out


def foreground(cur: np.ndarray, ref: np.ndarray, thr: float, min_area: int) -> np.ndarray:
    """差分 → 掩膜（含形态学清理）；参照尺寸和数据类型必须匹配。"""
    if cur is None or ref is None or cur.shape != ref.shape or cur.dtype != ref.dtype:
        raise ValueError("当前帧与参照帧的尺寸/类型不一致")
    if not np.isfinite(thr) or thr < 0 or min_area < 1:
        raise ValueError("差分阈值和最小面积无效")
    # Estimate only a small photometric drift from bright, near-matching background.
    # Dark/coloured objects are excluded; the original reference is never modified.
    current_small = cv2.resize(cur, (320, 240), interpolation=cv2.INTER_AREA).astype(np.float32)
    reference_small = cv2.resize(ref, (320, 240), interpolation=cv2.INTER_AREA).astype(np.float32)
    delta = current_small - reference_small
    if cur.ndim == 3:
        valid = ((current_small.min(axis=2) > 120) & (reference_small.min(axis=2) > 120)
                 & (current_small.max(axis=2) < 245) & (reference_small.max(axis=2) < 245)
                 & (np.abs(delta).max(axis=2) < 60))
    else:
        valid = ((current_small > 120) & (reference_small > 120)
                 & (current_small < 245) & (reference_small < 245) & (np.abs(delta) < 60))
    if np.count_nonzero(valid) >= 1000:
        offset = np.clip(np.median(delta[valid], axis=0), -12.0, 12.0)
        adjusted = np.clip(ref.astype(np.float32) + offset, 0, 255).astype(ref.dtype)
    else:
        adjusted = ref
    d = cv2.absdiff(cur, adjusted)
    gray = d.max(axis=2) if d.ndim == 3 else d
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = (gray > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    return keep

def reference_self_check(last_frame, reference, floor_polygon, thr, min_area,
                         fixture_max_area):
    """空盘参照自检：平底内还有大结构就拒绝写入。

    --capture-ref 与实时里的“重采空盘”共用这一份判据，避免两套标准。
    """
    floor_roi = np.zeros(last_frame.shape[:2], np.uint8)
    cv2.fillConvexPoly(floor_roi, np.rint(np.asarray(floor_polygon)).astype(np.int32), 1)
    residual = foreground(last_frame, reference, thr, min_area)
    structure = structure_foreground(last_frame, floor_roi)
    residual = (residual | structure) & (floor_roi > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(residual, 8)
    big = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)
           if stats[i, cv2.CC_STAT_AREA] >= max(min_area, fixture_max_area)]
    small = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)
             if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return big, small, structure


def tray_floor_polygon_px(cal: dict, extra_margin_mm: float = 0.0) -> np.ndarray:
    """Use the four black-dot FLOOR corners, not an outer-rim inset.

    ``slope_width_mm`` describes distances OUTSIDE this calibrated rectangle;
    subtracting it here would remove valid floor a second time. An explicit
    optional safety margin is the only permitted inset.
    """
    tray = cal["tray"]
    width, height = (float(v) for v in tray["size_mm"])
    if (not np.isfinite(extra_margin_mm) or extra_margin_mm < 0
            or 2 * extra_margin_mm >= min(width, height)):
        raise ValueError("平底安全边距必须非负、有限，且不能覆盖整个工作区")
    if extra_margin_mm == 0:
        return np.array([tray["corners_px"][k] for k in ("TL", "TR", "BR", "BL")], np.float32)
    margin = float(extra_margin_mm)
    corners_mm = np.array([[margin, margin], [width - margin, margin],
                           [width - margin, height - margin], [margin, height - margin]], np.float64)
    homography = tray.get("homography_mm_to_px")
    if homography is None:
        return np.array([tray["corners_px"][k] for k in ("TL", "TR", "BR", "BL")], np.float32)
    matrix = np.asarray(homography, np.float64)
    points = np.column_stack((corners_mm, np.ones(len(corners_mm)))) @ matrix.T
    return (points[:, :2] / points[:, 2][:, None]).astype(np.float32)


def structure_foreground(img: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Filled object regions from closed edges inside the working area.

    Needed for near-white solids on a white tray: their interior barely differs
    from the background, but the silhouette edge still exists. Only used inside
    the tray floor, so tray walls and frame borders are excluded.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    region = (np.asarray(roi) > 0).astype(np.uint8)
    if region.sum() < 100:
        return np.zeros(gray.shape, np.uint8)
    inner = cv2.erode(region, np.ones((15, 15), np.uint8))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 80, 200)
    edges = (edges & (inner > 0)).astype(np.uint8)
    edges = cv2.dilate(edges, np.ones((7, 7), np.uint8))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(edges, 8)
    filled = np.zeros(gray.shape, np.uint8)
    for i in range(1, count):
        component = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < 500.0:
            continue
        cv2.drawContours(filled, [contour], -1, 1, -1)
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return (filled & region).astype(np.uint8)


def adaptive_foreground(img: np.ndarray, roi: np.ndarray | None = None,
                        min_area: int = 2500) -> np.ndarray:
    """Reference-free fallback: Otsu split of dark objects against a bright tray.

    Weaker than an empty-tray difference (shadows and dark surroundings leak in),
    so it is only used when no usable reference frame exists.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    region = np.ones(gray.shape, np.uint8) if roi is None else (roi > 0).astype(np.uint8)
    values = gray[region > 0]
    if values.size < 100:
        return np.zeros(gray.shape, np.uint8)
    otsu, _ = cv2.threshold(values.astype(np.uint8), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu can return the dark level itself on near-bilevel images; use the midpoint
    # between the two class means so the dark class is actually included.
    low, high = values[values <= otsu], values[values > otsu]
    cut = (0.5 * (float(low.mean()) + float(high.mean()))
           if low.size and high.size else float(otsu))
    cut = min(cut, 150.0)
    mask = ((gray < cut) & (region > 0)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    h, w = mask.shape
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        if roi is None and (x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2):
            continue  # background touching the frame edge
        keep[labels == i] = 1
    return keep


@dataclass(frozen=True)
class FootprintGeometry:
    kind: str
    center_mm: tuple[float, float]
    polygon_mm: np.ndarray
    length_mm: float
    width_mm: float
    size_mm: float
    circularity: float
    vertices: int
    aspect: float
    edge_ratio: float
    regular: bool
    yaw_deg: float | None



@dataclass(frozen=True)
class SideAssociation:
    status: str
    component_id: int | None = None
    mask: np.ndarray | None = None


@dataclass
class Detection:
    instance_id: int
    contour_px: np.ndarray
    footprint: FootprintGeometry
    association: SideAssociation
    height_mm: float
    side_kind: str
    profile: tuple[float, ...]
    profile_error: float
    profile_margin: float
    height_residual_px: float
    label: str
    confidence: float
    grasp_z_mm: float = 0.0
    grasp_yaw_deg: float | None = None
    clearance_mm: float = 0.0
    rank: int = 0
    pick_candidate: bool = False
    robot_xyz_mm: tuple[float, float, float] | None = None
    grasp_score: float = 0.0
    color_name: str = "unknown"
    grasp_opening_mm: float = 0.0
    approach_z_mm: float = 0.0
    descend_z_mm: float = 0.0
    rejection_reasons: tuple[str, ...] = ()
    robot_target_ready: bool = False
    geometry_verified: bool = False
    shape_hypothesis: str = "unknown"
    contact_width_mm: float = 0.0
    model_fit_error_px: float | None = None
    model_fit_margin_px: float | None = None
    height_ok: bool = False

def dominant_color(bgr: np.ndarray, mask: np.ndarray) -> str:
    """Estimate a coarse appearance label; it is never used by geometry classification."""
    image = np.asarray(bgr)
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != binary.shape:
        return "unknown"
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    selected = hsv[binary > 0]
    if not len(selected):
        return "unknown"
    saturation = selected[:, 1]
    value = selected[:, 2]
    if float(np.median(saturation)) < 30.0:
        median_value = float(np.median(value))
        return "black" if median_value < 55 else "white" if median_value > 205 else "gray"
    # Hue wraps at 180 (red occurs near both 0 and 179); use circular-bin voting.
    names = ("red", "red_orange", "orange", "yellow", "yellow_green", "green",
             "green_cyan", "cyan", "blue", "indigo", "violet", "magenta")
    hue = selected[:, 0][saturation >= 30].astype(float)
    bins = np.floor((hue + 7.5) / 15.0).astype(int) % len(names)
    return names[int(np.argmax(np.bincount(bins, minlength=len(names))))]


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def pixels_to_mm(points_px: np.ndarray, px2mm: np.ndarray) -> np.ndarray:
    """Apply the tray-plane pixel-to-millimetre homography to N x 2 points."""
    points = np.asarray(points_px, np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ np.asarray(px2mm, np.float64).T
    denominator = homogeneous[:, 2]
    if np.any(np.abs(denominator) < 1e-9):
        raise ValueError("像素到毫米单应在轮廓上接近无穷远")
    return homogeneous[:, :2] / denominator[:, None]


def footprint(contour: np.ndarray, px2mm: np.ndarray) -> FootprintGeometry | None:
    """Measure footprint geometry in tray-plane millimetres, not image pixels."""
    if contour is None or len(contour) < 3:
        return None
    contour_px = np.asarray(contour, np.float32).reshape(-1, 2)
    contour_mm = pixels_to_mm(contour_px, px2mm).astype(np.float32)
    contour_mm_cv = contour_mm.reshape(-1, 1, 2)
    area = float(cv2.contourArea(contour_mm_cv))
    perimeter = float(cv2.arcLength(contour_mm_cv, True))
    if area <= 1e-6 or perimeter <= 1e-6:
        return None

    circularity = 4.0 * np.pi * area / (perimeter * perimeter)
    rect = cv2.minAreaRect(contour_mm_cv)
    (_, _), (rw, rh), rect_angle = rect
    yaw = rect_angle if rw >= rh else rect_angle + 90.0
    yaw = float((yaw + 90.0) % 180.0 - 90.0)
    length_mm, width_mm = sorted((float(rw), float(rh)), reverse=True)
    aspect = length_mm / max(width_mm, 1e-6)
    approx = cv2.approxPolyDP(contour_mm_cv, 0.02 * perimeter, True)
    vertices = len(approx)
    edge_lengths = np.linalg.norm(np.diff(np.vstack((approx.reshape(-1, 2), approx.reshape(-1, 2)[:1])), axis=0), axis=1)
    edge_ratio = float(edge_lengths.min() / max(float(edge_lengths.max()), 1e-6)) if len(edge_lengths) >= 3 else 0.0
    regular = False
    if 3 <= vertices <= 6 and len(edge_lengths) == vertices:
        edge_mean = float(edge_lengths.mean())
        regular = (edge_mean > 0.0
                   and float(edge_lengths.std() / edge_mean) <= 0.14
                   and aspect <= 1.35)
        if vertices == 3:
            regular = regular and edge_ratio >= 0.88
    if circularity >= 0.85 and vertices > 6:
        kind = "circle"
    elif 3 <= vertices <= 6:
        kind = f"poly{vertices}"
    else:
        kind = "unknown"

    moments = cv2.moments(np.asarray(contour, np.float32))
    if abs(moments["m00"]) < 1e-9:
        center_px = contour_px.mean(axis=0, keepdims=True)
    else:
        center_px = np.array([[moments["m10"] / moments["m00"],
                               moments["m01"] / moments["m00"]]])
    center_mm = pixels_to_mm(center_px, px2mm)[0]
    hull_mm = cv2.convexHull(contour_mm_cv).reshape(-1, 2)
    return FootprintGeometry(
        kind=kind,
        center_mm=(float(center_mm[0]), float(center_mm[1])),
        polygon_mm=hull_mm,
        length_mm=length_mm,
        width_mm=width_mm,
        size_mm=float(np.sqrt(length_mm * width_mm)),
        circularity=float(circularity),
        vertices=vertices,
        aspect=float(aspect),
        edge_ratio=edge_ratio,
        regular=regular,
        yaw_deg=None if kind == "circle" else yaw,
    )


def associate_side_components(mask_side: np.ndarray,
                              projected_bases: list[np.ndarray],
                              corridor_px: int = 12) -> list[SideAssociation]:
    """Associate each top-view footprint with one unique side-view component."""
    binary = (np.asarray(mask_side) > 0).astype(np.uint8)
    _, labels, _, _ = cv2.connectedComponentsWithStats(binary, 8)
    candidate_components: list[set[int]] = []
    for points in projected_bases:
        corridor = np.zeros(binary.shape, np.uint8)
        poly = np.round(np.asarray(points).reshape(-1, 2)).astype(np.int32)
        if len(poly) >= 2:
            cv2.polylines(corridor, [poly.reshape(-1, 1, 2)], True, 1,
                          thickness=max(3, 2 * corridor_px + 1))
        elif len(poly) == 1:
            cv2.circle(corridor, tuple(poly[0]), max(1, corridor_px), 1, -1)
        kernel_size = 2 * max(1, corridor_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        corridor = cv2.dilate(corridor, kernel)
        touched = set(int(i) for i in np.unique(labels[corridor > 0]) if i > 0)
        candidate_components.append(touched)

    owners: dict[int, set[int]] = {}
    for candidate, component_ids in enumerate(candidate_components):
        for component_id in component_ids:
            owners.setdefault(component_id, set()).add(candidate)

    result = []
    for component_ids in candidate_components:
        if not component_ids:
            result.append(SideAssociation("no_candidate"))
        elif len(component_ids) > 1:
            result.append(SideAssociation("multiple_candidates"))
        else:
            component_id = next(iter(component_ids))
            if len(owners[component_id]) > 1:
                result.append(SideAssociation("shared_component", component_id))
            else:
                result.append(SideAssociation("matched", component_id,
                                              (labels == component_id).astype(np.uint8)))
    return result


def estimate_supported_height(supported: np.ndarray, heights_mm: np.ndarray,
                              max_gap: int = 2, min_run: int = 3) -> tuple[float | None, str]:
    """Estimate the grounded supported run; reject isolated hits and scan clipping."""
    supported = np.asarray(supported, dtype=bool).reshape(-1)
    heights = np.asarray(heights_mm, dtype=np.float64).reshape(-1)
    if supported.size != heights.size or not supported.size:
        raise ValueError("高度支持序列与采样高度长度必须相同且非空")
    hits = np.flatnonzero(supported)
    if not hits.size:
        return None, "no_support"

    groups: list[tuple[int, int, int]] = []
    start = previous = int(hits[0])
    hit_count = 1
    for raw_index in hits[1:]:
        index = int(raw_index)
        if index - previous <= max_gap + 1:
            hit_count += 1
            previous = index
        else:
            groups.append((start, previous, hit_count))
            start = previous = index
            hit_count = 1
    groups.append((start, previous, hit_count))

    step = float(np.median(np.diff(heights))) if len(heights) > 1 else 1.0
    grounded = [g for g in groups if heights[g[0]] <= heights[0] + 2.0 * step and g[2] >= min_run]
    if not grounded:
        return None, "no_grounded_run"
    _, last, _ = grounded[0]
    if last >= len(heights) - 2:
        return None, "clipped_at_h_max"
    return float(heights[last]), "ok"


def classify_profile(profile: np.ndarray, max_error: float = 0.035,
                     min_margin: float = 0.012) -> tuple[str, float, float]:
    """Fit normalized side-width samples and reject poor or ambiguous matches."""
    values = np.asarray(profile, np.float64).reshape(-1)
    if values.size != 12 or not np.all(np.isfinite(values)):
        return "unknown", float("inf"), 0.0
    t_values = np.linspace(0.0, 0.95, len(values))
    errors = sorted((float(np.mean((values - np.array([model(t) for t in t_values])) ** 2)), name)
                    for name, model in PROFILE_MODELS.items())
    best_error, best_kind = errors[0]
    margin = errors[1][0] - best_error
    if best_error > max_error or margin < min_margin:
        return "unknown", best_error, margin
    return best_kind, best_error, margin


def side_profile(cal, Rt, tvec, R, T, up: float, mask_side: np.ndarray,
                 foot: FootprintGeometry, h_max: float = 120.0, step: float = 1.0):
    """Estimate height from the associated silhouette extreme, then fit its width profile."""
    mask = (np.asarray(mask_side) > 0).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if not len(xs) or foot.polygon_mm.shape[0] < 3 or step <= 0:
        return 0.0, "none", [], float("inf"), 0.0, float("inf")

    center = np.asarray(foot.center_mm, np.float64)
    center_px = side_project(cal, Rt, tvec, R, T,
                             np.array([[center[0], center[1], 0.0]], np.float64))[0]
    one_up_px = side_project(cal, Rt, tvec, R, T,
                             np.array([[center[0], center[1], up]], np.float64))[0]
    vertical_sign = float(np.sign(one_up_px[1] - center_px[1]))
    if vertical_sign == 0.0:
        return 0.0, "none", [], float("inf"), 0.0, float("inf")
    observed_extreme = float(np.max(ys) if vertical_sign > 0 else np.min(ys))

    support = cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0
    heights = np.arange(0.0, h_max + step * 0.5, step, dtype=np.float32)
    polygon = foot.polygon_mm.astype(np.float32)
    top_models = {
        "flat": polygon,
        "peak": center.reshape(1, 2).astype(np.float32),
    }
    center_px_xy = np.array([[center[0], center[1]]], np.float64)
    base_xy = np.vstack((center_px_xy, center_px_xy + [1.0, 0.0], center_px_xy + [0.0, 1.0]))
    base_px = side_project(cal, Rt, tvec, R, T,
                           np.column_stack((base_xy, np.zeros(3, np.float64))))
    across = np.array([base_px[1, 0] - base_px[0, 0],
                       base_px[2, 0] - base_px[0, 0]], np.float64)
    across /= max(float(np.linalg.norm(across)), 1e-9)
    half_width = max(float(np.max(np.abs((polygon - center) @ across))), foot.width_mm / 2)
    lateral = np.linspace(-half_width, half_width, 31)

    candidates = []
    for top_model, top_xy in top_models.items():
        xyz = np.empty((len(heights) * len(top_xy), 3), np.float32)
        xyz[:, :2] = np.tile(top_xy, (len(heights), 1))
        xyz[:, 2] = np.repeat(heights * up, len(top_xy))
        projected = side_project(cal, Rt, tvec, R, T, xyz).reshape(len(heights), len(top_xy), 2)
        extremes = (projected[:, :, 1].max(axis=1) if vertical_sign > 0
                    else projected[:, :, 1].min(axis=1))
        residuals = np.abs(extremes - observed_extreme)
        height_index = int(np.argmin(residuals))
        height_mm = float(heights[height_index])
        residual_px = float(residuals[height_index])
        if height_index >= len(heights) - 2:
            candidates.append((float("inf"), height_mm, "unknown", [], float("inf"), 0.0, residual_px))
            continue

        profile = []
        for fraction in np.linspace(0.0, 0.95, 12):
            xy = center[None, :] + lateral[:, None] * across[None, :]
            points = np.column_stack((xy, np.full(len(xy), up * fraction * height_mm)))
            px = side_project(cal, Rt, tvec, R, T, points)
            x, y = np.rint(px[:, 0]).astype(int), np.rint(px[:, 1]).astype(int)
            valid = (x >= 0) & (x < support.shape[1]) & (y >= 0) & (y < support.shape[0])
            profile.append(float(np.mean(support[y[valid], x[valid]])) if np.any(valid) else 0.0)
        kind, error, margin = classify_profile(np.asarray(profile))
        fit_cost = error if kind != "unknown" else 1.0
        candidates.append((residual_px + 20.0 * fit_cost, height_mm, kind,
                           profile, error, margin, residual_px))

    candidates.sort(key=lambda result: result[0])
    best = candidates[0]
    if best[6] > 4.0 or best[1] <= 2.0 or best[1] >= h_max - 2.0:
        return 0.0, "none", [], best[4], best[5], best[6]
    if len(candidates) > 1 and abs(candidates[1][0] - best[0]) < 0.5:
        return best[1], "unknown", best[3], best[4], best[5], best[6]
    return best[1], best[2], best[3], best[4], best[5], best[6]


SHAPE_CLASSES = (
    "cube", "cuboid", "cone", "cylinder", "ball", "regular_tetrahedron",
    "triangular_pyramid", "square_pyramid", "pentagonal_pyramid", "hexagonal_pyramid",
    "triangular_prism", "pentagonal_prism", "hexagonal_prism",
)


def fuse(foot: FootprintGeometry, side_kind: str, h_mm: float,
         profile_error: float, profile_margin: float,
         association_status: str = "matched") -> tuple[str, float]:
    """Fuse shape hypotheses; confidence is a quality score, not a probability."""
    if association_status != "matched":
        return f"unknown({association_status})", 0.0
    if side_kind in ("none", "unknown") or h_mm <= 2.0:
        return "unknown(侧面轮廓)", 0.0
    if foot.kind == "unknown":
        return "unknown(俯视足迹)", 0.0
    fit_quality = max(0.0, 1.0 - profile_error / 0.035)
    separation = min(1.0, profile_margin / 0.04)
    quality = float(0.5 + 0.5 * fit_quality * separation)
    height_ratio = h_mm / max(foot.width_mm, 1e-6)

    if foot.kind == "circle":
        if side_kind == "circle":
            return "ball", quality
        if side_kind == "triangle":
            return "cone", quality
        if side_kind == "rect":
            return "cylinder", quality
    if foot.kind == "poly4":
        if side_kind == "rect":
            cube_ratio = h_mm / max(foot.width_mm, 1e-6)
            is_cube = foot.aspect < 1.35 and 0.75 <= cube_ratio <= 1.35
            return ("cube" if is_cube else "cuboid"), quality
        if side_kind == "triangle":
            return "square_pyramid", quality
    if foot.kind == "poly3" and side_kind == "triangle":
        if foot.edge_ratio >= 0.88 and 0.78 <= height_ratio <= 0.96:
            return "regular_tetrahedron", quality
        return "triangular_pyramid", quality
    if foot.kind.startswith("poly"):
        sides = int(foot.kind[4:])
        if side_kind == "triangle":
            return {3: "triangular_pyramid", 4: "square_pyramid",
                    5: "pentagonal_pyramid", 6: "hexagonal_pyramid"}.get(
                        sides, f"{sides}-sided_pyramid"), quality
        if side_kind == "rect":
            return {3: "triangular_prism", 5: "pentagonal_prism",
                    6: "hexagonal_prism"}.get(sides, f"{sides}-sided_prism"), quality
    return f"unknown({foot.kind}+{side_kind})", 0.0

def _point_to_polygon_distance(points: np.ndarray, polygon: np.ndarray) -> float:
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    poly = np.asarray(polygon, np.float32).reshape(-1, 2)
    poly_cv = poly.reshape(-1, 1, 2)
    if cv2.pointPolygonTest(poly_cv, tuple(float(v) for v in pts[0]), False) >= 0:
        return 0.0
    starts = poly
    vectors = np.roll(poly, -1, axis=0) - starts
    delta = pts[:, None, :] - starts[None, :, :]
    lengths_sq = np.maximum(np.sum(vectors * vectors, axis=1), 1e-9)
    t = np.clip(np.sum(delta * vectors[None, :, :], axis=2) / lengths_sq[None, :], 0.0, 1.0)
    nearest = starts[None, :, :] + t[:, :, None] * vectors[None, :, :]
    return float(np.sqrt(np.sum((pts[:, None, :] - nearest) ** 2, axis=2)).min())

def polygon_clearance_mm(first: np.ndarray, second: np.ndarray) -> float:
    a = cv2.convexHull(np.asarray(first, np.float32).reshape(-1, 1, 2))
    b = cv2.convexHull(np.asarray(second, np.float32).reshape(-1, 1, 2))
    area, _ = cv2.intersectConvexConvex(a, b)
    if area > 1e-6:
        return 0.0
    return min(_point_to_polygon_distance(a[:, 0, :], b[:, 0, :]),
               _point_to_polygon_distance(b[:, 0, :], a[:, 0, :]))


def tray_point_to_robot(point_mm: tuple[float, float, float],
                        robot_from_tray_mm: np.ndarray | None) -> tuple[float, float, float] | None:
    if robot_from_tray_mm is None:
        return None
    transform = np.asarray(robot_from_tray_mm, np.float64)
    if (transform.shape != (4, 4) or not np.all(np.isfinite(transform))
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-6)):
        raise ValueError("robot_from_tray_mm 必须是有限的刚体齐次变换")
    if not np.all(np.isfinite(point_mm)):
        raise ValueError("目标坐标必须有限")
    # Transform uses raw PnP tray axes; caller converts positive height to signed z.
    point = transform @ np.array([*point_mm, 1.0], np.float64)
    return tuple(float(v) for v in point[:3])


def rank_grasps(detections: list[Detection], tray_size_mm: tuple[float, float],
                grasp_height_ratio: float = 0.5, min_clearance_mm: float = 5.0,
                robot_from_tray_mm: np.ndarray | None = None,
                gripper_opening_range_mm: tuple[float, float] | None = None,
                gripper_clearance_mm: float = 2.0,
                approach_clearance_mm: float = 20.0,
                tray_up_sign: float = 1.0) -> list[Detection]:
    """Rank geometric candidates. Robot execution remains pending hardware validation."""
    scalars = [grasp_height_ratio, min_clearance_mm, gripper_clearance_mm,
               approach_clearance_mm, *tray_size_mm, tray_up_sign]
    if (not np.all(np.isfinite(scalars)) or not 0 < grasp_height_ratio < 1
            or min_clearance_mm < 0 or gripper_clearance_mm < 0
            or approach_clearance_mm < 0 or min(tray_size_mm) <= 0
            or tray_up_sign not in (-1.0, 1.0)):
        raise ValueError("抓取参数应有限，距离非负，高度比例在 (0,1)，法向为 +/-1")
    if gripper_opening_range_mm is not None:
        if (len(gripper_opening_range_mm) != 2
                or not np.all(np.isfinite(gripper_opening_range_mm))
                or not 0 <= gripper_opening_range_mm[0] < gripper_opening_range_mm[1]):
            raise ValueError("夹爪开口范围应为有限且有序的非负数")
    tray_point_to_robot((0.0, 0.0, 0.0), robot_from_tray_mm)
    tray_w, tray_h = tray_size_mm
    for item in detections:
        values = [*item.footprint.center_mm, item.footprint.length_mm,
                  item.footprint.width_mm, item.height_mm, item.confidence]
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(item.footprint.polygon_mm)):
            raise ValueError("物体的坐标、轮廓、尺寸和评分必须有限")
    scene_height = max([0.0] + [d.height_mm for d in detections])
    scene_unresolved = any(not d.geometry_verified for d in detections)
    for index, detection in enumerate(detections):
        reasons: list[str] = list(detection.rejection_reasons)
        height_ok = detection.height_ok or detection.height_residual_px <= 4.0
        foot = detection.footprint
        polygon = foot.polygon_mm
        x, y = polygon[:, 0], polygon[:, 1]
        clearance = min(float(x.min()), float(y.min()),
                        tray_w - float(x.max()), tray_h - float(y.max()))
        for other_index, other in enumerate(detections):
            if index != other_index:
                clearance = min(clearance, polygon_clearance_mm(
                    polygon, other.footprint.polygon_mm))
        detection.clearance_mm = max(0.0, clearance)
        detection.grasp_yaw_deg = foot.yaw_deg
        axis_width = foot.width_mm if foot.yaw_deg is not None else min(foot.length_mm, foot.width_mm)
        if detection.side_kind == "triangle":
            axis_width *= 1.0 - grasp_height_ratio
        elif detection.label == "ball":
            axis_width *= 2.0 * np.sqrt(grasp_height_ratio * (1.0 - grasp_height_ratio))
        detection.contact_width_mm = float(axis_width)
        detection.grasp_opening_mm = axis_width + 2.0 * gripper_clearance_mm
        z_offset = detection.height_mm * grasp_height_ratio
        detection.grasp_z_mm = max(0.0, z_offset)
        detection.descend_z_mm = detection.grasp_z_mm
        detection.approach_z_mm = scene_height + approach_clearance_mm
        if scene_unresolved:
            _append_reason(reasons, "场景存在未确定三维几何的障碍物")
        detection.robot_xyz_mm = (tray_point_to_robot(
            (foot.center_mm[0], foot.center_mm[1], tray_up_sign * detection.descend_z_mm),
            robot_from_tray_mm) if detection.geometry_verified else None)
        if not detection.geometry_verified:
            _append_reason(reasons, "几何估计未完成三维校正")
        _append_reason(reasons, "待实机验证TCP、工具姿态、IK和碰撞")

        if detection.label not in SHAPE_CLASSES:
            _append_reason(reasons, "类别未确定(仅几何可用)")
        if detection.association.status != "matched":
            _append_reason(reasons, f"侧视关联:{detection.association.status}")
        if detection.height_mm <= 2.0 or not height_ok:
            _append_reason(reasons, "高度观测质量不足")
        if detection.clearance_mm < min_clearance_mm:
            _append_reason(reasons, "托盘边缘或邻块净空不足")
        if gripper_opening_range_mm is None:
            _append_reason(reasons, "夹爪开口范围未配置")
        else:
            min_opening, max_opening = gripper_opening_range_mm
            if detection.contact_width_mm < min_opening or detection.grasp_opening_mm > max_opening:
                _append_reason(reasons, "夹爪闭合或预张开超出配置范围")
        if robot_from_tray_mm is None:
            _append_reason(reasons, "机械臂基座到托盘变换未配置")
        detection.robot_target_ready = False  # No execution verification in this offline module.
        detection.pick_candidate = (
            detection.geometry_verified
            and detection.label in SHAPE_CLASSES
            and not scene_unresolved
            and height_ok
            and detection.association.status == "matched"
            and detection.height_mm > 2.0
            and detection.clearance_mm >= min_clearance_mm
            and gripper_opening_range_mm is not None
            and gripper_opening_range_mm[0] <= detection.contact_width_mm
            and detection.grasp_opening_mm <= gripper_opening_range_mm[1]
        )
        detection.rejection_reasons = tuple(reasons)
        detection.grasp_score = detection.confidence  # Diagnostic recognition quality; rank uses explicit priorities.

    # Reliability first, not a weighted trade-off: extra clearance must never
    # compensate for weaker recognition. Clearance/tool checks remain hard gates.
    ordered = sorted(detections,
                     key=lambda item: (
                         not item.pick_candidate,
                         -item.confidence,
                         -(item.model_fit_margin_px if item.model_fit_margin_px is not None else 0.0),
                         item.model_fit_error_px if item.model_fit_error_px is not None else item.height_residual_px,
                         -item.clearance_mm,
                         item.footprint.center_mm[1], item.footprint.center_mm[0]))
    rank = 0
    for detection in ordered:
        if detection.pick_candidate:
            rank += 1
            detection.rank = rank
        else:
            detection.rank = 0
    return ordered


def detections_to_payload(detections: list[Detection]) -> list[dict]:
    """Machine-readable tray-frame grasp targets and perception evidence."""
    payload = []
    for detection in detections:
        foot = detection.footprint
        payload.append({
            "object_id": detection.instance_id,
            "class": detection.label,
            "shape_hypothesis": detection.shape_hypothesis,
            "geometry_valid": detection.geometry_verified,
            "confidence_score": round(detection.confidence, 4),
            "appearance": {"color": detection.color_name},
            "position_mm": {"frame": "tray", "x": round(foot.center_mm[0], 2),
                            "y": round(foot.center_mm[1], 2), "z_base": 0.0},
            "dimensions_mm": {"length": round(foot.length_mm, 2),
                              "width": round(foot.width_mm, 2),
                              "height": round(detection.height_mm, 2)},
            "grasp_target": {
                "frame": "tray",
                "x_mm": round(foot.center_mm[0], 2),
                "y_mm": round(foot.center_mm[1], 2),
                "approach_z_mm": round(detection.approach_z_mm, 2),
                "bbox_center_z_mm": round(detection.height_mm / 2.0, 2),
                "descend_z_mm": round(detection.descend_z_mm, 2),
                "gripper_opening_mm": round(detection.grasp_opening_mm, 2),
                "yaw_deg": (round(detection.grasp_yaw_deg, 2)
                            if detection.grasp_yaw_deg is not None else None),
                "approach": "top_down",
                "status": "perception_candidate" if detection.pick_candidate else "hold",
            },
            "clearance_mm": round(detection.clearance_mm, 2),
            "rank": detection.rank,
            "pick_candidate": detection.pick_candidate,
            "rejection_reasons": list(detection.rejection_reasons),
            "robot_target_xyz_mm": (list(detection.robot_xyz_mm)
                                    if detection.robot_xyz_mm is not None else None),
            "robot_target_ready": detection.robot_target_ready,
            "evidence": {
                "side_association": detection.association.status,
                "model_fit_error_px": detection.model_fit_error_px,
                "model_fit_margin_px": detection.model_fit_margin_px,
                "side_profile_error": (round(detection.profile_error, 5)
                                        if np.isfinite(detection.profile_error) else None),
                "height_residual_px": (round(detection.height_residual_px, 3)
                                       if np.isfinite(detection.height_residual_px) else None),
            },
        })
        item = payload[-1]
        item["coordinate_convention"] = "tray x/y as calibrated; height positive upwards; robot transform uses raw PnP z"
        item["grasp_target"]["yaw_convention"] = "finger_long_axis_in_tray_xy; closing_axis=yaw+90deg"
        item["grasp_target"]["contact_width_mm"] = round(detection.contact_width_mm, 2)
        item["robot_command_ready"] = False
        if not detection.geometry_verified:
            item["unverified_estimates"] = {
                "position_mm": item["position_mm"], "dimensions_mm": item["dimensions_mm"],
                "grasp_target": item["grasp_target"],
            }
            item["position_mm"] = None
            item["dimensions_mm"] = None
            item["grasp_target"] = None
            item["clearance_mm"] = None
    return payload

def analyse_masks(mask_vertical: np.ndarray, mask_side: np.ndarray, cal: dict,
                  min_area: int = 1, side_corridor_px: int = 12,
                  h_max: float = 120.0, step: float = 1.0,
                  grasp_height_ratio: float = 0.5, min_clearance_mm: float = 5.0,
                  robot_from_tray_mm: np.ndarray | None = None,
                  gripper_opening_range_mm: tuple[float, float] | None = None,
                  gripper_clearance_mm: float | None = None,
                  approach_clearance_mm: float | None = None,
                  image_vertical: np.ndarray | None = None,
                  fit_cache: dict | None = None) -> list[Detection]:
    """Offline analysis entry point. Accepts masks and calibration; opens no devices."""
    if not 0.0 < grasp_height_ratio < 1.0:
        raise ValueError("grasp_height_ratio 必须在 0 和 1 之间")
    if (np.asarray(mask_vertical).ndim != 2 or np.asarray(mask_side).ndim != 2
            or not np.asarray(mask_vertical).size or not np.asarray(mask_side).size):
        raise ValueError("两路掩膜必须是非空二维数组")
    if (not np.all(np.isfinite([h_max, step, min_area, side_corridor_px]))
            or step <= 0 or h_max <= 2 * step or h_max / step > 2000
            or min_area < 1 or side_corridor_px < 0):
        raise ValueError("高度搜索步长/范围、最小面积和关联容差无效")
    if image_vertical is not None and np.asarray(image_vertical).shape != (*np.asarray(mask_vertical).shape, 3):
        raise ValueError("俯视原图尺寸必须与掩膜一致且具有三个通道")
    Rt, tvec, R, T, tray_w, tray_h, up = tray_pose(cal)
    px2mm = np.asarray(cal["tray"]["homography_px_to_mm"], np.float64)
    _corners = np.asarray([cal["tray"]["corners_px"][k]
                           for k in ("TL", "TR", "BR", "BL")], np.float64)
    _w, _h = (float(v) for v in cal["tray"]["size_mm"])
    px_per_mm = 0.5 * (np.linalg.norm(_corners[1] - _corners[0]) / _w
                       + np.linalg.norm(_corners[3] - _corners[0]) / _h)
    robot_cfg = cal.get("robot", {})
    gripper_cfg = robot_cfg.get("gripper", {})
    limits = gripper_cfg.get("opening_range_mm")
    if gripper_opening_range_mm is None and limits is not None:
        gripper_opening_range_mm = (float(limits[0]), float(limits[1]))
    gripper_clearance_mm = float(gripper_cfg.get("clearance_mm", 2.0)
                                  if gripper_clearance_mm is None else gripper_clearance_mm)
    approach_clearance_mm = float(gripper_cfg.get("approach_clearance_mm", 20.0)
                                   if approach_clearance_mm is None else approach_clearance_mm)
    if robot_from_tray_mm is None:
        transform = robot_cfg.get("T_robot_from_tray_mm")
        if transform is not None:
            robot_from_tray_mm = np.asarray(transform, np.float64)

    binary = (np.asarray(mask_vertical) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    candidates = []
    for label_id in range(1, count):
        if stats[label_id, cv2.CC_STAT_AREA] < min_area:
            continue
        component = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        measured = footprint(contour, px2mm)
        if measured is not None:
            candidates.append((label_id, contour, measured))

    projected_bases = [side_project(cal, Rt, tvec, R, T,
                                    np.column_stack((item[2].polygon_mm,
                                                     np.zeros(len(item[2].polygon_mm), np.float32))))
                       for item in candidates]
    associations = associate_side_components(mask_side, projected_bases, side_corridor_px)
    detections = []
    for (instance_id, contour, measured), association in zip(candidates, associations):
        if association.status == "matched" and association.mask is not None:
            height, side_kind, profile, error, margin, residual = side_profile(
                cal, Rt, tvec, R, T, up, association.mask, measured, h_max, step)
        else:
            height, side_kind, profile = 0.0, "none", []
            error, margin, residual = float("inf"), 0.0, float("inf")
        label, confidence = fuse(measured, side_kind, height, error, margin, association.status)
        detection = Detection(instance_id, contour, measured, association, height,
                              side_kind, tuple(profile), error, margin, residual,
                              label, confidence)
        detection.shape_hypothesis = label
        # A floor homography is an association seed, not a corrected elevated-object
        # measurement. Only an accepted dual-view 3D fit may promote valid XYZ.
        detection.label = "unknown(unclassified)"
        detection.confidence = 0.0
        if association.status == "matched" and association.mask is not None:
            side_contours, _ = cv2.findContours(association.mask, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
            side_contour = max(side_contours, key=cv2.contourArea) if side_contours else None
            # Cut or touching frame boundaries are unsupported for full-silhouette fitting.
            clipped = False
            for boundary, shape in ((contour, np.asarray(mask_vertical).shape),
                                     (side_contour, np.asarray(mask_side).shape)):
                if boundary is None:
                    clipped = True
                    break
                bx, by, bw, bh = cv2.boundingRect(boundary)
                clipped |= bx <= 1 or by <= 1 or bx+bw >= shape[1]-1 or by+bh >= shape[0]-1
            def _fit_contour(c):
                # A convex solid projects to a convex silhouette: rim/shadow noise makes
                # the measured boundary ragged, so fit the hull when it is nearly convex.
                if c is None:
                    return None
                area = float(cv2.contourArea(c))
                hull = cv2.convexHull(c)
                hull_area = float(cv2.contourArea(hull))
                if hull_area <= 0 or area / hull_area < 0.90:
                    return None
                return hull.reshape(-1, 2).astype(np.float64)
            fit_top, fit_side = _fit_contour(contour), _fit_contour(side_contour)
            if fit_top is None or fit_side is None:
                detection.rejection_reasons += ("轮廓非凸或粘连，跳过三维拟合",)
            cache_key = (round(measured.center_mm[0] / 8), round(measured.center_mm[1] / 8))
            fit_options = ({"warm_start": fit_cache.get(cache_key)} if fit_cache is not None else {})
            fit = (fit_two_view(fit_top, fit_side, cal, measured.center_mm,
                                (measured.length_mm, measured.width_mm), h_max, **fit_options)
                   if not clipped and fit_top is not None and fit_side is not None else None)
            if fit_cache is not None and fit is not None and fit.get("warm_start"):
                fit_cache[cache_key] = fit["warm_start"]
            if fit is not None:
                detection.model_fit_error_px = fit["error_px"]
                detection.model_fit_margin_px = fit["margin_px"]
                detection.shape_hypothesis = fit["label"]
            if fit is not None and fit["accepted"]:
                base = np.asarray(fit["polygon_mm"], np.float32)
                if fit["label"] in ("ball", "cylinder", "cone"):
                    yaw, length, width = None, fit["length_mm"], fit["width_mm"]
                else:
                    rectangle = cv2.minAreaRect(base.reshape(-1,1,2))
                    (_, _), (rw, rh), angle = rectangle
                    length, width = max(rw,rh), min(rw,rh)
                    yaw = float((angle + (90.0 if rh > rw else 0.0) + 90.0) % 180.0 - 90.0)
                detection.footprint = replace(measured, center_mm=tuple(fit["center_mm"]),
                    polygon_mm=base, length_mm=float(length), width_mm=float(width),
                    size_mm=float(np.sqrt(length*width)), aspect=float(length/max(width,1e-6)),
                    yaw_deg=yaw)
                detection.height_mm = fit["height_mm"]
                detection.height_residual_px = fit["error_px"]
                detection.side_kind = ("triangle" if fit["label"] in ("cone", "regular_tetrahedron")
                    or fit["label"].endswith("pyramid") else "circle" if fit["label"] == "ball" else "rect")
                detection.label = fit["label"]
                detection.confidence = float(min(1.0, max(0.0, 1.0 - (fit["error_px"] / px_per_mm) / 5.0)))
                detection.geometry_verified = True
                detection.height_ok = True
        if image_vertical is not None:
            object_mask = np.zeros(np.asarray(mask_vertical).shape, np.uint8)
            cv2.drawContours(object_mask, [contour], -1, 1, -1)
            detection.color_name = dominant_color(image_vertical, object_mask)
        detections.append(detection)
    return rank_grasps(detections, (tray_w, tray_h), grasp_height_ratio,
                       min_clearance_mm, robot_from_tray_mm,
                       gripper_opening_range_mm, gripper_clearance_mm,
                       approach_clearance_mm, tray_up_sign=up)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="双相机融合判定物块类型")
    parser.add_argument("--capture-ref", action="store_true", help="采空托盘参照帧后退出")
    parser.add_argument("--calibration", type=Path, default=JSON_PATH)
    parser.add_argument("--offline-vertical", type=Path, help="离线俯视图或掩膜")
    parser.add_argument("--offline-side", type=Path, help="离线侧视图或掩膜")
    parser.add_argument("--input-masks", action="store_true", help="离线输入为二值掩膜")
    parser.add_argument("--ref-dir", type=Path, default=REF_DIR)
    parser.add_argument("--output-json", type=Path, help="离线结果 JSON 路径")
    parser.add_argument("--exposure", type=float, default=-4.0)
    parser.add_argument("--thr", type=float, default=28.0, help="差分阈值")
    parser.add_argument("--min-area", type=int, default=2500, help="最小物块面积(px, 原图)")
    parser.add_argument("--grasp-height-ratio", type=float, default=0.5)
    parser.add_argument("--gripper-min-opening-mm", type=float, default=None)
    parser.add_argument("--gripper-max-opening-mm", type=float, default=None)
    parser.add_argument("--gripper-clearance-mm", type=float, default=None)
    parser.add_argument("--approach-clearance-mm", type=float, default=None)
    parser.add_argument("--jsonl", action="store_true", help="每秒输出一行机器可读 JSON")
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--min-clearance-mm", type=float, default=5.0)
    parser.add_argument("--fit-interval", type=float, default=2.0,
                        help="三维拟合最小间隔(秒)；拟合较慢，中间帧沿用上次结果")
    parser.add_argument("--fixture-max-area", type=int, default=12000,
                        help="采参照时允许忽略的静态固定结构面积(px)，更大的视为未清空")
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--session-dir", type=Path, help="保存实机原图、掩膜与逐帧诊断")
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--sample-limit", type=int, default=40)
    parser.add_argument("--max-result-age", type=float, default=8.0,
                        help="拟合来源帧最大时效(秒)，超时撤销目标")
    parser.add_argument("--skip-exposure-picker", action="store_true",
                        help="跳过启动时的选光照步骤，直接用 --exposure")
    args = parser.parse_args(argv)
    if (not np.all(np.isfinite([args.fit_interval, args.max_result_age, args.sample_interval]))
            or args.max_result_age <= 0 or args.sample_interval <= 0 or args.sample_limit < 0):
        parser.error("时效、采样间隔须为正有限数，采样上限须非负")

    if bool(args.offline_vertical) != bool(args.offline_side):
        parser.error("离线模式需要同时提供 --offline-vertical 与 --offline-side")
    if (args.gripper_min_opening_mm is None) != (args.gripper_max_opening_mm is None):
        parser.error("夹爪最小和最大开口必须同时提供")
    cal = json.loads(args.calibration.read_text(encoding="utf-8"))
    if args.offline_vertical:
        flag = cv2.IMREAD_GRAYSCALE if args.input_masks else cv2.IMREAD_COLOR
        vertical = cv2.imread(str(args.offline_vertical), flag)
        side = cv2.imread(str(args.offline_side), flag)
        if vertical is None or side is None:
            parser.error("离线输入图像读取失败")
        if args.input_masks:
            mv, ms = vertical > 0, side > 0
        else:
            references = [cv2.imread(str(args.ref_dir / f"ref_{role}.jpg"))
                          for role, _ in ROLES]
            if any(ref is None for ref in references):
                parser.error("离线差分模式需要两路空托盘参照图")
            from scripts.two_view_live import segment_pair
            fixture_file = args.ref_dir / "ref_static_vertical.png"
            fixture = cv2.imread(str(fixture_file), 0) if fixture_file.exists() else None
            if fixture is not None:
                fixture = cv2.dilate((fixture > 0).astype(np.uint8), np.ones((15, 15), np.uint8))
            masks = segment_pair(sys.modules[__name__], {"vertical": vertical, "side_a": side},
                                 dict(zip(("vertical", "side_a"), references)), cal,
                                 args.thr, args.min_area, fixture)
            mv, ms = masks["vertical"], masks["side_a"]
        from scripts.vision_workspace import work_area_polygon_px
        work_roi = np.zeros(mv.shape, np.uint8)
        cv2.fillConvexPoly(work_roi, np.rint(work_area_polygon_px(cal, mv.shape)).astype(np.int32), 1)
        mv = (mv & (work_roi > 0)).astype(np.uint8)
        opening = ((args.gripper_min_opening_mm, args.gripper_max_opening_mm)
                   if args.gripper_min_opening_mm is not None else None)
        detections = analyse_masks(
            mv, ms, cal, min_area=args.min_area,
            grasp_height_ratio=args.grasp_height_ratio, min_clearance_mm=args.min_clearance_mm,
            gripper_opening_range_mm=opening, gripper_clearance_mm=args.gripper_clearance_mm,
            approach_clearance_mm=args.approach_clearance_mm,
            image_vertical=None if args.input_masks else vertical)
        text = json.dumps(detections_to_payload(detections), ensure_ascii=False, indent=2,
                          allow_nan=False)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    Rt, tvec, R, T, _, _, up = tray_pose(cal)
    roi_px = tray_floor_polygon_px(cal)

    if args.capture_ref:
        references, lasts = capture_reference(args.exposure)
        big, small, structure = reference_self_check(
            lasts["vertical"], references["vertical"], roi_px,
            args.thr, args.min_area, args.fixture_max_area)
        if big:
            print(f"参照帧自检失败：平底内仍能看到物块 {big}，"
                  f"未写入参照帧；请清空托盘并保持不动后重采", file=sys.stderr)
            return 3
        REF_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(REF_DIR / "ref_static_vertical.png"), structure * 255)
        if small:
            print(f"已忽略 {len(small)} 处静态固定结构 {small}（作为背景减除）")
        for role, _ in ROLES:
            out = REF_DIR / f"ref_{role}.jpg"
            cv2.imwrite(str(out), references[role], [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"参照帧 {out.relative_to(ROOT)}")
        print("参照帧自检通过：平底内无物块，可直接实时运行。")
        return 0

    from scripts.two_view_live import run_live
    return run_live(sys.modules[__name__], args, cal)


if __name__ == "__main__":
    raise SystemExit(main())
