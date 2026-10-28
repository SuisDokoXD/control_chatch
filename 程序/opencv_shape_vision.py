"""用 OpenCV 传统视觉直接识别物块形状（不训练模型、不使用颜色作为类别）。

思路：
1. 托盘背景建模：只用"托盘色"像素做归一化卷积，得到平滑的亮度/色度曲面，
   用它抵消光照渐变；黑边、灰色台面不会污染模型。
2. 物体候选：色度明显偏离背景，或明显比背景暗（白色物块属于后者）。
3. 粘连分离：对候选区做距离变换取峰值作为种子，在颜色梯度图上做标记分水岭。
4. 形状描述：轮廓几何（圆度、凸性、外接矩形填充、多边形顶点数……）
   加上内部明暗结构（球面径向渐变 vs 平面圆盘的平坦度）。
5. 规则判定 9 类形状，输出标注图和 JSON。

颜色只用于"把物体和托盘分开"，从不参与形状判定；白色物块靠亮度差。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - 环境缺失时才触发
    print("缺少 OpenCV/NumPy，请先执行: python -m pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(2) from exc


ROOT = Path(__file__).resolve().parent.parent
RAW_IMAGES = ROOT / "dataset" / "yolo_cube" / "images" / "raw"
RAW_LABELS = ROOT / "dataset" / "yolo_cube" / "labels" / "raw"
DEFAULT_OUTPUT = ROOT / "captures" / "shape_cv"

SHAPES = (
    "ball", "cube", "cuboid", "cone", "tetrahedron",
    "pentagonal_prism", "hexagonal_prism", "cylinder", "triangular_prism",
)
UNKNOWN = "unknown"   # 掩膜质量不足时拒判，不硬套一个形状类别

# 托盘四角黑点的归一化坐标（顺序：左上 → 右上 → 右下 → 左下）。运行时把检测到的
# 角点对齐到这份基准，即使某个角点被物块挡住也能推断出完整托盘四边形。
#
# 2026-10-26：相机被转过 90°，旧基准（长边水平）已失效。现在**优先读标定结果**
# 配置/calibration_stereo.json 的 tray.corners_px —— 和"像素→毫米"用同一套点，
# 不会再出现"检测按一套、换算按另一套"的错位；读不到时才退回下面这份实测基准。
TRAY_DOT_REF_FALLBACK = (
    (0.1944, 0.0192),   # 左上
    (0.7932, 0.0296),   # 右上
    (0.7889, 0.8789),   # 右下
    (0.1963, 0.8848),   # 左下
)                       # 顺序：左上 → 右上 → 右下 → 左下


def _load_tray_dot_ref() -> tuple[tuple[float, float], ...]:
    """优先用标定结果里的托盘角点（和 tray 单应同源），失败则用内置基准。"""
    try:
        data = json.loads((ROOT / "配置" / "calibration_stereo.json").read_text(encoding="utf-8"))
        tray = data["tray"]
        w, h = tray["image_size"]
        return tuple((float(tray["corners_px"][k][0]) / w, float(tray["corners_px"][k][1]) / h)
                     for k in ("TL", "TR", "BR", "BL"))
    except Exception:
        return TRAY_DOT_REF_FALLBACK


TRAY_DOT_REF = _load_tray_dot_ref()

# 识别参数（已用 dataset/yolo_cube/labels/raw 的 4 张标注图做过量化筛选）
DEFAULTS = dict(
    scale=0.5,
    sigma=40.0,            # 托盘曲面估计的高斯尺度（半分辨率像素）
    chroma_min=11.0,       # 色度高于此值视为物体
    dark_min=26.0,         # 比背景暗超过此值视为物体（白色物块靠这一条）
    bright_min=999.0,      # 比背景亮超过此值视为物体；默认关闭（见 4.1 实测说明）
    peak_sigma=9.0,        # 距离变换平滑，抑制过度分裂
    peak_win=51,           # 距离变换峰值窗口
    min_radius=16.0,       # 种子的最小内切半径
    min_area_ratio=0.0030,
    max_area_ratio=0.08,   # 单块最大面积占比：特写图（物块 2~3 倍大）需要放宽到 0.08
    min_core=16.0,
    min_mean_chroma=8.0,   # 整块色度太低 → 判为托盘/台面残留
    drop_border=True,      # 贴着画面边缘的大块通常是台面
    tray_l_min=150.0,      # 判断托盘区域的亮度下限
    max_elong=3.5,         # 掩膜长宽比上限：超过就不是"一块物块"（本数据集真块 ≤2.1）
    min_extent=0.25,       # 掩膜填充自己外接框的比例下限：低于此值是物体边缘的细条
    tray_chroma_max=25.0,  # 判断托盘区域的色度上限
    tray_margin=0.03,      # 托盘内缩比例，避开边沿
    tray_min_ratio=0.05,   # 托盘少于画面该比例时不做 ROI 限制
    roi_min_overlap=0.8,   # 物块落在托盘内的最小面积比例
    tray_ref_l_min=165.0,  # 判定"白托盘"用的亮度门槛（比托盘表面更严）
    tray_ref_tol=12.0,     # 物块周围亮度低于托盘参考多少就判为不在托盘上
    dot_ref_pixels=2592 * 1944,  # 上面那些面积/尺寸阈值是按这个分辨率标定的
    dot_match_tol=0.10,    # 候选点归一化坐标与基准角点的最大距离，超过就不认
    # 托盘定位：优先用四角的黑点标定，找不到时退回上面那套"最亮低色度连通块"启发式
    tray_use_dots=True,    # 用托盘四角黑点定位工作区（比启发式准得多）
    dot_l_max=90.0,        # 黑点的亮度上限
    dot_area_min=250,      # 黑点面积下限（像素，原图分辨率）
    dot_area_max=4000,     # 黑点面积上限
    dot_size_max=150,      # 黑点外接边长上限
    dot_ring_in=6.0,       # "四周是亮托盘"判据：环带内半径（相对黑点半径的偏移）
    dot_ring_out=40.0,     # 环带外半径偏移
    dot_ring_min=140.0,    # 环带平均亮度下限，用来排掉台面和深色物块上的暗斑
    dot_ring_bright_frac=0.5,  # 环带里"亮托盘"像素的占比下限（比平均亮度更抗托盘外暗背景）
    dot_min_count=3,       # 至少匹配到 3 个角点才认这个四边形（2 点的相似变换保角，验不出来）
    dot_ring_band=25.0,    # 只保留"环带亮度距最亮者不超过这个差值"的候选
    dot_roi_expand=0.02,   # 黑点四边形向外扩张比例（黑点就在平底四角，只需一点容差；2026-10-26 由 0.08 改小）
    tray_quad_angle_tol=15.0,  # 托盘四边形内角允许偏离 90° 多少；超过就不认，退回启发式
    # 抓取排序：净空要求与形状确定性门槛
    clearance_ratio=0.35,   # 需要的净空 = 该比例 × 物体等效半径
    min_clearance_px=0.0,   # 净空绝对下限（半分辨率像素），0 表示只按比例
    max_ambiguity=0.35,     # 超过这个歧义度就不建议抓（形状不确定）
    weight_clearance=0.55,  # 排序权重：净空
    weight_confidence=0.30,  # 排序权重：分类置信度
    # 形状判定：掩膜质量门槛（低于门槛直接拒判，不硬分类）
    weight_regular=0.15,    # 排序权重：形状规整度（1-歧义度）
    min_mask_solidity=0.80,  # 掩膜实心度（area/凸包面积）低于此值 → unknown
    min_mask_convexity=0.75,  # 掩膜凸性（凸包周长/周长）低于此值 → unknown
    cyl_elong=1.15,          # 圆柱最小长宽比（俯视正圆盘不算圆柱，见 disc_ambig）
    dup_overlap=0.50,        # 两个候选掩膜重叠超过较小者的这个比例 → 判为同一物体的重复检出
    # 形状判定：几何阈值（全部由 dataset/yolo_cube 已标注实例标定，小样本）
    cuboid_fill=0.96,        # 长方体要求凸包几乎填满外接矩形（真矩形≈1.0；躺倒棱柱≈0.92~0.95）
    cuboid_elong=1.20,       # 长方体最小长宽比
    rect_ambig_fill=0.88,    # 矩形轮廓但填充率落在 [此值, cuboid_fill) → 形状歧义，拒判
    rect_ambig_max_v=4,      # 上面的歧义判据只对"四边形凸包"生效
    tri_ambig_band=0.20,     # 三角轮廓且 |径向亮度相关| 小于此值 → 四面体/三棱柱不可分，拒判
    round_radial=-0.60,      # 径向亮度相关低于此值 → 圆形家族（球/圆锥）
    ball_radial=-0.65,       # 圆形家族里低于此值判球，否则判圆锥
    poly_circle_min_v=7,     # 圆形家族补充判据：轮廓近似顶点数下限（圆≈8，六边形≈6）
    poly_circle_radial=-0.40,  # 上面的判据要求亮度也随半径下降
    # 俯视正圆盘：圆锥与圆柱都呈圆形且亮度平坦，判据不可靠 → 拒判
    disc_ambig=True,
    disc_min_circ=0.80,      # 圆度下限
    disc_radial_lo=-0.20,    # 径向亮度相关落在这个区间内才算"分不清锥/柱"
    prism_ambig_max_v=5,     # 凸包顶点数上限（6 边形就是六棱柱，不算歧义）
    poly_min_circularity=0.50,  # 规则多边形分支的圆度下限：太低说明是粘连的"哑铃形"块
    disc_radial_hi=0.25,
    # 立方体 / 五、六棱柱：轮廓都是 5~6 边形且填充率重叠 → 拒判
    prism_ambig=True,
    prism_ambig_min_v=5,     # 凸包顶点数下限
    prism_ambig_fill_lo=0.85,  # 凸包填充率区间
    prism_ambig_fill_hi=0.97,
    prism_ambig_max_elong=1.25,  # 只对"矮胖"的生效；细长棱柱靠长宽比能分开
    merge_colour_tol=22.0,   # 相邻碎块平均 Lab 距离小于此值才算"同色"
    merge_edge_max=20.0,     # 接触处平均梯度小于此值才算"没有接缝"，两条都满足才合并
    merge_adjacency_gap=2,   # 跨最多几像素判"相邻"（分水岭边界线不属于任何一块）
    # 形状级拆分：大块内部若有可见暗缝，就沿暗缝切开（拆开同色贴块）
    seam_split_area_ratio=1.6,   # 块面积超过中位数的这个倍数才尝试拆分
    seam_grad_percentile=97.0,   # 块内部梯度取这个分位以上当作"暗缝像素"
    seam_min_pixels=40,          # 暗缝像素太少就不拆
    seam_min_part_ratio=0.28,    # 切开后每一半至少占原块面积的比例
    seam_min_part_solidity=0.85,  # 切开后每一半的凸包实心度下限
    seam_erode_kernel=13,        # 先腐蚀掉边缘多少像素，只看块内部
    seam_min_line_ratio=0.35,    # 暗缝直线段长度下限（相对 sqrt(块面积)）
    seam_hough_threshold=25,     # Hough 直线检测阈值
    seam_max_line_gap=10,        # Hough 允许的线段间断长度
    seam_split_passes=2,         # 一个块可能藏 3 个物块，最多迭代拆几次
    seam_profile_reach=12,       # 沿切割线法线取样半径（像素）
    seam_profile_samples=40,     # 沿切割线取多少个采样点
    seam_min_dip=8.0,            # "暗缝"要求：中间比两侧暗多少灰度
    seam_max_step=25.0,          # "暗缝"要求：两侧亮度差不超过多少（超过就是面边界）
    seam_profile_min_samples=10,  # 有效采样点少于此值就放弃
    seam_profile_ratio=0.6,      # 满足"暗缝剖面"的采样点比例下限
    # 内部棱线（区分斜视角立方体/四面体 与 直立棱柱）
    edge_grad_percentile=96.0,   # 轮廓内部梯度取这个分位以上当作棱线像素
    edge_hough_threshold=20,     # Hough 直线检测阈值
    edge_min_line_ratio=0.30,    # 棱线长度下限（相对 sqrt(面积)）
    edge_max_line_gap=8,         # 允许的线段间断长度
    edge_angle_tol=12.0,         # 归并同一棱线的角度容差（度）
    edge_rho_tol=12.0,           # 归并同一棱线的距离容差（像素）
    # 注：曾尝试"顶面边数"（Otsu 分顶面/侧面 → 顶面多边形顶点数）区分立方体与
    # 五/六棱柱，实测无效（Otsu 分出的是高光斑而非几何顶面，球/圆锥也有 0.5~0.7
    # 的"顶面"占比），已从分类规则中移除；top_face_shape 仅保留为诊断特征。
)

@dataclass
class ShapeObject:
    bbox: tuple[int, int, int, int]
    area: int
    label: str
    confidence: float
    features: dict[str, float]
    mask: Any = None
    center: tuple[float, float] = (0.0, 0.0)
    # 抓取排序相关（由 rank_picks 填充）
    clearance_px: float = 0.0
    clearance_ratio: float = 0.0
    ambiguity: float = 0.0
    pickable: bool = False
    score: float = 0.0
    rank: int = 0
    reasons: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# 背景建模
# --------------------------------------------------------------------------- #
def lab_channels(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[:, :, 0], lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0

def tray_surface(
    L: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    sigma: float = 40.0,
    l_min: float = 110.0,
    chroma_max: float = 16.0,
) -> tuple[np.ndarray, np.ndarray]:
    """用托盘色像素做归一化卷积，估计平滑背景面（抵消光照渐变）。"""
    chroma = np.hypot(a, b)
    valid = ((L > l_min) & (chroma < chroma_max)).astype(np.float32)
    weight = cv2.GaussianBlur(valid, (0, 0), sigma)
    denom = np.maximum(weight, 1e-3)
    surfaces = [cv2.GaussianBlur(ch * valid, (0, 0), sigma) / denom for ch in (L, a, b)]
    return np.stack(surfaces, -1), weight


def objectness(lab: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """物体候选：色度偏离托盘、比托盘暗、或比托盘亮。

    第三条是补上的：白色物块如果比托盘还亮（过曝的白块、白圆柱），
    既不是"有色"也不是"更暗"，只靠前两条会整块漏掉。
    """
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    chroma = np.hypot(a, b)
    predicted, _ = tray_surface(L, a, b, cfg["sigma"])
    darker = predicted[:, :, 0] - L
    brighter = L - predicted[:, :, 0]
    raw = ((chroma > cfg["chroma_min"])
           | (darker > cfg["dark_min"])
           | (brighter > cfg["bright_min"])).astype(np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return raw, chroma, darker


def colour_gradient(lab: np.ndarray, blur_sigma: float = 2.0) -> np.ndarray:
    blur = cv2.GaussianBlur(lab, (0, 0), blur_sigma)
    grad = np.zeros(lab.shape[:2], np.float32)
    for c in range(3):
        gx = cv2.Sobel(blur[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
        grad = np.maximum(grad, cv2.magnitude(gx, gy))
    return grad


def _piece_labels(pieces: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """把若干掩膜合成一张标签图（0 表示背景）。"""
    labels = np.zeros(shape, np.int32)
    for index, mask in enumerate(pieces, start=1):
        labels[mask > 0] = index
    return labels


def _adjacency_stats(labels: np.ndarray, grad: np.ndarray,
                     gap: int = 1) -> dict[tuple[int, int], tuple[int, float]]:
    """统计相邻标签对：接触像素数 + 接触处平均梯度。

    接触处梯度大 = 两块之间有可见接缝（阴影或遮挡边）→ 是两个物体，不该合并；
    梯度小 = 同一个物体的不同面被切开 → 应该合并回去。

    `gap` 允许跨过最多 gap 个像素判相邻。**这是必须的**：分水岭会在两块之间留下
    一条 1~2 像素的边界线，那条线上的像素不属于任何一块，于是"严格相邻"永远为 0，
    合并判据就成了死代码（实测单张图上相邻对数量为 0）。
    """
    counts: dict[int, int] = {}
    sums: dict[int, float] = {}
    h, w = labels.shape
    spans = []
    for dy, dx in ((0, gap), (gap, 0), (gap, gap), (gap, -gap)):
        if dy == 0:
            spans.append((labels[:, :w - dx], labels[:, dx:], grad[:, :w - dx]))
        elif dx == 0:
            spans.append((labels[:h - dy, :], labels[dy:, :], grad[:h - dy, :]))
        elif dx > 0:
            spans.append((labels[:h - dy, :w - dx], labels[dy:, dx:], grad[:h - dy, :w - dx]))
        else:
            spans.append((labels[:h - dy, -dx:], labels[dy:, :dx], grad[:h - dy, -dx:]))
    for a, b, g in spans:
        x, y = a.ravel(), b.ravel()
        keep = (x > 0) & (y > 0) & (x != y)
        if not np.any(keep):
            continue
        lo = np.minimum(x[keep], y[keep]).astype(np.int64)
        hi = np.maximum(x[keep], y[keep]).astype(np.int64)
        key = lo * 100000 + hi
        values = g.ravel()[keep].astype(np.float64)
        unique, inverse = np.unique(key, return_inverse=True)
        pair_counts = np.bincount(inverse, minlength=len(unique))
        pair_sums = np.bincount(inverse, weights=values, minlength=len(unique))
        for k, c, s in zip(unique, pair_counts, pair_sums):
            counts[int(k)] = counts.get(int(k), 0) + int(c)
            sums[int(k)] = sums.get(int(k), 0.0) + float(s)
    return {(int(k // 100000), int(k % 100000)): (c, sums[k] / max(c, 1))
            for k, c in counts.items()}


def merge_pieces_by_colour(pieces: list[np.ndarray], lab: np.ndarray, grad: np.ndarray,
                           cfg: dict[str, Any]) -> list[np.ndarray]:
    """把"同色且接缝不明显"的相邻碎块合并回同一个物体。

    两个条件同时满足才合并：
    1. 平均 Lab 距离 ≤ `merge_colour_tol`（颜色接近）；
    2. 接触处平均梯度 ≤ `merge_edge_max`（没有可见接缝）。
    第 2 条是关键：两个同色物块贴在一起时颜色几乎一样，但它们之间有一条
    阴影/遮挡形成的暗缝，梯度明显更高；而同一个物块被切开的两块之间没有缝。
    """
    if len(pieces) <= 1:
        return pieces
    labels = _piece_labels(pieces, lab.shape[:2])
    means: dict[int, np.ndarray] = {}
    for index in range(1, len(pieces) + 1):
        region = labels == index
        means[index] = np.array([float(lab[:, :, c][region].mean()) for c in range(3)])

    parent = {i: i for i in range(1, len(pieces) + 1)}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), (_, edge_grad) in _adjacency_stats(labels, grad, cfg["merge_adjacency_gap"]).items():
        colour_gap = float(np.linalg.norm(means[a] - means[b]))
        if colour_gap <= cfg["merge_colour_tol"] and edge_grad <= cfg["merge_edge_max"]:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

    groups: dict[int, list[int]] = {}
    for index in range(1, len(pieces) + 1):
        groups.setdefault(find(index), []).append(index)
    merged = [np.isin(labels, members).astype(np.uint8) for members in groups.values()]
    return merged


def _piece_solidity(mask: np.ndarray) -> float:
    """块的凸包实心度：越接近 1 越像"一个完整物体"。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
    return area / hull_area


def _seam_profile_ratio(piece: np.ndarray, lab: np.ndarray, centre: np.ndarray,
                        direction: np.ndarray, normal: np.ndarray, length: float,
                        cfg: dict[str, Any]) -> float:
    """沿候选切割线抽样，判断它是"暗缝"还是"面与面之间的亮度台阶"。

    这是本方法最关键的一步。单个物块斜视角下，顶面和侧面之间也有一条很强的
    内部边缘，直接用梯度找线会把它误当成两块的分界，结果把好端端一个物块切开。
    两者的区别在亮度剖面上：
    - **暗缝**：中间一条细暗线，两侧都亮、而且两侧亮度接近（对称的下凹）；
    - **面边界**：一边亮一边暗，中间没有回弹（单调台阶）。
    返回满足"下凹且两侧接近"的采样点比例。
    """
    L = lab[:, :, 0]
    rows, cols = piece.shape
    reach = int(cfg["seam_profile_reach"])
    half = max(1.0, length / 2.0)     # 只在线段自身长度内取样
    good = total = 0
    for t in np.linspace(-half, half, int(cfg["seam_profile_samples"])):
        px = centre[0] + direction[0] * t
        py = centre[1] + direction[1] * t
        if not (0 <= px < cols and 0 <= py < rows) or piece[int(py), int(px)] == 0:
            continue
        left: list[float] = []
        right: list[float] = []
        middle: list[float] = []
        for s in range(-reach, reach + 1):
            x = int(round(px + normal[0] * s))
            y = int(round(py + normal[1] * s))
            if not (0 <= x < cols and 0 <= y < rows) or piece[y, x] == 0:
                continue
            value = float(L[y, x])
            if s <= -max(2, reach // 3):
                left.append(value)
            elif s >= max(2, reach // 3):
                right.append(value)
            elif abs(s) <= 2:
                middle.append(value)
        if not (left and right and middle):
            continue
        total += 1
        left_mean = float(np.mean(left))
        right_mean = float(np.mean(right))
        dip = min(left_mean, right_mean) - float(np.min(middle))
        step = abs(left_mean - right_mean)
        if dip >= cfg["seam_min_dip"] and step <= cfg["seam_max_step"]:
            good += 1
    if total < cfg["seam_profile_min_samples"]:
        return 0.0
    return good / total


def split_piece_along_seam(piece: np.ndarray, grad: np.ndarray, lab: np.ndarray,
                           cfg: dict[str, Any], total: int) -> list[np.ndarray]:
    """把一个大块沿内部可见暗缝切成两块；判定不通过就原样返回。

    原理：两个贴在一起的物块之间有一条"暗缝"（阴影或遮挡形成的暗线），
    它在梯度图上比物块内部强。步骤：
    1. 先把块边界腐蚀掉，排除外轮廓的强梯度，只看块内部；
    2. 取内部梯度较高的一批像素当"暗缝像素"；
    3. 用 Hough 变换找其中**最长的一条直线段**——真正的分界是一条横贯的线，
       而纹理/阴影斑点是零散的，Hough 加长度门槛能把它们过滤掉；
    4. 沿这条线的方向切开，并校验两半都够大、够"实"，否则放弃。
    """
    area = int(piece.sum())
    # 每一半至少要：≥ 最小物体面积，且 ≥ 原块的这个比例
    min_part = max(cfg["min_area_ratio"] * total, cfg["seam_min_part_ratio"] * area)
    if area < 2 * min_part:
        return [piece]
    erode_k = int(cfg["seam_erode_kernel"])
    inner = cv2.erode(piece, np.ones((erode_k, erode_k), np.uint8))
    if int(inner.sum()) < 200:
        return [piece]

    values = grad[inner > 0]
    threshold = float(np.percentile(values, cfg["seam_grad_percentile"]))
    seam = ((grad >= threshold) & (inner > 0)).astype(np.uint8) * 255
    if int(np.count_nonzero(seam)) < cfg["seam_min_pixels"]:
        return [piece]

    min_length = cfg["seam_min_line_ratio"] * float(np.sqrt(area))
    segments = cv2.HoughLinesP(seam, 1, np.pi / 180, threshold=cfg["seam_hough_threshold"],
                               minLineLength=min_length, maxLineGap=cfg["seam_max_line_gap"])
    if segments is None or len(segments) == 0:
        return [piece]

    best = max(segments, key=lambda s: float(np.hypot(s[0][2] - s[0][0], s[0][3] - s[0][1])))
    x1, y1, x2, y2 = (float(v) for v in best[0])
    direction = np.array([x2 - x1, y2 - y1], np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return [piece]
    direction /= norm
    normal = np.array([-direction[1], direction[0]], np.float32)
    centre = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], np.float32)

    rows, cols = piece.shape
    grid_y, grid_x = np.mgrid[0:rows, 0:cols]
    field = (grid_x - centre[0]) * normal[0] + (grid_y - centre[1]) * normal[1]
    part_a = ((piece > 0) & (field <= 0)).astype(np.uint8)
    part_b = ((piece > 0) & (field > 0)).astype(np.uint8)

    if min(int(part_a.sum()), int(part_b.sum())) < min_part:
        return [piece]
    if min(_piece_solidity(part_a), _piece_solidity(part_b)) < cfg["seam_min_part_solidity"]:
        return [piece]
    # 最关键的一关：这条线必须是"暗缝"（两侧都亮且接近），不能是物块自身的面边界
    if _seam_profile_ratio(piece, lab, centre, direction, normal, norm, cfg) < cfg["seam_profile_ratio"]:
        return [piece]
    return [part_a, part_b]


def split_pieces_along_seams(pieces: list[np.ndarray], grad: np.ndarray, lab: np.ndarray,
                             cfg: dict[str, Any], total: int) -> list[np.ndarray]:
    """对明显偏大的块反复尝试沿暗缝切开（一个块可能藏了 3 个物块）。"""
    if not pieces:
        return pieces
    for _ in range(int(cfg["seam_split_passes"])):
        areas = [int(p.sum()) for p in pieces]
        reference = float(np.median(areas)) if areas else 0.0
        changed = False
        out: list[np.ndarray] = []
        for piece in pieces:
            if int(piece.sum()) < cfg["seam_split_area_ratio"] * reference:
                out.append(piece)
                continue
            parts = split_piece_along_seam(piece, grad, lab, cfg, total)
            out.extend(parts)
            changed = changed or len(parts) > 1
        pieces = out
        if not changed:
            break
    return pieces


def split_components(raw: np.ndarray, grad: np.ndarray, lab: np.ndarray,
                     cfg: dict[str, Any]) -> list[np.ndarray]:
    """先用距离变换峰值做过分割，再按颜色把同色碎块合并回一个物体。
    背景：只做全局分水岭时，两个大小相近、贴在一起的物块常常只出一个种子，
    结果被当成一个块（漏检）；把种子调密会过分割，所以补一步"颜色相近就合并"。
    颜色只用于判断两块是不是同一个物体，不参与形状判定。
    """
    dist = cv2.distanceTransform(raw, cv2.DIST_L2, 5)
    smooth = cv2.GaussianBlur(dist, (0, 0), cfg["peak_sigma"])
    win = int(cfg["peak_win"])
    peaks = cv2.dilate(smooth, np.ones((win, win), np.uint8))
    seeds = ((smooth >= peaks - 1e-3) & (dist > cfg["min_radius"])).astype(np.uint8)
    count, markers = cv2.connectedComponents(seeds)
    markers = np.zeros(raw.shape, np.int32) if count <= 1 else markers.astype(np.int32)
    markers[raw == 0] = 0
    grad_img = cv2.cvtColor(np.clip(grad * 2, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(grad_img, markers)
    pieces: list[np.ndarray] = []
    for index in range(1, int(markers.max()) + 1):
        region = ((markers == index) & (raw > 0)).astype(np.uint8)
        if region.any():
            pieces.append(region)
    return merge_pieces_by_colour(pieces, lab, grad, cfg)




def _ring_stats(L: np.ndarray, x: float, y: float, r_in: float,
                r_out: float, l_min: float) -> tuple[float, float]:
    """以 (x, y) 为中心、半径 [r_in, r_out] 环带内的 (平均亮度, 亮像素占比)。

    只取平均亮度会误杀贴着托盘边缘的角点：那种点的环带里混进托盘外的暗背景，
    平均值被拉低；但它的环带大部分仍是亮托盘。所以判据要用"亮像素占比"。
    """
    h, w = L.shape
    x0, x1 = max(0, int(x - r_out)), min(w, int(x + r_out) + 1)
    y0, y1 = max(0, int(y - r_out)), min(h, int(y + r_out) + 1)
    patch = L[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.hypot(xx - x, yy - y)
    sel = (dist >= r_in) & (dist <= r_out)
    if not sel.any():
        return 0.0, 0.0
    vals = patch[sel]
    return float(vals.mean()), float((vals >= l_min).mean())


def find_dot_candidates(img: np.ndarray,
                        cfg: dict[str, Any]) -> list[tuple[float, float, float]]:
    """找出"像托盘角点"的暗斑，返回 [(x, y, 环带平均亮度)]，按平均亮度降序。

    两条判据缺一不可：
    1. 自身够暗、面积和外接尺寸像一个小圆点；
    2. **四周一圈大部分是亮的白托盘**（`dot_ring_bright_frac`）——这一条把灰色台面上的
       暗斑、深色物块上的暗斑和画面黑边都排掉，是本方法能稳定工作的关键。

    第 2 条用"亮像素占比"而不是平均亮度：紧贴托盘边缘的角点，环带里必然混进托盘外的
    暗背景，平均值会被拉低到门槛以下（实测侧视图里真角点均值 124、亮占比 0.63），
    用占比就不会误杀。

    面积/尺寸阈值按 `dot_ref_pixels` 标定，这里按当前图面积自动换算，与分辨率无关。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    dark = (L < cfg["dot_l_max"]).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    out: list[tuple[float, float, float]] = []
    scale = float(np.sqrt(L.size / max(float(cfg["dot_ref_pixels"]), 1.0)))
    a_min = cfg["dot_area_min"] * scale * scale
    a_max = cfg["dot_area_max"] * scale * scale
    s_max = cfg["dot_size_max"] * scale
    r_in_off = cfg["dot_ring_in"] * scale
    r_out_off = cfg["dot_ring_out"] * scale
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        bw = int(stats[index, cv2.CC_STAT_WIDTH])
        bh = int(stats[index, cv2.CC_STAT_HEIGHT])
        if not (a_min <= area <= a_max) or bw > s_max or bh > s_max:
            continue
        if not (0.35 <= bw / max(bh, 1) <= 2.8):
            continue
        x, y = float(centroids[index][0]), float(centroids[index][1])
        radius = max(bw, bh) * 0.5
        mean_ring, bright_frac = _ring_stats(L, x, y, radius + r_in_off,
                                             radius + r_out_off, cfg["dot_ring_min"])
        if bright_frac < cfg["dot_ring_bright_frac"]:
            continue
        out.append((x, y, mean_ring))
    out.sort(key=lambda item: -item[2])
    return out


def _quad_corner_angles(quad: np.ndarray) -> list[float]:
    """四边形四个内角（度），用于判断它像不像矩形。"""
    pts = np.asarray(quad, np.float64)
    angles = []
    for i in range(4):
        a, b, c = pts[(i - 1) % 4], pts[i], pts[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        angles.append(float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))))
    return angles


def find_tray_quad(img: np.ndarray, cfg: dict[str, Any] | None = None) -> np.ndarray | None:
    """用托盘四角的黑点定出工作区四边形，顺序为 左上→右上→右下→左下。

    做法：把检测到的角点按归一化坐标匹配到 `TRAY_DOT_REF` 基准布局，
    再用（4 点）单应 / （3 点）仿射把基准四边形变换到当前画面。
    这样即使某个角点被物块挡住，也能推出完整托盘范围；
    而且因为始终经过基准，不会产生跑到画面外的退化四边形。

    黑点太少或对不上基准时返回 None，调用方退回启发式 ROI。
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    h, w = img.shape[:2]
    cands = find_dot_candidates(img, cfg)
    if not cands:
        return None
    top = cands[0][2]
    strong = [c for c in cands if c[2] >= top - cfg["dot_ring_band"]]

    # 匹配到基准角点：每个角点只保留环带最亮的那个候选
    matched: dict[int, tuple[float, float]] = {}
    for x, y, _ in strong:
        nx, ny = x / w, y / h
        best_i, best_d = -1, cfg["dot_match_tol"]
        for i, (rx, ry) in enumerate(TRAY_DOT_REF):
            d = float(np.hypot(nx - rx, ny - ry))
            if d < best_d:
                best_i, best_d = i, d
        if best_i >= 0 and best_i not in matched:
            matched[best_i] = (x, y)
    if len(matched) < cfg["dot_min_count"]:
        return None

    ref = np.array(TRAY_DOT_REF, np.float32) * np.array([w, h], np.float32)
    idx = sorted(matched)
    src = np.array([TRAY_DOT_REF[i] for i in idx], np.float32) * np.array([w, h], np.float32)
    dst = np.array([matched[i] for i in idx], np.float32)
    quad = _tray_quad_from_matches(src, dst, ref)
    if quad is None:
        return None

    # 安全闸：只有接近矩形的四边形才可信。托盘四角黑点这套基准只对"近似垂直俯视"
    # 成立；侧视相机虽然也看得到黑点，但投影位置完全不同，套用基准会算出歪斜的四边形。
    # 用四个内角做判据，不接近 90° 就返回 None，让调用方退回启发式。
    angles = _quad_corner_angles(quad)
    if any(abs(a - 90.0) > cfg["tray_quad_angle_tol"] for a in angles):
        return None

    # 向外扩张，覆盖到托盘可用地面（黑点画在可用地面之内）
    centre = quad.mean(axis=0)
    quad = centre + (quad - centre) * (1.0 + cfg["dot_roi_expand"])
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    return quad.astype(np.float32)


def _tray_quad_from_matches(src: np.ndarray, dst: np.ndarray,
                            ref: np.ndarray) -> np.ndarray | None:
    """由匹配上的角点对推出完整托盘四边形（基准布局 → 当前画面）。

    4 点用单应、3 点用仿射、2 点用相似（旋转+等比缩放+平移）、1 点用平移。
    角点被物块挡住时靠降级变换兜住；点数不足时返回 None 由调用方退回启发式。
    """
    n = len(src)
    if n >= 4:
        matrix = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        return cv2.perspectiveTransform(ref.reshape(1, -1, 2), matrix).reshape(-1, 2)
    if n == 3:
        matrix = cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))
        return cv2.transform(ref.reshape(1, -1, 2), matrix).reshape(-1, 2)
    if n == 2:
        delta_src = src[1] - src[0]
        delta_dst = dst[1] - dst[0]
        norm_src = float(np.hypot(*delta_src))
        if norm_src < 1e-6:
            return None
        ratio = float(np.hypot(*delta_dst)) / norm_src
        theta = (np.arctan2(delta_dst[1], delta_dst[0])
                 - np.arctan2(delta_src[1], delta_src[0]))
        cos_t, sin_t = np.cos(theta) * ratio, np.sin(theta) * ratio
        centre_src = src.mean(axis=0)
        centre_dst = dst.mean(axis=0)
        matrix = np.array([
            [cos_t, -sin_t, centre_dst[0] - cos_t * centre_src[0] + sin_t * centre_src[1]],
            [sin_t, cos_t, centre_dst[1] - sin_t * centre_src[0] - cos_t * centre_src[1]],
        ], np.float32)
        return cv2.transform(ref.reshape(1, -1, 2), matrix).reshape(-1, 2)
    if n == 1:
        shift = dst[0] - src[0]
        matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], np.float32)
        return cv2.transform(ref.reshape(1, -1, 2), matrix).reshape(-1, 2)
    return None


def tray_quad_mask(quad: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """把托盘四边形变成 0/1 掩膜。"""
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 1)
    return mask


def find_tray_roi(small: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray | None, tuple[float, ...] | None]:
    """找出白色托盘的工作区（内缩后的四边形）。

    约定：只有托盘内部才是工作区，托盘边沿、左侧台面、画面黑边都属于干扰。
    找不到足够大的托盘时返回 (None, None)，此时退化为全图处理。
    """
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
    bright = ((L > cfg["tray_l_min"]) & (chroma < cfg["tray_chroma_max"])).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < cfg["tray_min_ratio"] * L.size:
        return None, None
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    shrink = max(0.1, 1.0 - 2.0 * cfg["tray_margin"])
    box = cv2.boxPoints(((cx, cy), (rw * shrink, rh * shrink), angle)).astype(np.int32)
    roi = np.zeros(L.shape, np.uint8)
    cv2.fillConvexPoly(roi, box, 1)
    return roi, (float(cx), float(cy), float(rw * shrink), float(rh * shrink), float(angle))


def tray_roi_mask(small: np.ndarray, cfg: dict[str, Any]) -> np.ndarray | None:
    """工作区掩膜：优先用四角黑点定出的托盘四边形，失败时退回启发式 ROI。

    启发式（找最亮的低色度大连通块）在部分图里会把左侧灰色台面和画面黑边
    一起算进工作区，导致台面被当成物块；黑点法直接由托盘自身的四个标记定界，
    没有这个问题，所以默认优先。
    """
    if cfg["tray_use_dots"]:
        quad = find_tray_quad(small, cfg)
        if quad is not None:
            return tray_quad_mask(quad, small.shape[:2])
    roi, _ = find_tray_roi(small, cfg)
    return roi


def segment_objects(small: np.ndarray, cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """在降采样图上找出物块实例，返回掩膜和基本量。"""
    cfg = {**DEFAULTS, **(cfg or {})}
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = np.dstack([lab[:, :, 0], lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0])
    raw, chroma, _ = objectness(lab, cfg)
    roi = tray_roi_mask(small, cfg)
    if roi is not None:
        raw = (raw & roi).astype(np.uint8)
    grad = colour_gradient(lab)

    h, w = raw.shape
    total = raw.size
    objects: list[dict[str, Any]] = []
    pieces = split_components(raw, grad, lab, cfg)
    # 形状级补刀：分水岭拆不开的粘连大块，再沿内部可见暗缝切一次
    pieces = split_pieces_along_seams(pieces, grad, lab, cfg, raw.size)
    for piece in pieces:
        # 一个 basin 里可能混进别的物体的边缘像素，先按连通块拆开再逐个过滤
        count, comps, stats, _ = cv2.connectedComponentsWithStats(piece, connectivity=8)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < cfg["min_area_ratio"] * total or area > cfg["max_area_ratio"] * total:
                continue
            blob = (comps == label).astype(np.uint8)
            core = float(cv2.distanceTransform(blob, cv2.DIST_L2, 5).max())
            if core < cfg["min_core"]:
                continue
            mask = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            ys, xs = np.nonzero(mask)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            if cfg["drop_border"] and (bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= w - 1 or bbox[3] >= h - 1):
                continue
            mean_chroma = float(chroma[mask > 0].mean())
            # 整块色度极低的基本是托盘/台面残留。
            # 注：实测过"放宽这一条以找回比托盘更亮的白块"，代价是引入大量
            # 高光杂块（精确率 0.53→0.43）却一个白块也没多匹配上，故保持严格；
            # 需要时可用 bright_min 打开"比托盘亮"这条候选门限。
            if mean_chroma < cfg["min_mean_chroma"]:
                continue
            if roi is not None and roi[mask > 0].mean() < cfg["roi_min_overlap"]:
                continue
            objects.append(dict(mask=mask, area=int(mask.sum()), core=core, bbox=bbox,
                                chroma=mean_chroma, center=(float(xs.mean()), float(ys.mean()))))
    return suppress_duplicates(objects, cfg["dup_overlap"])


def suppress_duplicates(candidates: list[dict[str, Any]], overlap: float = 0.5) -> list[dict[str, Any]]:
    """同一个物体被分水岭拆成多块时，只保留面积最大的那一块。"""
    kept: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda c: c["area"], reverse=True):
        mask = item["mask"]
        duplicate = False
        for other in kept:
            other_mask = other["mask"]
            intersection = int(np.count_nonzero(mask & other_mask))
            if not intersection:
                continue
            smaller = min(int(mask.sum()), int(other_mask.sum()))
            if intersection / max(smaller, 1) >= overlap:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


# --------------------------------------------------------------------------- #
# 形状描述
# --------------------------------------------------------------------------- #
def internal_edge_lines(mask: np.ndarray, lab: np.ndarray, cfg: dict[str, Any]) -> tuple[float, float]:
    """统计轮廓**内部**的长直线棱线：返回 (棱线条数, 最长棱线 ÷ 物体尺寸)。

    这是解决"立方体被误判成五棱柱"的关键判据：
    - 斜视角看立方体/四面体时，顶面与侧面之间的棱线落在轮廓内部；
    - 直立的五/六/三棱柱，顶面直接填满轮廓，内部没有棱线。
    两者轮廓的凸包顶点数都是 5～6，只有内部棱线能区分。
    """
    inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8))
    if int(inner.sum()) < 200:
        return 0.0, 0.0
    blur = cv2.GaussianBlur(lab[:, :, 0], (0, 0), 1.2)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    values = grad[inner > 0]
    if values.size < 100:
        return 0.0, 0.0
    threshold = float(np.percentile(values, cfg["edge_grad_percentile"]))
    edge = ((grad >= threshold) & (inner > 0)).astype(np.uint8) * 255
    size = float(np.sqrt(max(float(mask.sum()), 1.0)))
    min_length = cfg["edge_min_line_ratio"] * size
    segments = cv2.HoughLinesP(edge, 1, np.pi / 180, threshold=cfg["edge_hough_threshold"],
                               minLineLength=min_length, maxLineGap=cfg["edge_max_line_gap"])
    if segments is None or len(segments) == 0:
        return 0.0, 0.0

    # 方向相近、位置相近的线段归并成同一条棱线
    merged: list[dict[str, float]] = []
    for segment in segments:
        x1, y1, x2, y2 = (float(v) for v in segment[0])
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)
        mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        rho = mid_x * np.sin(np.radians(angle)) - mid_y * np.cos(np.radians(angle))
        for line in merged:
            delta_angle = abs(line["angle"] - angle)
            delta_angle = min(delta_angle, 180.0 - delta_angle)
            if delta_angle <= cfg["edge_angle_tol"] and abs(line["rho"] - rho) <= cfg["edge_rho_tol"]:
                count = line["n"]
                line["length"] += length
                line["angle"] = (line["angle"] * count + angle) / (count + 1)
                line["rho"] = (line["rho"] * count + rho) / (count + 1)
                line["n"] = count + 1
                break
        else:
            merged.append(dict(length=length, angle=angle, rho=rho, n=1.0))

    strong = [line for line in merged if line["length"] >= min_length]
    longest = max((line["length"] for line in strong), default=0.0)
    return float(len(strong)), longest / max(size, 1e-6)

def shape_features(mask: np.ndarray, lab: np.ndarray,
                   cfg: dict[str, Any] | None = None) -> dict[str, float] | None:
    """轮廓几何 + 内部明暗结构 + 内部棱线。全部与颜色名称无关。"""
    cfg = {**DEFAULTS, **(cfg or {})}
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < 200:
        return None
    peri = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    hull_peri = max(float(cv2.arcLength(hull, True)), 1e-6)
    (_, (rw, rh), _) = cv2.minAreaRect(contour)
    rect_area = max(rw * rh, 1.0)
    bw, bh = cv2.boundingRect(contour)[2:]
    # 掩膜必须是"整块物块"，不能是物体边缘的细条：
    # ① 长宽比过大 = 把白色板的立边、物体侧面细条当成了物体；
    # ② 填不满自己的外接框 = 一条斜着的细条，和旁边真正的块看起来像"重复检出"。
    # 这两种都不是靠"去重"能解决的，因为它们和真块的掩膜几乎不重叠。
    elong_rect = max(rw, rh) / max(min(rw, rh), 1e-6)
    extent = area / max(float(bw * bh), 1.0)
    if elong_rect > cfg["max_elong"] or extent < cfg["min_extent"]:
        return None
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    r_eq = float(np.sqrt(area / np.pi))
    pts = contour.reshape(-1, 2).astype(np.float32)
    rr = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    poly = cv2.approxPolyDP(hull, 0.02 * hull_peri, True).reshape(-1, 2)
    seg = np.linalg.norm(np.diff(np.vstack([poly, poly[:1]]), axis=0), axis=1)

    L = lab[:, :, 0]
    values = L[ys, xs]
    radius = np.hypot(xs - cx, ys - cy)
    radius = radius / max(float(radius.max()), 1e-6)
    if len(values) > 8 and float(values.std()) > 1e-6 and float(radius.std()) > 1e-6:
        radial_corr = float(np.corrcoef(radius, values)[0, 1])
    else:
        radial_corr = 0.0
    spec_excess = float(np.percentile(values, 99) - np.median(values))
    A = np.stack([xs, ys, np.ones_like(xs)], 1).astype(np.float32)
    coef, *_ = np.linalg.lstsq(A, values.astype(np.float32), rcond=None)
    plane_rmse = float(np.sqrt(((values - A @ coef) ** 2).mean()))
    edge_line_count, edge_longest_ratio = internal_edge_lines(mask, lab, cfg)
    face_vertices, face_ratio, face_circularity = top_face_shape(mask, lab)

    return dict(
        area=area, perimeter=peri, r_eq=r_eq,
        circularity=4 * np.pi * area / (peri * peri) if peri > 0 else 0.0,
        solidity=area / hull_area,
        convexity=hull_peri / max(peri, 1e-6),
        rect_fill=area / rect_area,
        # 凸包版本：掩膜缺角时凸包会把缺口补回来，比直接用掩膜面积稳得多
        hull_rect_fill=hull_area / rect_area,
        hull_extent=hull_area / max(bw * bh, 1),
        elong=float(max(rw, rh) / max(min(rw, rh), 1e-6)),
        circle_residual=float(np.sqrt(np.mean((rr - r_eq) ** 2)) / max(r_eq, 1e-6)),
        hull_vertices=float(len(poly)),
        poly_vertices=float(len(cv2.approxPolyDP(contour, 0.03 * peri, True))),
        linearity=float(seg.max() / max(seg.sum(), 1e-6)),
        radial_corr=radial_corr,
        spec_excess=spec_excess,
        plane_rmse=plane_rmse,
        edge_line_count=edge_line_count,
        edge_longest_ratio=edge_longest_ratio,
        top_face_vertices=face_vertices,
        top_face_ratio=face_ratio,
        top_face_circularity=face_circularity,
    )


def top_face_shape(mask: np.ndarray, lab: np.ndarray) -> tuple[float, float, float]:
    """把物块按亮度 Otsu 分成"亮区/暗区"，返回亮区的 (顶点数, 占轮廓比例, 圆度)。

    **实测结论：不能用来区分形状。** 原设想是"顶面边数 = 形状边数"（立方体 4、
    五棱柱 5、六棱柱 6、圆柱椭圆），但在本机单顶光下 Otsu 分出的是**高光斑**而不是
    几何顶面：五棱柱实测顶点数为 4/4/6，六棱柱为 4/4/4/4，而球/圆锥的"亮区"占比
    高达 0.49~0.72（它们根本没有顶面）。因此只作为诊断特征保留，不参与分类。
    """
    values = lab[:, :, 0][mask > 0]
    if values.size < 100:
        return 0.0, 0.0, 0.0
    threshold, _ = cv2.threshold(values.astype(np.uint8), 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    top = ((lab[:, :, 0] >= threshold) & (mask > 0)).astype(np.uint8)
    count, comps, stats, _ = cv2.connectedComponentsWithStats(top, connectivity=8)
    if count <= 1:
        return 0.0, 0.0, 0.0
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    face = (comps == biggest).astype(np.uint8)
    contours, _ = cv2.findContours(face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, 0.0, 0.0
    contour = max(contours, key=cv2.contourArea)
    face_area = float(cv2.contourArea(contour))
    mask_area = max(float(np.count_nonzero(mask)), 1.0)
    peri = float(cv2.arcLength(contour, True))
    circularity = 4 * np.pi * face_area / (peri * peri) if peri > 0 else 0.0
    vertices = float(len(cv2.approxPolyDP(contour, 0.03 * peri, True)))
    return vertices, face_area / mask_area, circularity


def classify_shape(f: dict[str, float], cfg: dict[str, Any] | None = None) -> tuple[str, float]:
    """按轮廓几何 + 内部明暗结构判定形状；返回 (类别, 置信度)。

    原则：
    1. 掩膜缺角会让填充率和顶点数同时失真，硬分类只会制造误检；
       实心度/凸性不足直接拒判（unknown）。
    2. 判断"像不像三角"用**凸包填充率**（凸包面积 ÷ 最小外接矩形面积），
       而不是掩膜填充率：凸包会把缺口补回来，三角 ≈0.5、四边形 ≈1.0 的差异才稳定。
    3. 单垂直视角下存在**原理上不可分**的形状对，宁可拒判也不硬分类：
       长方体 ↔ 躺倒的棱柱（轮廓都是矩形）、四面体 ↔ 三角棱柱（都是三角形）。

    规则来源：dataset/yolo_cube 已标注图片的 31 个匹配实例（见实验报告），
    阈值属于小样本标定，换物料或换相机后必须复核。
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    circ, solid, conv = f["circularity"], f["solidity"], f["convexity"]
    res, elong = f["circle_residual"], f["elong"]
    radial = f["radial_corr"]
    hull_fill = float(f.get("hull_rect_fill", f["rect_fill"]))
    hull_v = f["hull_vertices"]
    poly_v = float(f.get("poly_vertices", 0.0))

    # 掩膜缺角 → 拒判
    if solid < cfg["min_mask_solidity"] or conv < cfg["min_mask_convexity"]:
        return UNKNOWN, 0.0
    # 直立圆柱：圆形截面加侧壁，长宽比略大于 1、外接矩形填充高
    if elong >= cfg["cyl_elong"] and circ >= 0.83 and solid >= 0.97 and hull_fill >= 0.75:
        return "cylinder", 0.55

    # 圆形家族：球面靠"亮度随半径下降"识别；圆锥底面接近平面圆盘；
    # 规则多边形（正六边形）轮廓也很圆，所以要求凸包顶点足够多才算圆。
    round_like = (
        radial <= cfg["round_radial"]
        or (circ >= 0.82 and solid >= 0.97 and res <= 0.06 and hull_v >= 7)
    )
    if round_like:
        return ("ball", 0.75) if radial <= cfg["ball_radial"] else ("cone", 0.55)

    # 俯视正圆盘：圆锥（顶视有沿半径的梯度）与圆柱（平顶）从正上方看都是圆，
    # 亮度又都接近平坦，判据不可靠 → 拒判。直立圆柱已在上面被认走，不会走到这里。
    # 要求轮廓近似顶点数足够多（圆≈8；正六边形只有 5~6，圆度也能到 0.84，光看圆度分不开）。
    if (cfg["disc_ambig"] and circ >= cfg["disc_min_circ"] and solid >= 0.97
            and elong < cfg["cyl_elong"] and poly_v >= cfg["poly_circle_min_v"]
            and cfg["disc_radial_lo"] <= radial <= cfg["disc_radial_hi"]):
        return UNKNOWN, 0.0

    # 长方体：轮廓必须几乎填满外接矩形（真矩形 ≈1.0）。填充率 0.88~0.96 的
    # "矩形轮廓"更可能是**躺倒的棱柱**（斜切掉两角），单视角下与长方体不可分。
    if hull_fill >= cfg["cuboid_fill"] and elong >= cfg["cuboid_elong"]:
        return "cuboid", 0.65
    if (cfg["rect_ambig_fill"] <= hull_fill < cfg["cuboid_fill"]
            and elong >= cfg["cuboid_elong"] and hull_v <= cfg["rect_ambig_max_v"]):
        return UNKNOWN, 0.0

    # 立方体 / 五、六棱柱：单视角下轮廓都是 5~6 边形，填充率区间重叠。
    # 只有"矮胖"的（长宽比小）才落进这个重叠区；细长的棱柱靠长宽比就能分开。
    if (cfg["prism_ambig"] and cfg["prism_ambig_min_v"] <= hull_v <= cfg["prism_ambig_max_v"]
            and cfg["prism_ambig_fill_lo"] <= hull_fill < cfg["prism_ambig_fill_hi"]
            and elong < cfg["prism_ambig_max_elong"]):
        return UNKNOWN, 0.0

    # 三角轮廓：凸包填充率低才算三角；缺角的块已被上面的质量门槛拦掉
    if hull_fill < 0.70:
        # 四面体与三角棱柱从上方看都是三角形，唯一线索是亮度沿半径的趋势；
        # 四面体与三角棱柱从上方看都是三角形，唯一线索是亮度沿半径的趋势；
        if abs(radial) < cfg["tri_ambig_band"]:
            return UNKNOWN, 0.0
        if radial >= 0.10:
            return "tetrahedron", 0.45
        if hull_v <= 3:
            return "tetrahedron", 0.45
        return ("triangular_prism", 0.40) if elong >= 1.2 else ("tetrahedron", 0.40)

    # 亮度沿半径上升（顶部高光/棱线）：立方体的斜视角
    if radial >= 0.10:
        return "cube", 0.45

    # 圆形家族补充判据：轮廓近似顶点数多（圆≈8）且亮度随半径下降 → 圆锥。
    # 五/六棱柱的轮廓顶点数是 5/6，不会被误收。
    if poly_v >= cfg["poly_circle_min_v"] and radial <= cfg["poly_circle_radial"]:
        return "cone", 0.55

    # 规则多边形的顶面应当接近正多边形（正六边形圆度≈0.91、正五边形≈0.86，掩膜误差后
    # 也在 0.5 以上）。圆度很低的 5~6 边形轮廓多半是"两个物体粘在一起"的哑铃形块
    # ——距离变换只有一个峰，分水岭拆不开，靠这一条兜住，宁可不判。
    if hull_v >= 5 and circ < cfg["poly_min_circularity"]:
        return UNKNOWN, 0.0
    # 其余按凸包多边形顶点数
    if hull_v >= 6:
        return "hexagonal_prism", 0.45
    if hull_v == 5:
        return ("pentagonal_prism", 0.45) if elong < 1.35 else ("hexagonal_prism", 0.35)
    if hull_fill >= 0.97:              # 近乎完美的正方形 → 立方体顶面
        return "cube", 0.55
    return ("pentagonal_prism", 0.40) if hull_fill >= 0.85 else ("cube", 0.40)


def shape_confusion(f: dict[str, float], label: str) -> tuple[float, list[str]]:
    """只在已知混淆区里累计"判不准"的程度（0 表示形状无歧义）。

    判据取自 文档/OpenCV形状识别实验报告.md 记录的混淆矩阵，属于小样本标定，
    换物料或换相机后要复核。这个分数直接决定"要不要抓"。
    """
    score = 0.0
    reasons: list[str] = []
    circ = f["circularity"]
    res, elong, rect_fill = f["circle_residual"], f["elong"], f["rect_fill"]
    radial, hull_v = f["radial_corr"], f["hull_vertices"]

    # 球 / 圆锥：明暗结构落在判据附近
    if label in ("ball", "cone") and -0.68 <= radial <= -0.45:
        score += 0.35
        reasons.append("球/圆锥明暗相近")
    # 圆 / 正多边形：圆度与圆残差同时处在边界
    if (label in ("ball", "cone", "cylinder") and 0.78 <= circ <= 0.87
            and 0.02 <= res <= 0.09 and hull_v in (6.0, 7.0)):
        score += 0.25
        reasons.append("圆/正多边形边界")
    # 圆锥 / 圆柱：长宽比贴着分界
    if label in ("cone", "cylinder") and 1.10 <= elong <= 1.22:
        score += 0.20
        reasons.append("圆锥/圆柱长宽比接近")
    # 立方体 / 五、六棱柱：填充率重叠
    if label in ("cube", "pentagonal_prism", "hexagonal_prism") and 0.84 <= rect_fill <= 0.95:
        score += 0.25
        reasons.append("立方体/棱柱填充率接近")
    # 三角类：填充率贴近三角判据
    if label in ("tetrahedron", "triangular_prism") and 0.62 <= rect_fill <= 0.70:
        score += 0.20
        reasons.append("三角类填充率边界")
    return min(score, 1.0), reasons


def shape_ambiguity(f: dict[str, float], label: str, confidence: float) -> tuple[float, list[str]]:
    """形状不确定性总分：混淆区 + 轮廓质量 + 规则置信度。

    只用于排序和报告；决定"能不能抓"的是 shape_confusion（是否真的判不准）。
    轮廓粗糙说明掩膜边缘不干净，会拉低优先级，但不代表形状认错。
    """
    score, reasons = shape_confusion(f, label)
    if f["solidity"] < 0.92:
        score += 0.25
        reasons.append("轮廓缺口/不凸")
    if f["convexity"] < 0.90:
        score += 0.15
        reasons.append("边缘粗糙")
    if confidence < 0.45:
        score += 0.20
        reasons.append("规则置信度低")
    return min(score, 1.0), reasons


def radius_along(mask: np.ndarray, centre: tuple[float, float],
                 direction: tuple[float, float]) -> float:
    """从质心沿指定方向走到轮廓边界的距离（支撑半径）。

    长条形物块用等效半径会高估尺寸，导致净空恒为 0；沿连心线取各自的
    支撑半径，再相减才接近真实间隙。
    """
    rows, cols = mask.shape
    cx, cy = centre
    norm = float(np.hypot(direction[0], direction[1]))
    if norm < 1e-9:
        return 0.0
    ux, uy = direction[0] / norm, direction[1] / norm
    limit = int(np.hypot(rows, cols))
    last = 0.0
    for step in range(1, limit):
        x = int(round(cx + ux * step))
        y = int(round(cy + uy * step))
        if x < 0 or y < 0 or x >= cols or y >= rows or mask[y, x] == 0:
            break
        last = float(step)
    return last


def sits_on_tray(mask: np.ndarray, centre: tuple[float, float], lab: np.ndarray,
                 tray_l: np.ndarray, tol: float) -> tuple[bool, str]:
    """判断物块是不是真的摆在白托盘上。

    取离物体 9~25 像素的外环带（避开物体自身阴影和贴边的邻块），用 75 分位亮度
    和"只按高亮像素估计"的托盘参考亮度比较。左侧灰色台面明显比白托盘暗，
    靠这一条排除干扰物；但它仍然留在检出结果里，只是不建议抓。
    """
    mask = mask.astype(np.uint8)
    band = cv2.subtract(cv2.dilate(mask, np.ones((25, 25), np.uint8)),
                        cv2.dilate(mask, np.ones((9, 9), np.uint8)))
    if not np.any(band):
        return True, ""
    level = float(np.percentile(lab[:, :, 0][band > 0], 75))
    rows, cols = lab.shape[:2]
    cy = int(np.clip(centre[1], 0, rows - 1))
    cx = int(np.clip(centre[0], 0, cols - 1))
    reference = float(tray_l[cy, cx])
    if level < reference - tol:
        return False, f"不在白托盘上(周围{level:.0f}<参考{reference - tol:.0f})"
    return True, ""


def rank_picks(objects: list[ShapeObject], roi: np.ndarray | None,
               cfg: dict[str, Any] | None = None,
               lab: np.ndarray | None = None) -> list[ShapeObject]:
    """给每个物块算净空与形状歧义度，排出推荐抓取顺序。

    排序按现场需要：先"单独"——离邻块和托盘边沿的空隙要够；再看"形状规整、
    没有歧义"——歧义度低、置信度高。净空不足或形状存疑的块标成不建议抓
    （`pickable=False` 并写明原因），但仍然输出，便于人工复核。

    净空按"连心线上的支撑半径"计算：分水岭拆出来的相邻块在掩膜上本来就贴着，
    直接量掩膜间距会恒等于 1 像素；改用"质心距离 − 两侧各自的支撑半径"，
    并按 8 条射线量到托盘边沿的余量，才是真正能夹的空隙。
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    if not objects:
        return objects
    radii = [max(float(obj.features.get("r_eq", 1.0)), 1e-6) for obj in objects]
    centres = [(float(obj.center[0]), float(obj.center[1])) for obj in objects]
    tray_l = None
    if lab is not None:
        tray_pred, _ = tray_surface(lab[:, :, 0], lab[:, :, 1], lab[:, :, 2],
                                    cfg["sigma"], l_min=cfg["tray_ref_l_min"])
        tray_l = tray_pred[:, :, 0]

    for index, obj in enumerate(objects):
        r_eq = radii[index]
        gaps: list[float] = []
        for other in range(len(objects)):
            if other == index:
                continue
            dx = centres[other][0] - centres[index][0]
            dy = centres[other][1] - centres[index][1]
            centre_gap = float(np.hypot(dx, dy))
            if centre_gap < 1e-6:
                gaps.append(0.0)
                continue
            reach_here = radius_along(obj.mask, centres[index], (dx, dy))
            reach_other = radius_along(objects[other].mask, centres[other], (-dx, -dy))
            gaps.append(max(centre_gap - reach_here - reach_other, 0.0))
        if roi is not None:
            for step in range(8):
                angle = step * np.pi / 4
                direction = (float(np.cos(angle)), float(np.sin(angle)))
                room = radius_along(roi, centres[index], direction)
                reach = radius_along(obj.mask, centres[index], direction)
                gaps.append(max(room - reach, 0.0))
        clearance = min(gaps) if gaps else 4.0 * r_eq

        need = max(cfg["min_clearance_px"], cfg["clearance_ratio"] * r_eq)
        confusion, _ = shape_confusion(obj.features, obj.label)
        ambiguity, reasons = shape_ambiguity(obj.features, obj.label, obj.confidence)
        on_tray, tray_reason = True, ""
        if lab is not None and tray_l is not None:
            on_tray, tray_reason = sits_on_tray(obj.mask, centres[index], lab, tray_l,
                                                cfg["tray_ref_tol"])
        # 掩膜质量不足 → 形状拒判，直接排除在抓取候选之外
        shape_ok = obj.label != UNKNOWN
        if not shape_ok:
            reasons.insert(0, "轮廓质量差，形状拒判")
        obj.clearance_px = clearance
        obj.clearance_ratio = clearance / r_eq
        obj.ambiguity = confusion
        obj.pickable = (on_tray and shape_ok
                        and clearance >= need and confusion <= cfg["max_ambiguity"])
        if not on_tray:
            reasons.insert(0, tray_reason)
        if clearance < need:
            reasons.append(f"净空不足({clearance:.0f}<{need:.0f}px)")
        if confusion > cfg["max_ambiguity"]:
            reasons.append(f"形状存疑({confusion:.2f})")
        obj.reasons = tuple(reasons)
        clearance_norm = min(1.0, clearance / (2.5 * r_eq))
        obj.score = (cfg["weight_clearance"] * clearance_norm
                     + cfg["weight_confidence"] * obj.confidence
                     + cfg["weight_regular"] * (1.0 - ambiguity))

    order = sorted(objects, key=lambda o: (not o.pickable, -o.score, o.center[1], o.center[0]))
    for rank, obj in enumerate(order, start=1):
        obj.rank = rank if obj.pickable else 0
    return order


def analyse_image(bgr: np.ndarray, cfg: dict[str, Any] | None = None) -> tuple[np.ndarray, list[ShapeObject]]:
    """完整流程：降采样 → 分割 → 形状描述 → 分类。"""
    cfg = {**DEFAULTS, **(cfg or {})}
    scale = cfg["scale"]
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = np.dstack([lab[:, :, 0], lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0])
    results: list[ShapeObject] = []
    for item in segment_objects(small, cfg):
        f = shape_features(item["mask"], lab, cfg)
        if f is None:
            continue
        label, confidence = classify_shape(f, cfg)
        results.append(ShapeObject(bbox=item["bbox"], area=item["area"], label=label,
                                   confidence=confidence, features=f, mask=item["mask"],
                                   center=item["center"]))
    roi = tray_roi_mask(small, cfg)
    ordered = rank_picks(results, roi, cfg, lab)
    return small, ordered


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def annotate(small: np.ndarray, objects: Iterable[ShapeObject], scale: float = 1.0) -> np.ndarray:
    """画轮廓和标签：建议抓取的按优先级标成 #1、#2…，不建议抓的标成 x。

    形状原因（净空不足、形状存疑）只写进 JSON，因为 OpenCV 自带字体不支持中文。
    """
    vis = small.copy()
    for obj in objects:
        contours, _ = cv2.findContours(obj.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if obj.pickable:
            colour = (0, 200, 0) if obj.confidence >= 0.5 else (0, 200, 255)
            tag = f"#{obj.rank} {obj.label} {obj.confidence:.2f}"
        else:
            colour = (150, 150, 150)
            tag = f"x {obj.label} {obj.confidence:.2f}"
        cv2.drawContours(vis, contours, -1, colour, 2)
        x, y = obj.bbox[0], obj.bbox[1]
        cv2.putText(vis, tag, (x + 2, max(14, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)
    return vis


def objects_to_json(objects: Iterable[ShapeObject], scale: float) -> list[dict[str, Any]]:
    """导出结果；对象顺序就是推荐抓取顺序（不建议抓的排在后面）。"""
    payload = []
    for obj in objects:
        x1, y1, x2, y2 = obj.bbox
        payload.append(dict(
            rank=obj.rank,
            pickable=obj.pickable,
            label=obj.label,
            confidence=round(obj.confidence, 3),
            score=round(obj.score, 3),
            ambiguity=round(obj.ambiguity, 3),
            clearance_px=round(obj.clearance_px / scale, 1),
            clearance_ratio=round(obj.clearance_ratio, 3),
            reasons=list(obj.reasons),
            bbox_px=[round(v / scale, 1) for v in (x1, y1, x2, y2)],
            center_px=[round(obj.center[0] / scale, 1), round(obj.center[1] / scale, 1)],
            area_px=round(obj.area / (scale * scale), 1),
            features={k: round(float(v), 4) for k, v in obj.features.items()},
        ))
    return payload


# --------------------------------------------------------------------------- #
# 标注核对
# --------------------------------------------------------------------------- #
def load_yolo_boxes(image_path: Path, label_path: Path) -> list[tuple[str, tuple[int, int, int, int]]]:
    classes_path = label_path.parent / "classes.txt"
    classes = [x.strip() for x in classes_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        boxes.append((classes[int(parts[0])],
                      (int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                       int((cx + bw / 2) * w), int((cy + bh / 2) * h))))
    return boxes



def iou_box(a, b) -> float:
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def evaluate_labels(image_dir: Path = RAW_IMAGES, label_dir: Path = RAW_LABELS,
                    iou_min: float = 0.55, verbose: bool = True,
                    cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """用已有 YOLO 标注框核对形状识别准确率（只使用有标注的图片）。"""
    per_class: dict[str, dict[str, int]] = {}
    matched = correct = 0
    gt_total = 0
    stems = sorted(p.stem for p in label_dir.glob("*.txt") if p.name != "classes.txt")
    for stem in stems:
        image_path = image_dir / f"{stem}.jpg"
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path))
        boxes = load_yolo_boxes(image_path, label_dir / f"{stem}.txt")
        gt_total += len(boxes)
        _, results = analyse_image(image, cfg)
        scale = DEFAULTS["scale"]
        for truth, gt_box in boxes:
            scaled_gt = tuple(v * scale for v in gt_box)
            best, best_iou = None, 0.0
            for obj in results:
                iou = iou_box(scaled_gt, obj.bbox)
                if iou > best_iou:
                    best, best_iou = obj, iou
            if best is None or best_iou < iou_min:
                continue
            matched += 1
            correct += int(best.label == truth)
            row = per_class.setdefault(truth, {})
            row[best.label] = row.get(best.label, 0) + 1
    summary = dict(gt_total=gt_total, matched=matched, correct=correct,
                   accuracy=round(correct / matched, 4) if matched else 0.0,
                   per_class=per_class)
    if verbose:
        print(f"标注实例 {gt_total}，成功匹配 {matched}（IoU≥{iou_min}），"
              f"形状判定正确 {correct}（{summary['accuracy']:.1%}）")
        for truth in sorted(per_class):
            row = per_class[truth]
            total = sum(row.values())
            print(f"  {truth:<18} {row.get(truth, 0)}/{total}  {row}")
    return summary


# --------------------------------------------------------------------------- #
# 合成自检
# --------------------------------------------------------------------------- #
def render_synthetic_tray(shapes: list[str], size: tuple[int, int] = (1400, 960)) -> np.ndarray:
    """生成一张合成托盘图：亮底 + 指定形状的实心色块，用于自检。

    尺寸接近真实原图降采样后的量级，这样参数（最小面积/最小内切半径）
    处在和实物相同的区间。
    """
    w, h = size
    img = np.full((h, w, 3), 232, np.uint8)
    palette = [(60, 60, 200), (60, 200, 60), (200, 80, 60), (200, 200, 60)]
    for index, shape in enumerate(shapes):
        cx = 240 + (index % 3) * 460
        cy = 250 + (index // 3) * 460
        colour = palette[index % len(palette)]
        if shape in ("ball", "cone", "cylinder"):
            cv2.circle(img, (cx, cy), 95, colour, -1, cv2.LINE_AA)
        elif shape == "cube":
            cv2.rectangle(img, (cx - 85, cy - 85), (cx + 85, cy + 85), colour, -1)
        elif shape == "cuboid":
            cv2.rectangle(img, (cx - 115, cy - 50), (cx + 115, cy + 50), colour, -1)
        elif shape in ("tetrahedron", "triangular_prism"):
            pts = np.array([[cx, cy - 100], [cx + 100, cy + 80], [cx - 100, cy + 80]], np.int32)
            cv2.fillPoly(img, [pts], colour)
        elif shape == "pentagonal_prism":
            pts = np.array([[cx + int(96 * np.cos(np.pi / 2 + i * 2 * np.pi / 5)),
                             cy - int(96 * np.sin(np.pi / 2 + i * 2 * np.pi / 5))] for i in range(5)], np.int32)
            cv2.fillPoly(img, [pts], colour)
        elif shape == "hexagonal_prism":
            pts = np.array([[cx + int(96 * np.cos(np.pi / 6 + i * np.pi / 3)),
                             cy - int(96 * np.sin(np.pi / 6 + i * np.pi / 3))] for i in range(6)], np.int32)
            cv2.fillPoly(img, [pts], colour)
    return img


def run_self_test() -> int:
    """合成图自检：分割能分开物体，且圆/方/三角/五边形/六边形的判定正确。"""
    shapes = ["ball", "cube", "cuboid", "tetrahedron", "pentagonal_prism", "hexagonal_prism"]
    image = render_synthetic_tray(shapes)
    _, objects = analyse_image(image)
    found = [obj.label for obj in objects]
    print("合成图检出:", len(objects), found)
    checks = [
        ("检出数量", len(objects) >= 5),
        ("圆→ball/cone", any(lbl in ("ball", "cone") for lbl in found)),
        ("矩形→cube/cuboid", any(lbl in ("cube", "cuboid") for lbl in found)),
        ("三角→tetrahedron", "tetrahedron" in found),
        ("五边形→pentagonal_prism", "pentagonal_prism" in found),
        ("六边形→hexagonal_prism", "hexagonal_prism" in found),
    ]
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if failed:
        print("自检失败:", ", ".join(failed), file=sys.stderr)
        return 1
    print("合成图自检通过（合成色块，不代表真实物料精度）")
    return 0


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #
def parse_overrides(items: list[str]) -> dict[str, float]:
    """把 ["key=value", ...] 解析成 cfg 覆盖字典；键必须是 DEFAULTS 里已有的项。"""
    overrides: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set 需要 KEY=VALUE 形式，收到: {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in DEFAULTS:
            raise SystemExit(f"未知的配置项 {key}（可选：{', '.join(sorted(DEFAULTS))}）")
        overrides[key] = float(raw)
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenCV 形状识别（不训练模型、不用颜色作为类别）"
    )
    parser.add_argument("--image", type=Path, help="单张图片路径")
    parser.add_argument("--dir", type=Path, default=RAW_IMAGES, help="批量目录，默认 dataset/yolo_cube/images/raw")
    parser.add_argument("--limit", type=int, default=0, help="批量模式最多处理多少张，0 表示全部")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出目录")
    parser.add_argument("--save-images", action="store_true", help="保存标注图")
    parser.add_argument("--json", type=Path, help="把结果写入 JSON 文件")
    parser.add_argument("--self-test", action="store_true", help="合成图自检，不读磁盘")
    parser.add_argument("--eval-labels", action="store_true", help="用已有 YOLO 标注核对准确率")
    parser.add_argument("--clearance-ratio", type=float, default=DEFAULTS["clearance_ratio"],
                        help="所需净空 = 该比例 × 物体等效半径（默认 %(default)s）")
    parser.add_argument("--min-clearance-px", type=float, default=DEFAULTS["min_clearance_px"],
                        help="净空绝对下限（半分辨率像素，默认 %(default)s）")
    parser.add_argument("--max-ambiguity", type=float, default=DEFAULTS["max_ambiguity"],
                        help="形状歧义度上限，超过就不建议抓取（默认 %(default)s）")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="覆盖任意 DEFAULTS 阈值，可重复，如 --set cuboid_fill=0.90")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.eval_labels:
        evaluate_labels()
        return 0

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    if args.image:
        paths = [args.image if args.image.is_absolute() else ROOT / args.image]
    else:
        directory = args.dir if args.dir.is_absolute() else ROOT / args.dir
        paths = sorted(directory.glob("*.jpg"))
        if args.limit > 0:
            paths = paths[: args.limit]
    if not paths:
        print("没有找到图片", file=sys.stderr)
        return 2

    cfg = dict(clearance_ratio=args.clearance_ratio,
               min_clearance_px=args.min_clearance_px,
               max_ambiguity=args.max_ambiguity,
               **parse_overrides(args.set))
    payload: dict[str, Any] = {}
    started = time.perf_counter()
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"跳过无法读取的图片: {path}", file=sys.stderr)
            continue
        small, objects = analyse_image(image, cfg)
        payload[path.name] = objects_to_json(objects, DEFAULTS["scale"])
        counts: dict[str, int] = {}
        for obj in objects:
            counts[obj.label] = counts.get(obj.label, 0) + 1
        picks = [obj for obj in objects if obj.pickable]
        print(f"{path.name}: {len(objects)} 个物块，建议抓取 {len(picks)} 个 {counts}")
        if picks:
            order_text = " → ".join(
                f"#{o.rank} {o.label}(净空{o.clearance_ratio:.2f}r, 歧义{o.ambiguity:.2f})"
                for o in picks[:4]
            )
            print(f"   优先抓取: {order_text}")
        elif objects:
            best = max(objects, key=lambda o: o.score)
            why = best.reasons[0] if best.reasons else "条件不足"
            print(f"   没有满足条件的块；最接近的是 {best.label}({best.clearance_ratio:.2f}r, {why})")
        if args.save_images:
            vis = annotate(small, objects)
            cv2.imwrite(str(output / f"{path.stem}_shapes.jpg"), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    elapsed = time.perf_counter() - started
    print(f"共处理 {len(payload)} 张，用时 {elapsed:.1f}s（{elapsed / max(len(payload), 1):.2f}s/张）")

    if args.json:
        json_path = args.json if args.json.is_absolute() else ROOT / args.json
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"结果已写入 {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
