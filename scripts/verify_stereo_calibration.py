"""合成验证：用已知的内参/外参渲染标定板帧组，跑标定，看能否还原真值。

用法：python scripts/verify_stereo_calibration.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BOARD_JSON = ROOT / "captures" / "calibration" / "charuco_screen_244x170mm.json"
WORK = ROOT / "captures" / "calibration" / "_synthetic"

W, H = 2360, 1640          # 图像尺寸
TRUE_K1 = np.array([[2600.0, 0, 1180.0], [0, 2595.0, 820.0], [0, 0, 1]])
TRUE_K2 = np.array([[2450.0, 0, 1300.0], [0, 2448.0, 790.0], [0, 0, 1]])
TRUE_D1 = np.array([-0.12, 0.05, 0.001, -0.001, 0.0])
TRUE_D2 = np.array([-0.09, 0.02, -0.001, 0.001, 0.0])
# 侧视相机相对垂直相机：绕 x 轴俯仰约 40°，基线 400mm
_ax = 40.0 * np.pi / 180.0
TRUE_R = np.array([[1, 0, 0],
                   [0, np.cos(_ax), -np.sin(_ax)],
                   [0, np.sin(_ax), np.cos(_ax)]])
TRUE_T = np.array([-120.0, 260.0, 60.0])


def load_board():
    params = json.loads(BOARD_JSON.read_text(encoding="utf-8"))
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, params["dictionary"]))
    board = cv2.aruco.CharucoBoard(tuple(params["squares"]), params["square_mm"],
                                   params["marker_mm"], d)
    return board, params


def render(board_img, board_mm, rvec, tvec, K, dist, size=(W, H)):
    """把板子按给定位姿投影渲染到图像（含畸变），用 remap 实现。"""
def render(board_img, board_mm, rvec, tvec, K, dist, size=(W, H)):
    """把板子按给定位姿投影渲染到图像（含畸变）。

    remap 需要的是"目标像素 → 源坐标"的映射，所以对每个目标像素做反投影：
    去畸变 → 得到相机系射线 → 与板面求交 → 换算成板子图像上的像素坐标。
    """
    bw, bh = board_mm
    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, np.float64).reshape(3)
    normal = R @ np.array([0.0, 0.0, 1.0])          # 板面法向（相机坐标系）

    xs, ys = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    pixels = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32).reshape(-1, 1, 2)
    # 去畸变后得到归一化像平面坐标（z=1 的射线方向）
    norm = cv2.undistortPoints(pixels, K, dist).reshape(-1, 2).astype(np.float64)
    rays = np.concatenate([norm, np.ones((norm.shape[0], 1))], axis=1)

    denom = rays @ normal
    numer = float(normal @ t)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = numer / denom
    hit = rays * scale[:, None]                      # 射线与板面的交点（相机系）
    board_xyz = (hit - t) @ R                        # 转到板子坐标系 = Rᵀ(hit − t)

    map_x = (board_xyz[:, 0] * (board_img.shape[1] / bw)).astype(np.float32).reshape(size[1], size[0])
    map_y = (board_xyz[:, 1] * (board_img.shape[0] / bh)).astype(np.float32).reshape(size[1], size[0])
    canvas = np.full((size[1], size[0]), 170, np.uint8)
    cv2.remap(board_img, map_x, map_y, cv2.INTER_LINEAR, dst=canvas,
              borderMode=cv2.BORDER_CONSTANT, borderValue=170)
    return canvas


def main() -> int:
    board, params = load_board()
    board_mm = tuple(params["board_mm"])
    board_img = board.generateImage((int(board_mm[0] * 4), int(board_mm[1] * 4)), marginSize=0)


    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / "frames").mkdir(parents=True)

    rng = np.random.default_rng(7)
    n = 24
    print(f"生成 {n} 组合成帧（{W}×{H}）…")
    for i in range(1, n + 1):
        # 板子大致在垂直相机前方，姿态带随机倾斜（倾斜对内参可观测性是必需的）
        tvec1 = np.array([board_mm[0] / 2 - 180 + rng.uniform(-160, 160),
                          board_mm[1] / 2 - 120 + rng.uniform(-120, 120),
                          1250 + rng.uniform(-120, 120)], np.float64)
        rvec1 = np.array([rng.uniform(-0.55, 0.55),
                          rng.uniform(-0.55, 0.55),
                          rng.uniform(-0.45, 0.45)], np.float64)
        img1 = render(board_img, board_mm, rvec1, tvec1, TRUE_K1, TRUE_D1)
        # 侧视：把板子的位姿变换到相机 2 的坐标系
        R1, _ = cv2.Rodrigues(rvec1)
        R2 = TRUE_R @ R1
        t2 = TRUE_R @ tvec1 + TRUE_T
        rvec2, _ = cv2.Rodrigues(R2)
        img2 = render(board_img, board_mm, rvec2, t2, TRUE_K2, TRUE_D2)
        cv2.imwrite(str(WORK / "frames" / f"pair_{i:03d}_vertical.jpg"), img1,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(WORK / "frames" / f"pair_{i:03d}_side.jpg"), img2,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

    out_json = WORK / "result.json"
    cmd = [sys.executable, str(ROOT / "scripts" / "calibrate_stereo.py"),
           "--frames", str(WORK / "frames"), "--output", str(out_json),
           "--stage", "both"]
    print("运行标定：", " ".join(cmd[1:]))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(proc.stdout[-4000:] if proc.stdout else "")
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return 1

    res = json.loads(out_json.read_text(encoding="utf-8"))
    iv = res["intrinsics"]["vertical"]
    isd = res["intrinsics"]["side_a"]
    ex = res["extrinsics"]
    K1 = np.array(iv["mtx"]); K2 = np.array(isd["mtx"])
    R = np.array(ex["R"]); T = np.array(ex["T"])

    def rel(a, b):
        return abs(a - b) / max(abs(b), 1e-9)

    print("\n===== 与真值对比 =====")
    print(f"{'参数':22}{'真值':>14}{'还原值':>14}{'相对误差':>12}")
    rows = [
        ("垂直 fx", TRUE_K1[0, 0], K1[0, 0]), ("垂直 fy", TRUE_K1[1, 1], K1[1, 1]),
        ("垂直 cx", TRUE_K1[0, 2], K1[0, 2]), ("垂直 cy", TRUE_K1[1, 2], K1[1, 2]),
        ("侧视 fx", TRUE_K2[0, 0], K2[0, 0]), ("侧视 fy", TRUE_K2[1, 1], K2[1, 1]),
        ("侧视 cx", TRUE_K2[0, 2], K2[0, 2]), ("侧视 cy", TRUE_K2[1, 2], K2[1, 2]),
        ("基线 |t| (mm)", float(np.linalg.norm(TRUE_T)), float(np.linalg.norm(T))),
    ]
    for name, truth, got in rows:
        print(f"{name:22}{truth:14.2f}{got:14.2f}{rel(got, truth) * 100:11.3f}%")

    ang_true = np.degrees(np.arccos(np.clip((np.trace(TRUE_R) - 1) / 2, -1, 1)))
    ang_got = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"{'旋转角 (度)':22}{ang_true:14.2f}{ang_got:14.2f}{abs(ang_got - ang_true):11.3f}°")

    dir_true = TRUE_T / np.linalg.norm(TRUE_T)
    dir_got = T / np.linalg.norm(T)
    print(f"{'平移方向夹角 (度)':22}{'':14}{'':14}"
          f"{np.degrees(np.arccos(np.clip(np.dot(dir_true, dir_got), -1, 1))):11.3f}°")

    ok = (rel(K1[0, 0], TRUE_K1[0, 0]) < 0.02 and rel(K2[1, 1], TRUE_K2[1, 1]) < 0.02
          and abs(ang_got - ang_true) < 1.0
          and rel(float(np.linalg.norm(T)), float(np.linalg.norm(TRUE_T))) < 0.03)
    print(f"\n结论：{'通过（内参与外参都还原到 2~3% 以内）' if ok else '未通过，检查上面的偏差'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
