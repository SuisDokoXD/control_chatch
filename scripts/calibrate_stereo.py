"""ChArUco 双目标定：输出每台相机的内参+畸变、两相机之间的 R/t，以及重投影误差。

用法：
    # 全流程（内参 + 外参）
    python scripts/calibrate_stereo.py

    # 分两步：先用自由倾斜的帧标内参，再用平放的帧标外参
    python scripts/calibrate_stereo.py --stage intrinsics
    python scripts/calibrate_stereo.py --stage extrinsics --fix-intrinsics

    # 自动剔除重投影误差大的帧
    python scripts/calibrate_stereo.py --max-error 0.6

原理说明（为什么分两步）：
- 平面靶标的内参标定**需要倾斜**：所有姿态都在同一个平面上时焦距不可观测。
- 外参标定对平面靶标是**良态**的，所以平放在托盘里的帧就能标 R/t。
- 因此推荐：内参用"自由倾斜"的帧（不要求共同视野），外参用"平放在托盘"的帧
  （要求两台都看到），外参阶段用 `--fix-intrinsics` 固定内参。

产出 配置/calibration_stereo.json，含两台相机的内参、畸变、外参 R/t、重投影误差。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BOARD = ROOT / "captures" / "calibration" / "charuco_screen_244x170mm.json"
FRAMES_DIR = ROOT / "captures" / "calibration" / "frames"
OUT_PATH = ROOT / "配置" / "calibration_stereo.json"


def load_board(path: Path):
    params = json.loads(Path(path).read_text(encoding="utf-8"))
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, params["dictionary"]))
    board = cv2.aruco.CharucoBoard(tuple(params["squares"]), params["square_mm"],
                                   params["marker_mm"], dictionary)
    return board, cv2.aruco.CharucoDetector(board), params


def collect(board, detector, frames_dir: Path, min_corners: int):
    """读所有帧组，返回 [(序号, 垂直图, 侧视图, 垂直角点, 垂直id, 侧视角点, 侧视id)]。"""
    pairs = []
    stems = sorted({p.name.split("_vertical")[0] for p in frames_dir.glob("*_vertical.jpg")})
    for stem in stems:
        pv = frames_dir / f"{stem}_vertical.jpg"
        ps = frames_dir / f"{stem}_side.jpg"
        if not (pv.exists() and ps.exists()):
            continue
        iv = cv2.imread(str(pv), cv2.IMREAD_GRAYSCALE)
        isd = cv2.imread(str(ps), cv2.IMREAD_GRAYSCALE)
        if iv is None or isd is None:
            continue
        cv, idv, _, _ = detector.detectBoard(iv)
        cs, ids, _, _ = detector.detectBoard(isd)
        nv = 0 if cv is None else len(cv)
        ns = 0 if cs is None else len(cs)
        pairs.append(dict(stem=stem, image_v=iv, image_s=isd,
                          corners_v=cv, ids_v=idv, corners_s=cs, ids_s=ids,
                          n_v=nv, n_s=ns))
    return pairs


def object_points(board, ids):
    """按角点 id 取标定板坐标系下的三维点（单位 mm，板面 z=0）。"""
    all_pts = board.getChessboardCorners()
    return np.asarray(all_pts, np.float32)[np.asarray(ids).ravel()].reshape(-1, 1, 3)


def calibrate_single(board, pairs, key, min_corners, image_size, max_error=None, dist_flags=0):
    obj, img = [], []
    for p in pairs:
        if p[f"n_{key}"] < min_corners:
            continue
        obj.append(object_points(board, p[f"ids_{key}"]))
        img.append(np.asarray(p[f"corners_{key}"], np.float32).reshape(-1, 1, 2))
    if len(obj) < 3:
        return None
    flags = dist_flags
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, image_size, None, None, flags=flags)
    # 逐帧误差
    errors = []
    for i, (o, im) in enumerate(zip(obj, img)):
        proj, _ = cv2.projectPoints(o, rvecs[i], tvecs[i], mtx, dist)
        errors.append(float(np.sqrt(np.mean((proj.reshape(-1, 2) - im.reshape(-1, 2)) ** 2))))
    if max_error is not None:
        keep = [i for i, e in enumerate(errors) if e <= max_error]
        if len(keep) >= 3 and len(keep) < len(obj):
            obj = [obj[i] for i in keep]
            img = [img[i] for i in keep]
            rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, image_size, None, None, flags=flags)
            errors = []
            for i, (o, im) in enumerate(zip(obj, img)):
                proj, _ = cv2.projectPoints(o, rvecs[i], tvecs[i], mtx, dist)
                errors.append(float(np.sqrt(np.mean((proj.reshape(-1, 2) - im.reshape(-1, 2)) ** 2))))
    return dict(rms=float(rms), mtx=mtx, dist=dist, errors=errors, used=len(obj))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ChArUco 双目标定")
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--frames", type=Path, default=FRAMES_DIR)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--stage", choices=("both", "intrinsics", "extrinsics"), default="both")
    parser.add_argument("--fix-intrinsics", action="store_true",
                        help="外参阶段固定内参（读已有结果或本次内参阶段的结果）")
    parser.add_argument("--max-error", type=float, default=None,
                        help="单帧重投影误差超过此值就剔除（像素）")
    parser.add_argument("--dist-model", type=int, choices=(5, 4, 2), default=5,
                        help="畸变参数个数：5=k1,k2,p1,p2,k3（默认）；4=固定 k3；2=只留 k1,k2")
    parser.add_argument("--json", type=Path, help="把逐帧明细写出来")
    args = parser.parse_args(argv)

    # 畸变模型：默认 5 参数（k1,k2,p1,p2,k3）；镜头畸变极小时 5 参数容易过拟合，
    # 用 4（固定 k3）或 2（只留 k1,k2）更稳。
    dist_flags = 0
    if args.dist_model == 4:
        dist_flags |= cv2.CALIB_FIX_K3
    elif args.dist_model == 2:
        dist_flags |= cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST

    board_path = args.board if args.board.is_absolute() else ROOT / args.board
    frames_dir = args.frames if args.frames.is_absolute() else ROOT / args.frames
    out_path = args.output if args.output.is_absolute() else ROOT / args.output
    if not board_path.exists():
        print(f"找不到标定板参数 {board_path}", file=sys.stderr)
        return 2
    if not frames_dir.exists():
        print(f"找不到帧目录 {frames_dir}，先用 scripts/capture_stereo_frames.py 采图", file=sys.stderr)
        return 2

    board, detector, board_params = load_board(board_path)
    pairs = collect(board, detector, frames_dir, args.min_corners)
    if not pairs:
        print(f"{frames_dir} 里没有可用的帧组", file=sys.stderr)
        return 2
    def common_ids_count(p) -> int:
        if p["ids_v"] is None or p["ids_s"] is None:
            return 0
        return len(np.intersect1d(np.asarray(p["ids_v"]).ravel(), np.asarray(p["ids_s"]).ravel()))

    # 外参只用"两台都看到、且共同角点够多"的帧（两台检出的角点集合通常不同）
    both = [p for p in pairs
            if p["n_v"] >= args.min_corners and p["n_s"] >= args.min_corners
            and common_ids_count(p) >= args.min_corners]
    image_size = (pairs[0]["image_v"].shape[1], pairs[0]["image_v"].shape[0])
    print(f"帧组 {len(pairs)} 组，其中两台都达标 {len(both)} 组；图像 {image_size[0]}x{image_size[1]}")
    print(f"标定板 {board_params['label']}  格边长 {board_params['square_mm']}mm  "
          f"棋盘 {board_params['squares'][0]}×{board_params['squares'][1]}")
    if len(both) < 6:
        print(f"警告：两台都达标的只有 {len(both)} 组，外参标定至少需要 6 组（建议 15~20）")

    result: dict = {"board": board_params, "image_size": list(image_size),
                    "frames_total": len(pairs), "frames_both": len(both)}
    previous = {}
    if out_path.exists():
        try:
            previous = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}

    intrinsics = {}
    if args.stage in ("both", "intrinsics"):
        for key, name in (("v", "vertical"), ("s", "side_a")):
            r = calibrate_single(board, pairs, key, args.min_corners, image_size,
                                 args.max_error, dist_flags)
            if r is None:
                print(f"{name}: 有效帧不足，跳过", file=sys.stderr)
                continue
            intrinsics[name] = r
            print(f"\n[{name}] 用 {r['used']} 帧  重投影 RMS {r['rms']:.3f} px  "
                  f"单帧误差 中位 {np.median(r['errors']):.3f}  最大 {max(r['errors']):.3f}")
            print(f"  fx={r['mtx'][0,0]:.1f}  fy={r['mtx'][1,1]:.1f}  "
                  f"cx={r['mtx'][0,2]:.1f}  cy={r['mtx'][1,2]:.1f}")
            print(f"  畸变 {np.round(r['dist'].ravel(), 4).tolist()}")
        if intrinsics:
            result["intrinsics"] = {
                k: {"mtx": v["mtx"].tolist(), "dist": v["dist"].ravel().tolist(),
                    "rms": v["rms"], "used_frames": v["used"],
                    "dist_model": args.dist_model}
                for k, v in intrinsics.items()}

    # 外参阶段
    if args.stage in ("both", "extrinsics"):
        have_intr = intrinsics or (previous.get("intrinsics") or {})
        need_fix = args.fix_intrinsics or (args.stage == "extrinsics")
        if len(both) < 6:
            print("\n外参：两台都达标的帧不足 6 组，跳过", file=sys.stderr)
        elif need_fix and not ({"vertical", "side_a"} <= set(have_intr)):
            print("\n外参：需要先有内参结果（先跑 --stage intrinsics），跳过", file=sys.stderr)
        else:
            obj, img_v, img_s = [], [], []
            for p in both:
                # 两台相机检出的角点集合不同，立体标定必须只取共同角点，否则点数不匹配
                ids_v = np.asarray(p["ids_v"]).ravel().astype(int)
                ids_s = np.asarray(p["ids_s"]).ravel().astype(int)
                common = np.intersect1d(ids_v, ids_s)
                cv = np.asarray(p["corners_v"], np.float32).reshape(-1, 2)
                cs = np.asarray(p["corners_s"], np.float32).reshape(-1, 2)
                pos_v = {int(i): k for k, i in enumerate(ids_v)}
                pos_s = {int(i): k for k, i in enumerate(ids_s)}
                obj.append(object_points(board, common))
                img_v.append(cv[[pos_v[int(i)] for i in common]].reshape(-1, 1, 2))
                img_s.append(cs[[pos_s[int(i)] for i in common]].reshape(-1, 1, 2))
            flags = cv2.CALIB_FIX_INTRINSIC if need_fix else dist_flags
            mtx_v = np.array(have_intr["vertical"]["mtx"], np.float32) if need_fix else None
            dist_v = np.array(have_intr["vertical"]["dist"], np.float32) if need_fix else None
            mtx_s = np.array(have_intr["side_a"]["mtx"], np.float32) if need_fix else None
            dist_s = np.array(have_intr["side_a"]["dist"], np.float32) if need_fix else None
            rms, mv, dv, ms, ds, R, T, E, F = cv2.stereoCalibrate(
                obj, img_v, img_s, mtx_v, dist_v, mtx_s, dist_s, image_size,
                flags=flags, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
            print(f"\n[外参] 用 {len(obj)} 组（只取两台都检出的共同角点）  重投影 RMS {rms:.3f} px")
            print(f"  旋转 R =\n{np.round(R, 4)}")
            print(f"  平移 t = {np.round(T.ravel(), 2).tolist()} mm")
            print(f"  基线 |t| = {float(np.linalg.norm(T)):.1f} mm")
            result["extrinsics"] = {
                "R": R.tolist(), "T": T.ravel().tolist(),
                "baseline_mm": float(np.linalg.norm(T)),
                "rms": float(rms), "used_frames": len(both),
                "fixed_intrinsics": bool(need_fix),
            }
            if not need_fix:
                result.setdefault("intrinsics", {})
                result["intrinsics"]["vertical"] = {
                    "mtx": mv.tolist(), "dist": dv.ravel().tolist(), "rms": float(rms)}
                result["intrinsics"]["side_a"] = {
                    "mtx": ms.tolist(), "dist": ds.ravel().tolist(), "rms": float(rms)}

    if "intrinsics" not in result and previous.get("intrinsics"):
        result["intrinsics"] = previous["intrinsics"]
    if "extrinsics" not in result and previous.get("extrinsics"):
        result["extrinsics"] = previous["extrinsics"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果写入 {out_path}")

    if args.json:
        detail = args.json if args.json.is_absolute() else ROOT / args.json
        detail.write_text(json.dumps(
            [{"stem": p["stem"], "corners_v": p["n_v"], "corners_s": p["n_s"]} for p in pairs],
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"逐帧明细写入 {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
