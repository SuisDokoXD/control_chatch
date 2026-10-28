"""生成标定板：可打印的 PNG（按纸张尺寸），或按平板屏幕物理尺寸精确渲染的 PNG。

用法：
    # 打印用（默认 A4，300 DPI）
    python scripts/make_calibration_board.py --paper A4 --verify
    python scripts/make_calibration_board.py --paper A3

    # 平板用：给出屏幕的像素分辨率和物理尺寸，板子会按 1:1 物理尺寸渲染
    python scripts/make_calibration_board.py --screen-px 2360x1640 --screen-mm 237x165
    python scripts/make_calibration_board.py --screen-px 2732x2048 --screen-mm 275x206 --square-mm 25

产出 captures/calibration/charuco_<标签>.png 和同名 .json（参数，标定工具读它）。

**平板用的关键**：必须全屏、1:1 显示（不能被看图软件缩放），显示后用尺子量图上那根
100mm 标尺核对。板子的物理尺寸错了，标定出来的尺度就错了。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "captures" / "calibration"
DPI = 300

PAPERS = {
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "A5": (148.0, 210.0),
}


def build_board(page_mm: tuple[float, float], px_per_mm: float,
                cols: int, rows: int, square_mm: float,
                marker_ratio: float, dictionary: str,
                label: str, source: str) -> tuple[np.ndarray, dict]:
    pw, ph = page_mm
    board_w = cols * square_mm
    board_h = rows * square_mm
    margin_mm = 8.0
    ruler_mm = 24.0
    if board_w + 2 * margin_mm > pw or board_h + 2 * margin_mm + ruler_mm > ph:
        raise SystemExit(
            f"{label} 放不下：板 {board_w:.0f}×{board_h:.0f}mm + 边距，"
            f"画布 {pw:.0f}×{ph:.0f}mm。请减小 --squares 或 --square-mm。")

    def px(mm: float) -> int:
        return int(round(mm * px_per_mm))

    dictionary_id = getattr(cv2.aruco, dictionary)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard((cols, rows), square_mm, square_mm * marker_ratio, aruco_dict)

    img = board.generateImage((px(board_w), px(board_h)), marginSize=0, borderBits=1)

    page = np.full((px(ph), px(pw)), 255, np.uint8)
    ox, oy = px((pw - board_w) / 2), px((ph - board_h - ruler_mm) / 2)
    page[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img

    # 底部画一根 100mm 标尺，用来核对实际尺寸
    rx0, ry0 = px(10), px(ph - 12)
    rx1 = rx0 + px(100)
    cv2.line(page, (rx0, ry0), (rx1, ry0), 0, max(2, px(0.4)))
    for x in (rx0, rx1):
        cv2.line(page, (x, ry0 - px(2)), (x, ry0 + px(2)), 0, max(2, px(0.4)))
    cv2.putText(page, "100 mm", (rx0, ry0 - px(3)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.5, px(3.2) / 32.0), 0, max(1, px(0.3)))

    params = {
        "source": source,
        "label": label,
        "canvas_mm": [round(pw, 2), round(ph, 2)],
        "px_per_mm": round(px_per_mm, 5),
        "squares": [cols, rows],
        "square_mm": square_mm,
        "marker_mm": round(square_mm * marker_ratio, 3),
        "marker_ratio": marker_ratio,
        "dictionary": dictionary,
        "board_mm": [round(board_w, 2), round(board_h, 2)],
        "image_px": [int(page.shape[1]), int(page.shape[0])],
        "note": ("打印必须 100% 比例，打印后量底部 100mm 标尺核对。"
                 if source == "paper" else
                 "平板必须全屏 1:1 显示（不能被缩放），显示后量底部 100mm 标尺核对。"),
    }
    return page, params


def make_detector(params: dict):
    dictionary_id = getattr(cv2.aruco, params["dictionary"])
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(tuple(params["squares"]), params["square_mm"],
                                   params["marker_mm"], aruco_dict)
    return board, cv2.aruco.CharucoDetector(board)


def verify(params: dict, page: np.ndarray) -> bool:
    """合成回环：把板子做透视变换，看还能不能检测出角点。"""
    _, detector = make_detector(params)
    h, w = page.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[140, 90], [w - 60, 150], [w - 120, h - 70], [60, h - 120]])
    warped = cv2.warpPerspective(page, cv2.getPerspectiveTransform(src, dst), (w, h),
                                 borderValue=180)
    corners, ids, _, marker_ids = detector.detectBoard(warped)
    n_corners = 0 if corners is None else len(corners)
    n_markers = 0 if marker_ids is None else len(marker_ids)
    total = params["squares"][0] * params["squares"][1]
    print(f"  合成透视后：检出 {n_markers} 个 ArUco 码、{n_corners} 个棋盘角点"
          f"（板上共 {total} 个内角点）")
    cv2.imwrite(str(OUT_DIR / "verify_warped.png"), warped)
    if n_corners > 20:
        canvas = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        cv2.aruco.drawDetectedCornersCharuco(canvas, corners, ids, (0, 0, 255))
        cv2.imwrite(str(OUT_DIR / "verify_corners.png"), canvas)
        print("  回环验证通过（角点检测正常，图见 verify_corners.png）")
        return True
    print("  回环验证失败：角点太少")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 ChArUco 标定板")
    parser.add_argument("--paper", choices=sorted(PAPERS), default=None,
                        help="打印模式：纸张规格")
    parser.add_argument("--screen-px", default=None, help="平板模式：屏幕像素分辨率，如 2360x1640")
    parser.add_argument("--screen-mm", default=None, help="平板模式：屏幕物理尺寸，如 237x165")
    parser.add_argument("--squares", default=None, help="如 9x6（列x行）；默认自动排满")
    parser.add_argument("--square-mm", type=float, default=None, help="格边长；默认自动选")
    parser.add_argument("--marker-ratio", type=float, default=0.75)
    parser.add_argument("--dictionary", default="DICT_5X5_100")
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--measured-ruler", type=float, default=None,
                        help="实测标尺读数(mm)。图上标 100mm，量出来多少就填多少，"
                             "工具会按比例自动校正画布尺寸，让板子真正 1:1。")
    parser.add_argument("--verify", action="store_true", help="做一次合成回环验证")
    args = parser.parse_args(argv)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    if args.screen_px and args.screen_mm:
        sw, sh = (float(v) for v in args.screen_px.lower().split("x"))
        mw, mh = (float(v) for v in args.screen_mm.lower().split("x"))
        source = "screen"
        if args.measured_ruler:
            # 图上标 100mm、实测 M mm → 屏幕实际比标称大 M/100 倍，画布按比例放大
            factor = args.measured_ruler / 100.0
            print(f"按实测标尺 {args.measured_ruler:.1f}mm 校正：画布 {mw:.1f}×{mh:.1f}mm "
                  f"→ {mw * factor:.1f}×{mh * factor:.1f}mm（系数 {factor:.4f}）")
            mw, mh = mw * factor, mh * factor
        page_mm = (mw, mh)
        px_per_mm = sw / mw
        label = f"screen_{int(round(mw))}x{int(round(mh))}mm"
        if abs(px_per_mm - sh / mh) / px_per_mm > 0.02:
            print(f"提示：横纵 px/mm 不一致（{px_per_mm:.3f} vs {sh / mh:.3f}），"
                  f"请核对 --screen-px 与 --screen-mm 是否为同一块屏。", file=sys.stderr)
    else:
        paper = args.paper or "A4"
        page_mm = PAPERS[paper]
        px_per_mm = DPI / 25.4
        label = paper
        source = "paper"

    # 自动选格边长 / 网格：板子尽量占满画布（留 8mm 边距 + 24mm 标尺）
    square_mm = args.square_mm
    if square_mm is None:
        square_mm = 30.0 if source == "paper" else 25.0
        while square_mm > 8.0:
            c = int((page_mm[0] - 16) // square_mm)
            r = int((page_mm[1] - 40) // square_mm)
            if c >= 5 and r >= 5:
                break
            square_mm -= 1.0
    if args.squares:
        cols, rows = (int(v) for v in args.squares.lower().split("x"))
    else:
        cols = int((page_mm[0] - 16) // square_mm)
        rows = int((page_mm[1] - 40) // square_mm)

    page, params = build_board(page_mm, px_per_mm, cols, rows, square_mm,
                               args.marker_ratio, args.dictionary, label, source)
    png = out / f"charuco_{label}.png"
    cv2.imwrite(str(png), page)
    (out / f"charuco_{label}.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"已生成 {png}")
    print(f"  画布 {params['canvas_mm'][0]:.0f}×{params['canvas_mm'][1]:.0f}mm  "
          f"棋盘 {cols}×{rows} 格  格边长 {square_mm:.1f}mm  码边长 {params['marker_mm']}mm")
    print(f"  板实际尺寸 {params['board_mm'][0]:.0f}×{params['board_mm'][1]:.0f}mm  "
          f"图像 {params['image_px'][0]}×{params['image_px'][1]}px  "
          f"({px_per_mm:.3f} px/mm)")
    print(f"  参数写入 {out / f'charuco_{label}.json'}")
    print(f"  {params['note']}")

    if args.verify:
        print("\n合成回环验证：")
        return 0 if verify(params, page) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
