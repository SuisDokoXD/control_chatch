"""逐个打开摄像头编号，抓一帧存图，并生成一个对比页面让用户确认哪个是哪个视角。

用法：
    python scripts/probe_cameras_visual.py --max-index 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "captures" / "camera_probe"
SETTINGS = ROOT / "配置" / "camera_settings.json"


def load_settings() -> dict:
    if SETTINGS.exists():
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    return {}


def grab(index: int, settings: dict, warmup: int = 12):
    """尽量按配置分辨率打开并抓帧；失败就退回默认分辨率。"""
    want_w = int(settings.get("width") or 0)
    want_h = int(settings.get("height") or 0)
    fourcc = str(settings.get("fourcc") or "MJPG")
    for backend, tag in ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "ANY")):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        if want_w and want_h:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
        frame = None
        for _ in range(warmup):
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
            time.sleep(0.03)
        info = {
            "backend": tag,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": round(float(cap.get(cv2.CAP_PROP_FPS)), 1),
            "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
        cap.release()
        if frame is not None:
            return frame, info
    return None, None


def fourcc_text(value: int) -> str:
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="抓帧确认摄像头视角")
    parser.add_argument("--max-index", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    settings = load_settings()

    cards = []
    for index in range(args.max_index + 1):
        frame, info = grab(index, settings)
        if frame is None:
            print(f"编号 {index}: 打不开")
            continue
        name = f"index_{index}.jpg"
        cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"编号 {index}: {info['width']}x{info['height']} @{info['fps']}fps "
              f"backend={info['backend']} fourcc={fourcc_text(info['fourcc'])} -> {name}")
        cards.append({"index": index, "name": name, **info})

    rows = "\n".join(
        f'<figure><img src="{c["name"]}"><figcaption><b>编号 {c["index"]}</b>　'
        f'{c["width"]}×{c["height"]}　{c["fps"]}fps　{c["backend"]}　'
        f'{fourcc_text(c["fourcc"])}</figcaption></figure>'
        for c in cards
    )
    html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>摄像头视角确认</title>
<style>
 body {{ margin:0; background:#14171a; color:#e8eaed; font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif; }}
 h1 {{ font-size:17px; margin:12px 16px 4px; }}
 p {{ margin:0 16px 12px; color:#9aa0a6; }}
 figure {{ margin:0 16px 20px; }}
 img {{ width:min(88vw,1200px); display:block; border-radius:8px; border:1px solid #2c3238; }}
 figcaption {{ margin-top:6px; color:#8ab4f8; }}
</style></head><body>
<h1>哪个是垂直相机、哪个是侧视相机？</h1>
<p>下面每个编号对应一个摄像头。请告诉我：垂直（俯视托盘）= 编号几，侧视 = 编号几。</p>
{rows}
</body></html>
"""
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"\n对比页面：{out / 'index.html'}")
    return 0 if cards else 1


if __name__ == "__main__":
    raise SystemExit(main())
