"""在 OpenCV 图上写中文。

`cv2.putText` 只支持 ASCII，中文会整行变成 `??????`（本机窗口里实测如此），
所以这里优先用 Pillow + 系统字体（微软雅黑）绘制；Pillow 或字体缺失时退回
`cv2.putText`（中文仍会是 `??`，但至少不崩）。

用法：
    from 程序 import overlay_text
    img = overlay_text.draw_text(img, "已存 3 组  可以存（空格）", (10, 10), 24, (0, 220, 0))

注意 `org` 是文字**左上角**（Pillow 习惯），和 `cv2.putText` 的基线不同。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\Deng.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
)

_font_cache: dict[int, object] = {}

try:  # Pillow 可用时走真中文渲染
    from PIL import Image, ImageDraw, ImageFont
    _PIL = (Image, ImageDraw, ImageFont)
except Exception:  # pragma: no cover - 环境缺 Pillow 时的兜底
    _PIL = None


def _font(size: int):
    if _PIL is None:
        return None
    if size in _font_cache:
        return _font_cache[size]
    _, _, ImageFont = _PIL
    font = None
    for path in FONT_CANDIDATES:
        if path.exists():
            try:
                font = ImageFont.truetype(str(path), size)
                break
            except Exception:
                font = None
    _font_cache[size] = font
    return font


def available() -> bool:
    """True = 能正常渲染中文；False = 会退回 ASCII（中文变 ??）。"""
    return _PIL is not None and _font(20) is not None


def draw_text(img: np.ndarray, text: str, org: tuple[int, int], size: int = 24,
              color: tuple[int, int, int] = (0, 220, 0),
              shadow: bool = True) -> np.ndarray:
    """把 text 写到 img 上并返回新图（BGR）。org = 文字左上角。"""
    font = _font(size)
    if font is None:
        cv2.putText(img, text, (int(org[0]), int(org[1]) + size),
                    cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, color, 2)
        return img
    Image, ImageDraw, _ = _PIL
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    drawer = ImageDraw.Draw(pil)
    if shadow:
        drawer.text((org[0] + 1, org[1] + 1), text, font=font, fill=(0, 0, 0))
    drawer.text((org[0], org[1]), text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
