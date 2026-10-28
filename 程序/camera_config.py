"""相机配置解析：支持"公共默认值 + 每台相机覆盖"。

配置结构（`配置/camera_settings.json`）：

    {
      "width": 2592, "height": 1944, "fps": null, "fourcc": "MJPG",
      "auto_exposure": false, "exposure": -7,
      "auto_white_balance": true, "white_balance": null,
      "per_camera": {
        "vertical": {"index": 2, "exposure": -7},
        "side_a":   {"index": 1, "exposure": -7}
      }
    }

规则：
- 顶层键 = 所有相机的默认值；`per_camera.<角色>` 里的键只覆盖那一台。
- 键**不存在** → 用 `DEFAULTS`；
- 键**显式为 null** → 解析结果也是 None，意思是"不要去设置这一项"（例如这两台相机
  的 `CAP_PROP_FPS` 被驱动忽略、手动白平衡温度完全无效，就把它们设成 null 跳过）。
- 老格式（没有 per_camera）照旧可用，所以不会把已有脚本改坏。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CAMERA_SETTINGS_PATH = ROOT / "配置" / "camera_settings.json"
CAMERA_ROLES_PATH = ROOT / "配置" / "camera_roles.json"

# 每台相机可以被单独覆盖的键
PER_CAMERA_KEYS = (
    "index", "width", "height", "fps", "fourcc",
    "auto_exposure", "exposure", "gain",
    "auto_white_balance", "white_balance",
)

DEFAULTS: dict[str, Any] = {
    "width": 2592, "height": 1944, "fps": None, "fourcc": "MJPG",
    "auto_exposure": False, "exposure": None, "gain": None,
    "auto_white_balance": True, "white_balance": None,
}


def load_json(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_settings(path: Path | None = None) -> dict[str, Any]:
    return load_json(path or CAMERA_SETTINGS_PATH)


def load_roles(path: Path | None = None) -> dict[str, Any]:
    return load_json(path or CAMERA_ROLES_PATH)


def _pick(container: dict[str, Any], key: str, default: Any) -> Any:
    """键不存在 → 用默认值；键存在但显式为 null → 返回 None（表示"不要设置这一项"）。"""
    if key not in container:
        return default
    return container[key]


def resolve(settings: dict[str, Any], role: str | None = None,
            index: int | None = None) -> dict[str, Any]:
    """把顶层默认值 + `per_camera.<role>` 合并成一台相机实际使用的参数。

    `index` 显式给出时优先级最高（命令行覆盖）。
    """
    resolved = {k: _pick(settings, k, v) for k, v in DEFAULTS.items()}
    if role:
        override = (settings.get("per_camera") or {}).get(role) or {}
        for key in PER_CAMERA_KEYS:
            if key in override and override[key] is not None:
                resolved[key] = override[key]
    if index is not None:
        resolved["index"] = index
    return resolved


def describe(resolved: dict[str, Any]) -> str:
    wb = "自动" if resolved.get("auto_white_balance") else f"{resolved.get('white_balance')}K"
    exp = "自动" if resolved.get("auto_exposure") else str(resolved.get("exposure"))
    fps = f"@{resolved['fps']}fps" if resolved.get("fps") else "（不设帧率）"
    return (f"编号 {resolved.get('index')}  {resolved['width']}x{resolved['height']}"
            f"{fps}  {resolved['fourcc']}  曝光 {exp}  白平衡 {wb}")
