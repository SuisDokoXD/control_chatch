"""启动流程与显示层：1) 选光照  2) 全景 + 识别区域  3) 识别。

这里只做采集、曝光选择与窗口排版；分割阈值、模型拟合和接受门槛都不在这里。
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np


EXP_LOW, EXP_HIGH, EXP_STEP = -9.0, -1.0, 0.5


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def clamp_exposure(value, low=EXP_LOW, high=EXP_HIGH, step=EXP_STEP):
    value = min(high, max(low, float(value)))
    return float(np.round(value / step) * step)


def next_exposure(current, key, step=EXP_STEP):
    """']' 更亮（数值变大），'[' 更暗；方向来自实测 -3 过曝、-5 偏暗。"""
    if key in (ord("["), ord(","), ord("-")):
        return clamp_exposure(current - step)
    if key in (ord("]"), ord("."), ord("=")):
        return clamp_exposure(current + step)
    return clamp_exposure(current)


def brightness_stats(gray):
    values = np.asarray(gray, np.uint8)
    return {"mean": float(values.mean()),
            "over_pct": float((values >= 240).mean() * 100.0),
            "under_pct": float((values <= 35).mean() * 100.0)}


def photometric_offset(current, reference):
    """只读估计：当前画面相对空盘参照的整体亮度漂移（灰阶）。"""
    now = cv2.resize(current, (320, 240), interpolation=cv2.INTER_AREA).astype(np.float32)
    before = cv2.resize(reference, (320, 240), interpolation=cv2.INTER_AREA).astype(np.float32)
    delta = now - before
    valid = ((now.min(axis=2) > 120) & (before.min(axis=2) > 120)
             & (now.max(axis=2) < 245) & (before.max(axis=2) < 245)
             & (np.abs(delta).max(axis=2) < 60))
    if np.count_nonzero(valid) < 500:
        return None
    return float(np.median(delta[valid]))


def work_crop_rect(polygon, shape, margin_px=24):
    """识别工作区的外接矩形，裁剪到画面内。"""
    x, y, w, h = cv2.boundingRect(np.asarray(polygon, np.int32))
    if w < 8 or h < 8:
        raise ValueError("Recognition workspace is degenerate")
    x0, y0 = max(0, x - margin_px), max(0, y - margin_px)
    x1, y1 = min(shape[1], x + w + margin_px), min(shape[0], y + h + margin_px)
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError("Recognition workspace crop is degenerate")
    return int(x0), int(y0), int(x1), int(y1)


def apply_exposure(caps, exposure):
    for cap in caps.values():
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))


def windows_gone(*names):
    """只有窗口真的被销毁才算关闭；最小化或被移到屏幕外不算。"""
    alive = 0
    for name in names:
        try:
            cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
            alive += 1
        except cv2.error:
            continue
    return alive == 0


def read_pair(caps):
    grabs = {role: cap.grab() for role, cap in caps.items()}
    frames = {}
    for role, cap in caps.items():
        ok, image = cap.retrieve() if grabs[role] else (False, None)
        if not ok or image is None:
            raise RuntimeError(f"Camera read failed: {role}")
        frames[role] = image
    return frames


def open_cameras(api, exposure):
    settings, roles = api.camera_config.load_settings(), api.camera_config.load_roles()
    caps, resolved = {}, {}
    for role, _ in api.ROLES:
        config = api.camera_config.resolve(settings, role, roles.get(role, {}).get("index"))
        config.update(auto_exposure=False, exposure=exposure)
        cap = api.open_camera(config)
        if cap is None:
            raise RuntimeError(f"Cannot open camera {role}")
        caps[role] = cap
        resolved[role] = dict(config, actual_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                              actual_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                              actual_exposure=cap.get(cv2.CAP_PROP_EXPOSURE),
                              actual_fourcc=int(cap.get(cv2.CAP_PROP_FOURCC)))
    return caps, resolved


def recapture_references(api, args, cal, caps, exposure):
    """复用已验收的 --capture-ref 路径；结束后重开相机。"""
    for cap in caps.values():
        cap.release()
    caps.clear()
    try:
        references, lasts = api.capture_reference(exposure)
    except Exception as exc:                     # 相机异常要原样报出，不吞
        new_caps, _ = open_cameras(api, exposure)
        return False, f"采集失败：{exc}", None, None, new_caps
    big, small, structure = api.reference_self_check(
        lasts["vertical"], references["vertical"],
        api.tray_floor_polygon_px(cal),
        args.thr, args.min_area, args.fixture_max_area)
    if big:
        new_caps, _ = open_cameras(api, exposure)
        return False, f"平底内仍有物块 {big}，未写入参照帧；清空托盘后按 R 重试", None, None, new_caps
    args.ref_dir.mkdir(parents=True, exist_ok=True)
    for role in references:
        cv2.imwrite(str(args.ref_dir / f"ref_{role}.jpg"), references[role],
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(args.ref_dir / "ref_static_vertical.png"), structure * 255)
    atomic_json(args.ref_dir / "ref_meta.json",
                {"exposure": exposure, "captured_unix": time.time(),
                 "static_structures_px": small})
    fixture = (cv2.dilate((structure > 0).astype(np.uint8), np.ones((15, 15), np.uint8))
               if structure.any() else None)
    new_caps, _ = open_cameras(api, exposure)
    return True, "参照帧已更新（空盘自检通过）", references, fixture, new_caps


def choose_exposure(api, caps, cal, refs, fixture, exposure, args, chinese=True):
    """第一步：按当前光照选曝光。返回该步之后的会话资源。"""
    window = "1 选择光照 | [ 更暗  ] 更亮  Enter 确认  R 重采空盘  Q 退出"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1440, 660)
    exposure = clamp_exposure(exposure)
    apply_exposure(caps, exposure)
    message = ("用 [ ] 调到托盘不过曝、物块边缘清楚，再按 Enter" if chinese
               else "Adjust with [ ] until the tray is not blown out, then press Enter")
    while True:
        frames = read_pair(caps)
        panes = []
        for role in ("vertical", "side_a"):
            stats = brightness_stats(cv2.cvtColor(
                cv2.resize(frames[role], (480, 360), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY))
            pane = cv2.resize(frames[role], (700, 525), interpolation=cv2.INTER_AREA)
            good = stats["over_pct"] < 2.0 and 110.0 < stats["mean"] < 205.0
            color = (0, 220, 0) if good else (0, 180, 255)
            pane = api.overlay_text.draw_text(
                pane, f"{role}  均值 {stats['mean']:.0f}  过曝 {stats['over_pct']:.1f}%  "
                      f"过暗 {stats['under_pct']:.1f}%", (10, 8), 20, color)
            offset = photometric_offset(frames[role], refs[role])
            if offset is not None and abs(offset) > 8.0:
                pane = api.overlay_text.draw_text(
                    pane, f"比空盘参照{'暗' if offset < 0 else '亮'} {abs(offset):.0f} 灰阶，"
                          f"建议按 R 重采空盘参照", (10, 36), 18, (0, 90, 255))
            panes.append(pane)
        strip = np.full((128, 1400, 3), 22, np.uint8)
        strip = api.overlay_text.draw_text(
            strip, f"曝光 {exposure:+.1f}    [ 更暗    ] 更亮    Enter 确认使用",
            (12, 6), 26, (255, 255, 255))
        strip = api.overlay_text.draw_text(
            strip, ("目标：过曝 < 2%，均值 110~205；换灯、换机器后都要重选" if chinese
                    else "Aim for <2% overexposure, mean 110-205"), (12, 44), 18, (190, 190, 190))
        strip = api.overlay_text.draw_text(strip, message, (12, 76), 18, (0, 210, 255))
        banner = np.vstack([np.hstack(panes), strip])
        cv2.imshow(window, cv2.resize(
            banner, (1440, round(banner.shape[0] * 1440 / banner.shape[1]))))
        key = cv2.waitKey(1) & 255
        if key in (13, 10, 32):
            cv2.destroyWindow(window)
            return {"exposure": exposure, "caps": caps, "refs": refs,
                    "fixture": fixture, "aborted": False}
        if key in (ord("q"), 27):
            cv2.destroyWindow(window)
            return {"exposure": exposure, "caps": caps, "refs": refs,
                    "fixture": fixture, "aborted": True}
        if key == ord("r"):
            ok, text_, new_refs, new_fixture, new_caps = recapture_references(
                api, args, cal, caps, exposure)
            caps = new_caps
            if ok:
                refs, fixture = new_refs, new_fixture
            message = text_ if ok else f"重采失败：{text_}"
            continue
        updated = next_exposure(exposure, key)
        if updated != exposure:
            exposure = updated
            apply_exposure(caps, exposure)
            message = "已应用新曝光" if chinese else "exposure applied"


def build_panorama(api, frames, masks, work_polygon, min_area, workspace_label,
                   chinese, name_fn=None):
    """窗口 2：完整画面（俯视 + 侧视），不做识别区域裁剪。"""
    pane_w, pane_h = 700, 525
    panes = []
    for role in ("vertical", "side_a"):
        pane = cv2.resize(frames[role], (pane_w, pane_h), interpolation=cv2.INTER_AREA)
        scale = np.array([pane_w / frames[role].shape[1], pane_h / frames[role].shape[0]])
        contours, _ = cv2.findContours(masks[role], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(pane, [np.rint(contour * scale).astype(np.int32)],
                                 -1, (0, 180, 255), 1)
        panes.append(pane)
    vertical, side = panes
    scale = np.array([pane_w / frames["vertical"].shape[1],
                      pane_h / frames["vertical"].shape[0]])
    cv2.polylines(vertical, [np.rint(work_polygon * scale).astype(np.int32)], True,
                  (255, 220, 0), 2)
    vertical = api.overlay_text.draw_text(
        vertical, "全景·俯视" if chinese else "full·top", (8, 4), 20, (255, 220, 0))
    side = api.overlay_text.draw_text(
        side, "全景·侧视" if chinese else "full·side", (8, 4), 20, (255, 220, 0))
    vertical = api.overlay_text.draw_text(vertical, workspace_label, (8, 30), 16, (255, 220, 0))
    return np.hstack([vertical, side])


def build_workspace_view(api, frames, masks, visible, work_polygon, min_area, chinese,
                         status_lines, name_fn):
    """窗口 3：按四点工作区裁剪后的识别画面。"""
    x0, y0, x1, y1 = work_crop_rect(work_polygon, frames["vertical"].shape)
    crop = frames["vertical"][y0:y1, x0:x1].copy()
    offset = np.array([x0, y0], np.int32)
    cv2.polylines(crop, [np.rint(work_polygon).astype(np.int32) - offset], True,
                  (255, 220, 0), 3)
    contours, _ = cv2.findContours(masks["vertical"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(crop, [contour - offset], -1, (0, 180, 255), 2)
    for item in visible:
        color = (0, 230, 0) if item.geometry_verified else (0, 165, 255)
        cv2.drawContours(crop, [item.contour_px - offset], -1, color, 3)
        x, y, _, _ = cv2.boundingRect(item.contour_px - offset)
        crop = api.overlay_text.draw_text(
            crop, f"#{item.rank or '-'} {name_fn(item, chinese)}",
            (max(0, x), max(20, y - 26)), 26, color)
    scale = 1000.0 / max(1, crop.shape[1])
    if crop.shape[0] * scale > 780:
        scale = 780.0 / crop.shape[0]
    crop = cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)),
                      interpolation=cv2.INTER_AREA)
    crop = api.overlay_text.draw_text(
        crop, "识别区域（四点工作区内）" if chinese else "recognition area (inside 4 points)",
        (8, 4), 18, (255, 220, 0))
    width = max(int(crop.shape[1]), 900)
    if crop.shape[1] < width:
        crop = cv2.copyMakeBorder(crop, 0, 0, 0, width - crop.shape[1],
                                  cv2.BORDER_CONSTANT, value=(22, 22, 22))
    strip = np.full((36 * max(1, len(status_lines)) + 12, width, 3), 22, np.uint8)
    for index, line in enumerate(status_lines):
        strip = api.overlay_text.draw_text(
            strip, line, (10, 6 + index * 36), 18,
            (255, 255, 255) if index == 0 else (0, 200, 255))
    return np.vstack([crop, strip])