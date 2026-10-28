"""Measured camera runtime for the existing two-view analyser; never controls a robot.

Fit results are tied to the masks that produced them, not the latest video frame.
A changed scene or expired source invalidates both the overlay and published targets.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import cv2
import numpy as np

from scripts.vision_workspace import work_area_polygon_px, workspace_source
from scripts.live_ui import (atomic_json, build_panorama, build_workspace_view,
                             choose_exposure, clamp_exposure, open_cameras, read_pair,
                             windows_gone)

NAMES = {
    "ball": "球体", "cube": "正方体", "cuboid": "长方体", "cylinder": "圆柱",
    "cone": "圆锥", "regular_tetrahedron": "正四面体", "triangular_pyramid": "三棱锥",
    "square_pyramid": "四棱锥", "pentagonal_pyramid": "五棱锥", "hexagonal_pyramid": "六棱锥",
    "triangular_prism": "三棱柱", "pentagonal_prism": "五棱柱", "hexagonal_prism": "六棱柱",
}


def mask_signature(masks):
    return {role: cv2.resize((np.asarray(mask) > 0).astype(np.uint8), (320, 240),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            for role, mask in masks.items()}


def scene_matches(source, current, minimum_iou=0.92):
    if source is None or set(source) != set(current):
        return False
    for role in source:
        a, b = source[role], current[role]
        if a.shape != b.shape:
            return False
        union = np.count_nonzero(a | b)
        if union and np.count_nonzero(a & b) / union < minimum_iou:
            return False
        # Global IoU alone can hide a small disappearing object next to a big one.
        for old, new in ((a, b), (b, a)):
            count, labels, stats, _ = cv2.connectedComponentsWithStats(old.astype(np.uint8))
            for index in range(1, count):
                area = int(stats[index, cv2.CC_STAT_AREA])
                if area >= 12 and np.count_nonzero(new[labels == index]) / area < .85:
                    return False
    return True


def current_result(source, current, source_time, now, max_age):
    if source is None:
        return "pending"
    if now - source_time > max_age:
        return "expired"
    return "current" if scene_matches(source, current) else "scene_changed"


def segment_pair(api, frames, refs, cal, threshold, min_area, fixture=None):
    masks = {role: (api.foreground(frame, refs[role], threshold, min_area)
                    if refs is not None else api.adaptive_foreground(frame, None, min_area))
             for role, frame in frames.items()}
    roi = np.zeros(masks["vertical"].shape, np.uint8)
    cv2.fillConvexPoly(roi, np.rint(work_area_polygon_px(cal, roi.shape)).astype(np.int32), 1)
    structure = api.structure_foreground(frames["vertical"], roi)
    if fixture is not None and fixture.shape == roi.shape:
        structure &= (fixture == 0).astype(np.uint8)
    masks["vertical"] = ((masks["vertical"] | structure) & roi).astype(np.uint8)
    return masks




def object_is_current(item, masks):
    """Match the actual object's full contour, not total foreground area."""
    top = np.zeros(masks["vertical"].shape, np.uint8)
    cv2.drawContours(top, [item.contour_px], -1, 1, -1)
    sources = {"vertical": top}
    if item.association.mask is not None:
        sources["side_a"] = item.association.mask
    for role, source in sources.items():
        points = cv2.findNonZero((source > 0).astype(np.uint8))
        if points is None:
            return False
        x, y, w, h = cv2.boundingRect(points)
        x0, y0 = max(0, x-20), max(0, y-20)
        a = source[y0:y+h+20, x0:x+w+20] > 0
        b = masks[role][y0:y+h+20, x0:x+w+20] > 0
        union = np.count_nonzero(a | b)
        if not union or np.count_nonzero(a & b) / union < .92:
            return False
    return True


def display_name(item, chinese=True):
    confirmed = item.label in NAMES
    shape = item.label if confirmed else item.shape_hypothesis
    if shape not in NAMES:
        shape = "unclassified"
    if chinese:
        return ("确认:" if confirmed else "假设:") + NAMES.get(shape, "未分类")
    return ("confirmed:" if confirmed else "tentative:") + shape


def stamped_payload(api, detections, source_time, now, max_age):
    wall = time.time()
    source_wall = wall - max(0.0, now - source_time)
    payload = api.detections_to_payload(detections)
    for item in payload:
        item["observation"] = {"state": "current", "source_unix": source_wall,
                               "updated_unix": wall, "valid_until_unix": source_wall + max_age}
    return payload


def cache_scene_matches(source, current):
    """Seed reuse is not target validation; all eleven models still compete."""
    if source is None or set(source) != set(current):
        return False
    for role in source:
        union = np.count_nonzero(source[role] | current[role])
        if union and np.count_nonzero(source[role] & current[role]) / union < .80:
            return False
    return True


_FIT_CACHE = {}
_FIT_SIGNATURE = None
_FIT_COUNT = 0


def solve_frame(images, binary, signature, stamp, cal, options, root, started):
    # Separate process avoids GIL/array contention between capture and optimisation.
    from scripts import two_view_shape as analyser
    cv2.setNumThreads(1)
    global _FIT_SIGNATURE, _FIT_COUNT
    if not cache_scene_matches(_FIT_SIGNATURE, signature) or _FIT_COUNT % 10 == 0:
        _FIT_CACHE.clear()
    _FIT_SIGNATURE = signature
    _FIT_COUNT += 1
    begin = time.perf_counter()
    result = analyser.analyse_masks(binary["vertical"], binary["side_a"], cal,
                                   image_vertical=images["vertical"], fit_cache=_FIT_CACHE, **options)
    elapsed = (time.perf_counter() - begin) * 1000
    if root:
        with (root / "fits.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps({"source_s": stamp - started,
                "completed_s": time.perf_counter() - started, "fit_ms": elapsed,
                "detections": analyser.detections_to_payload(result)}, ensure_ascii=False,
                allow_nan=False) + "\n")
    return result, signature, stamp, elapsed


def run_live(api, args, cal):
    """Capture continuously while one worker runs the unchanged geometry solver."""
    refs = {role: cv2.imread(str(args.ref_dir / f"ref_{role}.jpg")) for role, _ in api.ROLES}
    if any(value is None for value in refs.values()):
        raise ValueError("Two readable reference frames are required; no silent fallback")
    fixture_path = args.ref_dir / "ref_static_vertical.png"
    fixture = cv2.imread(str(fixture_path), 0) if fixture_path.exists() else None
    if fixture is not None:
        fixture = cv2.dilate((fixture > 0).astype(np.uint8), np.ones((15, 15), np.uint8))
    root = args.session_dir
    if root:
        root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    wall_started = time.time()
    chinese = api.overlay_text.available()
    window_panorama = "2 全景（俯视 | 侧视）  q 退出  s 存图  e 重新选曝光"
    window_workspace = "3 识别区域（四点工作区内）"
    cv2.setNumThreads(1)
    executor = ProcessPoolExecutor(max_workers=1)
    pending = None
    detections, source_signature, source_time = [], None, 0.0
    last_fit_start, fit_ms, last_publish, last_sample = -1e9, 0.0, -1e9, -1e9
    sample_index, frame_count = 0, 0
    live_json = api.ROOT / "captures" / "preview" / "live_check" / "live_result.json"
    status_json = live_json.with_name("live_status.json")
    stream = None
    atomic_json(live_json, [])
    atomic_json(status_json, {"state": "starting", "updated_unix": time.time()})
    try:
        exposure = clamp_exposure(args.exposure)
        caps, resolved = open_cameras(api, exposure)
        if not args.skip_exposure_picker:
            chosen = choose_exposure(api, caps, cal, refs, fixture, exposure, args, chinese)
            caps, refs, fixture = chosen["caps"], chosen["refs"], chosen["fixture"]
            exposure = chosen["exposure"]
            if chosen["aborted"]:
                return 0
        if root:
            atomic_json(root / "session.json", {"cameras": resolved,
                        "calibration": str(args.calibration.resolve()),
                        "references": str(args.ref_dir.resolve()),
                        "started_unix": wall_started, "chinese_font": chinese,
                        "exposure": exposure,
                        "max_result_age_s": args.max_result_age,
                        "workspace_source": workspace_source(cal),
                        "workspace_polygon_px": work_area_polygon_px(cal).tolist()})
            stream = (root / "telemetry.jsonl").open("a", encoding="utf-8")
        print(json.dumps({"event": "cameras_open", "cameras": resolved, "exposure": exposure},
                         ensure_ascii=False), flush=True)
        cv2.namedWindow(window_panorama, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_workspace, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_panorama, 1400, 525)
        cv2.resizeWindow(window_workspace, 1000, 720)
        cv2.moveWindow(window_panorama, 0, 40)
        cv2.moveWindow(window_workspace, 760, 40)
        # All fit results carry the exact acquisition timestamp and scene signature.
        last_read = time.perf_counter()
        while not args.seconds or time.perf_counter() - started < args.seconds:
            tick = time.perf_counter()
            # Sequential grab/retrieve limits call skew, not hardware exposure synchronisation.
            frames = read_pair(caps)
            acquired = time.perf_counter()
            masks = segment_pair(api, frames, refs, cal, args.thr, args.min_area, fixture)
            sig = mask_signature(masks)
            now = time.perf_counter()
            if pending is not None and pending.done():
                try:
                    detections, source_signature, source_time, fit_ms = pending.result()
                except Exception as exc:
                    detections, source_signature = [], None
                    print(json.dumps({"event": "fit_error", "error": str(exc)}), flush=True)
                pending = None
            state = current_result(source_signature, sig, source_time, now, args.max_result_age)
            visible = []
            if state == "current":
                visible = detections
            elif state == "scene_changed":
                from dataclasses import replace
                visible = [replace(item, pick_candidate=False, rank=0, robot_target_ready=False,
                           rejection_reasons=item.rejection_reasons + ("场景变化：暂停抓取，等待重新拟合",))
                           for item in detections if object_is_current(item, masks)]
                if visible:
                    state = "partial_current"
            if pending is None and now - last_fit_start >= max(0.0, args.fit_interval):
                last_fit_start = now
                opening = ((args.gripper_min_opening_mm, args.gripper_max_opening_mm)
                           if args.gripper_min_opening_mm is not None else None)
                options = dict(min_area=args.min_area, grasp_height_ratio=args.grasp_height_ratio,
                               min_clearance_mm=args.min_clearance_mm,
                               gripper_opening_range_mm=opening,
                               gripper_clearance_mm=args.gripper_clearance_mm,
                               approach_clearance_mm=args.approach_clearance_mm)
                pending = executor.submit(solve_frame, frames, masks, sig, acquired,
                                          cal, options, root, started)
            work_polygon = work_area_polygon_px(cal, frames["vertical"].shape)
            workspace_label = ("手动四点工作区" if chinese else "manual 4-point workspace") \
                if workspace_source(cal).startswith("manual:") else \
                ("黑点平底工作区" if chinese else "calibrated floor workspace")
            panorama = build_panorama(api, frames, masks, work_polygon, args.min_area,
                                      workspace_label, chinese)
            head = (f"{state} | fit {fit_ms:.0f}ms | 曝光 {exposure:+.1f} | "
                    f"源帧 {now - source_time:.1f}s 前" if source_signature else
                    "正在拟合第一帧，暂无有效目标")
            status_lines = [head]
            for item in visible:
                reasons = "; ".join(item.rejection_reasons)
                status_lines.append(f"{display_name(item, chinese)}: {reasons or '可抓候选'}")
            workspace_view = build_workspace_view(api, frames, masks, visible, work_polygon,
                                                  args.min_area, chinese, status_lines,
                                                  display_name)
            cv2.imshow(window_panorama, panorama)
            cv2.imshow(window_workspace, workspace_view)
            count = int(sum(cv2.contourArea(c) >= args.min_area for c in cv2.findContours(masks['vertical'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]))
            telemetry = {"elapsed_s": now - started, "frame": frame_count, "state": state,
                         "segmentation_count": count, "fit_ms": fit_ms,
                         "loop_ms": (time.perf_counter() - tick) * 1000,
                         "frame_gap_ms": (acquired - last_read) * 1000,
                         "source_age_s": now-source_time if source_signature else None,
                         "updated_unix": time.time(),
                         "detections": stamped_payload(api, visible, source_time, now, args.max_result_age)}
            if now - last_publish >= 0.5:
                atomic_json(live_json, telemetry["detections"])
                atomic_json(status_json, {k: v for k, v in telemetry.items() if k != "detections"})
                if stream:
                    stream.write(json.dumps(telemetry, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                if args.jsonl:
                    print(json.dumps(telemetry, ensure_ascii=False, allow_nan=False), flush=True)
                last_publish = now
            save = root and now - last_sample >= args.sample_interval and sample_index < args.sample_limit
            if save:
                while (root / "samples" / f"{sample_index:04d}").exists():
                    sample_index += 1        # 复用 --session-dir 时接着往后编号，不覆盖旧证据
                folder = root / "samples" / f"{sample_index:04d}"
                folder.mkdir(parents=True, exist_ok=False)
                for role in frames:
                    cv2.imwrite(str(folder / f"{role}.jpg"), frames[role], [cv2.IMWRITE_JPEG_QUALITY, 98])
                    cv2.imwrite(str(folder / f"{role}_mask.png"), masks[role] * 255)
                cv2.imwrite(str(folder / "panorama.jpg"), panorama)
                cv2.imwrite(str(folder / "workspace.jpg"), workspace_view)
                atomic_json(folder / "capture.json", telemetry)
                sample_index += 1
                last_sample = now
            key = cv2.waitKey(1) & 255
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                target = root or api.ROOT / "captures" / "preview"
                target.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target / f"panorama_{time.time_ns()}.jpg"), panorama)
                cv2.imwrite(str(target / f"workspace_{time.time_ns()}.jpg"), workspace_view)
            if key == ord('e'):
                chosen = choose_exposure(api, caps, cal, refs, fixture, exposure, args, chinese)
                caps, refs = chosen["caps"], chosen["refs"]
                if chosen["fixture"] is not None:
                    fixture = chosen["fixture"]
                exposure = chosen["exposure"]
                if chosen["aborted"]:
                    break
            if windows_gone(window_panorama, window_workspace):
                break
            frame_count += 1
            last_read = acquired
        return 0
    finally:
        # Publish no target on shutdown, camera failure, or any unhandled exception.
        atomic_json(live_json, [])
        atomic_json(status_json, {"state": "stopped", "elapsed_s": time.perf_counter()-started})
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
        executor.shutdown(wait=True, cancel_futures=True)
        if stream:
            stream.close()
        if root:
            atomic_json(root / "summary.json", {"duration_s": time.perf_counter()-started,
                        "frames": frame_count, "samples": sample_index, "stopped_unix": time.time()})