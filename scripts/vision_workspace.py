"""Image detection workspace is independent of the calibrated metric floor.

outer_corners_px are the user's saved manual top-view workspace. They are NOT
substitutes for the black-dot pose or pixel-to-millimetre homography.
"""
from __future__ import annotations

import cv2
import numpy as np

CORNER_ORDER = ("TL", "TR", "BR", "BL")


def work_area_polygon_px(cal: dict, image_shape=None) -> np.ndarray:
    tray = cal["tray"]
    # An invalid manual selection must fail visibly, never silently choose another ROI.
    key = "outer_corners_px" if "outer_corners_px" in tray else "corners_px"
    try:
        polygon = np.array([tray[key][name] for name in CORNER_ORDER], np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid saved workspace: tray.{key}") from exc
    if (polygon.shape != (4, 2) or not np.all(np.isfinite(polygon))
            or not cv2.isContourConvex(polygon.reshape(-1, 1, 2))
            or abs(cv2.contourArea(polygon)) < 16):
        raise ValueError(f"Workspace must be four finite, ordered convex corners: tray.{key}")
    if image_shape is not None:
        height, width = image_shape[:2]
        if (np.any(polygon < 0) or np.any(polygon[:, 0] > width - 1)
                or np.any(polygon[:, 1] > height - 1)):
            raise ValueError("Saved workspace does not fit the current image size")
    return polygon


def workspace_source(cal: dict) -> str:
    return ("manual:tray.outer_corners_px" if "outer_corners_px" in cal["tray"]
            else "calibrated:tray.corners_px")