"""Offline fitting of upright, complete convex solids to TWO full silhouettes.

No camera access, image rasterization, or dependence on ``two_view_shape``.
All geometry is in tray millimetres; reported heights are positive. Only when
projecting do we convert height to raw tray z = sign(camera_tray_z) * height.
The sphere centre is (x, y, diameter/2); center_mm reports its ground projection.

Method references (official documentation):
* OpenCV projectPoints / solvePnP:
  https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
* SciPy bounded, numerical-Jacobian least_squares:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

Every residual evaluation projects a real 3D solid and takes its convex image
boundary. Fixed-count arc-length samples are compared to polygon SEGMENTS in
both directions, in both views (continuous subpixel distances, never a raster
mask). Multiple yaw seeds and bounded trust-region least squares are used.
This is deliberately an OFFLINE classifier, not a real-time detection loop.

Assumptions: calibrated synchronised views of one isolated, upright solid with
its flat regular base on z=0 (sphere tangent to z=0), no occlusion, no truncation.
A low absolute error AND separation from the next geometrically distinct model
are required. Cube/cuboid and regular_tetrahedron/triangular_pyramid are nested
families, so their aliases are not counted as independent runners-up. A pair of
silhouettes cannot prove absence of occlusion or distinguish every possible
solid: ambiguous or poorly explained inputs are rejected, not guaranteed
recognised. Strong non-convex lens distortion is outside this convex-boundary
model. Size is only an initial guess, never a measured silhouette cross-section.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

__all__ = ["fit_two_view"]

_SAMPLES = 64
_CONTACT_TRIM = 0.12   # ignore the lowest 12% of the side silhouette (contact/shadow)
_RING = np.column_stack((np.cos(np.linspace(0, 2*np.pi, 128, endpoint=False)),
                         np.sin(np.linspace(0, 2*np.pi, 128, endpoint=False))))
_FAMILIES = ("ball", "cuboid", "cylinder", "cone", "triangular_pyramid",
             "square_pyramid", "pentagonal_pyramid", "hexagonal_pyramid",
             "triangular_prism", "pentagonal_prism", "hexagonal_prism")
_SIDES = {"triangular": 3, "square": 4, "pentagonal": 5, "hexagonal": 6}


def _hull(points):
    """Find topology in float32, but preserve double precision for derivatives."""
    p = np.asarray(points, np.float64).reshape(-1, 2)
    indices = cv2.convexHull(p.astype(np.float32), returnPoints=False)
    return p[indices.ravel()]


def _sample(poly, count=_SAMPLES):
    edges = np.roll(poly, -1, axis=0) - poly
    lengths = np.linalg.norm(edges, axis=1)
    keep = lengths > 1e-10
    poly, edges, lengths = poly[keep], edges[keep], lengths[keep]
    cumulative = np.r_[0., np.cumsum(lengths)]
    positions = np.arange(count) * cumulative[-1] / count
    i = np.minimum(np.searchsorted(cumulative, positions, side="right") - 1,
                   len(lengths) - 1)
    return poly[i] + edges[i] * ((positions-cumulative[i])/lengths[i])[:, None]


def _segment_vectors(points, poly):
    edges = np.roll(poly, -1, axis=0) - poly
    delta = points[:, None, :] - poly[None, :, :]
    t = np.clip(np.sum(delta * edges[None, :, :], axis=2)
                / np.maximum(np.sum(edges*edges, axis=1), 1e-16), 0., 1.)
    vectors = delta - t[:, :, None] * edges[None, :, :]
    nearest = np.argmin(np.sum(vectors*vectors, axis=2), axis=1)
    return vectors[np.arange(len(points)), nearest]


def _contour(value):
    p = np.asarray(value, np.float64)
    if p.size < 6 or p.size % 2 or not np.all(np.isfinite(p)):
        raise ValueError("A finite polygon with at least three points is required")
    p = p.reshape(-1, 2)
    if len(p) > 1 and np.linalg.norm(p[0]-p[-1]) < 1e-9:
        p = p[:-1]
    area = abs(cv2.contourArea(p.astype(np.float32)))
    hull = _hull(p)
    hull_area = abs(cv2.contourArea(hull.astype(np.float32)))
    if hull_area < 16 or area/hull_area < .97:
        raise ValueError("Degenerate or non-convex/occluded contour")
    # Simplify measured segments below sensor precision; do not convexify them.
    # This bounds reverse-distance cost even for thousands of input pixels.
    boundary = cv2.approxPolyDP(p.astype(np.float32), .025, True).reshape(-1, 2)
    return boundary.astype(np.float64), _sample(p)


def _cameras(cal):
    w, h = np.asarray(cal["tray"]["size_mm"], np.float64)
    corners = np.array([cal["tray"]["corners_px"][k]
                        for k in ("TL", "TR", "BR", "BL")], np.float64)
    intrinsic = cal["intrinsics"]
    kv = np.asarray(intrinsic["vertical"]["mtx"], np.float64)
    dv = np.asarray(intrinsic["vertical"]["dist"], np.float64)
    ks = np.asarray(intrinsic["side_a"]["mtx"], np.float64)
    ds = np.asarray(intrinsic["side_a"]["dist"], np.float64)
    r = np.asarray(cal["extrinsics"]["R"], np.float64).reshape(3, 3)
    t = np.asarray(cal["extrinsics"]["T"], np.float64).reshape(3)
    arrays = (corners, kv, dv, ks, ds, r, t, np.array([w, h]))
    if not all(np.all(np.isfinite(a)) for a in arrays) or min(w, h) <= 0:
        raise ValueError("Invalid calibration")
    if (not np.allclose(r.T @ r, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(r), 1., atol=1e-4)):
        raise ValueError("Extrinsic R must be a rotation")
    world = np.array([[0., 0., 0.], [w, 0., 0.], [w, h, 0.], [0., h, 0.]])
    ok, rv, tv = cv2.solvePnP(world, corners, kv, dv, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("Tray pose unavailable")
    rt = cv2.Rodrigues(rv)[0]
    tv = tv.ravel()
    camera_z = (-rt.T @ tv)[2]
    if abs(camera_z) < 1e-6:
        raise ValueError("Camera lies in tray plane")
    up = float(np.sign(camera_z))
    cameras = [(rt, tv, kv, dv), (r @ rt, r @ tv + t, ks, ds)]
    # A zero/near-zero baseline does not provide independent height evidence.
    centres = [-a.T @ b for a, b, _, _ in cameras]
    if np.linalg.norm(centres[0]-centres[1]) < 1.:
        raise ValueError("Degenerate two-view baseline")
    return cameras, up, w, h


def _base(label, size):
    if label == "cuboid":
        l, w = size[:2]
        return np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
    if label in ("ball", "cylinder", "cone"):
        return _RING * size[0]/2
    n = _SIDES[label.split("_")[0]]
    # Parameter is circumdiameter; bases are genuinely regular, not stretched.
    angles = np.arange(n) * 2*np.pi/n
    return np.column_stack((np.cos(angles), np.sin(angles))) * size[0]/2


def _geometry(label, p):
    xy = p[:2]
    if label == "ball":
        size, height, yaw = [p[2]], p[2], 0.
    elif label == "cuboid":
        size, height, yaw = p[2:4], p[4], p[5]
    elif label in ("cylinder", "cone"):
        size, height, yaw = [p[2]], p[3], 0.
    else:
        size, height, yaw = [p[2]], p[3], p[4]
    c, s = np.cos(yaw), np.sin(yaw)
    base = _base(label, size) @ np.array([[c, s], [-s, c]]) + xy
    bottom = np.column_stack((base, np.zeros(len(base))))
    if label == "ball":
        return base, None, height, yaw
    if label == "cone" or label.endswith("pyramid"):
        vertices = np.vstack((bottom, [*xy, height]))
    else:
        vertices = np.vstack((bottom, bottom + [0., 0., height]))
    # Subdivide real 3D edges to account for moderate distortion in projected edges.
    if label == "cuboid" or label.endswith(("pyramid", "prism")):
        n = len(base)
        pairs = [(i, (i+1) % n) for i in range(n)]
        if label.endswith("pyramid"):
            pairs += [(i, n) for i in range(n)]
        else:
            pairs += [(i+n, (i+1) % n+n) for i in range(n)]
            pairs += [(i, i+n) for i in range(n)]
        fractions = np.array([.25, .5, .75])[:, None]
        midpoints = [vertices[i] + fractions*(vertices[j]-vertices[i]) for i, j in pairs]
        vertices = np.vstack([vertices, *midpoints])
    return base, vertices, height, yaw


def _projected(label, p, cameras, up):
    base, vertices, height, yaw = _geometry(label, p)
    polygons = []
    for rotation, translation, k, d in cameras:
        if label == "ball":
            # Exact 3D tangent circle for EACH perspective camera, not an equator
            # or a horizontal sphere cross-section. Only its angular sampling is discrete.
            centre = np.array([p[0], p[1], up*height/2])
            camera = -rotation.T @ translation
            v = camera-centre
            distance = np.linalg.norm(v)
            radius = height/2
            if distance <= radius:
                raise ValueError("Camera inside sphere")
            normal = v/distance
            axis = np.eye(3)[np.argmin(np.abs(normal))]
            u = np.cross(normal, axis)
            u /= np.linalg.norm(u)
            vv = np.cross(normal, u)
            ring_centre = centre + normal * radius**2/distance
            ring_radius = radius*np.sqrt(1-(radius/distance)**2)
            raw = ring_centre + ring_radius * (_RING[:, :1]*u + _RING[:, 1:]*vv)
        else:
            raw = vertices * [1., 1., up]
        camera_points = raw @ rotation.T + translation
        if np.min(camera_points[:, 2]) <= 1.:
            raise ValueError("Solid crosses camera plane")
        image = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3), k, d)[0]
        polygons.append(_hull(image))
    return polygons, base, height, yaw


def _residual(p, label, cameras, up, observed):
    polygons, _, _, _ = _projected(label, p, cameras, up)
    parts = []
    for model, (boundary, samples) in zip(polygons, observed):
        parts.extend((_segment_vectors(samples, model).ravel(),
                      _segment_vectors(_sample(model), boundary).ravel()))
    return np.concatenate(parts)


def fit_two_view(contour_vertical_px, contour_side_px, cal, initial_xy,
                 initial_size, h_max=120, *, warm_start=None) -> dict | None:
    """Fit calibrated full contours; return best fit including rejection evidence.

    ``initial_xy`` is (x,y) in tray mm. ``initial_size`` is a positive scalar
    diameter/edge or (length,width[,height]) in mm. Missing height starts at the
    smaller supplied dimension. A dict is returned for valid optimisations even
    if ``accepted`` is False; None means invalid/degenerate data or no solution.
    ``error_px`` is symmetric RMS Euclidean boundary distance (equal view weight);
    ``margin_px`` is runner-up error minus best error, after merging nested labels.
    ``polygon_mm`` is the fitted base/ground envelope, NOT an image silhouette
    mapped onto the tray. Sphere envelope is its ground disk, not its contact point.
    Ball/cylinder/cone yaw is 0; other yaws follow tray +x toward +y, modulo symmetry.
    No image dimensions or camera handles are required. Calibration errors,
    clipping and unsupported shapes must be handled upstream as well as by this
    conservative residual/margin gate; two silhouettes alone are not a proof.
    """
    try:
        observed = [_contour(contour_vertical_px), _contour(contour_side_px)]
        cameras, up, tray_w, tray_h = _cameras(cal)
        xy = np.asarray(initial_xy, np.float64).reshape(2)
        size = np.asarray(initial_size, np.float64).reshape(-1)
        h_max = float(h_max)
        if (size.size not in (1, 2, 3) or not np.all(np.isfinite(size))
                or np.min(size) <= 0 or not np.all(np.isfinite(xy))
                or not np.isfinite(h_max) or h_max <= 1.):
            return None
        length = float(size[0])
        width = float(size[1]) if size.size > 1 else length
        height = float(size[2]) if size.size > 2 else min(length, width)
        # Finite, generous physical bounds; perimeter/tray checks gate final fits.
        max_size = float(np.hypot(tray_w, tray_h))
        candidates = []
        for label in _FAMILIES:
            if label == "ball":
                p0 = [*xy, (length+width+height)/3]
                lo, hi = [0., 0., 1.], [tray_w, tray_h, h_max]
                seeds = [0.]
            elif label == "cuboid":
                p0 = [*xy, length, width, height, 0.]
                lo = [0., 0., 1., 1., 1., -np.pi]
                hi = [tray_w, tray_h, max_size, max_size, h_max, np.pi]
                seeds = np.deg2rad([0., 30., 60., 90.])
            elif label in ("cylinder", "cone"):
                p0 = [*xy, max(length, width), height]
                lo, hi = [0., 0., 1., 1.], [tray_w, tray_h, max_size, h_max]
                seeds = [0.]
            else:
                p0 = [*xy, max(length, width), height, 0.]
                period = 2*np.pi/_SIDES[label.split("_")[0]]
                lo = [0., 0., 1., 1., -period]
                hi = [tray_w, tray_h, max_size, h_max, period]
                seeds = np.arange(3)*period/3
            hint = warm_start.get(label) if isinstance(warm_start, dict) else None
            if hint is not None:
                hint = np.asarray(hint, np.float64).reshape(-1)
                if hint.size == len(p0) and np.all(np.isfinite(hint)):
                    p0 = hint.copy()
                    seeds = [None]
            best = None
            for yaw in seeds:
                start = np.asarray(p0, np.float64)
                if label not in ("ball", "cylinder", "cone") and yaw is not None:
                    start[-1] = yaw
                start = np.clip(start, np.array(lo)+1e-5, np.array(hi)-1e-5)
                try:
                    fit = least_squares(_residual, start, bounds=(lo, hi),
                                        args=(label, cameras, up, observed),
                                        x_scale="jac", max_nfev=75,
                                        loss="soft_l1", f_scale=2.0,
                                        ftol=2e-5, xtol=2e-5, gtol=2e-5)
                    error = float(np.sqrt(np.mean(fit.fun**2)*2))
                    if np.isfinite(error) and (best is None or error < best[0]):
                        best = (error, label, fit.x, lo, hi)
                except (ValueError, cv2.error, FloatingPointError):
                    continue
            if best is not None:
                candidates.append(best)
        if len(candidates) < 2:
            return None
        candidates.sort(key=lambda c: c[0])
        # Refine the winner and runner-up; initial passes bound the offline budget.
        for i in range(min(2, len(candidates))):
            error, label, p, lo, hi = candidates[i]
            fit = least_squares(_residual, p, bounds=(lo, hi),
                                args=(label, cameras, up, observed), x_scale="jac",
                                max_nfev=100, loss="soft_l1", f_scale=2.0,
                                ftol=1e-8, xtol=1e-8, gtol=1e-8)
            candidates[i] = (float(np.sqrt(np.mean(fit.fun**2)*2)), label, fit.x, lo, hi)
        candidates.sort(key=lambda c: c[0])
        error, label, p, _, _ = candidates[0]
        margin = max(0., float(candidates[1][0]-error))
        polygons, base, height, yaw = _projected(label, p, cameras, up)
        per_view, worst, tail = [], 0., 0.
        for model, (boundary, samples) in zip(polygons, observed):
            distances = np.r_[np.linalg.norm(_segment_vectors(samples, model), axis=1),
                              np.linalg.norm(_segment_vectors(_sample(model), boundary), axis=1)]
            per_view.append(float(np.sqrt(np.mean(distances**2))))
            worst = max(worst, float(distances.max()))
            # Shadows/rim pixels spoil isolated boundary points; judge the bulk of the
            # silhouette while still rejecting a systematically wrong model.
            tail = max(tail, float(np.percentile(distances, 95)))
        if label == "cuboid":
            l, w = map(float, p[2:4])
            if w > l:
                l, w, yaw = w, l, yaw+np.pi/2
            if (max(l, w, height)-min(l, w, height))/max(l, w, height) < .035:
                label = "cube"
            yaw %= np.pi/2 if label == "cube" else np.pi
        else:
            local = _base(label, [p[2]])
            l, w = map(float, np.ptp(local, axis=0))
            if label in ("ball", "cylinder", "cone"):
                l = w = float(p[2])
            elif label == "triangular_pyramid":
                edge = float(p[2])*np.sqrt(3)/2
                if abs(height-np.sqrt(2/3)*edge)/height < .035:
                    label = "regular_tetrahedron"
            if label not in ("ball", "cylinder", "cone"):
                n = 3 if label == "regular_tetrahedron" else _SIDES[label.split("_")[0]]
                yaw %= 2*np.pi/n
        scale = min(float(np.linalg.norm(np.ptp(o[1], axis=0))) for o in observed)
        corners_px = np.asarray([cal["tray"]["corners_px"][k]
                                 for k in ("TL", "TR", "BR", "BL")], np.float64)
        px_per_mm = 0.5 * (np.linalg.norm(corners_px[1] - corners_px[0]) / tray_w
                           + np.linalg.norm(corners_px[3] - corners_px[0]) / tray_h)
        # Judge in millimetres, not pixels: a 45 mm block spans ~600 px here, so a
        # 10 px boundary error is only about 1 mm and must not be rejected.
        threshold = max(6.0, 2.0 * float(px_per_mm))
        on_tray = (np.min(base[:, 0]) >= -.5 and np.max(base[:, 0]) <= tray_w+.5
                   and np.min(base[:, 1]) >= -.5 and np.max(base[:, 1]) <= tray_h+.5)
        # Round solids must actually look round from above; a tilted box projects a
        # polygon, so this keeps a tilted cuboid from being accepted as a sphere.
        if label in ("ball", "cylinder", "cone"):
            boundary, _ = observed[0]
            area = abs(cv2.contourArea(boundary.astype(np.float32)))
            perimeter = float(cv2.arcLength(boundary.astype(np.float32), True))
            roundness = 4*np.pi*area/(perimeter*perimeter) if perimeter > 0 else 0.0
            if roundness < 0.93:
                label = "unknown(non_round)"
        accepted = bool(on_tray and label != "unknown(non_round)"
                        and max(per_view) <= threshold and tail <= 3*threshold
                        and margin >= max(.5, .12*error)
                        and height < h_max-.05)
        return {"label": label, "center_mm": (float(p[0]), float(p[1])),
                "length_mm": l, "width_mm": w, "height_mm": float(height),
                "yaw_deg": float(np.rad2deg(yaw)), "polygon_mm": base.tolist(),
                "error_px": error, "margin_px": margin, "tail_px": tail,
                "worst_px": worst, "per_view_px": per_view, "accepted": accepted,
                "warm_start": {candidate[1]: candidate[2].tolist() for candidate in candidates}}
    except (ValueError, TypeError, KeyError, IndexError, cv2.error, FloatingPointError):
        return None
