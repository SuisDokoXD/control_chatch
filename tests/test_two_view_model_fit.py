"""Deterministic offline tests; fixtures project independent real 3D geometry.

Run with Python -B to avoid changing any bytecode files. No camera is accessed.
"""
from __future__ import annotations

import unittest
import cv2
import numpy as np

from scripts.two_view_model_fit import fit_two_view


def calibration(up=-1):
    k = np.array([[820., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
    rv = np.zeros(3) if up == -1 else np.array([np.pi, 0., 0.])
    rt = cv2.Rodrigues(rv)[0]
    tv = np.array([-80., -60., 530.])
    # A physically different side camera, looking at the tray centre.
    camera = np.array([65., -380., up*260.])
    forward = np.array([80., 60., up*20.])-camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., float(up)])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rs = np.vstack((right, down, forward))
    ts = -rs @ camera
    relative = rs @ rt.T
    offset = ts-relative @ tv
    corners = np.array([[0., 0., 0.], [180., 0., 0.],
                        [180., 140., 0.], [0., 140., 0.]])
    pixels = cv2.projectPoints(corners, rv, tv, k, np.zeros(5))[0].reshape(-1, 2)
    cal = {"tray": {"size_mm": [180., 140.],
                    "corners_px": dict(zip(("TL", "TR", "BR", "BL"), pixels.tolist()))},
           "intrinsics": {name: {"mtx": k.tolist(), "dist": [0.]*5}
                          for name in ("vertical", "side_a")},
           "extrinsics": {"R": relative.tolist(), "T": offset.tolist()}}
    return cal, [(rt, tv, k), (rs, ts, k)]


def project_solid(points, cameras, noise=0.):
    """Only test-side cv2 projection and convex hull; no fitter geometry helpers."""
    rng = np.random.default_rng(20261027)
    contours = []
    for rotation, translation, k in cameras:
        rv = cv2.Rodrigues(rotation)[0]
        image = cv2.projectPoints(np.asarray(points, np.float64), rv, translation,
                                  k, np.zeros(5))[0].reshape(-1, 2)
        hull = cv2.convexHull(image.astype(np.float32)).reshape(-1, 2).astype(np.float64)
        # Independent dense boundary sampling, not a bbox or horizontal section.
        edges = np.roll(hull, -1, axis=0)-hull
        boundary = np.vstack([a+np.arange(8)[:, None]/8*e for a, e in zip(hull, edges)])
        if noise:
            boundary += rng.normal(0., noise, boundary.shape)
        contours.append(boundary)
    return contours


def sphere_points(xy, diameter, up=-1):
    longitude = np.linspace(0, 2*np.pi, 320, endpoint=False)
    latitude = np.linspace(-np.pi/2, np.pi/2, 161)
    u, v = np.meshgrid(longitude, latitude)
    r = diameter/2
    return np.column_stack((xy[0]+r*np.cos(v.ravel())*np.cos(u.ravel()),
                            xy[1]+r*np.cos(v.ravel())*np.sin(u.ravel()),
                            up*(r+r*np.sin(v.ravel()))))


def box_points(xy, dimensions, yaw, up=-1, tilt=0.):
    l, w, h = dimensions
    points = np.array([[x, y, z] for z in (0., h)
                       for y in (-w/2, w/2) for x in (-l/2, l/2)])
    if tilt:
        angle = np.deg2rad(tilt)
        rotation = np.array([[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)],
                             [0., np.sin(angle), np.cos(angle)]])
        points = points @ rotation.T
        points[:, 2] -= points[:, 2].min()
    a = np.deg2rad(yaw)
    points[:, :2] = points[:, :2] @ np.array([[np.cos(a), np.sin(a)],
                                                           [-np.sin(a), np.cos(a)]])+xy
    points[:, 2] *= up
    return points


class TwoViewModelFitTests(unittest.TestCase):
    def assert_fit(self, result, label, xy, height):
        self.assertIsNotNone(result)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["label"], label, result)
        self.assertLess(np.linalg.norm(np.array(result["center_mm"])-xy), 2., result)
        self.assertLess(abs(result["height_mm"]-height), 2., result)
        self.assertGreaterEqual(result["margin_px"], .15)
        self.assertGreater(result["height_mm"], 0.)
        self.assertEqual(np.asarray(result["polygon_mm"]).shape[1], 2)
        print(f"{label}: centre={result['center_mm']}, height={result['height_mm']:.4f}, "
              f"error={result['error_px']:.5f}px, margin={result['margin_px']:.5f}px", flush=True)

    def test_real_3d_sphere(self):
        cal, cameras = calibration()
        xy, diameter = np.array([93., 72.]), 42.
        contours = project_solid(sphere_points(xy, diameter), cameras)
        result = fit_two_view(*contours, cal, xy+[5., -4.], 36.)
        self.assert_fit(result, "ball", xy, diameter)
        self.assertEqual(result["yaw_deg"], 0.)
        self.assertLess(abs(result["length_mm"]-diameter), 2.)

    def test_real_3d_cube_top_is_perspectively_larger(self):
        cal, cameras = calibration()
        xy = np.array([112., 80.])
        points = box_points(xy, (38., 38., 38.), 27.)
        rotation, translation, k = cameras[0]
        image = cv2.projectPoints(points, cv2.Rodrigues(rotation)[0], translation,
                                  k, np.zeros(5))[0].reshape(-1, 2)
        self.assertGreater(np.ptp(image[4:, 0]), np.ptp(image[:4, 0])*1.07)
        contours = project_solid(points, cameras, noise=.015)
        result = fit_two_view(*contours, cal, xy+[-4., 3.], (33., 40., 31.))
        self.assert_fit(result, "cube", xy, 38.)
        self.assertLess(abs(result["length_mm"]-38.), 2.)
        # Ground polygon must represent the real 38 mm base, not magnified top.
        self.assertAlmostEqual(abs(cv2.contourArea(np.float32(result["polygon_mm"]))),
                               38.**2, delta=50.)

    def test_positive_raw_up_and_asymmetric_cuboid(self):
        cal, cameras = calibration(up=1)
        xy = np.array([85., 65.])
        contours = project_solid(box_points(xy, (46., 27., 35.), 63., up=1), cameras)
        result = fit_two_view(*contours, cal, xy+[3., -3.], (40., 30., 30.))
        self.assert_fit(result, "cuboid", xy, 35.)
        self.assertLess(abs(result["width_mm"]-27.), 2.)

    def test_inconsistent_full_contours_rejected(self):
        cal, cameras = calibration()
        contours = project_solid(box_points([95., 70.], (36., 36., 36.), 19.), cameras)
        contours[1] += [65., -25.]
        result = fit_two_view(*contours, cal, (95., 70.), 36.)
        self.assertTrue(result is None or not result["accepted"], result)

    def test_tilted_solid_rejected(self):
        cal, cameras = calibration()
        contours = project_solid(box_points([90., 70.], (46., 28., 38.), 21., tilt=31.), cameras)
        result = fit_two_view(*contours, cal, (90., 70.), (46., 28., 38.))
        self.assertTrue(result is None or not result["accepted"], result)

    def test_regular_prisms_pyramids_and_round_solids(self):
        cal, cameras = calibration()
        xy, diameter = np.array([96., 74.]), 44.
        cases = [("cylinder", 512, False, 39.), ("cone", 512, True, 39.),
                 ("triangular_prism", 3, False, 39.),
                 ("pentagonal_prism", 5, False, 39.),
                 ("hexagonal_prism", 6, False, 39.),
                 ("triangular_pyramid", 3, True, 44.),
                 ("regular_tetrahedron", 3, True, diameter/np.sqrt(2)),
                 ("square_pyramid", 4, True, 39.),
                 ("pentagonal_pyramid", 5, True, 39.),
                 ("hexagonal_pyramid", 6, True, 39.)]
        for label, sides, pyramid, height in cases:
            with self.subTest(label=label):
                angles = np.arange(sides)*2*np.pi/sides+np.deg2rad(17.)
                base = np.column_stack((np.cos(angles), np.sin(angles)))*diameter/2+xy
                bottom = np.column_stack((base, np.zeros(sides)))
                points = (np.vstack((bottom, [*xy, -height])) if pyramid else
                          np.vstack((bottom, bottom+[0., 0., -height])))
                contours = project_solid(points, cameras)
                result = fit_two_view(*contours, cal, xy+[2., -3.], (40., 40., 35.))
                self.assert_fit(result, label, xy, height)

    def test_small_ambiguous_round_model_is_rejected_on_margin(self):
        cal, cameras = calibration()
        a = np.arange(512)*2*np.pi/512
        base = np.column_stack((np.cos(a), np.sin(a)))*2.5+[96., 74.]
        bottom = np.column_stack((base, np.zeros(512)))
        contours = project_solid(np.vstack((bottom, bottom+[0., 0., -12.])), cameras)
        result = fit_two_view(*contours, cal, (96., 74.), (5., 5., 12.))
        self.assertIsNotNone(result)
        self.assertLess(result["error_px"], .1)
        self.assertLess(result["margin_px"], .15)
        self.assertFalse(result["accepted"])

    def test_invalid_and_obviously_occluded_contours(self):
        cal, _ = calibration()
        concave = np.array([[0., 0.], [30., 0.], [30., 30.],
                            [15., 10.], [0., 30.], [0., 10.]])
        for bad in ([], [[np.nan, 0.]]*8, np.zeros((8, 2)), concave):
            with self.subTest(bad=str(bad)[:30]):
                self.assertIsNone(fit_two_view(bad, concave, cal, (80., 60.), 30.))


if __name__ == "__main__":
    unittest.main(verbosity=2)
