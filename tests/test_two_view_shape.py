"""Offline tests for two-view geometry and mask fusion."""

from __future__ import annotations

import unittest
import json
import tempfile
import contextlib
import io
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scripts.two_view_shape import (
    PROFILE_MODELS,
    Detection,
    FootprintGeometry,
    SideAssociation,
    SHAPE_CLASSES,
    analyse_masks,
    adaptive_foreground,
    associate_side_components,
    classify_profile,
    detections_to_payload,
    dominant_color,
    estimate_supported_height,
    footprint,
    fuse,
    pixels_to_mm,
    tray_floor_polygon_px,
    rank_grasps,
    side_project,
    tray_point_to_robot,
    tray_pose,
    main,
)

def calibration_fixture() -> dict:
    width_mm, height_mm, depth_mm = 160.0, 120.0, 600.0
    vertical_k = np.array([[600.0, 0.0, 200.0],
                           [0.0, 600.0, 160.0],
                           [0.0, 0.0, 1.0]], np.float64)
    side_k = np.array([[600.0, 0.0, 250.0],
                       [0.0, 600.0, 250.0],
                       [0.0, 0.0, 1.0]], np.float64)
    rvec = np.zeros(3, np.float64)
    tvec = np.array([0.0, 0.0, depth_mm], np.float64)
    tray_points = np.array([[0.0, 0.0, 0.0],
                            [width_mm, 0.0, 0.0],
                            [width_mm, height_mm, 0.0],
                            [0.0, height_mm, 0.0]], np.float32)
    image_points = cv2.projectPoints(tray_points, rvec, tvec, vertical_k,
                                     np.zeros(5))[0].reshape(-1, 2)
    angle = np.deg2rad(30.5)
    rotation = np.array([[1.0, 0.0, 0.0],
                         [0.0, np.cos(angle), np.sin(angle)],
                         [0.0, -np.sin(angle), np.cos(angle)]], np.float64)
    return {
        "tray": {
            "size_mm": [width_mm, height_mm],
            "corners_px": dict(zip(("TL", "TR", "BR", "BL"), image_points.tolist())),
            "homography_px_to_mm": [[1.0, 0.0, -200.0],
                                     [0.0, 1.0, -160.0],
                                     [0.0, 0.0, 1.0]],
            "homography_mm_to_px": [[1.0, 0.0, 200.0],
                                     [0.0, 1.0, 160.0],
                                     [0.0, 0.0, 1.0]],
        },
        "intrinsics": {
            "vertical": {"mtx": vertical_k.tolist(), "dist": [0.0] * 5},
            "side_a": {"mtx": side_k.tolist(), "dist": [0.0] * 5},
        },
        "robot": {
            "T_robot_from_tray_mm": np.eye(4).tolist(),
            "gripper": {"opening_range_mm": [10.0, 80.0],
                        "clearance_mm": 2.0, "approach_clearance_mm": 20.0},
        },
        "extrinsics": {"R": rotation.tolist(), "T": [-50.0, 0.0, 0.0]},
    }


class FootprintTests(unittest.TestCase):
    def test_rectangle_dimensions_are_measured_after_homography(self) -> None:
        mask = np.zeros((500, 500), np.uint8)
        cv2.rectangle(mask, (250, 210), (280, 240), 1, -1)
        contour = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)[0][0]
        h = np.array([[0.1, 0.0, 0.0],
                      [0.0, 0.1, 0.0],
                      [0.0, 0.0, 1.0]], np.float64)

        result = footprint(contour, h)

        self.assertIsNotNone(result)
        self.assertEqual(result.kind, "poly4")
        self.assertAlmostEqual(result.length_mm, 3.0, places=3)
        self.assertAlmostEqual(result.width_mm, 3.0, places=3)
        self.assertAlmostEqual(result.size_mm, 3.0, places=3)

    def test_circle_size_is_its_physical_diameter(self) -> None:
        mask = np.zeros((300, 300), np.uint8)
        cv2.circle(mask, (150, 150), 50, 1, -1)
        contour = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)[0][0]
        h = np.array([[0.1, 0.0, 0.0],
                      [0.0, 0.1, 0.0],
                      [0.0, 0.0, 1.0]], np.float64)

        result = footprint(contour, h)

        self.assertIsNotNone(result)
        self.assertEqual(result.kind, "circle")
        self.assertAlmostEqual(result.size_mm, 10.0, delta=0.4)


class SideAssociationTests(unittest.TestCase):
    def test_separate_components_match_separate_projected_bases(self) -> None:
        mask = np.zeros((180, 260), np.uint8)
        cv2.rectangle(mask, (30, 40), (55, 135), 1, -1)
        cv2.rectangle(mask, (195, 40), (220, 135), 1, -1)
        bases = [np.array([[42, 130], [45, 130]], np.float32),
                 np.array([[207, 130], [210, 130]], np.float32)]

        matches = associate_side_components(mask, bases, corridor_px=3)

        self.assertEqual([match.status for match in matches], ["matched", "matched"])
        self.assertNotEqual(matches[0].component_id, matches[1].component_id)
        self.assertFalse(np.array_equal(matches[0].mask, matches[1].mask))

    def test_shared_side_component_is_rejected_for_both_instances(self) -> None:
        mask = np.zeros((160, 240), np.uint8)
        cv2.rectangle(mask, (25, 55), (215, 125), 1, -1)
        bases = [np.array([[42, 120], [45, 120]], np.float32),
                 np.array([[195, 120], [198, 120]], np.float32)]

        matches = associate_side_components(mask, bases, corridor_px=3)

        self.assertEqual([match.status for match in matches],
                         ["shared_component", "shared_component"])
        self.assertTrue(all(match.mask is None for match in matches))

    def test_missing_side_component_is_reported(self) -> None:
        mask = np.zeros((80, 100), np.uint8)
        bases = [np.array([[40, 60], [50, 60]], np.float32)]
        self.assertEqual(associate_side_components(mask, bases)[0].status,
                         "no_candidate")


class ProfileAndHeightTests(unittest.TestCase):
    def test_known_templates_are_identified_and_bad_fit_is_rejected(self) -> None:
        ts = np.linspace(0.0, 0.95, 12)
        for expected in ("rect", "circle", "triangle"):
            profile = np.array([PROFILE_MODELS[expected](t) for t in ts])
            kind, error, margin = classify_profile(profile)
            self.assertEqual(kind, expected)
            self.assertLessEqual(error, 0.035)
            self.assertGreaterEqual(margin, 0.012)

        kind, _, _ = classify_profile(np.full(12, 0.5))
        self.assertEqual(kind, "unknown")

    def test_height_tolerates_short_gaps_and_ignores_isolated_high_hit(self) -> None:
        heights = np.arange(0.0, 41.0)
        supported = np.zeros(len(heights), dtype=bool)
        supported[:21] = True
        supported[8:10] = False
        supported[35] = True

        height, status = estimate_supported_height(supported, heights)

        self.assertEqual(status, "ok")
        self.assertEqual(height, 20.0)

    def test_height_rejects_missing_base_and_scan_clipping(self) -> None:
        heights = np.arange(0.0, 21.0)
        floating = np.zeros(len(heights), dtype=bool)
        floating[8:15] = True
        clipped = np.zeros(len(heights), dtype=bool)
        clipped[:20] = True

        self.assertEqual(estimate_supported_height(floating, heights)[1],
                         "no_grounded_run")
        self.assertEqual(estimate_supported_height(clipped, heights)[1],
                         "clipped_at_h_max")


class GeometryClassificationTests(unittest.TestCase):
    def foot(self, kind: str, length: float = 30.0, width: float = 30.0,
             regular: bool = True) -> FootprintGeometry:
        polygon = np.array([[-length / 2, -width / 2],
                            [length / 2, -width / 2],
                            [length / 2, width / 2],
                            [-length / 2, width / 2]], np.float32)
        if kind == "poly3":
            polygon = np.array([[0.0, 17.32], [-15.0, -8.66], [15.0, -8.66]], np.float32)
        vertices = int(kind[4:]) if kind.startswith("poly") else 0
        return FootprintGeometry(kind, (80.0, 60.0), polygon, length, width,
                                 float(np.sqrt(length * width)), 0.78, vertices,
                                 length / width, 0.99 if regular else 0.55,
                                 regular, 0.0)

    def test_requested_rule_shape_families_are_in_contract(self) -> None:
        self.assertTrue({"cube", "cuboid", "cone", "cylinder", "ball",
                         "regular_tetrahedron", "triangular_pyramid",
                         "pentagonal_pyramid", "hexagonal_pyramid"}.issubset(SHAPE_CLASSES))

    def test_fusion_maps_polygon_footprints_and_side_models(self) -> None:
        cases = [
            (self.foot("poly4"), "rect", 30.0, "cube", "cube"),
            (self.foot("poly3", regular=True), "triangle", 25.0,
             "regular_tetrahedron", "regular_tetrahedron"),
            (self.foot("poly3", regular=False), "triangle", 25.0,
             "triangular_pyramid", "triangular_pyramid"),
            (self.foot("poly5"), "triangle", 25.0,
             "pentagonal_pyramid", "pentagonal_pyramid"),
            (self.foot("poly6"), "triangle", 25.0,
             "hexagonal_pyramid", "hexagonal_pyramid"),
            (self.foot("poly3"), "rect", 30.0,
             "triangular_prism", "triangular_prism"),
        ]
        for geometry, side, height, expected, _ in cases:
            label, confidence = fuse(geometry, side, height, 0.001, 0.08)
            self.assertEqual(label, expected)
            self.assertGreater(confidence, 0.0)


class GraspContractTests(unittest.TestCase):
    def test_hsv_color_is_reported_independently_from_shape(self) -> None:
        mask = np.ones((40, 40), np.uint8)
        red = np.zeros((40, 40, 3), np.uint8)
        red[:] = (0, 0, 255)
        self.assertEqual(dominant_color(red, mask), "red")
        white = np.full((40, 40, 3), 245, np.uint8)
        self.assertEqual(dominant_color(white, mask), "white")

    def test_grasp_has_opening_and_tool_configuration_gates_command_ready(self) -> None:
        geometry = FootprintGeometry("poly4", (80.0, 60.0),
                                     np.array([[65, 45], [95, 45], [95, 75], [65, 75]], np.float32),
                                     30.0, 30.0, 30.0, 0.78, 4, 1.0, 1.0, True, 0.0)
        association = SideAssociation("matched", 1, np.ones((4, 4), np.uint8))
        detection = Detection(1, np.empty((0, 1, 2), np.int32), geometry, association,
                              30.0, "rect", (1.0,) * 12, 0.001, 0.08, 0.5,
                              "cube", 0.99)
        transform = np.eye(4)
        transform[:3, 3] = [100.0, 200.0, 300.0]

        detection.geometry_verified = True  # Independent ideal 3-D fixture, not mask inference.
        ranked = rank_grasps([detection], (160.0, 120.0), 0.5, 5.0,
                             transform, (10.0, 80.0), 2.0, 20.0)
        self.assertTrue(ranked[0].pick_candidate)
        self.assertFalse(ranked[0].robot_target_ready)
        self.assertEqual(ranked[0].grasp_opening_mm, 34.0)
        self.assertEqual(ranked[0].approach_z_mm, 50.0)
        self.assertEqual(ranked[0].descend_z_mm, 15.0)
        self.assertEqual(ranked[0].robot_xyz_mm, (180.0, 260.0, 315.0))
        payload = detections_to_payload(ranked)[0]
        self.assertEqual(payload["grasp_target"]["descend_z_mm"], 15.0)
        self.assertEqual(payload["robot_target_xyz_mm"], [180.0, 260.0, 315.0])
        self.assertFalse(payload["robot_target_ready"])
        self.assertFalse(payload["robot_command_ready"])

    def test_unconfigured_tool_and_robot_transform_are_held(self) -> None:
        geometry = FootprintGeometry("circle", (80.0, 60.0),
                                     np.array([[70, 50], [90, 50], [90, 70], [70, 70]], np.float32),
                                     20.0, 20.0, 20.0, 0.9, 12, 1.0, 1.0, False, None)
        detection = Detection(2, np.empty((0, 1, 2), np.int32), geometry,
                              SideAssociation("matched", 2, np.ones((2, 2), np.uint8)),
                              20.0, "circle", (1.0,) * 12, 0.001, 0.08, 0.5,
                              "ball", 0.9)
        rank_grasps([detection], (160.0, 120.0))
        self.assertFalse(detection.pick_candidate)
        self.assertFalse(detection.robot_target_ready)
        self.assertIsNone(detection.robot_xyz_mm)
        self.assertIn("夹爪开口范围未配置", detection.rejection_reasons)
        self.assertIn("机械臂基座到托盘变换未配置", detection.rejection_reasons)
    def test_robot_frame_homogeneous_transform(self) -> None:
        transform = np.eye(4)
        transform[:3, 3] = [5.0, -3.0, 8.0]
        self.assertEqual(tray_point_to_robot((2.0, 4.0, 6.0), transform),
                         (7.0, 1.0, 14.0))



class OfflineAnalysisTests(unittest.TestCase):
    def test_analysis_entry_point_returns_association_and_measurements(self) -> None:
        cal = calibration_fixture()
        rt, tvec, rotation, translation, _, _, up = tray_pose(cal)
        vertical = np.zeros((500, 500), np.uint8)
        cv2.rectangle(vertical, (250, 210), (280, 240), 1, -1)
        contour = cv2.findContours(vertical, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)[0][0]
        h = np.asarray(cal["tray"]["homography_px_to_mm"], np.float64)
        measured = footprint(contour, h)
        prism = np.array([[x, y, z] for x in (50.0, 80.0)
                          for y in (50.0, 80.0)
                          for z in (0.0, up * 30.0)], np.float32)
        projected = side_project(cal, rt, tvec, rotation, translation, prism)
        hull = cv2.convexHull(np.round(projected).astype(np.int32))
        side = np.zeros((800, 500), np.uint8)
        cv2.fillConvexPoly(side, hull, 1)

        # Exercise heuristic-only fallback: the hand-drawn top mask is not a true 3-D projection.
        with patch("scripts.two_view_shape.fit_two_view", return_value=None):
            result = analyse_masks(vertical, side, cal, min_area=10, side_corridor_px=10,
                                   h_max=60.0, step=1.0)

        self.assertEqual(len(result), 1)
        detection = result[0]
        self.assertEqual(detection.association.status, "matched")
        self.assertAlmostEqual(detection.footprint.length_mm, 30.0, delta=0.2)
        self.assertAlmostEqual(detection.footprint.width_mm, 30.0, delta=0.2)
        self.assertAlmostEqual(detection.height_mm, 30.0, delta=3.0)
        self.assertEqual(detection.side_kind, "rect")
        self.assertEqual(detection.shape_hypothesis, "cube")
        self.assertTrue(detection.label.startswith("unknown"))
        self.assertEqual(detection.confidence, 0.0)
        self.assertFalse(detection.geometry_verified)  # no accepted 3-D projection fit
        self.assertFalse(detection.pick_candidate)
        self.assertFalse(detection.robot_target_ready)
        payload = detections_to_payload(result)[0]
        self.assertIsNone(payload["grasp_target"])
        self.assertIsNone(payload["position_mm"])
        self.assertIn("unverified_estimates", payload)
        self.assertAlmostEqual(payload["unverified_estimates"]["grasp_target"]["descend_z_mm"],
                               15.0, delta=2.0)


class OfflineCliTests(unittest.TestCase):
    def test_mask_replay_writes_json_without_opening_camera_or_gui(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            cal_path = folder / "cal.json"
            cal_path.write_text(json.dumps(calibration_fixture()), encoding="utf-8")
            top, side = folder / "top.png", folder / "side.png"
            output = folder / "result.json"
            cv2.imwrite(str(top), np.zeros((500, 500), np.uint8))
            cv2.imwrite(str(side), np.zeros((800, 500), np.uint8))
            with patch("scripts.two_view_shape.open_camera", side_effect=AssertionError("camera opened")), \
                 patch("scripts.two_view_shape.cv2.imshow", side_effect=AssertionError("GUI opened")), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = main(["--calibration", str(cal_path), "--offline-vertical", str(top),
                             "--offline-side", str(side), "--input-masks",
                             "--output-json", str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), [])

    def test_color_bins_cover_twelve_hues_and_red_wrap(self):
        names = set()
        for hue in range(0, 180, 15):
            hsv = np.full((10, 10, 3), [hue, 220, 220], np.uint8)
            names.add(dominant_color(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), np.ones((10, 10),np.uint8)))
        self.assertEqual(len(names), 12)
        hsv = np.full((10, 10, 3), [179, 220, 220], np.uint8)
        hsv[:5, :, 0] = 1
        self.assertEqual(dominant_color(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), np.ones((10, 10),np.uint8)), "red")


class ReliabilityRankingTests(unittest.TestCase):
    def object(self, identity, xy, confidence, margin):
        x,y = xy
        geometry = FootprintGeometry("poly4",xy,
            np.float32([[x-5,y-5],[x+5,y-5],[x+5,y+5],[x-5,y+5]]),
            10.,10.,10.,.78,4,1.,1.,True,0.)
        return Detection(identity,np.empty((0,1,2),np.int32),geometry,
            SideAssociation("matched"),10.,"rect",(1.,)*12,.001,.08,.2,
            "cube",confidence,geometry_verified=True,
            model_fit_error_px=.2,model_fit_margin_px=margin)

    def test_clearance_cannot_outweigh_clearer_classification(self):
        certain = self.object(1,(12.,50.),.97,2.)  # Only 7 mm edge clearance, still valid.
        roomy = self.object(2,(65.,50.),.75,2.)
        ranked = rank_grasps([roomy,certain],(120.,100.),
                             gripper_opening_range_mm=(0.,40.))
        self.assertTrue(all(d.pick_candidate for d in ranked))
        self.assertLess(certain.clearance_mm,roomy.clearance_mm)
        self.assertEqual(ranked[0].instance_id,1)
        self.assertEqual(certain.rank,1)

    def test_class_separation_breaks_equal_quality_tie_before_clearance(self):
        clear = self.object(1,(12.,50.),.9,3.)
        ambiguous = self.object(2,(65.,50.),.9,.2)
        ranked = rank_grasps([ambiguous,clear],(120.,100.),
                             gripper_opening_range_mm=(0.,40.))
        self.assertEqual(ranked[0].instance_id,1)

    def test_insufficient_clearance_is_still_a_hard_gate(self):
        blocked = self.object(1,(7.,50.),.99,4.)
        valid = self.object(2,(65.,50.),.8,.5)
        ranked = rank_grasps([blocked,valid],(120.,100.),
                             gripper_opening_range_mm=(0.,40.))
        self.assertEqual(ranked[0].instance_id,2)
        self.assertFalse(blocked.pick_candidate)
        self.assertEqual(blocked.rank,0)


class IntegratedSolidTests(unittest.TestCase):
    def test_true_3d_cube_recovers_position_height_and_grasp_payload(self):
        from test_two_view_model_fit import calibration, box_points, project_solid
        cal, cameras = calibration()
        contours = project_solid(box_points([112.,80.],(38.,38.,38.),27.), cameras)
        corners = np.float32([cal["tray"]["corners_px"][k] for k in ("TL","TR","BR","BL")])
        cal["tray"]["homography_px_to_mm"] = cv2.getPerspectiveTransform(
            corners,np.float32([[0,0],[180,0],[180,140],[0,140]])).tolist()
        cal["robot"] = {"T_robot_from_tray_mm":np.eye(4).tolist(),
                          "gripper":{"opening_range_mm":[0.,100.]}}
        masks = []
        for contour in contours:
            mask = np.zeros((700,900),np.uint8)
            cv2.fillConvexPoly(mask,np.rint(contour).astype(np.int32),1)
            masks.append(mask)
        payload = detections_to_payload(analyse_masks(*masks,cal,min_area=10))[0]
        self.assertEqual(payload["class"],"cube")
        self.assertTrue(payload["geometry_valid"])
        self.assertAlmostEqual(payload["position_mm"]["x"],112.,delta=2.)
        self.assertAlmostEqual(payload["position_mm"]["y"],80.,delta=2.)
        self.assertAlmostEqual(payload["dimensions_mm"]["height"],38.,delta=2.)
        self.assertAlmostEqual(payload["grasp_target"]["descend_z_mm"],19.,delta=1.)
        self.assertTrue(payload["pick_candidate"])
        self.assertEqual(payload["rank"],1)
        self.assertFalse(payload["robot_command_ready"])
        json.dumps(payload, allow_nan=False)


class TrayFloorRoiTests(unittest.TestCase):
    def test_floor_polygon_uses_black_dots_without_second_slope_inset(self):
        cal = calibration_fixture()
        cal["tray"]["slope_width_mm"] = {"left_mm": 27.58, "right_mm": 29.10,
                                         "bottom_mm": 16.83, "top_mm": 6.38}
        floor = tray_floor_polygon_px(cal)
        expected = np.asarray([cal["tray"]["corners_px"][key]
                               for key in ("TL", "TR", "BR", "BL")], np.float32)
        np.testing.assert_array_equal(floor, expected)
        # A valid object near the right floor edge must not be discarded.
        self.assertGreater(cv2.pointPolygonTest(floor.reshape(-1, 1, 2), (350., 220.), False), 0)

    def test_only_explicit_margin_may_inset_the_floor(self):
        cal = calibration_fixture()
        floor = tray_floor_polygon_px(cal, extra_margin_mm=2.)
        mm = pixels_to_mm(floor, np.asarray(cal["tray"]["homography_px_to_mm"]))
        np.testing.assert_allclose(mm.min(axis=0), [2., 2.], atol=1e-4)
        np.testing.assert_allclose(mm.max(axis=0), [158., 118.], atol=1e-4)
        for margin in (-1., float("nan"), 60.):
            with self.assertRaises(ValueError):
                tray_floor_polygon_px(cal, extra_margin_mm=margin)

    def test_wall_band_stays_outside_the_working_area(self):
        cal = calibration_fixture()
        cal["tray"]["slope_width_mm"] = {"left_mm": 27.58, "right_mm": 29.10,
                                         "bottom_mm": 16.83, "top_mm": 6.38}
        floor = tray_floor_polygon_px(cal)
        self.assertEqual(cv2.pointPolygonTest(floor.astype(np.float32).reshape(-1, 1, 2),
                                              (510.0, 400.0), False), -1.0)


class AdaptiveFallbackTests(unittest.TestCase):
    def test_otsu_fallback_keeps_tray_object(self):
        image = np.full((300, 400, 3), 210, np.uint8)
        cv2.rectangle(image, (120, 100), (200, 170), (40, 40, 40), -1)
        mask = adaptive_foreground(image, min_area=500)
        self.assertEqual(int(mask[130, 150]), 1)
        self.assertEqual(int(mask[10, 10]), 0)

    def test_otsu_fallback_drops_object_touching_frame_edge(self):
        image = np.full((300, 400, 3), 210, np.uint8)
        image[:, 0:30] = (30, 30, 30)          # dark band touching the frame edge
        mask = adaptive_foreground(image, min_area=500)
        self.assertEqual(int(mask.sum()), 0)

    def test_otsu_fallback_reports_nothing_on_blank_tray(self):
        self.assertEqual(int(adaptive_foreground(np.full((200, 200, 3), 205, np.uint8)).sum()), 0)


class RegressionTests(unittest.TestCase):
    def test_nonrigid_or_nonfinite_robot_transform_is_rejected(self):
        for transform in (np.full((4, 4), np.nan), np.diag([2., 2., 2., 1.])):
            with self.assertRaises(ValueError):
                tray_point_to_robot((0., 0., 0.), transform)

    def test_negative_clearances_are_rejected_even_with_no_objects(self):
        for kwargs in ({"gripper_clearance_mm": -8.}, {"approach_clearance_mm": -40.},
                       {"min_clearance_mm": -1.}, {"gripper_opening_range_mm": (22., 10.)}):
            with self.assertRaises(ValueError):
                rank_grasps([], (160., 120.), **kwargs)

    def test_up_sign_uses_tray_coordinates_after_rotation(self):
        cal = calibration_fixture()
        obj = np.float32([[0,0,0], [160,0,0], [160,120,0], [0,120,0]])
        k = np.asarray(cal["intrinsics"]["vertical"]["mtx"])
        px = cv2.projectPoints(obj, np.array([np.pi, 0., 0.]),
                               np.array([0., 0., 600.]), k, np.zeros(5))[0].reshape(-1, 2)
        cal["tray"]["corners_px"] = dict(zip(("TL","TR","BR","BL"), px.tolist()))
        self.assertEqual(tray_pose(cal)[-1], 1.)

    def test_jaws_check_contact_width_and_signed_robot_z(self):
        f = FootprintGeometry("poly4", (80.,60.),
                              np.float32([[55,50],[105,50],[105,70],[55,70]]),
                              50.,20.,np.sqrt(1000),.7,4,2.5,.4,False,0.)
        d = Detection(1, np.empty((0,1,2), np.int32), f, SideAssociation("matched"),
                      30., "rect", (1.,)*12, .001,.08,.5,"cuboid",.99,
                      geometry_verified=True)
        rank_grasps([d], (160.,120.), robot_from_tray_mm=np.eye(4),
                    gripper_opening_range_mm=(22.,80.), tray_up_sign=-1.)
        self.assertFalse(d.pick_candidate)
        self.assertEqual(d.robot_xyz_mm, (80.,60.,-15.))
        self.assertIn("夹爪闭合或预张开超出配置范围", d.rejection_reasons)
        self.assertFalse(d.robot_target_ready)

    def test_true_3d_sphere_projection_is_not_promoted_to_robot_target(self):
        cal = calibration_fixture()
        rt, tv, _, _, _, _, up = tray_pose(cal)
        a = np.deg2rad(20.)
        r = np.array([[1.,0.,0.],[0.,np.cos(a),np.sin(a)],
                      [0.,-np.sin(a),np.cos(a)]])
        cal["extrinsics"]["R"] = r.tolist()
        cal["extrinsics"]["T"] = (np.array([0.,0.,600.]) - r @ np.array([80.,60.,600.])).tolist()
        rt, tv, r, translation, _, _, up = tray_pose(cal)
        ph, th = np.meshgrid(np.linspace(0,np.pi,35), np.linspace(0,2*np.pi,75))
        xyz = np.column_stack((80+20*np.sin(ph).ravel()*np.cos(th).ravel(),
                               60+20*np.sin(ph).ravel()*np.sin(th).ravel(),
                               up*(20+20*np.cos(ph).ravel())))
        top, side = np.zeros((500,500),np.uint8), np.zeros((600,600),np.uint8)
        k = np.asarray(cal["intrinsics"]["vertical"]["mtx"])
        px = cv2.projectPoints(xyz, cv2.Rodrigues(rt)[0],tv,k,np.zeros(5))[0]
        cv2.fillConvexPoly(top, cv2.convexHull(np.rint(px).astype(np.int32)),1)
        px = side_project(cal,rt,tv,r,translation,xyz)
        cv2.fillConvexPoly(side, cv2.convexHull(np.rint(px).astype(np.int32)),1)
        result = analyse_masks(top,side,cal,h_max=100.)
        self.assertEqual(len(result),1)
        payload = detections_to_payload(result)[0]
        self.assertTrue(payload["geometry_valid"])
        self.assertEqual(payload["class"], "ball")
        self.assertAlmostEqual(payload["position_mm"]["x"],80.,delta=2.)
        self.assertAlmostEqual(payload["position_mm"]["y"],60.,delta=2.)
        self.assertAlmostEqual(payload["dimensions_mm"]["height"],40.,delta=2.)
        self.assertTrue(payload["pick_candidate"])
        self.assertFalse(payload["robot_target_ready"])
        self.assertFalse(payload["robot_command_ready"])
        self.assertIsNotNone(payload["grasp_target"])
        self.assertLess(payload["robot_target_xyz_mm"][2],0.)


if __name__ == "__main__":
    unittest.main()
