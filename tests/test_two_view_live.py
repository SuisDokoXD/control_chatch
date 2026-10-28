"""Runtime safety regressions derived from the 2026-10-27 real-camera baseline."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts import two_view_shape as shape
from scripts.two_view_live import (atomic_json, current_result, display_name,
                                   mask_signature, scene_matches, segment_pair)
import test_two_view_shape as fixtures
calibration_fixture = fixtures.calibration_fixture

class FreshnessTests(unittest.TestCase):
    def scene(self):
        mask = np.zeros((240, 320), np.uint8)
        mask[80:120, 100:150] = 1
        return mask_signature({'vertical': mask, 'side_a': mask.copy()})

    def test_unchanged_source_is_current(self):
        source = self.scene()
        self.assertEqual(current_result(source, source, 10, 11, 8), 'current')

    def test_removed_object_invalidates_result_immediately(self):
        source = self.scene()
        empty = {role: np.zeros_like(mask) for role, mask in source.items()}
        self.assertEqual(current_result(source, empty, 10, 10.1, 8), 'scene_changed')

    def test_moved_side_view_invalidates_even_with_identical_top(self):
        source = self.scene()
        changed = {k: v.copy() for k, v in source.items()}
        changed['side_a'] = np.roll(changed['side_a'], 40, axis=1)
        self.assertFalse(scene_matches(source, changed))

    def test_old_source_expires_even_if_scene_unchanged(self):
        source = self.scene()
        self.assertEqual(current_result(source, source, 10, 19, 8), 'expired')

    def test_no_source_is_pending_not_empty_confirmed_scene(self):
        self.assertEqual(current_result(None, self.scene(), 0, 20, 8), 'pending')

    def test_json_remains_a_list_and_atomic_write_has_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.json'
            atomic_json(path, [{'class': 'pentagonal_prism'}])
            atomic_json(path, [])
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), [])
            self.assertFalse(path.with_suffix('.json.tmp').exists())

    def test_tentative_model_never_displayed_as_confirmed(self):
        item = SimpleNamespace(label='unknown(unclassified)', shape_hypothesis='pentagonal_prism')
        self.assertEqual(display_name(item), '假设:五棱柱')
        self.assertEqual(display_name(item, False), 'tentative:pentagonal_prism')
        item.label = 'pentagonal_prism'
        self.assertEqual(display_name(item), '确认:五棱柱')

    def test_ascii_fallback_never_contains_question_marks_or_chinese(self):
        item = SimpleNamespace(label='unknown(unclassified)', shape_hypothesis='unknown(ambiguous)')
        name = display_name(item, False)
        self.assertTrue(name.isascii())
        self.assertNotIn('?', name)


class GeometryPromotionTests(unittest.TestCase):
    def test_failed_3d_fit_never_promotes_low_profile_residual_to_valid_xyz(self):
        cal = calibration_fixture()
        top = np.zeros((500, 600), np.uint8)
        side = np.zeros_like(top)
        cv2.rectangle(top, (260, 200), (290, 230), 1, -1)
        cv2.rectangle(side, (280, 240), (340, 300), 1, -1)
        association = shape.SideAssociation('matched', 1, side)
        with patch.object(shape, 'associate_side_components', return_value=[association]), \
             patch.object(shape, 'side_profile', return_value=(30., 'rect', [], 0.001, .1, 1.0)), \
             patch.object(shape, 'fit_two_view', return_value=None):
            result = shape.analyse_masks(top, side, cal, min_area=10)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].geometry_verified)
        payload = shape.detections_to_payload(result)[0]
        self.assertIsNone(payload['position_mm'])
        self.assertIsNone(payload['grasp_target'])
        self.assertIsNotNone(payload['unverified_estimates']['position_mm'])

    def test_unknown_class_is_not_a_pick_candidate_even_with_valid_geometry(self):
        foot = fixtures.GeometryClassificationTests().foot('poly4')
        foot = shape.replace(foot, polygon_mm=foot.polygon_mm + np.array([80., 60.]))
        detection = shape.Detection(1, np.zeros((4, 1, 2), np.int32), foot,
            shape.SideAssociation('matched'), 30., 'rect', (), .001, .1, 1.,
            'unknown(unclassified)', .9, geometry_verified=True, height_ok=True)
        shape.rank_grasps([detection], (160., 120.), gripper_opening_range_mm=(0., 100.))
        self.assertFalse(detection.pick_candidate)
        self.assertEqual(detection.rank, 0)


class IlluminationTests(unittest.TestCase):
    def test_uniform_background_drift_does_not_create_foreground(self):
        reference = np.full((300, 400, 3), 200, np.uint8)
        current = np.full_like(reference, 190)
        self.assertEqual(int(shape.foreground(current, reference, 8, 50).sum()), 0)

    def test_dark_objects_remain_after_drift_correction(self):
        reference = np.full((300, 400, 3), 200, np.uint8)
        current = np.full_like(reference, 195)
        current[90:170, 140:220] = (120, 50, 70)
        mask = shape.foreground(current, reference, 28, 50)
        self.assertGreater(float(mask[95:165,145:215].mean()), .99)
        self.assertEqual(int(mask[:60].sum()), 0)

    def test_large_lighting_change_cannot_be_silently_normalised_away(self):
        reference = np.full((300, 400, 3), 200, np.uint8)
        current = np.full_like(reference, 150)
        self.assertGreater(float(shape.foreground(current, reference, 28, 50).mean()), .99)

    def test_real_side_shadow_bridge_is_removed_without_changing_threshold(self):
        root = Path(__file__).resolve().parents[1]
        current_path = root / 'reports/live_tuning_20261027/run01/samples/0000/side_a.jpg'
        if not current_path.exists():
            self.skipTest('Recorded camera fixture is not available in this checkout')
        current = cv2.imread(str(current_path))
        reference = cv2.imread(str(root / 'captures/calibration/ref_side_a.jpg'))
        mask = shape.foreground(current, reference, 28, 2500)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        # Independently recorded centres of the two large raw-difference regions.
        first, second = int(labels[657, 1333]), int(labels[1015, 1798])
        self.assertGreater(first, 0)
        self.assertGreater(second, 0)
        self.assertNotEqual(first, second)
        self.assertGreater(stats[first, 4], 100000)
        self.assertGreater(stats[second, 4], 200000)


if __name__ == '__main__':
    unittest.main()