"""流程层测试：选光照 → 双窗口 → 识别。判据函数必须可离线验证。"""
from __future__ import annotations
import unittest

import cv2
import numpy as np

from scripts.live_ui import (brightness_stats, clamp_exposure, next_exposure,
                             photometric_offset, work_crop_rect)
from scripts import two_view_shape as shape

FLOOR = np.array([[20, 20], [580, 20], [580, 580], [20, 580]], np.float32)


class ExposureStepTests(unittest.TestCase):
    def test_brackets_move_in_the_measured_direction(self):
        self.assertEqual(next_exposure(-4.0, ord(']')), -3.5)   # 更亮
        self.assertEqual(next_exposure(-4.0, ord('[')), -4.5)   # 更暗

    def test_other_keys_leave_exposure_untouched(self):
        for key in (ord('x'), 255, ord('\r')):
            self.assertEqual(next_exposure(-5.0, key), -5.0)

    def test_bounds_and_step_grid(self):
        self.assertEqual(clamp_exposure(-99), -9.0)
        self.assertEqual(clamp_exposure(99), -1.0)
        self.assertEqual(clamp_exposure(-4.26), -4.5)
        self.assertEqual(clamp_exposure(-4.24), -4.0)


class BrightnessTests(unittest.TestCase):
    def test_overexposed_and_dark_frames_are_reported(self):
        blown = brightness_stats(np.full((60, 60), 255, np.uint8))
        self.assertEqual(blown['over_pct'], 100.0)
        dark = brightness_stats(np.full((60, 60), 10, np.uint8))
        self.assertEqual(dark['under_pct'], 100.0)
        mid = brightness_stats(np.full((60, 60), 160, np.uint8))
        self.assertEqual(mid['mean'], 160.0)
        self.assertEqual((mid['over_pct'], mid['under_pct']), (0.0, 0.0))

    def test_drift_sign_follows_the_scene_change(self):
        ref = np.full((300, 400, 3), 200, np.uint8)
        darker = np.full_like(ref, 190)
        brighter = np.full_like(ref, 210)
        self.assertLess(photometric_offset(darker, ref), -8)
        self.assertGreater(photometric_offset(brighter, ref), 8)

    def test_unrelated_scene_returns_no_estimate(self):
        ref = np.full((300, 400, 3), 200, np.uint8)
        self.assertIsNone(photometric_offset(np.zeros((300, 400, 3), np.uint8), ref))


class WorkspaceCropTests(unittest.TestCase):
    def test_crop_covers_the_polygon_and_respects_margin(self):
        polygon = np.array([[100, 80], [400, 80], [400, 300], [100, 300]], np.float32)
        x0, y0, x1, y1 = work_crop_rect(polygon, (720, 1280, 3), margin_px=20)
        self.assertEqual((x0, y0, x1, y1), (80, 60, 421, 321))

    def test_crop_is_clipped_to_the_frame(self):
        polygon = np.array([[0, 0], [300, 0], [300, 200], [0, 200]], np.float32)
        self.assertEqual(work_crop_rect(polygon, (200, 300, 3), margin_px=50), (0, 0, 300, 200))

    def test_degenerate_workspace_is_rejected(self):
        with self.assertRaises(ValueError):
            work_crop_rect(np.array([[5, 5], [5, 5], [5, 5], [5, 5]], np.float32),
                           (200, 200, 3))


class ReferenceSelfCheckTests(unittest.TestCase):
    def frame(self, box=False):
        image = np.full((600, 600, 3), 210, np.uint8)
        if box:
            cv2.rectangle(image, (200, 200), (330, 330), (60, 60, 60), -1)
        return image

    def test_empty_tray_passes(self):
        image = self.frame()
        big, small, structure = shape.reference_self_check(image, image.copy(), FLOOR,
                                                          28.0, 2500, 12000)
        self.assertEqual(big, [])
        self.assertEqual(int(structure.sum()), 0)

    def test_visible_object_is_refused(self):
        big, _, _ = shape.reference_self_check(self.frame(box=True), self.frame(),
                                               FLOOR, 28.0, 2500, 12000)
        self.assertTrue(big)
        self.assertGreater(max(big), 12000)


class ExposurePickerFlowTests(unittest.TestCase):
    def api_stub(self):
        from types import SimpleNamespace
        draw = lambda img, text, org, size, color=None, shadow=True: img
        return SimpleNamespace(overlay_text=SimpleNamespace(draw_text=draw))

    def run_picker(self, keys, start=-4.0):
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts import live_ui
        frames = {role: np.full((600, 600, 3), 170, np.uint8)
                  for role in ('vertical', 'side_a')}
        refs = {role: np.full((600, 600, 3), 175, np.uint8) for role in frames}
        pressed = list(keys)
        with patch.object(live_ui, 'read_pair', return_value=frames), \
             patch.object(live_ui.cv2, 'namedWindow'), \
             patch.object(live_ui.cv2, 'resizeWindow'), \
             patch.object(live_ui.cv2, 'imshow'), \
             patch.object(live_ui.cv2, 'destroyWindow'), \
             patch.object(live_ui.cv2, 'waitKey', side_effect=pressed + [255]):
            return live_ui.choose_exposure(
                self.api_stub(), {}, None, refs, None, start,
                SimpleNamespace(thr=28.0, min_area=2500, fixture_max_area=12000), True)

    def test_enter_confirms_current_exposure(self):
        result = self.run_picker([13])
        self.assertFalse(result['aborted'])
        self.assertEqual(result['exposure'], -4.0)

    def test_bracket_adjusts_before_confirm(self):
        result = self.run_picker([ord(']'), 13])
        self.assertEqual(result['exposure'], -3.5)

    def test_q_aborts(self):
        self.assertTrue(self.run_picker([ord('q')])['aborted'])

    def test_space_also_confirms(self):
        self.assertFalse(self.run_picker([32])['aborted'])


class TwoWindowLayoutTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        self.api = SimpleNamespace(overlay_text=SimpleNamespace(
            draw_text=lambda img, text, org, size, color=None, shadow=True: img))
        self.frames = {role: np.full((600, 800, 3), 180, np.uint8)
                       for role in ('vertical', 'side_a')}
        self.masks = {role: np.zeros((600, 800), np.uint8) for role in self.frames}
        cv2.rectangle(self.masks['vertical'], (200, 150), (320, 260), 1, -1)

    def test_panorama_keeps_full_frames(self):
        from scripts.live_ui import build_panorama
        polygon = np.array([[120, 100], [600, 100], [600, 500], [120, 500]], np.float32)
        view = build_panorama(self.api, self.frames, self.masks, polygon, 2500,
                              '手动四点工作区', True)
        self.assertEqual(view.shape[0], 525)
        self.assertEqual(view.shape[1], 1400)

    def test_workspace_view_is_a_crop_of_the_same_frame(self):
        from scripts.live_ui import build_workspace_view, work_crop_rect
        polygon = np.array([[120, 100], [600, 100], [600, 500], [120, 500]], np.float32)
        x0, y0, x1, y1 = work_crop_rect(polygon, self.frames['vertical'].shape)
        self.assertLess((x1 - x0) * (y1 - y0),
                        self.frames['vertical'].shape[0] * self.frames['vertical'].shape[1])
        view = build_workspace_view(self.api, self.frames, self.masks, [], polygon,
                                    2500, True, ['state current'], lambda item, zh: '测试')
        self.assertGreater(view.shape[0], 0)
        self.assertGreater(view.shape[1], 0)


if __name__ == '__main__':
    unittest.main()