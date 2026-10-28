"""Evidence checks for warm starts and frame-bound result lifetimes."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from scripts.two_view_live import mask_signature, scene_matches, stamped_payload
from scripts.two_view_model_fit import fit_two_view
import test_two_view_model_fit as model_fixtures


class RuntimeCacheTests(unittest.TestCase):
    def test_small_disappearing_instance_is_not_hidden_by_global_iou(self):
        mask = np.zeros((240, 320), np.uint8)
        mask[10:200, 10:200] = 1
        mask[210:215, 250:255] = 1
        before = mask_signature({'vertical':mask})
        mask[210:215, 250:255] = 0
        after = mask_signature({'vertical':mask})
        self.assertFalse(scene_matches(before, after))

    def test_output_expiry_is_anchored_to_source_not_publish_time(self):
        api = SimpleNamespace(detections_to_payload=lambda values: [{'class':'cube'}])
        with patch('scripts.two_view_live.time.time', return_value=1000.):
            item = stamped_payload(api, [], 50., 53., 8.)[0]
        self.assertEqual(item['observation']['source_unix'],997.)
        self.assertEqual(item['observation']['valid_until_unix'],1005.)
        self.assertEqual(item['observation']['updated_unix'],1000.)

    def test_warm_start_compares_every_family_and_keeps_cube_geometry(self):
        cal,cameras = model_fixtures.calibration()
        xy=np.array([112.,80.])
        contours=model_fixtures.project_solid(model_fixtures.box_points(xy,(38.,38.,38.),27.),cameras)
        cold=fit_two_view(*contours,cal,xy,(38.,38.,38.))
        self.assertTrue(cold['accepted'])
        warm=fit_two_view(*contours,cal,xy,(38.,38.,38.),warm_start=cold['warm_start'])
        self.assertEqual(set(cold['warm_start']),set(warm['warm_start']))
        self.assertEqual(len(warm['warm_start']),11)
        self.assertTrue(warm['accepted'])
        self.assertEqual(warm['label'],'cube')
        self.assertLess(abs(warm['height_mm']-38.),.1)
        self.assertLess(np.linalg.norm(np.array(warm['center_mm'])-xy),.1)

    def test_invalid_warm_hint_falls_back_to_normal_search(self):
        cal,cameras=model_fixtures.calibration()
        xy=np.array([112.,80.])
        contours=model_fixtures.project_solid(model_fixtures.box_points(xy,(38.,38.,38.),27.),cameras)
        result=fit_two_view(*contours,cal,xy,38.,warm_start={'cuboid':[float('nan')]})
        self.assertTrue(result['accepted'])
        self.assertEqual(result['label'],'cube')


class ObjectValidityTests(unittest.TestCase):
    def make_scene(self):
        import cv2
        top = np.zeros((300, 400), np.uint8)
        side = np.zeros_like(top)
        cv2.rectangle(top, (100,100), (150,150), 1, -1)
        cv2.rectangle(side, (120,120), (170,170), 1, -1)
        contour = cv2.findContours(top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        item = SimpleNamespace(contour_px=contour, association=SimpleNamespace(mask=side))
        return item, {'vertical':top, 'side_a':side}

    def test_unrelated_background_noise_does_not_hide_a_stable_target(self):
        from scripts.two_view_live import object_is_current
        item, masks = self.make_scene()
        masks['vertical'][250:260,300:310] = 1
        self.assertTrue(object_is_current(item, masks))

    def test_removed_target_is_not_displayed_from_cached_fit(self):
        from scripts.two_view_live import object_is_current
        item, masks = self.make_scene()
        masks['vertical'][:] = 0
        self.assertFalse(object_is_current(item, masks))

    def test_changed_side_silhouette_invalidates_target(self):
        from scripts.two_view_live import object_is_current
        item, masks = self.make_scene()
        masks['side_a'] = np.roll(masks['side_a'],30,axis=0)
        self.assertFalse(object_is_current(item, masks))


if __name__=='__main__':
    unittest.main()