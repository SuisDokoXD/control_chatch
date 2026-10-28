"""User-selected image workspace must not overwrite black-dot metric calibration."""
import copy
from types import SimpleNamespace
import unittest
import cv2
import numpy as np

from scripts.vision_workspace import work_area_polygon_px, workspace_source
from scripts.two_view_live import segment_pair


class ManualWorkspaceTests(unittest.TestCase):
    def fixture(self):
        return {'tray':{
            'corners_px':dict(zip(('TL','TR','BR','BL'),[[80,80],[220,80],[220,220],[80,220]])),
            'outer_corners_px':dict(zip(('TL','TR','BR','BL'),[[20,20],[280,20],[280,280],[20,280]])),
            'slope_width_mm':{'left_mm':30,'right_mm':30},
            'homography_px_to_mm':np.eye(3).tolist(),
        }}

    def test_manual_points_are_used_exactly_and_calibration_is_not_modified(self):
        cal=self.fixture(); original=copy.deepcopy(cal)
        expected=np.float32([[20,20],[280,20],[280,280],[20,280]])
        np.testing.assert_array_equal(work_area_polygon_px(cal,(300,300)),expected)
        self.assertEqual(workspace_source(cal),'manual:tray.outer_corners_px')
        self.assertEqual(cal,original)

    def test_missing_manual_points_falls_back_to_calibrated_four_points(self):
        cal=self.fixture();del cal['tray']['outer_corners_px']
        np.testing.assert_array_equal(work_area_polygon_px(cal),[[80,80],[220,80],[220,220],[80,220]])
        self.assertEqual(workspace_source(cal),'calibrated:tray.corners_px')

    def test_invalid_manual_points_never_silently_select_another_roi(self):
        for points in ([[0,0]]*4,[[0,0],[50,50],[0,50],[50,0]],[[0,0],[float('nan'),0],[10,10],[0,10]]):
            cal=self.fixture();cal['tray']['outer_corners_px']=dict(zip(('TL','TR','BR','BL'),points))
            with self.assertRaises(ValueError):work_area_polygon_px(cal)
        with self.assertRaises(ValueError):work_area_polygon_px(self.fixture(),(100,100))

    def test_segmentation_keeps_object_inside_manual_area_outside_black_dot_floor(self):
        mask=np.zeros((300,300),np.uint8)
        cv2.rectangle(mask,(230,110),(260,150),1,-1)
        cv2.rectangle(mask,(0,0),(10,10),1,-1)
        api=SimpleNamespace(foreground=lambda *args:mask.copy(),
                            structure_foreground=lambda image,roi:np.zeros_like(roi))
        frames={k:np.zeros((300,300,3),np.uint8) for k in ('vertical','side_a')}
        masks=segment_pair(api,frames,frames,self.fixture(),28,10)
        self.assertEqual(int(masks['vertical'][130,245]),1)
        self.assertEqual(int(masks['vertical'][5,5]),0)
        self.assertEqual(int(masks['side_a'][5,5]),1)  # no invented side manual points


if __name__=='__main__':unittest.main()