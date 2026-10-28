"""opencv_shape_vision 的形状判定与实例分割测试。"""

from __future__ import annotations

import unittest

import cv2
import numpy as np
from 程序.opencv_shape_vision import (
    DEFAULTS,
    TRAY_DOT_REF,
    UNKNOWN,
    ShapeObject,
    analyse_image,
    classify_shape,
    find_dot_candidates,
    find_tray_quad,
    find_tray_roi,
    merge_pieces_by_colour,
    rank_picks,
    parse_overrides,
    render_synthetic_tray,
    segment_objects,
    shape_ambiguity,
    shape_features,
    suppress_duplicates,
)

def feature_vector(**overrides) -> dict[str, float]:
    """一个"典型立方体"的特征字典，测试时只覆盖关心的字段。"""
    base = dict(
        area=5000.0, perimeter=300.0, r_eq=40.0,
        circularity=0.70, solidity=0.95, convexity=0.95,
        rect_fill=0.85, extent=0.85, elong=1.05,
        circle_residual=0.15, hull_vertices=5.0, poly_vertices=5.0,
        linearity=0.3, radial_corr=0.0, spec_excess=20.0, plane_rmse=8.0,
    )
    base.update(overrides)
    return base


class ClassifyShapeTests(unittest.TestCase):
    def test_sphere_shading_is_ball(self) -> None:
        label, _ = classify_shape(feature_vector(circularity=0.88, solidity=0.99,
                                                 circle_residual=0.02, rect_fill=0.79,
                                                 elong=1.03, hull_vertices=8, radial_corr=-0.82))
        self.assertEqual(label, "ball")

    def test_flat_disc_is_cone(self) -> None:
        label, _ = classify_shape(feature_vector(circularity=0.88, solidity=0.99,
                                                 circle_residual=0.02, rect_fill=0.79,
                                                 elong=1.03, hull_vertices=8, radial_corr=-0.30))
        self.assertEqual(label, "cone")

    def test_round_but_stretched_is_cylinder(self) -> None:
        label, _ = classify_shape(feature_vector(circularity=0.85, solidity=0.99,
                                                 circle_residual=0.07, rect_fill=0.78,
                                                 elong=1.17, hull_vertices=8, radial_corr=0.12))
        self.assertEqual(label, "cylinder")

    def test_long_high_fill_rectangle_is_cuboid(self) -> None:
        label, _ = classify_shape(feature_vector(rect_fill=0.96, elong=2.05,
                                                 hull_vertices=4, circularity=0.68))
        self.assertEqual(label, "cuboid")

    def test_perfect_square_is_cube(self) -> None:
        label, _ = classify_shape(feature_vector(rect_fill=1.0, elong=1.0, hull_vertices=4,
                                                 circularity=0.78, solidity=1.0))
        self.assertEqual(label, "cube")

    def test_triangle_low_fill_is_tetrahedron(self) -> None:
        label, _ = classify_shape(feature_vector(rect_fill=0.50, elong=1.16, hull_vertices=3,
                                                 circularity=0.60, solidity=0.97, radial_corr=0.32))
        self.assertEqual(label, "tetrahedron")

    def test_five_sided_outline_is_pentagonal_prism(self) -> None:
        label, _ = classify_shape(feature_vector(rect_fill=0.70, elong=1.04, hull_vertices=5,
                                                 circularity=0.79, solidity=0.99,
                                                 circle_residual=0.067, radial_corr=0.0))
        self.assertEqual(label, "pentagonal_prism")

    def test_six_sided_outline_is_hexagonal_prism(self) -> None:
        label, _ = classify_shape(feature_vector(rect_fill=0.76, elong=1.14, hull_vertices=6,
                                                 circularity=0.84, solidity=0.99,
                                                 circle_residual=0.044, radial_corr=0.0))
        self.assertEqual(label, "hexagonal_prism")

    def test_mid_fill_rectangle_is_unknown(self) -> None:
        """矩形轮廓但填充率不够（0.88~0.96）→ 可能是躺倒的棱柱，宁可拒判。"""
        label, _ = classify_shape(feature_vector(rect_fill=0.93, elong=1.5, hull_vertices=4,
                                                 circularity=0.66, solidity=0.96))
        self.assertEqual(label, UNKNOWN)

    def test_near_perfect_rectangle_is_cuboid(self) -> None:
        """真矩形（凸包几乎填满外接矩形）才是长方体，不能和上面的歧义区混为一谈。"""
        label, _ = classify_shape(feature_vector(rect_fill=0.98, elong=1.5, hull_vertices=4,
                                                 circularity=0.66, solidity=0.96))
        self.assertEqual(label, "cuboid")

    def test_triangle_without_shading_trend_is_unknown(self) -> None:
        """四面体与三角棱柱从上方看都是三角形；亮度沿半径无趋势时拒判。"""
        label, _ = classify_shape(feature_vector(rect_fill=0.60, elong=1.30, hull_vertices=4,
                                                 circularity=0.60, solidity=0.96, radial_corr=0.05))
        self.assertEqual(label, UNKNOWN)

    def test_triangle_with_shading_trend_is_classified(self) -> None:
        """亮度明显沿半径变化时，三角族仍然给出结论。"""
        label, _ = classify_shape(feature_vector(rect_fill=0.62, elong=1.29, hull_vertices=5,
                                                 circularity=0.63, solidity=0.96, radial_corr=-0.35))
        self.assertEqual(label, "triangular_prism")

    def test_round_contour_with_shading_is_cone(self) -> None:
        """轮廓近似顶点数多（圆≈8）且亮度随半径下降 → 圆锥，而不是六棱柱。"""
        label, _ = classify_shape(feature_vector(rect_fill=0.79, elong=1.10, hull_vertices=7,
                                                 poly_vertices=8, circularity=0.58,
                                                 solidity=0.92, radial_corr=-0.59))
        self.assertEqual(label, "cone")

    def test_polygon_outline_is_not_taken_as_cone(self) -> None:
        """六棱柱的轮廓顶点数是 6，不能被"圆形"判据收走。"""
        label, _ = classify_shape(feature_vector(rect_fill=0.77, elong=1.13, hull_vertices=7,
                                                 poly_vertices=6, circularity=0.57,
                                                 solidity=0.86, radial_corr=-0.56))
        self.assertEqual(label, "hexagonal_prism")

    def test_flat_disc_between_cone_and_cylinder_is_unknown(self) -> None:
        """俯视正圆盘：圆锥与圆柱都呈圆形且亮度平坦，分不清就拒判。"""
        label, _ = classify_shape(feature_vector(circularity=0.84, solidity=0.99,
                                                 circle_residual=0.08, rect_fill=0.80,
                                                 elong=1.06, hull_vertices=7,
                                                 poly_vertices=7, radial_corr=0.09))
        self.assertEqual(label, UNKNOWN)

    def test_round_hexagon_is_not_taken_as_flat_disc(self) -> None:
        """正六边形圆度也能到 0.84，但轮廓顶点只有 5~6，不能被圆盘判据收走。"""
        label, _ = classify_shape(feature_vector(circularity=0.84, solidity=0.99,
                                                 circle_residual=0.05, rect_fill=0.88,
                                                 elong=1.05, hull_vertices=6,
                                                 poly_vertices=6, radial_corr=0.05))
        self.assertEqual(label, "hexagonal_prism")

    def test_compact_five_sided_blob_is_unknown(self) -> None:
        """5 边形轮廓 + 中高填充率：斜视角立方体与五棱柱不可分 → 拒判。"""
        label, _ = classify_shape(feature_vector(circularity=0.75, solidity=0.97,
                                                 rect_fill=0.94, hull_vertices=5,
                                                 poly_vertices=5, elong=1.12,
                                                 radial_corr=-0.28))
        self.assertEqual(label, UNKNOWN)

    def test_elongated_five_sided_blob_is_still_classified(self) -> None:
        """细长的 5 边形块靠长宽比能和立方体分开，不该被歧义判据吞掉。"""
        label, _ = classify_shape(feature_vector(circularity=0.70, solidity=0.96,
                                                 rect_fill=0.93, hull_vertices=5,
                                                 poly_vertices=5, elong=1.37,
                                                 radial_corr=-0.43))
        self.assertEqual(label, "hexagonal_prism")

    def test_lumpy_polygon_is_unknown(self) -> None:
        """两个物体粘成的哑铃形：圆度很低，规则多边形不可能这么低 → 拒判。"""
        label, _ = classify_shape(feature_vector(circularity=0.42, solidity=0.83,
                                                 convexity=0.77, rect_fill=0.82,
                                                 hull_vertices=8, poly_vertices=10,
                                                 elong=1.77, radial_corr=-0.20))
        self.assertEqual(label, UNKNOWN)


    def test_confidence_is_reported(self) -> None:
        label, confidence = classify_shape(feature_vector(rect_fill=0.96, elong=2.05, hull_vertices=4))
        self.assertEqual(label, "cuboid")
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class ParseOverridesTests(unittest.TestCase):
    def test_parses_known_keys(self) -> None:
        self.assertEqual(parse_overrides(["cuboid_fill=0.90", "tri_ambig_band=0.0"]),
                         {"cuboid_fill": 0.90, "tri_ambig_band": 0.0})

    def test_rejects_unknown_key(self) -> None:
        with self.assertRaises(SystemExit):
            parse_overrides(["no_such_key=1"])

    def test_rejects_missing_equals(self) -> None:
        with self.assertRaises(SystemExit):
            parse_overrides(["cuboid_fill"])


class TrayDotTests(unittest.TestCase):
    """托盘四角黑点定位。"""

    SIZE = (1296, 972)   # 面积是标定分辨率的 1/4，阈值会自动换算

    def _tray_with_dots(self, skip: tuple[int, ...] = ()) -> np.ndarray:
        w, h = self.SIZE
        img = np.full((h, w, 3), 235, np.uint8)      # 亮托盘
        for i, (nx, ny) in enumerate(TRAY_DOT_REF):
            if i in skip:
                continue
            cv2.circle(img, (int(nx * w), int(ny * h)), 8, (10, 10, 10), -1)
        return img

    def test_four_dots_give_the_reference_quad(self) -> None:
        quad = find_tray_quad(self._tray_with_dots(), DEFAULTS)
        self.assertIsNotNone(quad)
        w, h = self.SIZE
        expect = np.array(TRAY_DOT_REF, np.float32) * np.array([w, h], np.float32)
        centre = expect.mean(axis=0)
        expect = centre + (expect - centre) * (1.0 + DEFAULTS["dot_roi_expand"])
        self.assertTrue(np.allclose(quad, expect, atol=6.0),
                        f"期望 {expect.tolist()}，实际 {quad.tolist()}")

    def test_one_dot_hidden_is_completed(self) -> None:
        """角点被物块挡住时，靠基准布局补全，四边形仍应接近参考位置。"""
        quad = find_tray_quad(self._tray_with_dots(skip=(0,)), DEFAULTS)
        self.assertIsNotNone(quad)
        w, h = self.SIZE
        expect = np.array(TRAY_DOT_REF, np.float32) * np.array([w, h], np.float32)
        centre = expect.mean(axis=0)
        expect = centre + (expect - centre) * (1.0 + DEFAULTS["dot_roi_expand"])
        self.assertTrue(np.allclose(quad, expect, atol=12.0))

    def test_dot_on_dark_background_is_ignored(self) -> None:
        """四周不是亮托盘的暗斑不能当角点——灰色台面上的暗斑就靠这条排掉。"""
        w, h = self.SIZE
        img = np.full((h, w, 3), 60, np.uint8)       # 暗背景
        for nx, ny in TRAY_DOT_REF:
            cv2.circle(img, (int(nx * w), int(ny * h)), 8, (10, 10, 10), -1)
        self.assertIsNone(find_tray_quad(img, DEFAULTS))

    def test_blank_tray_has_no_quad(self) -> None:
        w, h = self.SIZE
        self.assertIsNone(find_tray_quad(np.full((h, w, 3), 235, np.uint8), DEFAULTS))

    def test_dot_near_tray_edge_is_still_found(self) -> None:
        """角点紧贴托盘边缘时，环带里会混进托盘外的暗背景。

        判据必须用"环带亮像素占比"而不是平均亮度，否则真角点会被误杀
        （实测侧视图里真角点环带均值只有 124，但亮占比 0.63）。
        """
        w, h = self.SIZE
        img = np.full((h, w, 3), 235, np.uint8)
        # 托盘只占上半部分，下半部分是暗背景
        img[h // 2:, :] = 30
        for nx, ny in TRAY_DOT_REF:
            cx, cy = int(nx * w), int(ny * h)
            if cy >= h // 2:
                cy = h // 2 - 12          # 让角点紧贴托盘边缘，环带一半落在暗背景里
            cv2.circle(img, (cx, cy), 8, (10, 10, 10), -1)
        cands = find_dot_candidates(img, DEFAULTS)
        self.assertGreaterEqual(len(cands), 2, f"紧贴边缘的角点被误杀了：{cands}")


class ShapeFeatureTests(unittest.TestCase):
    def _mask(self, draw) -> np.ndarray:
        mask = np.zeros((400, 400), np.uint8)
        draw(mask)
        return mask

    def _lab(self, mask: np.ndarray, gradient: bool = False) -> np.ndarray:
        lab = np.full((400, 400, 3), 230.0, np.float32)
        if gradient:
            ys, xs = np.nonzero(mask)
            cy, cx = ys.mean(), xs.mean()
            radius = np.hypot(xs - cx, ys - cy)
            lab[ys, xs, 0] = 240.0 - 70.0 * (radius / max(radius.max(), 1e-6))
        return lab

    def test_circle_has_high_circularity(self) -> None:
        mask = self._mask(lambda m: cv2.circle(m, (200, 200), 120, 1, -1))
        f = shape_features(mask, self._lab(mask))
        self.assertIsNotNone(f)
        self.assertGreater(f["circularity"], 0.85)
        self.assertLess(f["circle_residual"], 0.05)

    def test_sphere_gradient_gives_negative_radial_correlation(self) -> None:
        mask = self._mask(lambda m: cv2.circle(m, (200, 200), 120, 1, -1))
        sphere = shape_features(mask, self._lab(mask, gradient=True))
        flat = shape_features(mask, self._lab(mask, gradient=False))
        self.assertLess(sphere["radial_corr"], -0.5)
        self.assertGreater(abs(flat["radial_corr"]), -0.5)

    def test_square_fills_its_bounding_rect(self) -> None:
        mask = self._mask(lambda m: cv2.rectangle(m, (100, 100), (300, 300), 1, -1))
        f = shape_features(mask, self._lab(mask))
        self.assertGreater(f["rect_fill"], 0.97)
        self.assertAlmostEqual(f["elong"], 1.0, places=1)

    def test_tiny_mask_returns_none(self) -> None:
        mask = self._mask(lambda m: cv2.circle(m, (10, 10), 3, 1, -1))
        self.assertIsNone(shape_features(mask, self._lab(mask)))


class DetectionTests(unittest.TestCase):
    def test_finds_and_names_synthetic_shapes(self) -> None:
        image = render_synthetic_tray(["ball", "cube", "cuboid", "tetrahedron"])
        _, objects = analyse_image(image)
        labels = {obj.label for obj in objects}
        self.assertEqual(len(objects), 4)
        self.assertTrue(labels & {"ball", "cone"})
        self.assertTrue(labels & {"cube", "cuboid"})
        self.assertIn("tetrahedron", labels)

    def test_tray_roi_excludes_the_area_outside_the_tray(self) -> None:
        image = np.full((800, 1200, 3), 40, np.uint8)          # 深色台面
        cv2.rectangle(image, (250, 150), (950, 650), (238, 238, 238), -1)  # 白色托盘
        cv2.circle(image, (110, 700), 60, (60, 60, 200), -1)   # 托盘外的干扰物
        cv2.circle(image, (600, 400), 60, (60, 200, 60), -1)   # 托盘内的物块
        small = cv2.resize(image, None, fx=DEFAULTS["scale"], fy=DEFAULTS["scale"])
        roi, box = find_tray_roi(small, DEFAULTS)
        self.assertIsNotNone(roi)
        self.assertIsNotNone(box)
        objects = segment_objects(small)
        self.assertEqual(len(objects), 1)
        centre_x = objects[0]["center"][0]
        self.assertGreater(centre_x, small.shape[1] * 0.3)

    def test_duplicate_masks_are_suppressed(self) -> None:
        mask = np.zeros((100, 100), np.uint8)
        cv2.circle(mask, (50, 50), 30, 1, -1)
        smaller = np.zeros_like(mask)
        cv2.circle(smaller, (50, 50), 25, 1, -1)
        kept = suppress_duplicates([
            dict(mask=mask, area=int(mask.sum())),
            dict(mask=smaller, area=int(smaller.sum())),
        ])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["area"], int(mask.sum()))



class RankingTests(unittest.TestCase):
    """抓取排序：先看单独（净空），再看形状是否规整无歧义。"""

    def make_object(self, centre, r_eq=40.0, label="cuboid", **overrides):
        features = feature_vector(**overrides)
        features["r_eq"] = r_eq
        mask = np.zeros((400, 400), np.uint8)
        cv2.circle(mask, (int(centre[0]), int(centre[1])), int(r_eq), 1, -1)
        bbox = (int(centre[0] - r_eq), int(centre[1] - r_eq),
                int(centre[0] + r_eq), int(centre[1] + r_eq))
        return ShapeObject(bbox=bbox, area=int(mask.sum()), label=label,
                           confidence=0.65, features=features, mask=mask, center=centre)

    def test_isolated_block_outranks_crowded_ones(self) -> None:
        isolated = self.make_object((300, 300))
        crowded_a = self.make_object((80, 80))
        crowded_b = self.make_object((150, 80))

        ordered = rank_picks([isolated, crowded_a, crowded_b], roi=None, cfg={})

        self.assertTrue(ordered[0].pickable)
        self.assertEqual(ordered[0].rank, 1)
        self.assertIs(ordered[0], isolated)
        self.assertFalse(crowded_a.pickable)
        self.assertFalse(crowded_b.pickable)

    def test_clearance_is_centroid_gap_minus_radii(self) -> None:
        left = self.make_object((100, 100), r_eq=40)
        right = self.make_object((260, 100), r_eq=40)

        rank_picks([left, right], roi=None, cfg={})

        self.assertAlmostEqual(left.clearance_px, 80.0, places=1)
        self.assertAlmostEqual(left.clearance_ratio, 2.0, places=1)

    def test_tray_edge_counts_as_obstacle(self) -> None:
        obj = self.make_object((60, 200), r_eq=40)
        roi = np.zeros((400, 400), np.uint8)
        roi[20:380, 20:380] = 1

        rank_picks([obj], roi, cfg={})

        self.assertLess(obj.clearance_px, 5.0)
        self.assertFalse(obj.pickable)

    def test_confusable_shape_is_not_pickable(self) -> None:
        # 明暗结构落在"球/圆锥"判据边界，长宽比又贴着"圆锥/圆柱"分界
        obj = self.make_object((300, 300), label="cone", radial_corr=-0.50, elong=1.15)

        ordered = rank_picks([obj], roi=None, cfg={})

        self.assertFalse(ordered[0].pickable)
        self.assertGreater(ordered[0].ambiguity, 0.35)
        self.assertTrue(any("形状存疑" in reason for reason in ordered[0].reasons))

    def test_ragged_contour_stays_pickable_but_ranks_lower(self) -> None:
        clean = self.make_object((300, 300))
        ragged = self.make_object((80, 80), solidity=0.85, convexity=0.85)

        ordered = rank_picks([clean, ragged], roi=None, cfg={})

        self.assertTrue(ragged.pickable)      # 轮廓粗糙不等于形状判错
        self.assertIs(ordered[0], clean)      # 但优先级排在干净那块之后

    def test_clean_shape_has_no_ambiguity(self) -> None:
        obj = self.make_object((300, 300), circularity=0.88, solidity=0.99,
                               convexity=0.96, circle_residual=0.02, rect_fill=0.79,
                               elong=1.03, hull_vertices=8, radial_corr=-0.82)
        ambiguity, reasons = shape_ambiguity(obj.features, "ball", 0.75)
        self.assertEqual(ambiguity, 0.0)
        self.assertEqual(reasons, [])

class MergePiecesTests(unittest.TestCase):
    """过分割后的合并判据：同色 + 没有可见接缝才合并。"""

    def build(self, seam_gradient: float):
        mask_left = np.zeros((200, 200), np.uint8)
        mask_right = np.zeros((200, 200), np.uint8)
        mask_left[60:140, 40:100] = 1
        mask_right[60:140, 100:160] = 1
        lab = np.zeros((200, 200, 3), np.float32)
        lab[:, :, 0] = 200.0
        grad = np.zeros((200, 200), np.float32)
        grad[60:140, 99:101] = seam_gradient
        return [mask_left, mask_right], lab, grad

    def test_pieces_without_seam_are_merged(self) -> None:
        pieces, lab, grad = self.build(seam_gradient=0.0)
        merged = merge_pieces_by_colour(pieces, lab, grad, DEFAULTS)
        self.assertEqual(len(merged), 1)

    def test_pieces_with_visible_seam_stay_separate(self) -> None:
        pieces, lab, grad = self.build(seam_gradient=120.0)
        merged = merge_pieces_by_colour(pieces, lab, grad, DEFAULTS)
        self.assertEqual(len(merged), 2)

    def test_different_colours_stay_separate(self) -> None:
        pieces, lab, grad = self.build(seam_gradient=0.0)
        lab[60:140, 100:160, 0] = 120.0          # 右块颜色明显不同
        merged = merge_pieces_by_colour(pieces, lab, grad, DEFAULTS)
        self.assertEqual(len(merged), 2)

    def test_pieces_split_by_a_gap_are_still_merged(self) -> None:
        """分水岭会在两块之间留一条 1 像素的边界线，必须跨过它判相邻。"""
        pieces, lab, grad = self.build(seam_gradient=0.0)
        pieces[0] = pieces[0].copy()
        pieces[1] = pieces[1].copy()
        pieces[0][:, 99:100] = 0          # 挖掉 1 像素，模拟分水岭边界线
        merged = merge_pieces_by_colour(pieces, lab, grad, DEFAULTS)
        self.assertEqual(len(merged), 1)

    def test_gap_adjacency_can_be_disabled(self) -> None:
        pieces, lab, grad = self.build(seam_gradient=0.0)
        pieces[0][:, 99:100] = 0
        merged = merge_pieces_by_colour(pieces, lab, grad,
                                       {**DEFAULTS, "merge_adjacency_gap": 1})
        self.assertEqual(len(merged), 2)

    def test_sliver_mask_is_not_an_object(self) -> None:
        """物体边缘的细条不是物块：长宽比过大或填不满外接框都应被丢掉。"""
        mask = np.zeros((400, 400), np.uint8)
        mask[100:340, 180:196] = 1            # 240x16 的细条，长宽比 15
        lab = np.full((400, 400, 3), 230.0, np.float32)
        self.assertIsNone(shape_features(mask, lab, DEFAULTS))

class DamagedMaskTests(unittest.TestCase):
    """掩膜缺角时拒判，不硬分类。"""

    def test_damaged_mask_is_unknown(self) -> None:
        features = feature_vector(solidity=0.60, convexity=0.55)
        label, confidence = classify_shape(features)
        self.assertEqual(label, UNKNOWN)
        self.assertEqual(confidence, 0.0)

    def test_damaged_mask_object_is_not_pickable(self) -> None:
        features = feature_vector(solidity=0.60, convexity=0.55)
        features["r_eq"] = 40.0
        mask = np.zeros((400, 400), np.uint8)
        cv2.circle(mask, (300, 300), 40, 1, -1)
        obj = ShapeObject(bbox=(260, 260, 340, 340), area=int(mask.sum()), label=UNKNOWN,
                          confidence=0.0, features=features, mask=mask, center=(300.0, 300.0))

        ordered = rank_picks([obj], roi=None, cfg={})

        self.assertFalse(ordered[0].pickable)
        self.assertTrue(any("拒判" in reason for reason in ordered[0].reasons))

    def test_clean_mask_still_classifies(self) -> None:
        features = feature_vector(solidity=0.99, convexity=0.96, rect_fill=0.96,
                                  elong=2.05, hull_vertices=4)
        label, _ = classify_shape(features)
        self.assertEqual(label, "cuboid")


class HullFillTests(unittest.TestCase):
    """凸包填充率：掩膜缺角时仍然稳定。"""

    def test_triangle_hull_fill_is_low(self) -> None:
        mask = np.zeros((300, 300), np.uint8)
        cv2.fillPoly(mask, [np.array([[150, 60], [250, 240], [50, 240]], np.int32)], 1)
        f = shape_features(mask, np.full((300, 300, 3), 230.0, np.float32))
        self.assertLess(f["hull_rect_fill"], 0.70)

    def test_square_hull_fill_is_high(self) -> None:
        mask = np.zeros((300, 300), np.uint8)
        cv2.rectangle(mask, (80, 80), (220, 220), 1, -1)
        f = shape_features(mask, np.full((300, 300, 3), 230.0, np.float32))
        self.assertGreater(f["hull_rect_fill"], 0.97)

    def test_hull_fill_survives_a_chipped_mask(self) -> None:
        mask = np.zeros((300, 300), np.uint8)
        cv2.rectangle(mask, (80, 80), (220, 220), 1, -1)
        mask[80:150, 80:120] = 0                 # 挖掉一角，模拟被邻块吃掉
        f = shape_features(mask, np.full((300, 300, 3), 230.0, np.float32))
        # 掩膜填充率明显下降，凸包填充率仍然很高
        # 掩膜填充率明显下降，凸包填充率基本不受影响（缺口被凸包补回来）
        self.assertLess(f["rect_fill"], 0.90)
        self.assertGreater(f["hull_rect_fill"], f["rect_fill"] + 0.05)
        self.assertGreater(f["hull_rect_fill"], 0.90)

if __name__ == "__main__":
    unittest.main()
