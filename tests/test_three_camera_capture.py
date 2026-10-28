from __future__ import annotations

import json
import cv2
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from 程序.three_camera_capture import (
    CAMERA_ROLES,
    FrameCaptureError,
    capture_frame_bundle,
    compose_preview,
    configure_camera,
    main,
    resolve_camera_specs,
    save_frame_bundle,
)


class FakeCamera:
    def __init__(self, role: str, events: list[tuple[str, str]], *, grab_ok: bool = True, retrieve_ok: bool = True):
        self.role = role
        self.events = events
        self.grab_ok = grab_ok
        self.retrieve_ok = retrieve_ok
        self.frame = np.full((48, 64, 3), len(role), dtype=np.uint8)

    def grab(self) -> bool:
        self.events.append(("grab", self.role))
        return self.grab_ok

    def retrieve(self) -> tuple[bool, np.ndarray]:
        self.events.append(("retrieve", self.role))
        return self.retrieve_ok, self.frame


class ThreeCameraCaptureTests(unittest.TestCase):
    def camera_config(self, indices: tuple[int | None, int | None, int | None]) -> dict:
        return {
            role: {"index": index, "device_name": f"device-{role}"}
            for role, index in zip(CAMERA_ROLES, indices)
        }

    def test_resolves_configured_and_overridden_indices(self) -> None:
        config = self.camera_config((1, None, 3))
        specs = resolve_camera_specs(config, {"vertical": None, "side_a": 2, "side_b": None})

        self.assertEqual([specs[role].index for role in CAMERA_ROLES], [1, 2, 3])
        self.assertEqual(specs["side_a"].device_name, "device-side_a")

    def test_requires_all_three_camera_indices(self) -> None:
        config = self.camera_config((1, None, None))

        with self.assertRaisesRegex(ValueError, "side_a"):
            resolve_camera_specs(config, {})

    def test_rejects_duplicate_camera_indices(self) -> None:
        config = self.camera_config((1, 1, 2))

        with self.assertRaisesRegex(ValueError, "不同|unique"):
            resolve_camera_specs(config, {})

    def test_grabs_every_camera_before_retrieving_any_frame(self) -> None:
        events: list[tuple[str, str]] = []
        cameras = {role: FakeCamera(role, events) for role in CAMERA_ROLES}
        ticks = iter((10.0, 10.012, 10.031))

        bundle = capture_frame_bundle(cameras, sequence=7, clock=lambda: next(ticks))

        self.assertEqual(
            events,
            [("grab", role) for role in CAMERA_ROLES]
            + [("retrieve", role) for role in CAMERA_ROLES],
        )
        self.assertEqual(bundle.sequence, 7)
        self.assertEqual(tuple(bundle.frames), CAMERA_ROLES)
        self.assertAlmostEqual(bundle.grab_spread_ms, 31.0, places=3)

    def test_failed_grab_names_camera_and_skips_retrieval(self) -> None:
        events: list[tuple[str, str]] = []
        cameras = {role: FakeCamera(role, events) for role in CAMERA_ROLES}
        cameras["side_a"].grab_ok = False

        with self.assertRaisesRegex(FrameCaptureError, "side_a.*grab"):
            capture_frame_bundle(cameras, sequence=1)

        self.assertEqual(events, [("grab", "vertical"), ("grab", "side_a")])

    def test_failed_retrieve_names_camera(self) -> None:
        events: list[tuple[str, str]] = []
        cameras = {role: FakeCamera(role, events) for role in CAMERA_ROLES}
        cameras["side_b"].retrieve_ok = False

        with self.assertRaisesRegex(FrameCaptureError, "side_b.*retrieve"):
            capture_frame_bundle(cameras, sequence=1)

    def test_preview_keeps_all_three_views_within_width(self) -> None:
        frames = {
            "vertical": np.zeros((100, 200, 3), dtype=np.uint8),
            "side_a": np.zeros((80, 80, 3), dtype=np.uint8),
            "side_b": np.zeros((60, 120, 3), dtype=np.uint8),
        }

        preview = compose_preview(frames, target_height=120, max_width=300)

        self.assertEqual(preview.ndim, 3)
        self.assertEqual(preview.shape[2], 3)
        self.assertLessEqual(preview.shape[1], 300)
        self.assertGreater(preview.shape[0], 0)

    def test_saves_three_images_and_unsynchronized_metadata(self) -> None:
        events: list[tuple[str, str]] = []
        cameras = {role: FakeCamera(role, events) for role in CAMERA_ROLES}
        bundle = capture_frame_bundle(cameras, sequence=4, clock=lambda: 1.0)
        specs = resolve_camera_specs(self.camera_config((1, 2, 3)), {})

        with tempfile.TemporaryDirectory() as temporary_dir:
            frame_dir = save_frame_bundle(bundle, Path(temporary_dir), specs)
            metadata = json.loads((frame_dir / "metadata.json").read_text(encoding="utf-8"))

            self.assertTrue(all((frame_dir / f"{role}.jpg").is_file() for role in CAMERA_ROLES))
            self.assertEqual(metadata["sequence"], 4)
            self.assertFalse(metadata["hardware_synchronized"])
            self.assertEqual(metadata["sync_method"], "sequential_grab_then_retrieve")
            self.assertEqual(metadata["cameras"]["side_b"]["index"], 3)
            self.assertEqual(metadata["captured_at_utc"], bundle.captured_at_utc)
            self.assertAlmostEqual(metadata["grab_spread_ms"], bundle.grab_spread_ms, places=3)
            self.assertEqual(metadata["grab_return_times_monotonic_s"], bundle.grab_times)


    def test_exposure_controls_are_opt_in_and_auto_white_balance_is_disabled_first(self) -> None:
        class FakeProperties:
            def __init__(self) -> None:
                self.calls: list[tuple[int, float]] = []
                self.values: dict[int, float] = {}

            def set(self, property_id: int, value: float) -> bool:
                self.calls.append((property_id, value))
                self.values[property_id] = value
                return True

            def get(self, property_id: int) -> float:
                return self.values.get(property_id, 0.0)

        settings = {
            "width": 640,
            "height": 480,
            "fps": 30,
            "fourcc": "MJPG",
            "auto_exposure": False,
            "exposure": -6,
            "gain": 2,
            "auto_white_balance": False,
            "white_balance": 4600,
        }
        camera = FakeProperties()
        exposure_properties = {
            cv2.CAP_PROP_AUTO_EXPOSURE,
            cv2.CAP_PROP_EXPOSURE,
            cv2.CAP_PROP_GAIN,
            cv2.CAP_PROP_AUTO_WB,
            cv2.CAP_PROP_WB_TEMPERATURE,
        }

        configure_camera(camera, settings)
        self.assertFalse(any(property_id in exposure_properties for property_id, _ in camera.calls))

        camera.calls.clear()
        configure_camera(camera, settings, apply_exposure_settings=True)
        properties = [property_id for property_id, _ in camera.calls]
        self.assertLess(properties.index(cv2.CAP_PROP_AUTO_WB), properties.index(cv2.CAP_PROP_WB_TEMPERATURE))

    def test_custom_roles_config_path_is_used_in_validation_error(self) -> None:
        config = self.camera_config((1, None, None))

        with self.assertRaisesRegex(ValueError, "custom_camera_roles.json"):
            resolve_camera_specs(config, {}, config_path=Path("custom_camera_roles.json"))

    def test_failed_image_write_leaves_no_partial_frame_set(self) -> None:
        events: list[tuple[str, str]] = []
        cameras = {role: FakeCamera(role, events) for role in CAMERA_ROLES}
        bundle = capture_frame_bundle(cameras, sequence=4, clock=lambda: 1.0)
        specs = resolve_camera_specs(self.camera_config((1, 2, 3)), {})
        original_imwrite = cv2.imwrite

        def fail_on_side_b(path: str, *args: object) -> bool:
            if Path(path).name == "side_b.jpg":
                return False
            return original_imwrite(path, *args)

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "frames"
            with patch("程序.three_camera_capture.cv2.imwrite", side_effect=fail_on_side_b):
                with self.assertRaises(OSError):
                    save_frame_bundle(bundle, output_dir, specs)

            self.assertTrue(output_dir.is_dir())
            self.assertEqual(list(output_dir.iterdir()), [])

    def test_invalid_camera_format_is_rejected_before_opening_devices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            roles_path = Path(temporary_dir) / "roles.json"
            settings_path = Path(temporary_dir) / "settings.json"
            roles_path.write_text(json.dumps(self.camera_config((1, 2, 3))), encoding="utf-8")
            settings_path.write_text(json.dumps({"width": 0, "height": 480, "fps": 30}), encoding="utf-8")

            with patch("程序.three_camera_capture.open_cameras", side_effect=RuntimeError("opened too early")) as opener:
                result = main([
                    "--camera-roles", str(roles_path),
                    "--camera-settings", str(settings_path),
                    "--headless", "--frames", "1",
                ])

            self.assertEqual(result, 2)
            opener.assert_not_called()
if __name__ == "__main__":
    unittest.main()
