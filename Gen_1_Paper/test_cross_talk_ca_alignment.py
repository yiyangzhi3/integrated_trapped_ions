"""Headless regression checks for both notebooks' manual alignment."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

import numpy as np
import tifffile


def notebook_functions(filename):
    path = Path(__file__).with_name(filename)
    notebook = json.loads(path.read_text())
    module = types.ModuleType(path.stem)
    for cell in notebook["cells"][:5]:
        if "functions" in cell.get("metadata", {}).get("tags", []):
            exec(compile("".join(cell["source"]), str(path), "exec"), module.__dict__)
    return module


class ManualAlignmentTests(unittest.TestCase):
    functions = notebook_functions("cross_talk_analysis_ca_2.ipynb")

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = np.zeros((81, 121, 3), dtype=np.uint8)
        self.image[30:33, 70:73] = [120, 80, 40]
        self.reference = self.root / "focused.tiff"
        tifffile.imwrite(self.reference, self.image, photometric="rgb")
        self.dataset = dict(name="test_729nm", reference_path=self.reference)

    def test_zoom_boxes_preserve_full_image_and_origin_pixel_coordinates(self):
        b = self.functions
        angle_box = np.array([[10., 10.], [100., 50.]])
        angle_points = np.array([[20., 25.], [80., 31.]])
        origin_box = np.array([[30., 20.], [90., 65.]])
        origin_points = np.array([[40., 42.], [80., 42.]])
        with patch.object(b.GateImageProfiler, "_require_interactive_backend"), patch.object(
            b.GateImageProfiler, "_pick_two_points",
            side_effect=[angle_box, angle_points, origin_box, origin_points],
        ) as pick:
            record = b.select_alignment(self.dataset, self.root / "alignments")

        self.assertEqual(pick.call_count, 4)
        for call in pick.call_args_list:
            self.assertEqual(call.args[0].shape, self.image.shape)
        np.testing.assert_array_equal(pick.call_args_list[1].kwargs["roi"], angle_box)
        np.testing.assert_array_equal(pick.call_args_list[3].kwargs["roi"], origin_box)
        self.assertAlmostEqual(record["angle_deg"], np.degrees(np.arctan2(6, 60)))
        np.testing.assert_array_equal(record["origin_px"], [60., 42.])
        np.testing.assert_array_equal(record["rotation_roi_px"], angle_box)
        np.testing.assert_array_equal(record["origin_roi_px"], origin_box)
        rotated = b.rotate_pixels(self.image, record["angle_deg"])
        np.testing.assert_array_equal(pick.call_args_list[2].args[0], rotated)
        np.testing.assert_array_equal(pick.call_args_list[3].args[0], rotated)
        np.testing.assert_array_equal(tifffile.imread(self.reference), self.image)
        self.assertEqual(b.load_alignment(self.dataset, self.root / "alignments"), record)

        # Rotation never transforms the axes; only the selected origin and
        # calibration determine their values on the original full-size grid.
        x, y, extent = b.image_axes(rotated.shape, record)
        unrotated_axes = b.image_axes(self.image.shape, dict(record, angle_deg=0.))
        np.testing.assert_array_equal(x, unrotated_axes[0])
        np.testing.assert_array_equal(y, unrotated_axes[1])
        self.assertEqual(extent, unrotated_axes[2])
        self.assertEqual((len(y), len(x)), self.image.shape[:2])
        self.assertEqual(x[60], 0.)
        self.assertEqual(y[42], 0.)
        np.testing.assert_allclose(np.diff(x), 5.3 / 50.)
        np.testing.assert_allclose(np.diff(y), 5.3 / 50.)

    def test_rotation_moves_only_intensity_on_the_original_canvas(self):
        b = self.functions
        image = np.zeros((61, 81, 3), dtype=np.uint8)
        image[30, 50] = [30, 60, 90]
        rotated = b.rotate_pixels(image, 90.)
        self.assertEqual(rotated.shape, image.shape)
        self.assertEqual(rotated.dtype, image.dtype)
        np.testing.assert_array_equal(rotated[20, 40], [30, 60, 90])
        np.testing.assert_array_equal(image[30, 50], [30, 60, 90])
        self.assertEqual(np.count_nonzero(rotated), 3)

    def test_cancelled_selection_preserves_saved_alignment(self):
        b = self.functions
        directory = self.root / "alignments"
        saved = b.save_alignment(self.dataset, dict(angle_deg=1.), directory)
        path = directory / "test_729nm.json"
        before = path.read_bytes()
        with patch.object(b.GateImageProfiler, "_require_interactive_backend"), patch.object(
            b.GateImageProfiler, "_pick_two_points", side_effect=RuntimeError("cancelled"),
        ):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                b.select_alignment(self.dataset, directory)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(b.load_alignment(self.dataset, directory), saved)


class BaManualAlignmentTests(ManualAlignmentTests):
    functions = notebook_functions("cross_talk_analysis.ipynb")


class CurrentBaManualAlignmentTests(ManualAlignmentTests):
    functions = notebook_functions("cross_talk_analysis_ba.ipynb")

    def test_brightness_boost_reaches_zoom_views_only(self):
        b = self.functions
        points = [np.array([[10., 10.], [100., 50.]]),
                  np.array([[20., 25.], [80., 31.]]),
                  np.array([[30., 20.], [90., 65.]]),
                  np.array([[40., 42.], [80., 42.]])]
        with patch.object(b.GateImageProfiler, "_require_interactive_backend"), patch.object(
            b.GateImageProfiler, "_pick_two_points", side_effect=points,
        ) as pick:
            record = b.select_alignment(self.dataset, self.root / "alignments", zoom_brightness_gain=4.)
        self.assertEqual(record["zoom_display_brightness_gain"], 4.)
        self.assertNotIn("display_gain", pick.call_args_list[0].kwargs)
        self.assertNotIn("display_gain", pick.call_args_list[2].kwargs)
        self.assertEqual(pick.call_args_list[1].kwargs["display_gain"], 4.)
        self.assertEqual(pick.call_args_list[3].kwargs["display_gain"], 4.)
        np.testing.assert_array_equal(pick.call_args_list[1].args[0], self.image)
        np.testing.assert_array_equal(pick.call_args_list[3].args[0], b.rotate_pixels(self.image, record["angle_deg"]))

    def test_zoom_display_is_brighter_without_changing_coordinates_or_counts(self):
        b = self.functions
        image = np.full((24, 32, 3), [10, 20, 30], dtype=np.uint8)
        before = image.copy()
        roi = np.array([[5., 4.], [25., 20.]])
        points = np.array([[8., 9.], [22., 14.]])

        def inspect_display(*args, **kwargs):
            axis = b.plt.gcf().axes[0]
            np.testing.assert_array_equal(axis.images[0].get_array()[10, 10], [40, 80, 120])
            np.testing.assert_allclose(axis.get_xlim(), [5., 25.])
            np.testing.assert_allclose(axis.get_ylim(), [20., 4.])
            return points

        with patch.object(b.plt, "show"), patch.object(b.plt, "ginput", side_effect=inspect_display):
            chosen = b.GateImageProfiler._pick_two_points(image, "Pick edges", roi=roi, display_gain=4.)
        np.testing.assert_array_equal(chosen, points)
        np.testing.assert_array_equal(image, before)
        rgba = np.array([[[90, 10, 0, 128]]], dtype=np.uint8)
        np.testing.assert_array_equal(b.GateImageProfiler._scale_display_brightness(rgba, 4.),
                                      [[[255, 40, 0, 128]]])


class CaDisplayBrightnessTests(unittest.TestCase):
    functions = notebook_functions("cross_talk_analysis_ca_2.ipynb")
    setUp = ManualAlignmentTests.setUp
    test_brightness_boost_reaches_zoom_views_only = (
        CurrentBaManualAlignmentTests.test_brightness_boost_reaches_zoom_views_only
    )
    test_zoom_display_is_brighter_without_changing_coordinates_or_counts = (
        CurrentBaManualAlignmentTests.test_zoom_display_is_brighter_without_changing_coordinates_or_counts
    )


if __name__ == "__main__":
    unittest.main()
