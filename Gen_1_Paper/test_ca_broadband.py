"""Run from repo root: MPLBACKEND=Agg python -m unittest discover -s Gen_1_Paper -p test_ca_broadband.py"""

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from Analysis.Experimental_Analysis import Experimental_Analysis

from Analysis.ca_3d_import import (
    CaHeightScan, WAVELENGTHS_NM, peak_count_rate_vs_height, discover_broadband_scans,
    load_height_scan,
)
from Analysis.ca_3d_plotting import (
    _rotate_points, alignment_from_points, align_scans_manual, aligned_axes,
    rotate_image, select_height_plane, plot_publication_grid, export_publication,
    plot_peak_count_rates,
    fit_beam_sizes, plot_beam_sizes, plot_beam_profile_fits,
)


class CaBroadbandTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        reference = self.root / "focused.tiff"
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[39:42, 49:52] = 200
        Image.fromarray(image).save(reference)
        frame = self.root / "frame_007.tiff"
        Image.fromarray(image).save(frame)
        self.scan = CaHeightScan(
            "M1", 405, self.root, reference, {}, 1.0, 0.5,
            pd.DataFrame([dict(image_index=7, path=frame, height_um=75.003,
                               exposure_ms=4.0, stage_position_mm=1.150006)]),
        )
        self.alignment = alignment_from_points(
            self.scan, [[20, 40], [80, 40]], pixel_size_um=1, magnification=1,
        )

    def test_manual_points_rotate_horizontal_and_match_pixel_transform(self):
        points = np.array([[20, 20], [70, 45]])
        alignment = alignment_from_points(self.scan, points)
        self.assertAlmostEqual(alignment["angle_deg"], np.degrees(np.arctan2(25, 50)))
        rotated = _rotate_points(points, (80, 100), alignment["angle_deg"])
        self.assertAlmostEqual(rotated[0, 1], rotated[1, 1])
        np.testing.assert_allclose(alignment["origin_px"], rotated.mean(axis=0))
        # Verify the analytical forward transform against scipy's image rotation.
        marker = np.zeros((80, 100), dtype=np.float32)
        marker[19:22, 19:22] = 1
        actual = np.argwhere(rotate_image(marker, alignment["angle_deg"]) > 0).mean(axis=0)[::-1]
        np.testing.assert_allclose(actual, rotated[0], atol=0.6)
        reverse = alignment_from_points(self.scan, points[::-1])
        self.assertAlmostEqual(reverse["angle_deg"], alignment["angle_deg"])
        np.testing.assert_allclose(reverse["origin_px"], alignment["origin_px"])

    def test_axes_use_rotated_midpoint_and_upward_y(self):
        x, y = aligned_axes(self.alignment)
        self.assertEqual(x[50], 0)
        self.assertEqual(y[40], 0)
        self.assertEqual(x[51], 1)
        self.assertEqual(y[39], 1)

    def test_invalid_points_rejected(self):
        for points in ([[2, 2], [2, 2]], [[-1, 3], [5, 6]], [[0, 0], [100, 80]]):
            with self.assertRaises(ValueError):
                alignment_from_points(self.scan, points)

    def test_manual_selection_saved_reused_and_reference_change_detected(self):
        scans = dict.fromkeys(WAVELENGTHS_NM)
        scans[405] = self.scan
        directory = self.root / "alignments"
        with patch("Analysis.ca_3d_plotting._require_interactive_backend"), patch(
            "Analysis.ca_3d_plotting._pick_two_points", return_value=np.array([[20, 40], [80, 40]])
        ) as picker:
            first = align_scans_manual(scans, directory)
            second = align_scans_manual(scans, directory)
            self.assertEqual(picker.call_count, 1)
        self.assertEqual(first, second)
        self.assertIsNone(first[450])
        Image.fromarray(np.ones((80, 100, 3), dtype=np.uint8)).save(self.scan.reference_path)
        with self.assertRaisesRegex(ValueError, "reference/calibration changed"):
            align_scans_manual(scans, directory)

    def test_nearest_height_rgb_sum_and_raw_stack_are_equivalent(self):
        plane = select_height_plane(self.scan, self.alignment, 75)
        self.assertEqual(plane["frame_index"], 7)
        self.assertEqual(plane["exposure_ms"], 4)
        self.assertEqual(plane["selected_height_um"], 75.003)
        self.assertEqual(plane["image"].max(), 600)
        self.assertEqual(plane["image"].dtype, np.float32)
        stack = np.full((1, 80, 100), 30, dtype=np.float32)
        from_stack = select_height_plane(self.scan, self.alignment, 75, raw_stack=stack,
                                         center_um=(0, 0), background_counts=10)
        self.assertEqual(from_stack["image"].max(), 20)
        self.assertTrue(np.all(stack == 30))
        with self.assertRaisesRegex(ValueError, "closest is"):
            select_height_plane(self.scan, self.alignment, 90)

    def test_peak_roi_and_center_override(self):
        stack = np.zeros((1, 80, 100), dtype=np.float32)
        stack[0, 5, 5] = 900  # Distractor excluded by the selected search region.
        stack[0, 40, 50] = 600
        plane = select_height_plane(self.scan, self.alignment, 75, raw_stack=stack,
                                     peak_roi_um=(-5, 5, -5, 5))
        self.assertEqual(plane["center_um"], [0, 0])
        fixed = select_height_plane(self.scan, self.alignment, 75, center_um=(3, 4))
        self.assertEqual(fixed["center_um"], [3, 4])

    def test_mixed_devices_use_independent_ions_and_reserved_wavelengths(self):
        def dataset(root, ion, wl):
            directory = root / f"ion_{ion}" / f"{wl}_nm"
            directory.mkdir(parents=True)
            (directory / "focused_example.tiff").write_bytes(self.scan.reference_path.read_bytes())
            (directory / "focused_example.json").write_text(json.dumps({
                "focus_state": {"result": {"final_z_mm": 1.0}}
            }))
            (directory / "height_scan").mkdir()
            (directory / "height_scan/frame.tiff").write_bytes(self.scan.reference_path.read_bytes())
            pd.DataFrame([dict(image_index=0, stage_position_mm=1.15,
                               exposure_ms=4, filename="frame.tiff")]).to_csv(
                directory / "height_scan/exposure_log.csv", index=False)
        ca, ba = self.root / "R2C2/profile", self.root / "R3C6"
        for wl in (405, 729, 880):
            dataset(ca, "M1", wl)
        for wl in (450, 515):
            dataset(ba, "P1", wl)
        dataset(ca, "M1", 450)  # Wrong device must not fill a reserved Ba tile.
        scans = discover_broadband_scans(ca, ba, "M1", barium_ion_index="P1")
        self.assertEqual(tuple(scans), WAVELENGTHS_NM)
        self.assertIsNone(scans[635])
        for wl, scan in scans.items():
            if scan is None:
                continue
            barium = wl in (450, 515, 635)
            self.assertEqual(scan.device_name, "R3C6" if barium else "R2C2")
            self.assertEqual(scan.ion_index, "P1" if barium else "M1")
            self.assertEqual(scan.flip_vertical, barium)
        dataset(ba, "P1", 635)  # A completed upload fills the sixth tile on rediscovery.
        self.assertIsNotNone(discover_broadband_scans(
            ca, ba, "M1", barium_ion_index="P1")[635])
        # Load a chip without an ion subfolder, then ensure its override does
        # not even inspect a broken original 729 nm source.
        dataset(self.root / "standalone", "temporary", 729)
        directory = self.root / "standalone/ion_temporary/729_nm"
        standalone = self.root / "Gen1_OG/729_nm"
        standalone.parent.mkdir()
        directory.rename(standalone)
        replacement = load_height_scan(standalone, wavelength_nm=729, device_name="Gen1_OG")
        self.assertEqual(replacement.ion_index, "unlabeled")
        self.assertFalse(replacement.flip_vertical)
        self.assertAlmostEqual(replacement.frames.height_um.iloc[0], 75)
        (ca / "ion_M1/729_nm/focused_example.json").write_text("broken")
        mixed = discover_broadband_scans(
            ca, ba, "M1", barium_ion_index="P1", wavelength_overrides={729: replacement})
        self.assertIs(mixed[729], replacement)
        for wl in (405, 450, 515, 635, 880):
            self.assertEqual(mixed[wl].device_name, "R3C6" if wl in (450, 515, 635) else "R2C2")
        with self.assertRaisesRegex(ValueError, "override"):
            discover_broadband_scans(ca, ba, "M1", wavelength_overrides={729: self.scan})
        # A standalone source retains the loader's completeness checks.
        (standalone / "height_scan/frame.tiff").unlink()
        with self.assertRaisesRegex(ValueError, "TIFF/log mismatch"):
            load_height_scan(standalone, wavelength_nm=729)

    def test_vertical_reflection_after_rotation_preserves_x_and_raw_data(self):
        stack = np.zeros((1, 80, 100), dtype=np.float32)
        stack[0, 30, 50] = 600
        stack[0, 34, 52] = 300  # Asymmetric marker below/right of the peak.
        alignment = alignment_from_points(
            self.scan, [[20, 35], [80, 45]], pixel_size_um=1, magnification=1,
        )
        original = stack.copy()
        ca = select_height_plane(self.scan, alignment, 75, raw_stack=stack)
        ba_scan = replace(self.scan, device_name="R3C6", flip_vertical=True)
        ba = select_height_plane(ba_scan, alignment, 75, raw_stack=stack)
        np.testing.assert_array_equal(stack, original)
        np.testing.assert_array_equal(ba["image"], ca["image"][::-1])
        np.testing.assert_allclose(ba["x_um"], ca["x_um"])
        np.testing.assert_allclose(ba["y_um"], -ca["y_um"][::-1])
        np.testing.assert_allclose(ba["center_um"], np.array(ca["center_um"])*[1, -1])
        self.assertTrue(np.all(np.diff(ba["y_um"]) < 0))
        from_tiff = select_height_plane(ba_scan, alignment, 75)
        raw_tiff = select_height_plane(self.scan, alignment, 75)
        np.testing.assert_array_equal(from_tiff["image"], raw_tiff["image"][::-1])
        np.testing.assert_allclose(from_tiff["center_um"],
                                   np.array(raw_tiff["center_um"])*[1, -1])
        fig, table, crops = plot_publication_grid({405: ba}, half_width_um=10)
        self.addCleanup(plt.close, fig)
        self.assertEqual(table.device_name.iloc[0], "R3C6")
        self.assertTrue(table.flip_vertical.iloc[0])
        # The dimmer marker was below the peak in the camera frame; after
        # reflection it must appear above it in exported physical coordinates.
        marker_rows, marker_cols = np.where(crops[405]["intensity"] == 300)
        self.assertGreater(len(marker_rows), 0)
        self.assertTrue(np.all(crops[405]["y_um"][marker_rows] > 0))
        self.assertTrue(np.all(crops[405]["x_um"][marker_cols] > 0))

    def test_export_contains_exact_plotted_arrays_and_alignment(self):
        plane = select_height_plane(self.scan, self.alignment, 75)
        fig, table, crops = plot_publication_grid({405: plane}, half_width_um=10)
        self.addCleanup(plt.close, fig)
        self.assertEqual(crops[405]["normalized"].max(), 1)
        self.assertEqual(len(fig.axes), 7)  # Six wavelength slots plus one colorbar.
        self.assertEqual(table.normalization_peak_counts.iloc[0], 600)
        paths = export_publication(fig, table, crops, self.root / "exports", "test")
        self.assertEqual({p.suffix for p in paths}, {".png", ".pdf", ".svg", ".npz", ".json"})
        self.assertTrue(all(p.stat().st_size > 0 for p in paths))
        svg = ET.parse(self.root / "exports/test.svg").getroot()
        ns = {"svg": "http://www.w3.org/2000/svg"}
        # The measured heatmap stays embedded; colorbar fills are vectors.
        images = svg.findall(".//svg:image", ns)
        self.assertEqual(len(images), 1)
        self.assertTrue(images[0].get("{http://www.w3.org/1999/xlink}href").startswith("data:image/png;base64,"))
        labels = svg.findall(".//svg:text", ns)
        self.assertTrue(labels)
        self.assertTrue(all("Arial" in label.get("style", "") for label in labels))
        with np.load(self.root / "exports/test.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["wl_405_normalized"], crops[405]["normalized"])
        metadata = json.loads((self.root / "exports/test.json").read_text())[0]
        self.assertEqual(metadata["alignment"]["points_px"], self.alignment["points_px"])
        self.assertEqual(metadata["stage_to_height_ratio"], 0.5)
        with self.assertRaisesRegex(ValueError, "extends outside"):
            plot_publication_grid({405: plane}, half_width_um=100)

    def test_peak_count_rate_uses_rgb_sum_and_each_frames_exposure(self):
        # The second frame has more counts but a lower exposure-normalized rate.
        second = self.root / "frame_003.tiff"
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[10, 20] = 250
        Image.fromarray(image).save(second)
        self.scan.frames = pd.concat([self.scan.frames, pd.DataFrame([
            dict(image_index=3, path=second, height_um=74.004,
                 exposure_ms=10.0, stage_position_mm=1.148008)
        ])], ignore_index=True)
        table = peak_count_rate_vs_height(self.scan)
        np.testing.assert_array_equal(table.image_index, [7, 3])
        np.testing.assert_allclose(table.height_um, [75.003, 74.004])
        np.testing.assert_allclose(table.peak_counts, [600, 750])
        np.testing.assert_allclose(table.peak_count_rate_per_s, [150000, 75000])
        self.assertEqual(table.peak_column_px.iloc[1], 20)
        self.assertEqual(table.peak_row_px.iloc[1], 10)
        fig, peaks = plot_peak_count_rates({405: table})
        self.addCleanup(plt.close, fig)
        self.assertEqual(len(fig.axes), 6)
        self.assertEqual(peaks.image_index.iloc[0], 7)
        self.assertEqual(fig.get_facecolor(), (1, 1, 1, 1))
        np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), [74.004, 75.003])

    def test_peak_count_rate_grayscale_and_zero_exposure(self):
        gray = np.array([[10, 40], [20, 30]], dtype=np.uint16)
        Image.fromarray(gray).save(self.scan.frames.path.iloc[0])
        table = peak_count_rate_vs_height(self.scan)
        self.assertEqual(table.peak_counts.iloc[0], 40)
        self.assertEqual(table.peak_count_rate_per_s.iloc[0], 10000)
        self.scan.frames.loc[0, "exposure_ms"] = 0
        with self.assertRaisesRegex(ValueError, "Exposure must be positive"):
            peak_count_rate_vs_height(self.scan)

    def test_gaussian_radii_on_peak_slices_and_legacy_saved_scan(self):
        x = np.linspace(-12, 12, 481)
        y = np.linspace(20, -20, 801)
        image = 17 + 650*np.exp(-2*((x[None, :]-1.2)/1.3)**2
                               -2*((y[:, None]+2.4)/6.1)**2)
        source = image.copy()
        analysis = Experimental_Analysis.from_image(image, x, y, height_um=75, exposure_ms=4)
        fits = analysis.fit_beam_profiles(half_width_um=10)
        self.assertAlmostEqual(fits["axial"]["radius_um"], 1.3, places=6)
        self.assertAlmostEqual(fits["radial"]["radius_um"], 6.1, places=6)
        for direction in ("axial", "radial"):
            self.assertAlmostEqual(fits[direction]["fwhm_um"],
                                   np.sqrt(2*np.log(2))*fits[direction]["radius_um"])
            self.assertAlmostEqual(fits[direction]["background_counts"], 17, places=5)
            self.assertGreater(fits[direction]["r_squared"], .999999)
        np.testing.assert_array_equal(image, source)
        np.savez(self.root / "legacy.npz", intensity=image[None], x_um=x, y_um=y,
                 height_um=[75], exposure_ms=[4], angle_deg=0)
        legacy = Experimental_Analysis(self.root / "legacy.npz").select_height(75, normalize=False)
        np.testing.assert_array_equal(legacy.axial_slice, analysis.axial_slice)
        np.testing.assert_array_equal(legacy.radial_slice, analysis.radial_slice)
        reflected = Experimental_Analysis.from_image(image[::-1], x, -y[::-1])
        reflected_fits = reflected.fit_beam_profiles(half_width_um=10)
        for direction in ("axial", "radial"):
            self.assertAlmostEqual(reflected_fits[direction]["radius_um"], fits[direction]["radius_um"])

    def test_beam_size_table_plot_and_failure_reporting(self):
        plane = select_height_plane(self.scan, self.alignment, 75)
        x, y = plane["x_um"], plane["y_um"]
        plane["image"] = 20 + 600*np.exp(-2*(x[None, :]/2)**2-2*(y[:, None]/6)**2)
        failed = dict(plane, wavelength_nm=450, image=np.ones_like(plane["image"]))
        table, fits = fit_beam_sizes({405: plane, 450: failed}, half_width_um=15)
        self.assertEqual(table.status.tolist(), ["ok", "fit failed"])
        self.assertIn("nonconstant", table.fit_error.iloc[1])
        self.assertTrue(np.isnan(table.radial_radius_um.iloc[1]))
        self.assertAlmostEqual(table.axial_radius_um.iloc[0], 2)
        self.assertAlmostEqual(table.radial_radius_um.iloc[0], 6)
        fig = plot_beam_sizes(table, quantity="radius")
        self.addCleanup(plt.close, fig)
        self.assertEqual(len(fig.axes[0].collections), 2)
        np.testing.assert_allclose(fig.axes[0].collections[0].get_offsets(), [[405, 6]])
        np.testing.assert_allclose(fig.axes[0].collections[1].get_offsets(), [[405, 2]])
        fwhm_fig = plot_beam_sizes(table, quantity="fwhm")
        self.addCleanup(plt.close, fwhm_fig)
        np.testing.assert_allclose(fwhm_fig.axes[0].collections[0].get_offsets(),
                                   [[405, 6*np.sqrt(2*np.log(2))]])
        diagnostics = plot_beam_profile_fits(fits)
        self.addCleanup(plt.close, diagnostics)
        self.assertEqual(len(diagnostics.axes), 12)
        with self.assertRaises(ValueError):
            plot_beam_sizes(table, quantity="diameter")

    def test_gaussian_fit_rejects_invalid_or_insufficient_data(self):
        for x, counts in ((np.arange(10), np.ones(10)), (np.arange(4), np.arange(4)),
                          (np.arange(10), -np.ones(10))):
            with self.assertRaises(ValueError):
                Experimental_Analysis.fit_gaussian_profile(x, counts)


if __name__ == "__main__":
    unittest.main()
