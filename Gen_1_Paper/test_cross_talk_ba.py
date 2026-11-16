"""Check the new Ba notebook's HDR thresholds and exposure selection."""

import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest

import numpy as np
import pandas as pd
import tifffile
from PIL import Image


NOTEBOOK = Path(__file__).with_name("cross_talk_analysis_ba.ipynb")
notebook = json.loads(NOTEBOOK.read_text())
b = types.ModuleType("ba_4c_hdr")
for cell in notebook["cells"][:5]:
    if "functions" in cell.get("metadata", {}).get("tags", []):
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), b.__dict__)


class BaHDRTests(unittest.TestCase):
    def test_longest_valid_exposure_original_mean_and_manual_noise(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = dict(camera_id=1, pixel_clock_mhz=15, width=4, height=2,
                            pixel_format="BGR8_PACKED")
            reference = root / "focused.tiff"
            tifffile.imwrite(reference, np.zeros((2, 4, 3), dtype=np.uint8), photometric="rgb")
            dark_path = root / "dark"
            dark_path.mkdir()
            # R+G+B mean=3, sample SD=2; the 10 ms image has mean=6, SD=4.
            for name, factor in (("0p2_ms.bmp", 1), ("1_ms.bmp", 1), ("10_ms.bmp", 2)):
                dark_rgb = np.zeros((2, 4, 3), dtype=np.uint8)
                dark_rgb[..., 0] = factor*np.array([0, 1, 2, 3, 3, 4, 5, 6]).reshape(2, 4)
                Image.fromarray(dark_rgb).save(dark_path / name)
            original_path = root / "dark_calibration.npz"
            np.savez(original_path, actual_exposures_ms=[.01, 20.], dark_counts=[2., 2.],
                     exposure_unit="ms", dark_count_definition="mean over frames and pixels of B+G+R",
                     metadata_json=json.dumps(metadata))
            rgb = np.array([
                [[11, 0, 0], [9, 0, 0], [0, 0, 230], [200, 200, 200],
                 [230, 0, 0], [0, 0, 0], [11, 0, 0], [13, 0, 0]],
                [[101, 0, 0], [10, 0, 0], [229, 0, 0], [200, 200, 200],
                 [0, 230, 0], [0, 0, 0], [81, 0, 0], [0, 0, 0]],
                [[201, 0, 0], [9, 0, 0], [255, 0, 0], [229, 0, 0],
                 [0, 0, 255], [0, 0, 0], [14, 0, 0], [201, 202, 0]],
            ], dtype=np.uint8).reshape(3, 2, 4, 3)
            rows = []
            for index, (nominal, actual) in enumerate(((.2, .25), (1., 2.), (10., 10.))):
                frame_path = root / f"frame_{index}.tiff"
                tifffile.imwrite(frame_path, rgb[index], photometric="rgb")
                rows.append(dict(height_index=0, target_height_um=75., requested_exposure_ms=nominal,
                                 actual_exposure_ms=actual, path=frame_path))
            frames = pd.DataFrame(rows)
            frames.to_csv(root / "exposure_log.csv", index=False)
            dataset = dict(name="ba_4c_test", run_dir=root, frames=frames,
                           metadata=metadata, reference_path=reference,
                           reference_metadata=dict(exposure_ms=1.))
            alignment = dict(angle_deg=0., origin_px=[1.5, .5],
                             pixel_pitch_um=[5.3, 5.3], magnification=50.)
            dark = b.load_dark_calibration(original_path, dark_path)
            result = b.process_dataset(dataset, alignment, dark, cache_dir=root / "cache")
            # Original mean=2 is subtracted (not MANUAL means=3/6). Counts must
            # exceed 3*MANUAL SD (6/12). Exactly 12 at the longest frame is rejected.
            # Select longest valid even when a shorter valid frame has a larger rate.
            np.testing.assert_allclose(result["rate"][0],
                                       [[19900., 4000., 113500., 22700.], [0., 0., 39500., 40100.]])
            np.testing.assert_array_equal(result["source"][0], [[2, 1, 1, 2], [-1, -1, 1, 2]])
            self.assertEqual(result["hdr_selection"], "longest_valid_exposure")
            self.assertFalse(result["fallback"].any())
            self.assertEqual(result["fallback_method"], "none")
            self.assertEqual(result["summary"].iloc[0].valid_pixels_pct, 75.)
            self.assertEqual(result["summary"].iloc[0].positive_pixels_pct, 75.)
            cached = b.process_dataset(dataset, alignment, dark, cache_dir=root / "cache")
            np.testing.assert_array_equal(cached["source"], result["source"])
            np.testing.assert_array_equal(cached["rate"], result["rate"])
            changed = b.process_dataset(dataset, alignment, dark, max_rgb=220., cache_dir=root / "cache")
            self.assertEqual(changed["source"][0, 0, 2], -1)
            self.assertEqual(changed["source"][0, 0, 3], 1)
            excluded = b.process_dataset(dataset, alignment, dark, excluded_exposures_ms=[10.])
            np.testing.assert_array_equal(excluded["source"][0], [[1, 1, 1, 1], [-1, -1, 1, 0]])
            np.testing.assert_allclose(excluded["rate"][0, 0, 0], 49500.)
            self.assertTrue(np.isnan(b.ratio_db(result["rate"][0, 1, 1], 100.)))
            tighter = b.process_dataset(dataset, alignment, dark, signal_sigma_multiplier=4., cache_dir=root / "cache")
            self.assertEqual(tighter["source"][0, 0, 1], -1)
            self.assertTrue(result["comparisons"].signal_threshold_applied.all())

    def test_dark_sum_sample_sd_exact_exposure_and_content_fingerprint(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)
            rgb = np.zeros((2, 2, 3), dtype=np.uint8)
            rgb[..., 0] = [[0, 2], [4, 6]]
            rgb[..., 1] = rgb[..., 0]  # correlated channels: SD must be of the sum
            Image.fromarray(rgb).save(path / "1_ms.bmp")
            Image.fromarray(rgb + 1).save(path / "10_ms.bmp")
            dark = b.load_manual_dark_statistics(path)
            np.testing.assert_allclose(dark["counts"], [6., 9.])
            np.testing.assert_allclose(dark["sigma_counts"], np.std([0., 4., 8., 12.], ddof=1))
            mean, sd = b.dark_statistics_at(dark, [1., 10.])
            np.testing.assert_allclose(mean, [6., 9.])
            np.testing.assert_allclose(sd, dark["sigma_counts"])
            with self.assertRaisesRegex(ValueError, "matching dark BMP"):
                b.dark_statistics_at(dark, [2.])
            # Scalar mean interpolation is separate from exact exposure matching of MANUAL noise.
            self.assertEqual(float(b.dark_at(dark, 4.)), 7.)
            rgb[0, 0, 0] += 1
            Image.fromarray(rgb).save(path / "1_ms.bmp")
            self.assertNotEqual(dark["fingerprint"], b.load_manual_dark_statistics(path)["fingerprint"])

    def test_original_mean_interpolation_is_independent_of_manual_mean(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            noise_path = root / "manual"
            noise_path.mkdir()
            rgb = np.zeros((2, 2, 3), dtype=np.uint8)
            rgb[..., 0] = [[1, 3], [5, 7]]
            for name in ("1_ms.bmp", "10_ms.bmp"):
                Image.fromarray(rgb).save(noise_path / name)
            original_path = root / "original.npz"
            def save_original(means):
                np.savez(original_path, actual_exposures_ms=[.1, 10.], dark_counts=means,
                         exposure_unit="ms", dark_count_definition="mean over frames and pixels of B+G+R",
                         metadata_json=json.dumps(dict(width=2, height=2)))
            save_original([.2, 20.])
            dark = b.load_dark_calibration(original_path, noise_path)
            np.testing.assert_allclose(b.dark_at(dark, [1.5, 2.]), [3., 4.])
            _, sigma = b.dark_statistics_at(dark["noise"], [1., 10.])
            np.testing.assert_allclose(sigma, np.std([1., 3., 5., 7.], ddof=1))
            # Shifting the MANUAL mean changes provenance but neither subtraction nor SD.
            Image.fromarray(rgb+5).save(noise_path / "1_ms.bmp")
            shifted = b.load_dark_calibration(original_path, noise_path)
            np.testing.assert_array_equal(shifted["counts"], dark["counts"])
            np.testing.assert_allclose(shifted["noise"]["sigma_counts"], dark["noise"]["sigma_counts"])
            self.assertNotEqual(shifted["fingerprint"], dark["fingerprint"])
            # Original calibration content also participates in cache provenance.
            save_original([.4, 40.])
            self.assertNotEqual(shifted["fingerprint"], b.load_dark_calibration(original_path, noise_path)["fingerprint"])

    def test_configuration_preserves_hdr_and_channel_selection(self):
        settings = "".join(notebook["cells"][10]["source"])
        ns = dict(b.__dict__)
        exec(settings, ns)
        self.assertNotIn("USE_HDR_FALLBACK", ns)
        self.assertEqual(ns["SIGNAL_SIGMA_MULTIPLIER"], 3.)
        self.assertEqual(ns["DARK_PATH"].name, "dark_calibration.npz")
        self.assertEqual(ns["DARK_PATH"].parent.name, "dark_20261002_113902_15MHz")
        self.assertEqual(ns["NOISE_PATH"].name, "MANUAL")
        self.assertEqual(ns["MAX_RGB_CHANNEL"], 255*.9)
        self.assertEqual(ns["EXCLUDED_EXPOSURES_MS"], {-2: (), 1: (), 2: ()})
        self.assertEqual(ns["DATASET_FOLDERS"], {-2: "ion_M2", 1: "ion_P1", 2: "ion_P2"})
        self.assertEqual(ns["SELECTED_IONS"], (-2, 1, 2))
        self.assertEqual(ns["OUTPUT_DIR"].name, "M2_P1_P2")
        self.assertEqual(ns["OUTPUT_DIR"].parent.name, "cross_talk_results_ba")
        self.assertEqual(ns["PUBLICATION_OFFSETS_UM"],
                         {1: (-10., -5.), -1: (5., 10.), 0: (-5., 5.)})
        self.assertEqual(ns["ION_LOCATION_LABELS"], {-2: 1, 1: -1, 2: 0})
        self.assertEqual(ns["PUBLICATION_ION_ORDER"], (1, -1, 0))
        self.assertEqual(ns["ION_INDEX_X_DIRECTION"], 1)

    def test_nonconsecutive_channel_indices_keep_offsets_and_all_six_ratios(self):
        ns = dict(b.__dict__, display=lambda *args: None)
        exec("".join(notebook["cells"][10]["source"]), ns)
        overlay = next(c for c in notebook["cells"] if c.get("id") == "ba-common-y-overlay")
        overlay_source = "".join(overlay["source"])
        exec(overlay_source[:overlay_source.index('if "results" not in globals():')], ns)
        cut = ns["COMMON_SLICE_Y_UM"]
        x, y = np.arange(-30., 30.01, .1), np.array([cut-.1, cut, cut+.1])
        results = {}
        for ion, center, scale in ((-2, 10., 100.), (1, -5., 300.), (2, -10., 700.)):
            profile = scale*np.exp(-((x-center)/2)**2)+.01
            image = np.broadcast_to(profile, (1, 3, len(x))).astype(np.float32).copy()
            results[ion] = dict(rate=image, source=np.zeros(image.shape, dtype=np.int16),
                                x_um=x, y_um=y, heights_um=np.array([75.]),
                                dataset=dict(name=f"ion_{ion}"))
        matrix = b.neighboring_matrices(results, pitch_um=5., index_x_direction=-1)
        self.assertEqual(matrix["ions"], [-2, 1, 2])
        for point in matrix["table"].itertuples():
            self.assertEqual(point.requested_offset_um, -(point.measured_ion-point.driven_ion)*5.)
        ns["results"] = results
        for identifier in ("ba-common-y-overlay", "ba-common-y-crosstalk"):
            cell = next(c for c in notebook["cells"] if c.get("id") == identifier)
            source = "".join(cell["source"])
            if identifier == "ba-common-y-overlay":
                # Sampling is independent of interactive edits to visual markers.
                source = source[:source.index("# Anchor markers when their requested")]
            exec(source, ns)
        table = ns["common_slice_crosstalk_table"]
        self.assertEqual(len(table), 6)
        self.assertEqual(set(table.driven_ion), {-1, 0, 1})
        self.assertEqual(set(table.driven_dataset_key), {-2, 1, 2})
        self.assertEqual(set(table.measured_location), {-1, 0, 1})
        self.assertEqual(list(ns["common_slice_crosstalk_matrix_db"].index), [-1, 0, 1])
        np.testing.assert_allclose(sorted(ns["common_slice_separation_table"].separation_um),
                                   [5., 15., 20.])
        b.plt.close("all")

    def test_physical_locations_determine_neighbors_without_changing_acquisition_keys(self):
        x, y = np.arange(-15., 15.01, .1), np.array([0., 1.])
        labels = {-2: 1, 1: -1, 2: 0}
        results = {}
        for key, location in labels.items():
            profile = 100*np.exp(-2*((x-5*location)/1.5)**2)+.1
            rate = np.broadcast_to(profile, (1, 2, len(x))).copy()
            results[key] = dict(rate=rate, source=np.zeros(rate.shape,dtype=np.int16),
                heights_um=np.array([75.]), x_um=x, y_um=y, dataset=dict(name=str(key)))
        physical = b.results_by_ion_location(results, labels)
        self.assertEqual(list(physical), [1, -1, 0])
        self.assertEqual(set(results), {-2, 1, 2})
        for key, location in labels.items():
            self.assertEqual(physical[location]["source_dataset_key"], key)
            self.assertIs(physical[location]["rate"], results[key]["rate"])
        scan = b.neighboring_matrices(physical, pitch_um=5., index_x_direction=1)
        self.assertEqual(scan["ions"], [-1, 0, 1])
        table = scan["table"][scan["table"].measured_ion != scan["table"].driven_ion]
        self.assertEqual(len(table), 6)
        for row in table.itertuples():
            self.assertAlmostEqual(row.requested_offset_um, 5*(row.measured_ion-row.driven_ion))
            self.assertAlmostEqual(row.sample_x_um, 5*row.measured_ion)
        self.assertEqual(set(table.requested_offset_um), {-10., -5., 5., 10.})
        with self.assertRaisesRegex(ValueError, "unique"):
            b.results_by_ion_location(results, {-2: 1, 1: 1, 2: 0})

    def test_center_anchor_and_outer_thresholds_use_plane_normalization(self):
        cell = next(c for c in notebook["cells"] if c.get("id") == "ba-common-y-overlay")
        node = next(node for node in ast.parse("".join(cell["source"])).body
                    if isinstance(node, ast.FunctionDef) and node.name == "place_center_anchored_markers")
        ns = dict(b.__dict__)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"), ns)
        x = np.arange(-4., 7.)
        profiles = {}
        for ion, center in zip((1, 2, -2), (-1., 1., 3.)):
            values = np.maximum(.1, 1.-.8*np.abs(x-center))
            profiles[ion] = dict(x_um=x, normalized_intensity=values, peak_x_um=center,
                                 slice_peak_relative_to_plane=.8)
        place = ns["place_center_anchored_markers"]
        best = place(profiles, [1, 2, -2], center_ion=2, target_spacing_um=3.,
                     minimum_plane_intensity=.61)
        np.testing.assert_allclose([best["positions_um"][ion] for ion in (1, 2, -2)],
                                   [-1.296875, 1., 3.296875])
        self.assertAlmostEqual(best["plane_normalized_intensities"][1], .61)
        self.assertAlmostEqual(best["plane_normalized_intensities"][-2], .61)
        self.assertAlmostEqual(best["normalized_intensities"][1], .7625)
        self.assertEqual(best["positions_um"][2], 1.)
        # Missing pixels must not be bridged while finding threshold crossings.
        profiles[1]["normalized_intensity"][x == -2.] = np.nan
        missing = place(profiles, [1, 2, -2], center_ion=2, target_spacing_um=3.,
                        minimum_plane_intensity=.61)
        self.assertEqual(missing["positions_um"][1], -1.)


if __name__ == "__main__":
    unittest.main()
