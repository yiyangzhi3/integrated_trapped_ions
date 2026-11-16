"""Headless checks for the independent original-chip single-channel notebook."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).with_name("cross_talk_analysis_ca_3.ipynb")
notebook = nbformat.read(NOTEBOOK, as_version=4)
b = types.ModuleType("single_channel_crosstalk_functions")
for cell in notebook.cells:
    if "functions" in cell.metadata.get("tags", []):
        exec(compile(cell.source, str(NOTEBOOK), "exec"), b.__dict__)


def example_results():
    results = {}
    x, y, heights = np.arange(-20., 20.01, .4), np.arange(-6., 7.), np.array([65., 75., 85.])
    for wl, width in ((729, 2.), (830, 3.)):
        cube = np.empty((3, len(y), len(x)), dtype=np.float32)
        for hi in range(3):
            for yi, radial in enumerate(y):
                center = (hi-1)*.4 + radial*.08
                cube[hi, yi] = 1000*np.exp(-((x-center)/width)**2-radial**2/25)
        frames = pd.DataFrame(dict(requested_exposure_ms=[1., 1., 1.], actual_exposure_ms=[1., 1., 1.]))
        results[wl] = dict(rate=cube, source=np.broadcast_to(np.arange(3)[:, None, None], cube.shape).copy(),
            x_um=x.copy(), y_um=y.copy(), heights_um=heights.copy(),
            dataset=dict(name=f"{wl}_nm_crosstalk", frames=frames, run_dir=Path("signal_test"),
                         reference_path=Path("focused_test.tiff")),
            alignment=dict(angle_deg=0.), min_signal=5, max_rgb=230,
            hdr_selection="longest_valid_exposure", excluded_exposures_ms=[])
    return results


class SingleChannelTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_independent_single_drive_nearest_pixels_and_missing_samples(self):
        results = example_results()
        table = b.single_channel_crosstalk(results)
        self.assertEqual(len(table), 2*3*3)
        self.assertEqual(set(table.driven_channel), {0})
        self.assertEqual(set(table.wavelength_nm), {729, 830})
        self.assertEqual(set(table.offset_um), {-5, 0, 5})
        np.testing.assert_allclose(table.loc[table.offset_um == 0, "crosstalk_db"], 0)
        for row in table.itertuples():
            result = results[row.wavelength_nm]
            slot = b.height_slot(result, row.height_um)
            profile, px, _, peak = b.axial_profile(result, slot)
            sample = np.argmin(np.abs(result["x_um"]-row.target_x_um))
            self.assertAlmostEqual(row.sample_x_um, result["x_um"][sample])
            self.assertAlmostEqual(row.crosstalk_db, 10*np.log10(profile[sample]/peak))
        # Other wavelengths' origins and brightness cannot change a result.
        shifted = {wl: dict(r) for wl, r in results.items()}
        shifted[830] = dict(results[830], x_um=results[830]["x_um"]+123,
                            rate=results[830]["rate"]*7)
        changed = b.single_channel_crosstalk(shifted)
        np.testing.assert_allclose(changed.crosstalk_db, table.crosstalk_db, atol=2e-5)
        row = table[(table.wavelength_nm == 729) & (table.height_um == 75) & (table.offset_um == 5)].iloc[0]
        result = results[729]
        yi = np.argmin(abs(result["y_um"]-row.sample_y_um))
        xi = np.argmin(abs(result["x_um"]-row.sample_x_um))
        result["source"][1, yi, xi] = -1
        missing = b.single_channel_crosstalk(results)
        selected = missing[(missing.wavelength_nm == 729) & (missing.height_um == 75) & (missing.offset_um == 5)]
        self.assertTrue(np.isnan(selected.crosstalk_db.iloc[0]))

    def test_spatial_matrix_labels_and_coverage(self):
        table = b.single_channel_crosstalk(example_results())
        matrix = b._single_matrix_values(table, 75)
        self.assertEqual(matrix.shape, (3, 2))
        self.assertEqual(list(matrix.columns), [729, 830])
        self.assertEqual(list(matrix.index), [-5, 0, 5])
        fig = b.plot_single_channel_matrix(table, 75, show=False)
        self.assertEqual(fig.axes[0].images[0].get_array().shape, (3, 2))
        self.assertEqual([t.get_text() for t in fig.axes[0].get_xticklabels()],
                         ["729 nm\nchannel 0", "830 nm\nchannel 0"])
        b.plot_single_channel_scan(table, show=False)
        b.plot_single_channel_matrix_grid(table, show=False)
        with self.assertRaisesRegex(ValueError, "every wavelength"):
            b._single_matrix_values(table[~((table.wavelength_nm == 830) & (table.height_um == 75))], 75)

    def test_four_reference_maps_export_and_wavelength_provenance(self):
        results = example_results()
        scan = b.analyze_reference_windows(results, {wl: (-5., 5.) for wl in results},
                                           radial_step_um=1., height_step_um=1.)
        self.assertEqual(len(scan["baseline"]), 4)
        self.assertEqual(set(scan["anchors"].wavelength_nm), {729, 830})
        self.assertNotIn("ion", scan["baseline"].columns)
        for wl, result in results.items():
            self.assertEqual(scan["maps"][wl]["db"].shape, (2, 21, 11))
            self.assertTrue(np.isin(scan["maps"][wl]["sampled_heights_um"], result["heights_um"]).all())
        fig = b.plot_height_radial_maps(scan, show=False)
        self.assertEqual([ax.get_title() for ax in fig.axes[:4]],
                         ["729 nm | Δx = -5 µm", "729 nm | Δx = +5 µm",
                          "830 nm | Δx = -5 µm", "830 nm | Δx = +5 µm"])
        profiles = b.plot_reference_axial_profiles(scan, show=False)
        self.assertTrue(profiles.axes[0].get_title().startswith("729 nm"))
        with TemporaryDirectory() as temp:
            quality = b.export_reference_windows(scan, results, temp)
            self.assertEqual(len(quality), 4)
            settings = json.loads((Path(temp)/"analysis_parameters.json").read_text())
            self.assertEqual(settings["driven_channels_per_wavelength"], 1)
            self.assertEqual(set(settings["datasets"]), {"729", "830"})
            exported = pd.read_csv(Path(temp)/"height_radial_crosstalk.csv")
            self.assertEqual(set(exported.wavelength_nm), {729, 830})
            self.assertTrue((Path(temp)/"729_nm_height_radial_arrays.npz").exists())
            paths = b.export_single_channel_figures(b.single_channel_crosstalk(results), scan, temp,
                                                    matrix_height_um=75, formats=("svg",), dpi=72)
            self.assertEqual(len(paths), 3)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))
        with self.assertRaisesRegex(ValueError, "height window"):
            b.analyze_reference_windows(results, {wl: (-5., 5.) for wl in results},
                                         reference_height_um=85, height_half_range_um=10)


if __name__ == "__main__":
    unittest.main()
