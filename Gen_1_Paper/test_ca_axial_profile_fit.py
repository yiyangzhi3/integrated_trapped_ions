"""Check Gaussian-center annotations without replacing measured profiles."""

import ast
import json
from pathlib import Path
import types
import unittest

import matplotlib.pyplot as plt
import numpy as np


NOTEBOOK = Path(__file__).with_name("cross_talk_analysis_ca_2.ipynb")
notebook = json.loads(NOTEBOOK.read_text())
b = types.ModuleType("ca_axial_profile_functions")
for cell in notebook["cells"][:5]:
    if "functions" in cell.get("metadata", {}).get("tags", []):
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), b.__dict__)


class GaussianCenterTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_subpixel_centers_and_fixed_reference_leave_measurements_unchanged(self):
        x = np.arange(-12.5, 12.5, .106)
        y = np.array([19.4, 19.5, 19.6])
        centers = {0: -1.137, 1: 3.593}
        results = {}
        for ion in (0, 1):
            radius = 2.1 if ion == 0 else 1.6
            values = 2. + (1000.+500.*ion)*np.exp(-2*((x-centers[ion])/radius)**2)
            image = np.broadcast_to(values, (1, 3, len(x))).copy()
            results[ion] = dict(rate=image, source=np.zeros(image.shape, dtype=np.int16),
                                x_um=x, y_um=y, heights_um=np.array([75.]),
                                dataset=dict(name=f"ion{ion}", wavelength_nm=729))
        ns = dict(b.__dict__, results=results, WAVELENGTH_NM=729,
                  PUBLICATION_REFERENCE_HEIGHT_UM=75., display=lambda *args: None)
        cell = next(c for c in notebook["cells"] if c.get("id") == "common-y-slice-overlay")
        exec("".join(cell["source"]), ns)
        self.assertAlmostEqual(ns["common_slice_separation_um"], 4.73, places=5)
        self.assertGreater(abs(ns["common_slice_peak_separation_um"]-4.73), .001)
        ax = ns["common_slice_figure"].axes[0]
        curves = [line for line in ax.lines if line.get_label().startswith("Ion location")]
        guides = [line for line in ax.lines if line.get_linestyle() == ":"]
        markers = [line for line in ax.lines if line.get_marker() == "o"]
        self.assertEqual(len(curves), 2)
        self.assertEqual(len(guides), 2)
        self.assertEqual(len(markers), 2)
        self.assertEqual(len(ax.lines), len(curves)+len(guides)+len(markers))
        placement = ns["common_slice_fixed_spacing"]
        self.assertAlmostEqual(placement["right_x_um"]-placement["left_x_um"], 5., places=12)
        for ion, curve, guide in zip((0, 1), curves, guides):
            fit = ns["common_slice_gaussian_fits"][ion]
            self.assertAlmostEqual(fit["center_um"], centers[ion], places=5)
            self.assertGreater(fit["r_squared"], .999999)
            self.assertGreater(fit["counts"].max(), 900.)
            np.testing.assert_allclose(guide.get_xdata(),
                                       ns["common_slice_profiles"][ion]["reference_position_x_um"])
            np.testing.assert_array_equal(curve.get_xdata(), x)
            measured = results[ion]["rate"][0, 1]
            np.testing.assert_allclose(curve.get_ydata(), 10*np.log10(measured/measured.max()))
            self.assertEqual(ns["common_slice_profiles"][ion]["peak_x_um"], x[np.argmax(measured)])

    def test_fixed_spacing_finds_intersection_optimum_without_bridging_invalid_pixels(self):
        cell = next(c for c in notebook["cells"] if c.get("id") == "common-y-slice-overlay")
        node = next(node for node in ast.parse("".join(cell["source"])).body
                    if isinstance(node, ast.FunctionDef) and node.name == "place_fixed_spacing_markers")
        ns = dict(b.__dict__)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), "exec"), ns)
        place = ns["place_fixed_spacing_markers"]
        x = np.array([0., 1., 2., 3.])
        left, right = np.array([.2, 1., .2, .1]), np.array([.1, .2, 1., .2])
        best = place(x, left, x, right, spacing_um=1.5)
        self.assertAlmostEqual(best["left_x_um"], .75)
        self.assertAlmostEqual(best["right_x_um"], 2.25)
        self.assertAlmostEqual(best["weaker_normalized_intensity"], .8)
        right[2] = np.nan
        missing = place(x, left, x, right, spacing_um=1.5)
        self.assertAlmostEqual(missing["right_x_um"], 3.)
        self.assertAlmostEqual(missing["weaker_normalized_intensity"], .2)


if __name__ == "__main__":
    unittest.main()
