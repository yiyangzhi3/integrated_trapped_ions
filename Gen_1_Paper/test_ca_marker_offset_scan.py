"""Check the ratio of two moving endpoints in an exact-spacing window."""

import ast
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


notebook = json.loads(Path(__file__).with_name("cross_talk_analysis_ca_2.ipynb").read_text())
namespace = dict(np=np, pd=pd)
for cell_id, names in (("f3ef4a75", {"sample_linear", "ratio_db"}),
                       ("ca-marker-offset-scan", {"marker_shift_crosstalk_scan"})):
    source = "".join(next(c for c in notebook["cells"] if c.get("id") == cell_id)["source"])
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "marker_scan", "exec"), namespace)
scan = namespace["marker_shift_crosstalk_scan"]


class MarkerOffsetTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(-10., 11.)
        self.profiles = {
            0: dict(x_um=self.x, normalized_intensity=np.maximum(.01, 1-.2*abs(self.x))),
            1: dict(x_um=self.x, normalized_intensity=np.maximum(.01, 1-.25*abs(self.x-5))),
        }

    def test_both_endpoints_move_and_exact_crossings_do_not_depend_on_display_grid(self):
        original = {i: p["normalized_intensity"].copy() for i, p in self.profiles.items()}
        table, crossings = scan(self.profiles, 0, 1, 0., 5., np.arange(-4., 5.))
        np.testing.assert_allclose(table.right_marker_x_um-table.left_marker_x_um, 5.)
        np.testing.assert_allclose(table.left_marker_x_um, table.axial_offset_um)
        threshold = 10**(-5/10)
        nearest = crossings.loc[crossings.groupby("driven_ion").axial_offset_um.apply(lambda s: s.abs().idxmin())]
        self.assertAlmostEqual(nearest.query("driven_ion == 0").iloc[0].axial_offset_um,
                               -threshold/(.2*(1+threshold)), places=10)
        self.assertAlmostEqual(nearest.query("driven_ion == 1").iloc[0].axial_offset_um,
                               (.23+threshold)/(.24+.25*threshold), places=10)
        np.testing.assert_allclose(crossings.crosstalk_db, -5., atol=1e-9)
        # At shift +2, the orange left endpoint is .25 and its right endpoint is .5.
        value = table.query("driven_ion == 1 and axial_offset_um == 2").iloc[0]
        self.assertAlmostEqual(value.offsite_intensity, .25)
        self.assertAlmostEqual(value.onsite_intensity, .5)
        self.assertAlmostEqual(value.intensity_ratio, .5)
        self.assertAlmostEqual(value.crosstalk_db, 10*np.log10(.5))
        _, dense_crossings = scan(self.profiles, 0, 1, 0., 5., np.linspace(-4., 4., 801))
        np.testing.assert_allclose(dense_crossings.axial_offset_um, crossings.axial_offset_um)
        for ion in self.profiles:
            np.testing.assert_array_equal(self.profiles[ion]["normalized_intensity"], original[ion])

    def test_missing_pixels_are_not_interpolated_across(self):
        self.profiles[0]["normalized_intensity"][self.x == 3.] = np.nan
        table, crossings = scan(self.profiles, 0, 1, 0., 5., np.arange(-4., 4.01, .5))
        self.assertTrue(crossings.query("driven_ion == 0").empty)
        self.assertTrue(np.isnan(table.query("driven_ion == 0 and axial_offset_um == -1.5").iloc[0].intensity_ratio))
        self.assertTrue((crossings.query("driven_ion == 1").axial_offset_um > 0).any())

    def test_invalid_denominator_is_not_used(self):
        self.profiles[0]["normalized_intensity"][self.x == -1.] = np.nan
        table, crossings = scan(self.profiles, 0, 1, 0., 5., np.arange(-4., 4.01, .5))
        self.assertTrue(np.isnan(table.query("driven_ion == 0 and axial_offset_um == -1.5").iloc[0].intensity_ratio))
        self.assertTrue(crossings.query("driven_ion == 0 and -2 < axial_offset_um < 0").empty)

    def test_curve_scaling_cancels_in_endpoint_ratios(self):
        table, crossings = scan(self.profiles, 0, 1, 0., 5., np.arange(-4., 5.))
        scaled = {i: dict(p, normalized_intensity=p["normalized_intensity"]*scale)
                  for (i, p), scale in zip(self.profiles.items(), (17., .043))}
        scaled_table, scaled_crossings = scan(scaled, 0, 1, 0., 5., np.arange(-4., 5.))
        np.testing.assert_allclose(scaled_table.intensity_ratio, table.intensity_ratio)
        np.testing.assert_allclose(scaled_crossings.axial_offset_um, crossings.axial_offset_um)

    def test_common_coordinate_translation_does_not_change_required_offsets(self):
        table, crossings = scan(self.profiles, 0, 1, 0., 5., np.arange(-4., 5.))
        shifted = {i: dict(p, x_um=p["x_um"]+2.3) for i, p in self.profiles.items()}
        translated_table, translated_crossings = scan(shifted, 0, 1, 2.3, 5., np.arange(-4., 5.))
        np.testing.assert_allclose(translated_table.intensity_ratio, table.intensity_ratio)
        np.testing.assert_allclose(translated_crossings.axial_offset_um, crossings.axial_offset_um)


if __name__ == "__main__":
    unittest.main()
