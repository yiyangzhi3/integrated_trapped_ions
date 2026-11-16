import types
import nbformat
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

# Execute only the notebook's reusable function cells, never measurement/GUI cells.
notebook_path = Path(__file__).with_name("cross_talk_analysis.ipynb")
notebook = nbformat.read(notebook_path, as_version=4)
b = types.ModuleType("cross_talk_batch_functions")
for cell in notebook.cells:
    if "functions" in cell.metadata.get("tags", []):
        exec(compile(cell.source, str(notebook_path), "exec"), b.__dict__)


def test_matrix_and_peak_rows():
    x = np.arange(-15., 16.)
    results = {}
    widths = {-1: 2., 0: 3., 1: 4.}
    for ion in [-1, 0, 1]:
        image = np.ones((2, 3, len(x)), dtype=np.float32)
        for h in range(2):
            image[h, ion+1] = 100*np.exp(-((x-ion*5)/(widths[ion]+h))**2)
        results[ion] = dict(rate=image, source=np.zeros(image.shape, dtype=np.int16),
            heights_um=np.array([75., 76.+ion*1e-13]), x_um=x,
            y_um=np.array([-10., 0., 20.]), dataset={"name": str(ion)})
    scan = b.neighboring_matrices(results)
    assert scan["db"].shape == (2, 3, 3)
    for h in range(2):
        for row, measured in enumerate([-1, 0, 1]):
            for column, driven in enumerate([-1, 0, 1]):
                expected = -10/np.log(10)*((measured-driven)*5/(widths[driven]+h))**2
                assert np.isclose(scan["db"][h, row, column], expected, atol=1e-5)
    assert np.allclose(np.diagonal(scan["db"], axis1=1, axis2=2), 0)
    # Matrix orientation: columns normalize to the corresponding driven profile.
    assert not np.isclose(scan["db"][0, 1, 0], scan["db"][0, 0, 1])
    # Changing any other dataset's coordinate origin must not change a matrix column.
    shifted = {ion: dict(result, x_um=result["x_um"]+shift)
               for (ion,result),shift in zip(results.items(),[-80.,23.,150.])}
    shifted_scan = b.neighboring_matrices(shifted,pitch_um=5)
    assert np.allclose(shifted_scan["db"],scan["db"])
    assert np.allclose(shifted_scan["table"].requested_offset_um,
                       (shifted_scan["table"].measured_ion-shifted_scan["table"].driven_ion)*5)
    profile, px, py, peak = b.axial_profile(results[-1], 0)
    assert px == -5 and py == -10 and peak == 100
    # Never substitute a higher off-row pixel when sampling an axial profile.
    results[-1]["rate"][0, 2, x==0] = 99
    table = b.crosstalk_by_height(results[-1])
    assert np.isclose(table.loc[0, "crosstalk_plus_db"], scan["db"][0, 1, 0])
    results[-1]["source"][0, 0, x==0] = -1
    missing = b.neighboring_matrices(results)
    assert np.isnan(missing["db"][0, 1, 0])
    with patch.object(b.plt, "show"):
        b.plot_neighbor_matrix(scan, 75)
        b.plot_neighbor_scan(scan)
        b.plot_matrix_grid(scan)
    print("PASS: asymmetric matrix orientation, normalization, all heights, driven-only peak rows, invariance to other datasets' origins, and invalid samples")


def test_streaming_discovery_cache():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        dataset_dir = root/"sample"
        run = dataset_dir/"signal_test"
        run.mkdir(parents=True)
        metadata = dict(status="complete", started_utc="2026-01-01T01:00:00+00:00",
            planned_frames=6, camera_id=1, pixel_clock_mhz=15, width=4, height=2,
            pixel_format="BGR8_PACKED")
        (run/"run_metadata.json").write_text(json.dumps(metadata))
        reference = dataset_dir/"focused_test.tiff"
        tifffile.imwrite(reference, np.zeros((2, 4, 3), dtype=np.uint8), photometric="rgb")
        reference.with_suffix(".json").write_text(json.dumps(dict(capture_kind="focused",
            frame_received_utc="2026-01-01T00:59:00+00:00", raw_tiff=reference.name, exposure_ms=5)))
        # Counts deliberately disagree across exposures. Rate, not counts, wins.
        counts = np.array([[100,10,5,10,10,5,6,10],
                           [150,100,6,255,50,4,300,220],
                           [200,200,100,200,250,3,150,300]], dtype=np.uint16).reshape(3,2,4)
        rows=[]
        for height in range(2):
            for exp_index, exposure in enumerate([.2,1.,10.]):
                rgb=np.zeros((2,4,3),dtype=np.uint8)
                rgb[...,0]=np.minimum(counts[exp_index],255).astype(np.uint8)
                if exp_index==1:
                    rgb[1,2]=[100,100,100]
                if exp_index==2:
                    rgb[1,3]=[100,100,100]
                name=f"frame_{height}_{exp_index}.tiff"
                tifffile.imwrite(run/name,rgb,photometric="rgb")
                rows.append(dict(height_index=height,target_height_um=75+height,
                    actual_exposure_ms=exposure,requested_exposure_ms=exposure,
                    time_index=0,exposure_index=exp_index,frame_index=0,filename=name))
        pd.DataFrame(rows).to_csv(run/"exposure_log.csv",index=False)
        dark_path=root/"dark.npz"
        np.savez(dark_path,actual_exposures_ms=np.array([.01,100.]),dark_counts=np.zeros(2),
            metadata_json=json.dumps(metadata),dark_count_definition="mean over frames and pixels of B+G+R",
            exposure_unit="ms")
        dataset=b.discover_dataset(dataset_dir)
        dark=b.load_dark_calibration(dark_path)
        alignment=b.save_alignment(dataset,dict(angle_deg=0,origin_px=[1.5,.5],
            pixel_pitch_um=[5.3,5.3],magnification=50,method="test"),root/"alignments")
        assert b.load_alignment(dataset,root/"alignments")==alignment
        assert b._fingerprint(dataset,alignment,dark,5,220)==b._fingerprint(dataset,alignment,dark,5.0,220.0)
        output=b.process_dataset(dataset,alignment,dark,cache_dir=root/"cache")
        expected=np.array([[500000,100000,10000,50000],[50000,0,300000,50000]])
        assert np.allclose(output["rate"][0],expected)
        assert output["source"][0,1,1]==-1
        assert output["source"][0,1,0]==0  # exact tie -> shorter exposure
        assert output["summary"].peak_uses_0p2_ms.all()
        assert output["saturation"].all_channels_below_sat.all()
        cached=b.process_dataset(dataset,alignment,dark,cache_dir=root/"cache")
        assert np.array_equal(output["rate"],cached["rate"])
        assert np.array_equal(output["source"],cached["source"])
        assert len(cached["comparisons"])==6
        assert b._fingerprint(dataset,alignment,dark,5,220)!=b._fingerprint(dataset,alignment,dark,5,220,[1.])
        excluded=b.process_dataset(dataset,alignment,dark,cache_dir=root/"cache",excluded_exposures_ms=[1.])
        selected=excluded["source"][excluded["source"]>=0]
        assert not np.isclose(dataset["frames"].iloc[selected].requested_exposure_ms,1.).any()
        assert not np.isclose(excluded["comparisons"].requested_exposure_ms,1.).any()
        assert np.isclose(excluded["rate"][0,0,1],50000.)
        assert np.isclose(excluded["rate"][0,1,2],30000.)
        assert excluded["excluded_exposures_ms"]==[1.]
        assert b.dark_at(dark,5)==0
        try:
            b.dark_at(dark,200)
        except ValueError:
            pass
        else:
            raise AssertionError("Extrapolation must be rejected")
        with patch.object(b.plt,"show"):
            b.plot_hdr(output,[75])
        print("PASS: automatic discovery, independent alignment cache, streaming HDR, strict thresholds, saturation, dark units, cache round trip, and plotting")


def test_reference_height_radial_windows():
    x = np.arange(-20., 21.)
    heights = np.array([65., 75., 85.])
    offsets = {-1: (5., 10.), 0: (-5., 5.), 1: (-5., -10.)}
    results = {}
    for ion in [-1, 0, 1]:
        y = np.arange(-6., 7.) + 3*ion
        image = np.empty((len(heights), len(y), len(x)), dtype=np.float32)
        for hi, height in enumerate(heights):
            for ri, actual_y in enumerate(y):
                dy = actual_y-3*ion
                center = 2*ion + (height-75)/10 + np.sign(dy)
                width = 3 + .02*(height-75) + .05*dy
                amplitude = 1000*np.exp(-dy**2/8)*(1+.01*(height-75))
                image[hi,ri] = amplitude*np.exp(-((x-center)/width)**2)
        results[ion] = dict(rate=image, source=np.zeros(image.shape, dtype=np.int16),
            heights_um=heights, x_um=x, y_um=y,
            dataset=dict(name=str(ion), frames=pd.DataFrame(dict(
                actual_exposure_ms=[.1997],requested_exposure_ms=[.2]))))
    analysis = b.analyze_reference_windows(results, offsets)
    assert len(analysis["baseline"]) == 6
    for ion, scan in analysis["maps"].items():
        assert scan["reference_x_um"] == 2*ion
        assert scan["reference_y_um"] == 3*ion
        assert scan["db"].shape == (2,3,11)
        assert np.array_equal(scan["heights_um"], heights)
        assert np.array_equal(scan["radial_offset_um"], np.arange(-5.,6.))
        for hi,height in enumerate(heights):
            for ri,dy in enumerate(scan["radial_offset_um"]):
                center = 2*ion+(height-75)/10+np.sign(dy)
                assert scan["peak_x_um"][hi,ri] == center
                width = 3+.02*(height-75)+.05*dy
                for oi,offset in enumerate(offsets[ion]):
                    expected = -10/np.log(10)*(offset/width)**2
                    assert np.isclose(scan["db"][oi,hi,ri], expected, atol=2e-5)
        for _,row in analysis["baseline"].query("ion == @ion").iterrows():
            assert np.isclose(row.crosstalk_db,-10/np.log(10)*(row.offset_um/3)**2,atol=2e-5)
    # Fractional axial positions use the nearest measured pixel (lower on an exact tie).
    fractional = dict(offsets)
    fractional[-1] = (5.5,10.)
    interpolated = b.analyze_reference_windows(results,fractional)
    expected_ratio = np.exp(-(5/3)**2)
    assert np.isclose(interpolated["maps"][-1]["ratio"][0,1,5],expected_ratio)
    # Missing a neighboring pixel must create a missing sample, never a bridge.
    missing_col = int(np.where(x == (2*(-1)+5))[0][0])
    results[-1]["source"][1,6,missing_col] = -1
    missing = b.analyze_reference_windows(results,fractional)
    assert np.isnan(missing["maps"][-1]["db"][0,1,5])
    assert np.isfinite(missing["maps"][-1]["db"][1,1,5])
    print("PASS: 75-um anchors, independent radial origins, all six signed offsets, moving axial peaks, local normalization, nearest axial samples, and invalid-pixel masks")


def test_one_um_sampling_grid():
    x = np.arange(-20.,21.)
    y = np.arange(-15.,16.)*.4
    heights = np.arange(65.,86.)
    offsets = {-1:(5.,10.),0:(-5.,5.),1:(-5.,-10.)}
    results = {}
    for ion in [-1,0,1]:
        image = np.empty((len(heights),len(y),len(x)),dtype=np.float32)
        for hi,height in enumerate(heights):
            for ri,radial in enumerate(y):
                center = 2*ion+.1*(height-75)+.25*radial
                image[hi,ri] = (1000*np.exp(-radial**2/10)
                    * np.exp(-((x-center)/(3+.04*radial))**2))
        results[ion] = dict(rate=image,source=np.zeros(image.shape,dtype=np.int16),
            heights_um=heights,x_um=x,y_um=y,
            dataset=dict(name=str(ion),frames=pd.DataFrame(dict(
                actual_exposure_ms=[.1997],requested_exposure_ms=[.2]))))
    sampled = b.analyze_reference_windows(results,offsets,radial_step_um=1.,height_step_um=1.)
    for ion,scan in sampled["maps"].items():
        assert scan["db"].shape==(2,21,11)
        assert np.allclose(np.diff(scan["radial_y_um"]),1.)
        assert np.allclose(np.diff(scan["heights_um"]),1.)
        assert np.allclose(scan["radial_offset_um"],np.arange(-5.,6.))
        # Select the nearest measured radial row before peak selection.
        radial_index=int(np.argmin(np.abs(y-1.)))
        profile=results[ion]["rate"][10,radial_index]
        col=int(np.argmax(profile))
        assert scan["peak_x_um"][10,6]==x[col]
        for oi,offset in enumerate(offsets[ion]):
            expected=np.interp(x[col]+offset,x,profile)/profile[col]
            assert np.isclose(scan["ratio"][oi,10,6],expected)
        assert np.isclose(scan["sampled_radial_y_um"][6],y[radial_index])
        assert (scan["radial_interpolation_fraction"]==0).all()
        assert (scan["interpolation_fraction"]==0).all()
    # Snap a missing requested height to the closest acquired height and record both.
    broken=dict(results)
    broken[0]=dict(results[0],heights_um=heights[::2],rate=results[0]["rate"][::2],
                   source=results[0]["source"][::2])
    nearest=b.analyze_reference_windows(broken,offsets,radial_step_um=1.,height_step_um=1.)
    assert nearest["maps"][0]["heights_um"][1]==66
    assert nearest["maps"][0]["sampled_heights_um"][1]==65
    print("PASS: nominal 1 um grid, nearest measured radial/height samples, zero interpolation weights, and actual-coordinate provenance")


def test_three_channel_common_cut():
    """Keep original coordinates and use six independently normalized samples."""
    cells = {c.get("id"): c.source for c in notebook.cells if c.cell_type == "code"}
    results = {}
    for ion, center, row_shift, width, scale in (
        (-1, -4.2, 0., 1.5, 100.),
        (0, .6, .03, 2., 300.),
        (1, 5.1, -.04, 2.5, 700.),
    ):
        x = np.arange(-15., 15.01, .1) + row_shift
        y = np.array([9., 10., 11.]) + row_shift
        selected = scale*np.exp(-((x-center)/width)**2) + .02
        # The brightest off-slice row must never set the cut or normalization.
        image = np.stack([100*scale*np.exp(-((x-11)/2)**2)+1,
                          selected, np.ones_like(x)])[None].astype(np.float32)
        results[ion] = dict(rate=image, source=np.zeros(image.shape, dtype=np.int16),
                            heights_um=np.array([75.]), x_um=x, y_um=y,
                            dataset=dict(name=f"ion_{ion}"))
    ns = dict(b.__dict__, results=results, PUBLICATION_REFERENCE_HEIGHT_UM=75.,
              display=lambda *args: None)
    for cell in ("ba-common-y-overlay", "ba-common-y-crosstalk"):
        exec(cells[cell], ns)
    table = ns["common_slice_crosstalk_table"]
    assert len(table) == 6
    matrix = ns["common_slice_crosstalk_matrix_db"]
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(np.diag(matrix), 0)
    curves = [line for line in ns["common_slice_figure"].axes[0].lines
              if line.get_label().startswith("Ion location")]
    assert len(curves) == 3
    for line, (ion, result) in zip(curves, sorted(results.items())):
        np.testing.assert_array_equal(line.get_xdata(), result["x_um"])
        np.testing.assert_allclose(np.nanmax(line.get_ydata()), 0)
        assert line.get_linestyle() == "-"
        selected_y = ns["common_slice_table"].set_index("ion").loc[ion, "sampled_y_um"]
        assert selected_y == result["y_um"][1]
    for point in table.itertuples():
        result = results[point.driven_ion]
        profile = result["rate"][0, 1]
        index = int(np.argmin(abs(result["x_um"]-point.target_peak_x_um)))
        expected = 10*np.log10(profile[index]/profile.max())
        assert np.isclose(point.crosstalk_db, expected, atol=2e-5)
        assert np.isclose(matrix.loc[point.measured_location, point.driven_ion], expected, atol=2e-5)
    assert len(ns["common_slice_separation_table"]) == 3

    # An invalid neighbor pixel stays missing instead of using interpolation.
    target = ns["common_slice_profiles"][1]["peak_x_um"]
    index = int(np.argmin(abs(results[-1]["x_um"]-target)))
    results[-1]["source"][0, 1, index] = -1
    for cell in ("ba-common-y-overlay", "ba-common-y-crosstalk"):
        exec(cells[cell], ns)
    assert np.isnan(ns["common_slice_crosstalk_matrix_db"].loc[1, -1])
    assert np.isfinite(ns["common_slice_crosstalk_matrix_db"].loc[-1, 1])
    with tempfile.TemporaryDirectory() as temp:
        ns["OUTPUT_DIR"] = Path(temp)
        exec(cells["ba-common-y-svg"], ns)
        assert ns["common_slice_svg_path"].is_file()
        assert ns["common_slice_svg_path"].suffix == ".svg"
    print("PASS: three-channel common cut, nearest rows, unshifted coordinates, all six directed ratios, matrix orientation, missing samples, and SVG export")


if __name__ == "__main__":
    test_matrix_and_peak_rows()
    test_streaming_discovery_cache()
    test_reference_height_radial_windows()
    test_one_um_sampling_grid()
    test_three_channel_common_cut()
