"""Focus analysis and plotting for Lumerical XY_cut_<height>.mat scans.

Lumerical exports P[x, y]. Convert explicitly to image[y, x], then optionally
mirror the physical x coordinate to match the laboratory comparison convention.
"""
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

from Simulation_Import_Analysis import import_lumerical_mat, Simulation_Import_Analysis


COLORS = {"data": "#245C84", "fit": "#DC861D", "focus": "#257E73"}


def set_figure_style(font_family="Arial", font_size=12):
    """Consistent notebook/export styling, with an installed font fallback."""
    try:
        font_manager.findfont(font_manager.FontProperties(family=font_family), fallback_to_default=False)
    except ValueError:
        font_family = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": font_family, "font.size": font_size,
        "axes.titlesize": font_size + 1, "axes.labelsize": font_size,
        "figure.titlesize": font_size + 2, "legend.fontsize": font_size - 2,
        "xtick.labelsize": font_size - 1, "ytick.labelsize": font_size - 1,
        "figure.dpi": 110, "savefig.dpi": 600, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.edgecolor": "#7A858E", "lines.linewidth": 2,
        "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
        "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _r_squared(observed, predicted):
    total = np.sum((observed - np.mean(observed))**2)
    return 1 - np.sum((observed - predicted)**2) / total if total > 0 else np.nan


def _fit_profile(position, intensity):
    parameters, fitted = Simulation_Import_Analysis._fit_gaussian(position, intensity)
    peak = int(np.argmax(intensity))
    level = parameters["background"] + parameters["amplitude"] * np.exp(-2)
    resolved = bool(np.any(intensity[:peak] <= level) and np.any(intensity[peak + 1:] <= level))
    return {**parameters, "r_squared": _r_squared(intensity, fitted),
            "curve": fitted, "window_resolved": resolved}


def load_focus_scan(directory, chip_surface_um=6.0, mirror_x=True, power_axes="xy"):
    """Discover numeric file heights and fit peak-centered x/y intensity slices.

    power_axes='xy' is the export from P=zeros(Nx, Ny). Use 'yx' only for a
    deliberately transposed export. Raw peak intensity is retained across planes.
    """
    directory = Path(directory).expanduser()
    if power_axes not in ("xy", "yx"):
        raise ValueError("power_axes must be 'xy' or 'yx'.")
    files = []
    for path in directory.glob("XY_cut_*.mat"):
        match = re.fullmatch(r"XY_cut_(-?\d+(?:\.\d+)?)\.mat", path.name)
        if match:
            files.append((float(match.group(1)), path))
    files.sort()
    if not files:
        raise FileNotFoundError(f"No XY_cut_<height>.mat exports in {directory}")
    heights = np.array([height for height, _ in files])
    if len(np.unique(heights)) != len(heights):
        raise ValueError(f"Duplicate numeric file heights in {directory}")
    planes = []
    for height, path in files:
        data = import_lumerical_mat(path)
        x, y = np.asarray(data["x"], dtype=float).ravel(), np.asarray(data["y"], dtype=float).ravel()
        raw = np.asarray(data["P"])
        if np.iscomplexobj(raw) or not np.all(np.isfinite(raw)):
            raise ValueError(f"{path}: P must be finite and real.")
        expected = (x.size, y.size) if power_axes == "xy" else (y.size, x.size)
        if raw.shape != expected:
            raise ValueError(f"{path}: expected P{expected} for axes {power_axes}, found {raw.shape}")
        image = raw.T.copy() if power_axes == "xy" else raw.copy()
        x_um = (-x if mirror_x else x) * 1e6
        y_um = y * 1e6
        ix, iy = np.argsort(x_um), np.argsort(y_um)
        x_um, y_um, image = x_um[ix], y_um[iy], image[np.ix_(iy, ix)]
        if not (np.all(np.diff(x_um) > 0) and np.all(np.diff(y_um) > 0)):
            raise ValueError(f"{path}: coordinates must be distinct.")
        peak = float(image.max())
        if peak <= 0:
            raise ValueError(f"{path}: peak intensity must be positive.")
        image_normalized = image / peak
        row, col = np.unravel_index(np.argmax(image), image.shape)
        profiles = {"x": image_normalized[row, :], "y": image_normalized[:, col]}
        fits = {axis: _fit_profile(position, profiles[axis])
                for axis, position in (("x", x_um), ("y", y_um))}
        planes.append({"path": path, "height_um": height,
                       "height_above_chip_um": height - chip_surface_um,
                       "x_um": x_um, "y_um": y_um, "intensity": image,
                       "normalized_intensity": image_normalized, "peak_intensity": peak,
                       "peak_x_um": x_um[col], "peak_y_um": y_um[row],
                       "profiles": profiles, "fits": fits})
    return {"directory": directory, "chip_surface_um": chip_surface_um,
            "height_um": heights, "height_above_chip_um": heights - chip_surface_um,
            "planes": planes, "max_intensity": np.array([p["peak_intensity"] for p in planes]),
            "radii_um": {axis: np.array([p["fits"][axis]["waist_um"] for p in planes])
                         for axis in ("x", "y")},
            "mirror_x": mirror_x, "power_axes": power_axes}


def waist_model(height, w0, focus, rayleigh):
    return w0 * np.sqrt(1 + ((height - focus) / rayleigh)**2)


def intensity_model(height, peak, focus, rayleigh, exponent=0.5):
    return peak / (1 + ((height - focus) / rayleigh)**2)**exponent


def fit_focus_scan(scan, focus_axis="y", intensity_exponent=0.5):
    """Fit propagation radius and raw peak intensity versus height above chip.

    The two fits are independent phenomenological Gaussian propagation fits;
    they need not give identical focal heights. No diffraction/M² constraint is
    imposed. y is the narrow axis in these Lumerical exports.
    """
    if focus_axis not in ("x", "y") or intensity_exponent <= 0:
        raise ValueError("Choose focus_axis x/y and a positive intensity exponent.")
    h = scan["height_above_chip_um"]
    if h.size < 5:
        raise ValueError("A focus scan needs at least five planes.")
    radii, intensity = scan["radii_um"][focus_axis], scan["max_intensity"]
    span = np.ptp(h)
    wp, wc = curve_fit(waist_model, h, radii,
                      p0=[radii.min(), h[np.argmin(radii)], span / 2],
                      bounds=([1e-9, -np.inf, 1e-9], [np.inf, np.inf, np.inf]), maxfev=30000)
    scale = intensity.max()
    def model(height, peak, focus, rayleigh):
        return intensity_model(height, peak, focus, rayleigh, intensity_exponent)
    ip, ic = curve_fit(model, h, intensity / scale,
                      p0=[1, h[np.argmax(intensity)], wp[2]],
                      bounds=([0, -np.inf, 1e-9], [np.inf, np.inf, np.inf]), maxfev=30000)
    scan["focus_fit"] = {
        "axis": focus_axis, "intensity_exponent": intensity_exponent,
        "waist_parameters": wp, "waist_covariance": wc,
        "intensity_parameters": ip, "intensity_covariance": ic, "intensity_scale": scale,
        "waist_r_squared": _r_squared(radii, waist_model(h, *wp)),
        "intensity_r_squared": _r_squared(intensity, scale * model(h, *ip)),
        "waist_focus_above_chip_um": wp[1], "intensity_focus_above_chip_um": ip[1],
        "minimum_radius_um": wp[0], "sampled_peak_above_chip_um": h[np.argmax(intensity)],
        "waist_focus_bracketed": bool(h.min() < wp[1] < h.max()),
        "intensity_focus_bracketed": bool(h.min() < ip[1] < h.max()),
    }
    return scan


def _style_axis(ax):
    ax.grid(color="#D8DFE5", alpha=0.65, linewidth=0.7)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.margins(x=0.03)


def plot_focus_plane(scan, label, height_um=None):
    """Three panels: normalized XY map and peak-centered x/y Gaussian cuts."""
    index = int(np.argmax(scan["max_intensity"])) if height_um is None else int(np.argmin(abs(scan["height_um"] - height_um)))
    plane = scan["planes"][index]
    fig = plt.figure(figsize=(16, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 0.045, 1, 1], height_ratios=[0.10, 1], wspace=0.08)
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.set_axis_off()
    axes = [fig.add_subplot(grid[1, col]) for col in (0, 2, 3)]
    color_ax = fig.add_subplot(grid[1, 1])
    color_ax.set_box_aspect(1 / 0.045)
    image = axes[0].pcolormesh(plane["x_um"], plane["y_um"], plane["normalized_intensity"],
                              shading="auto", cmap="inferno", vmin=0, vmax=1, rasterized=True)
    axes[0].axhline(plane["peak_y_um"], color="#58D8E0", ls="--", lw=1)
    axes[0].axvline(plane["peak_x_um"], color="#58D8E0", ls=":", lw=1)
    axes[0].plot(plane["peak_x_um"], plane["peak_y_um"], "+", color="white", ms=9, mew=1.5)
    axes[0].set(xlabel="x (µm)", ylabel="y (µm)", title="(a) XY intensity")
    axes[0].set_aspect("equal", adjustable="box")
    fig.colorbar(image, cax=color_ax, label="Intensity / plane maximum", ticks=[0, 0.5, 1])
    for ax, axis, letter in zip(axes[1:], ("x", "y"), ("b", "c")):
        position, profile, fit = plane[f"{axis}_um"], plane["profiles"][axis], plane["fits"][axis]
        dense = np.linspace(position.min(), position.max(), 600)
        params = [fit[key] for key in ("amplitude", "center_um", "waist_um", "background")]
        ax.plot(position, profile, "o", ms=3.5, mfc="white", mec=COLORS["data"], mew=0.9, label="Simulation")
        ax.plot(dense, Simulation_Import_Analysis.gaussian(dense, *params), color=COLORS["fit"], lw=2, label="Gaussian fit")
        fixed = "y" if axis == "x" else "x"
        fixed_position = plane[f"peak_{fixed}_um"]
        fixed_position = 0.0 if abs(fixed_position) < 1e-9 else fixed_position
        ax.set(xlabel=f"{axis} (µm)", ylabel="Normalized intensity", ylim=(-0.03, 1.35),
               title=f"({letter}) {axis} cut at {fixed} = {fixed_position:.2f} µm")
        note = fr"$w_{axis}$ = {fit['waist_um']:.3f} µm" + "\n" + fr"$R^2$ = {fit['r_squared']:.4f}"
        if not fit["window_resolved"]:
            note += "\nWindow-limited profile"
        ax.text(0.04, 0.96, note, transform=ax.transAxes, va="top", fontsize=10,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#DFE5EA", "alpha": 0.92})
        _style_axis(ax)
        ax.set_box_aspect(1)
    handles, labels = axes[1].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", ncol=2, handlelength=2.2, columnspacing=2)
    fig.suptitle(f"{label}  ·  height above chip {plane['height_above_chip_um']:g} µm"
                 f"  (absolute z = {plane['height_um']:g} µm)")
    return fig, plane


def plot_ion0_axial_radial_map(scan, height_um=None, axial_axis="y",
                             x_limits_um=None, y_limits_um=None, cmap=None,
                             transpose_intensity=False):
    """Show a beam-centered ion0 map with axial x and radial y plot axes.

    Select the sampled intensity-maximum plane by default. The axial export
    axis is the narrow direction used for the waist analysis. Gray outside the
    exported rectangle represents unmeasured coordinates, not zero intensity.
    transpose_intensity transposes only the displayed values on fixed axes.
    """
    if axial_axis not in ("x", "y"):
        raise ValueError("Choose the exported x or y axis for the axial coordinate.")
    index = int(np.argmax(scan["max_intensity"])) if height_um is None else int(np.argmin(abs(scan["height_um"] - height_um)))
    plane = scan["planes"][index]
    radial_axis = "x" if axial_axis == "y" else "y"
    axial = plane[f"{axial_axis}_um"] - plane[f"peak_{axial_axis}_um"]
    radial = plane[f"{radial_axis}_um"] - plane[f"peak_{radial_axis}_um"]
    image = plane["normalized_intensity"].T if axial_axis == "y" else plane["normalized_intensity"]
    if transpose_intensity:
        image = image.T
    if image.shape != (radial.size, axial.size):
        raise ValueError("Transposing intensity on fixed coordinates requires matching axis sizes.")
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("wavelength_729nm", ["black", "#b10000", "white"])
    full_window = x_limits_um is None and y_limits_um is None
    fig, ax = plt.subplots(figsize=(7, 6) if full_window else (6, 8), constrained_layout=True)
    ax.set_facecolor("#E6E8EB")
    mesh = ax.pcolormesh(axial, radial, image, shading="auto", cmap=cmap,
                         vmin=0, vmax=1, rasterized=True)
    bounds = mesh.get_datalim(ax.transData)
    if x_limits_um is None:
        x_limits_um = (bounds.x0, bounds.x1)
    if y_limits_um is None:
        y_limits_um = (bounds.y0, bounds.y1)
    ax.set(xlabel="Axial x (µm)", ylabel="Radial y (µm)",
           xlim=x_limits_um, ylim=y_limits_um,
           title=r"Ion0 · 729 nm · $\lambda_n/18$ with z refinement")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_xticks(np.linspace(*x_limits_um, 5))
    ax.set_yticks(np.linspace(*y_limits_um, 5))
    if y_limits_um[0] < bounds.y0:
        ax.text(0.5, 0.025, "Gray: outside exported window", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9, color="#626B75")
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.04)
    colorbar.set_label("Normalized intensity")
    colorbar.set_ticks(np.linspace(0, 1, 6))
    print(f"Ion0 map: absolute z = {plane['height_um']:g} µm; "
          f"height above chip = {plane['height_above_chip_um']:g} µm.")
    print(f"Beam-centered axial x uses export {axial_axis}; radial y uses export {radial_axis}.")
    print(f"Exported centered extent: axial [{axial.min():.3f}, {axial.max():.3f}] µm; "
          f"radial [{radial.min():.3f}, {radial.max():.3f}] µm. Gray is outside the exported window.")
    return fig, plane


def plot_focus_propagation(scan, label):
    """Peak intensity and narrow-axis radius, with independently fitted focus."""
    fit = scan["focus_fit"]
    h = scan["height_above_chip_um"]
    dense = np.linspace(h.min(), h.max(), 700)
    axis = fit["axis"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), constrained_layout=True)
    specifications = [
        (scan["max_intensity"], fit["intensity_scale"] * intensity_model(dense, *fit["intensity_parameters"], exponent=fit["intensity_exponent"]),
         fit["intensity_focus_above_chip_um"], fit["intensity_r_squared"], "Peak intensity (exported units)", "(a) Intensity focus"),
        (scan["radii_um"][axis], waist_model(dense, *fit["waist_parameters"]),
         fit["waist_focus_above_chip_um"], fit["waist_r_squared"], fr"$w_{axis}$, 1/e² intensity radius (µm)", f"(b) {axis}-radius focus"),
    ]
    for ax, (samples, curve, focus, r2, ylabel, title) in zip(axes, specifications):
        ax.plot(h, samples, "o", ms=5, mfc="white", mec=COLORS["data"], mew=1.4, label="Simulation")
        ax.plot(dense, curve, color=COLORS["fit"], label="Propagation fit")
        if h.min() <= focus <= h.max():
            ax.axvline(focus, color=COLORS["focus"], ls="--", lw=1.2)
        bracket = "" if h.min() < focus < h.max() else "\nOutside scanned interval"
        note = f"Focus = {focus:.2f} µm\n" + fr"$R^2$ = {r2:.4f}" + bracket
        ax.text(0.04, 0.96, note, transform=ax.transAxes, va="top", fontsize=11,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#DFE5EA", "alpha": 0.94})
        ax.set(xlabel="Height above chip (µm)", ylabel=ylabel, title=title)
        _style_axis(ax)
        lower, upper = min(samples.min(), curve.min()), max(samples.max(), curve.max())
        pad = 0.2 * (upper - lower) if upper > lower else 1
        ax.set_ylim(max(0, lower - pad), upper + 1.2 * pad)
        ax.legend(loc="lower center", ncol=2)
    fig.suptitle(f"{label}  ·  focus scan")
    return fig


def report_focus_scan(scan, label):
    fit = scan["focus_fit"]
    print(f"{label}: {len(scan['planes'])} planes; absolute z {scan['height_um'][0]:g}–{scan['height_um'][-1]:g} µm")
    print(f"  Chip surface z = {scan['chip_surface_um']:g} µm; focus radius axis = {fit['axis']}")
    print(f"  Sampled intensity maximum: {fit['sampled_peak_above_chip_um']:.3f} µm above chip")
    print(f"  Intensity-model focus:     {fit['intensity_focus_above_chip_um']:.3f} µm; R² = {fit['intensity_r_squared']:.6f}")
    print(f"  Radius-model focus:        {fit['waist_focus_above_chip_um']:.3f} µm; R² = {fit['waist_r_squared']:.6f}")
    print(f"  Minimum fitted w_{fit['axis']}:       {fit['minimum_radius_um']:.4f} µm (1/e² radius)")
    for axis in ("x", "y"):
        clipped = sum(not plane["fits"][axis]["window_resolved"] for plane in scan["planes"])
        if clipped:
            print(f"  {axis} profile: {clipped}/{len(scan['planes'])} planes do not resolve both 1/e² edges; widths rely on extrapolation.")
    if not (fit["intensity_focus_bracketed"] and fit["waist_focus_bracketed"]):
        print("  A fitted focus lies outside the scanned interval; extend the height scan.")


def report_focus_offsets(scan, reference_position_um=0.0):
    """Report the focus-axis Gaussian center relative to a stated coordinate.

    Interpolate centers linearly in height at each fitted focus; do not
    extrapolate beyond the measured planes. Coordinates use the scan's
    declared mirror/axis convention.
    """
    fit = scan["focus_fit"]
    axis = fit["axis"]
    heights = scan["height_above_chip_um"]
    centers = np.array([plane["fits"][axis]["center_um"] for plane in scan["planes"]])
    results = {}
    print(f"Axial offset uses Gaussian-fit {axis} centers relative to {axis} = {reference_position_um:g} µm.")
    for name, key in (("Intensity focus", "intensity_focus_above_chip_um"),
                      ("Minimum-waist focus", "waist_focus_above_chip_um")):
        height = float(fit[key])
        if not heights.min() <= height <= heights.max():
            raise ValueError(f"{name} lies outside the scan; its position cannot be interpolated.")
        center = float(np.interp(height, heights, centers))
        offset = center - reference_position_um
        results[name] = {"axis": axis, "height_above_chip_um": height,
                         "absolute_height_um": height + scan["chip_surface_um"],
                         "center_um": center, "reference_position_um": reference_position_um,
                         "offset_um": offset}
        print(f"  {name}: height = {height:.3f} µm above chip; "
              f"{axis} center = {center:+.4f} µm; axial offset = {offset:+.4f} µm")
    return results


def plot_focus_comparison(scans):
    """Compare sampled scan curves and fit-derived focal heights across meshes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    palette = ["#E59A36", "#4C78A8", "#269B84"]
    for (label, scan), color in zip(scans.items(), palette):
        h, fit = scan["height_above_chip_um"], scan["focus_fit"]
        dense = np.linspace(h.min(), h.max(), 400)
        axes[0].plot(h, scan["max_intensity"], "o", ms=3.8, mfc="white", color=color)
        axes[0].plot(dense, fit["intensity_scale"] * intensity_model(dense, *fit["intensity_parameters"], exponent=fit["intensity_exponent"]), color=color, label=label)
        axes[1].plot(h, scan["radii_um"][fit["axis"]], "o", ms=3.8, mfc="white", color=color)
        axes[1].plot(dense, waist_model(dense, *fit["waist_parameters"]), color=color, label=label)
    for ax, ylabel in zip(axes, ("Peak intensity", "Axial beam waist")):
        ax.set(xlabel="Height above chip (µm)", ylabel=ylabel)
        _style_axis(ax)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("Focus Analysis across Mesh Settings")
    return fig
