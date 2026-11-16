"""Manual lens alignment and height-plane publication plots for Ca scans."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .Experimental_Import import _pick_two_points, _require_interactive_backend
from .ca_3d_import import WAVELENGTHS_NM


WAVELENGTH_COLORS = {
    405: "#3b00ff", 450: "#001aff", 515: "#00ff59",
    635: "#ff000d", 729: "#ff210f", 880: "#c780ff",
}


def rotate_image(image, angle_deg):
    """Same cropped nearest-neighbor rotation as Experimental_Import."""
    return ndimage.rotate(image, angle_deg, axes=(1, 0), reshape=False, order=0)


def _rotate_points(points, shape, angle_deg):
    center = (np.array(shape[:2][::-1], dtype=float) - 1) / 2
    angle = np.deg2rad(angle_deg)
    # Forward transform for image coordinates (columns right, rows down).
    matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    return (np.asarray(points) - center) @ matrix.T + center


def alignment_from_points(scan, points, *, pixel_size_um=5.3, magnification=50.0):
    """Rotate any two manually chosen points so their line becomes horizontal.

    Point order does not matter. The smallest correction within +/-90 degrees
    is used, avoiding a 180-degree flip. Origin is their rotated midpoint.
    """
    image = scan.read_reference()
    points = np.asarray(points, dtype=float)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("Choose exactly two finite (column, row) points.")
    if np.any(points < 0) or np.any(points > np.array(image.shape[:2][::-1]) - 1):
        raise ValueError("Both points must lie inside the reference image.")
    delta = points[1] - points[0]
    if np.linalg.norm(delta) < 1:
        raise ValueError("Choose two distinct points at least one pixel apart.")
    if not np.isfinite([pixel_size_um, magnification]).all() or min(pixel_size_um, magnification) <= 0:
        raise ValueError("Pixel size and magnification must be positive and finite.")
    angle = (np.degrees(np.arctan2(delta[1], delta[0])) + 90) % 180 - 90
    rotated = _rotate_points(points, image.shape, angle)
    return {
        "version": 1, "method": "manual two points to horizontal",
        "ion_index": scan.ion_index, "wavelength_nm": scan.wavelength_nm,
        "reference_path": str(scan.reference_path.resolve()),
        "reference_sha256": hashlib.sha256(scan.reference_path.read_bytes()).hexdigest(),
        "shape": list(image.shape[:2]), "points_px": points.tolist(),
        "angle_deg": float(angle), "origin_px": rotated.mean(axis=0).tolist(),
        "pixel_size_um": float(pixel_size_um), "magnification": float(magnification),
        "rotation_order": 0, "reshape": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def align_scans_manual(scans, directory, *, reselect=(), pixel_size_um=5.3, magnification=50.0):
    """Reuse saved manual selections or collect two clicks for each measured scan.

    Uses the existing importer's interactive-backend check and two-point picker.
    Escape / closing the selector cancels without saving that wavelength.
    A changed reference or camera calibration requires explicit re-selection.
    """
    directory = Path(directory)
    alignments = dict.fromkeys(WAVELENGTHS_NM)
    for wl, scan in scans.items():
        if scan is None:
            continue
        prefix = f"{scan.device_name}_" if scan.device_name else ""
        path = directory / f"{prefix}ion_{scan.ion_index}_{wl}_nm.json"
        if path.exists() and wl not in reselect:
            record = json.loads(path.read_text())
            signature = hashlib.sha256(scan.reference_path.read_bytes()).hexdigest()
            if (
                record.get("version") != 1
                or record.get("ion_index") != scan.ion_index
                or record.get("wavelength_nm") != wl
                or record.get("reference_path") != str(scan.reference_path.resolve())
                or record.get("reference_sha256") != signature
                or record.get("pixel_size_um") != pixel_size_um
                or record.get("magnification") != magnification
            ):
                raise ValueError(f"{path.name}: reference/calibration changed. Add {wl} to RESELECT_WAVELENGTHS.")
            print(f"{scan.ion_index}, {wl} nm: reused manual rotation {record['angle_deg']:.4f} degrees")
        else:
            _require_interactive_backend()
            points = _pick_two_points(
                scan.read_reference(),
                f"Ion {scan.ion_index}, {wl} nm: click TWO lens points to make horizontal",
            )
            record = alignment_from_points(scan, points, pixel_size_um=pixel_size_um, magnification=magnification)
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"{scan.ion_index}, {wl} nm: saved manual rotation {record['angle_deg']:.4f} degrees")
        alignments[wl] = record
    return alignments


def aligned_axes(alignment):
    rows, cols = alignment["shape"]
    x0, y0 = alignment["origin_px"]
    spacing = alignment["pixel_size_um"] / alignment["magnification"]
    return (np.arange(cols) - x0) * spacing, -(np.arange(rows) - y0) * spacing


def preview_alignments(scans, alignments):
    """Show chosen points and their rotated positions for visual verification."""
    import matplotlib.pyplot as plt

    figures = []
    for wl, scan in scans.items():
        if scan is None or alignments.get(wl) is None:
            continue
        record = alignments[wl]
        image = scan.read_reference()
        points = np.asarray(record["points_px"])
        rotated_points = _rotate_points(points, image.shape, record["angle_deg"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        axes[0].imshow(image, origin="upper", cmap="gray")
        axes[0].plot(*points.T, "c+-", markersize=12)
        axes[0].set_title("Your two selected points")
        axes[1].imshow(rotate_image(image, record["angle_deg"]), origin="upper", cmap="gray")
        axes[1].plot(*rotated_points.T, "c+-", markersize=12)
        axes[1].plot(*record["origin_px"], "rx")
        axes[1].axhline(record["origin_px"][1], color="cyan", ls="--", lw=.7)
        axes[1].set_title(f"Rotated {record['angle_deg']:.3f}°; red × = lens origin")
        for ax in axes:
            ax.set(xlabel="Column (pixels)", ylabel="Row (pixels)")
        fig.suptitle(f"Ion {scan.ion_index} · {wl} nm")
        figures.append(fig)
    return figures


def select_height_plane(scan, alignment, height_um, *, raw_stack=None, max_height_error_um=0.6,
                        center_um=None, peak_roi_um=None, background_counts=0.0):
    """Read nearest measured height, rotate it, and locate its beam center.

    Axes before recentering use the manual lens midpoint, x right and y up.
    Scans flagged flip_vertical are reflected about lens y=0 after rotation,
    before ROI selection and centering. Raw frames/stacks stay unchanged.
    Optional peak_roi_um = (xmin,xmax,ymin,ymax) restricts peak detection in
    those lens coordinates. center_um overrides detection with an (x,y) pair.
    Background is a user-specified scalar R+G+B count, subtracted before
    clipping to zero. No automatic background or exposure correction.
    """
    if not np.isfinite(height_um) or not np.isfinite(max_height_error_um) or max_height_error_um < 0:
        raise ValueError("Height must be finite and its allowed error nonnegative.")
    if not np.isfinite(background_counts) or background_counts < 0:
        raise ValueError("Background counts must be finite and nonnegative.")
    if (alignment["ion_index"], alignment["wavelength_nm"]) != (scan.ion_index, scan.wavelength_nm):
        raise ValueError("Alignment belongs to another scan.")
    slot = int(np.argmin(np.abs(scan.frames.height_um.to_numpy() - height_um)))
    row = scan.frames.iloc[slot]
    if abs(row.height_um - height_um) > max_height_error_um:
        raise ValueError(
            f"{scan.wavelength_nm} nm: requested {height_um:g} um; closest is {row.height_um:.4f} um. "
            f"Measured range {scan.frames.height_um.min():.4f}–{scan.frames.height_um.max():.4f} um."
        )
    if raw_stack is None:
        raw = scan.read_frame(int(row.image_index)).astype(np.float32)
        if raw.ndim == 3:
            raw = raw[..., :3].sum(axis=2)
    else:
        if raw_stack.shape != (len(scan.frames), *alignment["shape"]):
            raise ValueError("Raw stack must match this scan's frame count and image dimensions.")
        raw = np.asarray(raw_stack[slot], dtype=np.float32)
    if raw.shape != tuple(alignment["shape"]):
        raise ValueError("Scan frame and reference dimensions differ.")
    image = rotate_image(raw, alignment["angle_deg"])
    image = np.maximum(image - background_counts, 0)
    x, y = aligned_axes(alignment)
    if scan.flip_vertical:
        image = image[::-1].copy()
        y = -y[::-1]
    if center_um is None:
        mask = np.ones(image.shape, dtype=bool)
        if peak_roi_um is not None:
            xmin, xmax, ymin, ymax = peak_roi_um
            if not np.isfinite(peak_roi_um).all() or xmin >= xmax or ymin >= ymax:
                raise ValueError("Peak ROI must be (xmin,xmax,ymin,ymax) with increasing limits.")
            mask &= ((y >= ymin) & (y <= ymax))[:, None] & ((x >= xmin) & (x <= xmax))[None, :]
        if not mask.any() or image[mask].max() <= 0:
            raise ValueError(f"{scan.wavelength_nm} nm: no positive signal in the peak search region.")
        search_image = np.where(mask, image, -np.inf)
        # Preserve the same peak pixel when equal maxima span multiple rows.
        # A reflection must not change the tie-breaking camera-row order.
        if scan.flip_vertical:
            iy, ix = np.unravel_index(np.argmax(search_image[::-1]), image.shape)
            iy = image.shape[0] - 1 - iy
        else:
            iy, ix = np.unravel_index(np.argmax(search_image), image.shape)
        center_um = (x[ix], y[iy])
    center_um = np.asarray(center_um, dtype=float)
    if center_um.shape != (2,) or not np.isfinite(center_um).all():
        raise ValueError("Beam center must be a finite (x,y) pair in lens coordinates.")
    return dict(
        ion_index=scan.ion_index, wavelength_nm=scan.wavelength_nm,
        device_name=scan.device_name, flip_vertical=scan.flip_vertical,
        requested_height_um=float(height_um), selected_height_um=float(row.height_um),
        frame_index=int(row.image_index), exposure_ms=float(row.exposure_ms),
        stage_position_mm=float(row.stage_position_mm), focus_stage_mm=scan.focus_stage_mm,
        stage_to_height_ratio=scan.stage_to_height_ratio, source_path=str(row.path),
        center_um=center_um.tolist(), background_counts=float(background_counts),
        peak_roi_um=None if peak_roi_um is None else list(peak_roi_um),
        alignment=alignment, image=image, x_um=x-center_um[0], y_um=y-center_um[1],
    )


def plot_publication_grid(planes, *, half_width_um=30.0, font_size=13, show_heights=True,
                          x_shift_um=0.0, y_shift_um=0.0):
    """Six slots, wavelength colormaps, independent displayed-plane normalization.

    Displayed intensity is divided by its maximum within the displayed crop.
    Return figure, metadata table, and the exact cropped arrays for export.
    Empty slots remain visible; no absent measurements are synthesized.
    Global shifts add to displayed coordinates: positive x moves beams right,
    positive y moves them up. Shift before cropping; never resample pixels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if not np.isfinite(half_width_um) or half_width_um <= 0:
        raise ValueError("half_width_um must be finite and positive.")
    if not np.isfinite([x_shift_um, y_shift_um]).all():
        raise ValueError("Global x and y shifts must be finite values in microns.")
    crops, records = {}, []
    with plt.rc_context({"font.family": "Arial", "font.size": font_size,
                         "svg.fonttype": "none", "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white",
                         "savefig.transparent": False, "text.color": "black",
                         "axes.labelcolor": "black", "axes.edgecolor": "black",
                         "xtick.color": "black", "ytick.color": "black"}):
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2), constrained_layout=True,
                                 facecolor="white")
        for ax, wl in zip(axes.flat, WAVELENGTHS_NM):
            plane = planes.get(wl)
            if plane is None:
                ax.set_title(f"{wl} nm")
                ax.text(.5, .5, "Not measured yet", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            x = plane["x_um"] + x_shift_um
            y = plane["y_um"] + y_shift_um
            dx, dy = x[1]-x[0], y[1]-y[0]
            if min(-x.min(), x.max(), -y.min(), y.max()) < half_width_um:
                raise ValueError(f"{wl} nm: the ±{half_width_um:g} um crop extends outside the image. Reduce HALF_WIDTH_UM or adjust its center.")
            cols, rows = np.abs(x) <= half_width_um, np.abs(y) <= half_width_um
            image = plane["image"][np.ix_(rows, cols)]
            if not image.size or image.max() <= 0:
                raise ValueError(f"{wl} nm: no positive signal in the displayed crop.")
            peak = float(image.max())
            normalized = image / peak
            cx, cy = x[cols], y[rows]
            extent = [cx[0]-dx/2, cx[-1]+dx/2, cy[-1]+dy/2, cy[0]-dy/2]
            cmap = LinearSegmentedColormap.from_list(f"wl_{wl}", ["black", WAVELENGTH_COLORS[wl]])
            im = ax.imshow(normalized, extent=extent, origin="upper", cmap=cmap,
                           vmin=0, vmax=1, interpolation="nearest", aspect="equal")
            ax.set(xlim=(-half_width_um, half_width_um), ylim=(-half_width_um, half_width_um),
                   xlabel="x (µm)", ylabel="y (µm)")
            ticks = np.linspace(-half_width_um, half_width_um, 7)
            ax.set(xticks=ticks, yticks=ticks)
            title = f"{wl} nm"
            if show_heights:
                title += f" · h = {plane['selected_height_um']:.2f} µm"
            ax.set_title(title)
            fig.colorbar(im, ax=ax, ticks=[0, .5, 1], fraction=.046, pad=.03)
            crops[wl] = dict(intensity=image, normalized=normalized, x_um=cx, y_um=cy)
            record = {k: v for k, v in plane.items() if k not in ("image", "x_um", "y_um")}
            record["normalization_peak_counts"] = peak
            record["half_width_um"] = half_width_um
            record["global_x_shift_um"] = float(x_shift_um)
            record["global_y_shift_um"] = float(y_shift_um)
            records.append(record)
    return fig, pd.DataFrame(records), crops


def save_publication_figure(fig, path, *, dpi=600):
    """Save editable Arial text, embedded images, and vector colorbar fills."""
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    # Colormap updates recreate these meshes, so disable rasterization at save
    # time as well. Vector fills avoid missing colorbar images in SVG editors.
    for ax in fig.axes:
        for image in ax.images:
            colorbar = image.colorbar
            if colorbar is not None and colorbar.solids is not None:
                colorbar.solids.set_rasterized(False)
                colorbar.solids.set_edgecolor("face")
    for text in fig.findobj(match=Text):
        text.set_fontfamily("Arial")
    with plt.rc_context({"svg.fonttype": "none", "svg.image_inline": True,
                         "pdf.fonttype": 42, "font.family": "Arial"}):
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", transparent=False)


def export_publication(fig, table, crops, directory, stem):
    """Export PNG/PDF/SVG plus exact plotted arrays and acquisition metadata."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "pdf", "svg"):
        path = directory / f"{stem}.{extension}"
        save_publication_figure(fig, path)
        paths.append(path)
    arrays = {f"wl_{wl}_{key}": value for wl, crop in crops.items() for key, value in crop.items()}
    array_path = directory / f"{stem}.npz"
    np.savez_compressed(array_path, **arrays)
    metadata_path = directory / f"{stem}.json"
    metadata_path.write_text(json.dumps(table.to_dict(orient="records"), indent=2) + "\n")
    return paths + [array_path, metadata_path]


def fit_beam_sizes(planes, *, half_width_um=30.0):
    """Fit peak-crossing axial/radial profiles of the selected aligned planes.

    Return a per-wavelength table (both 1/e² radii and FWHM) and exact fitted
    profiles. Failed fits remain explicit table rows with NaN widths.
    """
    from .Experimental_Analysis import Experimental_Analysis

    if not np.isfinite(half_width_um) or half_width_um <= 0:
        raise ValueError("Fit half-width must be positive and finite.")
    records, fits = [], {}
    for wl in WAVELENGTHS_NM:
        plane = planes.get(wl)
        if plane is None:
            continue
        record = {key: plane[key] for key in (
            "wavelength_nm", "device_name", "ion_index", "selected_height_um",
            "requested_height_um", "frame_index", "exposure_ms", "flip_vertical", "source_path",
        )}
        record.update(fit_half_width_um=float(half_width_um),
                      axial_radius_um=np.nan, radial_radius_um=np.nan,
                      axial_fwhm_um=np.nan, radial_fwhm_um=np.nan)
        try:
            analysis = Experimental_Analysis.from_image(
                plane["image"], plane["x_um"], plane["y_um"],
                height_um=plane["selected_height_um"], exposure_ms=plane["exposure_ms"],
            )
            fitted = analysis.fit_beam_profiles(half_width_um=half_width_um)
            record.update(peak_x_um=float(analysis.x0), peak_y_um=float(analysis.y0),
                          peak_counts=float(analysis.height_slice[analysis.iy, analysis.ix]))
            for direction, fit in fitted.items():
                for key in ("radius_um", "fwhm_um", "r_squared", "background_counts", "half_max_covered"):
                    record[f"{direction}_{key}"] = fit[key]
            covered = all(f["half_max_covered"] for f in fitted.values())
            record.update(status="ok" if covered else "check fit window", fit_error="")
            fits[wl] = fitted
        except (ValueError, RuntimeError, FloatingPointError) as error:
            record.update(status="fit failed", fit_error=str(error))
        records.append(record)
    return pd.DataFrame(records), fits


def plot_beam_sizes(table, *, quantity="radius", font_size=18, annotate=True):
    """Paper-style wavelength scatter: radial circles and axial diamonds.

    quantity='radius' plots 1/e² intensity radii; 'fwhm' matches the original
    paper figure's width convention. Colors match the six image tiles.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if quantity not in ("radius", "fwhm"):
        raise ValueError("quantity must be 'radius' or 'fwhm'.")
    with plt.rc_context({"font.family": "Arial", "font.size": font_size,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "text.color": "black", "axes.labelcolor": "black",
                         "xtick.color": "black", "ytick.color": "black"}):
        fig, ax = plt.subplots(figsize=(5.8, 4.4), dpi=150, constrained_layout=True)
        widths = []
        for _, row in table.iterrows():
            wl = int(row.wavelength_nm)
            for direction, marker in (("radial", "o"), ("axial", "D")):
                value = row[f"{direction}_{quantity}_um"]
                if not np.isfinite(value):
                    continue
                widths.append(value)
                ax.scatter(wl, value, s=70, marker=marker,
                           color=WAVELENGTH_COLORS[wl], edgecolor="none", zorder=3)
                if annotate:
                    # Stagger neighboring labels; keep narrow-width labels
                    # close enough to their points to stay above y=0.
                    below = wl in (405, 515)
                    small = value <= 1.0
                    offset_y = (-3 if small else -9) if below else 8
                    ax.annotate(f"{value:.2f}", (wl, value), xytext=(0 if small else 3, offset_y),
                                textcoords="offset points", fontsize=font_size-2,
                                ha="center" if small else "left", va="top" if below else "bottom")
        label = "Beam radius (µm)" if quantity == "radius" else "Beam FWHM (µm)"
        ax.set(xlim=(350, 950), ylim=(0, max(10, np.ceil(1.2*max(widths, default=0)))),
               xlabel="Wavelength (nm)", ylabel=label, xticks=[400, 500, 600, 700, 800, 900])
        ax.xaxis.label.set_fontsize(font_size)
        ax.yaxis.label.set_fontsize(font_size)
        ax.grid(True, alpha=.25)
        ax.legend(handles=[Line2D([], [], color="black", marker=marker, linestyle="none",
                                 label=label, markersize=7, markerfacecolor="none",
                                 markeredgecolor="black", markeredgewidth=1.2)
                           for marker, label in (("o", "Radial (y)"), ("D", "Axial (x)"))],
                  frameon=False, fontsize=11, loc="upper right")
    return fig


def plot_beam_profile_fits(fits):
    """Show measured slices and Gaussian fits, radial above axial, per wavelength."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 6, figsize=(18, 5.5), constrained_layout=True,
                             facecolor="white", squeeze=False)
    for column, wl in enumerate(WAVELENGTHS_NM):
        for row, direction in enumerate(("radial", "axial")):
            ax = axes[row, column]
            fit = fits.get(wl, {}).get(direction)
            if fit is None:
                ax.text(.5, .5, "No fit", transform=ax.transAxes, ha="center", va="center")
                ax.set_axis_off()
                continue
            x = fit["coordinates_um"]-fit["center_um"]
            scale = fit["amplitude_counts"]+fit["background_counts"]
            ax.plot(x, fit["counts"]/scale, ".", color=WAVELENGTH_COLORS[wl], ms=3, label="Measured")
            ax.plot(x, fit["fitted_counts"]/scale, "k--", lw=1, label="Gaussian")
            ax.set(title=f"{wl} nm · {direction}\nw = {fit['radius_um']:.2f} µm; R² = {fit['r_squared']:.3f}",
                   xlabel="Offset from fitted center (µm)", ylabel="Relative counts")
            ax.set_facecolor("white")
            ax.grid(alpha=.2)
            if column == 0:
                ax.legend(fontsize=8, frameon=False)
    return fig


def plot_peak_count_rates(tables, *, y_scale="linear", font_size=12):
    """Six wavelength slots showing per-frame peak camera counts/s vs height.

    Keep absolute exposure-normalized rates and independent vertical axes.
    Mark the largest measured rate in each curve and return its metadata.
    """
    import matplotlib.pyplot as plt

    if y_scale not in ("linear", "log"):
        raise ValueError("y_scale must be 'linear' or 'log'.")
    peaks = []
    with plt.rc_context({
        "font.family": "Arial", "font.size": font_size,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.transparent": False,
        "text.color": "black", "axes.labelcolor": "black",
        "axes.edgecolor": "black", "xtick.color": "black", "ytick.color": "black",
    }):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True, facecolor="white")
        for ax, wl in zip(axes.flat, WAVELENGTHS_NM):
            ax.set_title(f"{wl} nm")
            table = tables.get(wl)
            if table is None or table.empty:
                ax.text(.5, .5, "Not measured yet", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            ordered = table.sort_values("height_um")
            rates = ordered.peak_count_rate_per_s.to_numpy()
            if not np.isfinite(rates).all() or (rates < 0).any():
                raise ValueError(f"{wl} nm: expected finite, nonnegative count rates.")
            if y_scale == "log" and (rates <= 0).any():
                raise ValueError(f"{wl} nm: log scale requires positive count rates; use linear.")
            ax.plot(ordered.height_um, rates, ".-", color=WAVELENGTH_COLORS[wl], lw=1.2, ms=3)
            peak = ordered.iloc[int(np.argmax(rates))]
            ax.plot(peak.height_um, peak.peak_count_rate_per_s, "o", ms=7,
                    mfc="none", mec="black", label=f"Max at {peak.height_um:.2f} µm")
            ax.set(xlabel="Height above focus (µm)", ylabel="Peak count rate (counts/s)", yscale=y_scale)
            if y_scale == "linear":
                ax.set_ylim(bottom=0)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
            ax.grid(alpha=.25)
            ax.legend(fontsize=9, loc="best")
            peaks.append(peak.to_dict())
        fig.suptitle("Peak intensity / exposure time versus height")
    return fig, pd.DataFrame(peaks)
