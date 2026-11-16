"""Import the formal-retake Ca scans without alignment or intensity correction.

Layout: Ca_3D/ion_{M1,P1}/{wavelength}_nm/{focused_*.tiff,
focused_*.json,height_scan/*.tiff,height_scan/exposure_log.csv}.
Images stay on disk until requested; all paths and metadata are retained.
"""

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


WAVELENGTHS_NM = (405, 450, 515, 635, 729, 880)
ION_INDICES = ("M1", "0", "P1")


def read_tiff(path):
    """Read original camera pixels, preserving channels, dtype, and orientation."""
    with Image.open(path) as image:
        return np.array(image)


@dataclass
class CaHeightScan:
    ion_index: str
    wavelength_nm: int
    directory: Path
    reference_path: Path
    reference_metadata: dict
    focus_stage_mm: float
    stage_to_height_ratio: float
    frames: pd.DataFrame
    device_name: str = ""
    flip_vertical: bool = False

    def read_reference(self):
        """Return the unrotated focused reference TIFF (the alignment candidate)."""
        return read_tiff(self.reference_path)

    def read_frame(self, frame_index=0):
        """Read by acquisition image_index, not by physical height or row offset."""
        matches = self.frames.loc[self.frames.image_index == frame_index, "path"]
        if len(matches) != 1:
            raise KeyError(f"No unique frame {frame_index} in {self.directory}")
        return read_tiff(matches.iloc[0])

    def load_intensity(self):
        """Load float32 [frame, row, column] raw counts in frames-table order.

        Match Experimental_Import's R+G+B convention (grayscale stays scalar).
        No rotation, background subtraction, exposure division, or normalization.
        Heights and exposures are in self.frames, in exactly the same order.
        This allocates about 5 MiB per 1280 x 1024 frame.
        """
        stack = None
        for index, path in enumerate(self.frames.path):
            image = read_tiff(path).astype(np.float32)
            if image.ndim == 3 and image.shape[2] in (3, 4):
                image = image[..., :3].sum(axis=2)
            if image.ndim != 2:
                raise ValueError(f"Expected grayscale or RGB(A) TIFF: {path}")
            if stack is None:
                stack = np.empty((len(self.frames), *image.shape), dtype=np.float32)
            if image.shape != stack.shape[1:]:
                raise ValueError(f"Scan image dimensions changed: {path}")
            stack[index] = image
        return stack


def load_height_scan(directory, *, wavelength_nm, ion_index="unlabeled",
                     stage_to_height_ratio=0.5, device_name="", flip_vertical=False):
    """Load one wavelength folder containing a focused reference and height_scan/.

    Accept the same validated TIFF/log format as discover_ca_scans, without
    requiring an ion subfolder. A source without an ion label stays unlabeled.
    """
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    if wavelength_nm not in WAVELENGTHS_NM:
        raise ValueError(f"Unsupported wavelength: {wavelength_nm}")
    if not np.isfinite(stage_to_height_ratio) or stage_to_height_ratio <= 0:
        raise ValueError("stage_to_height_ratio must be finite and positive")
    references = sorted(directory.glob("focused_*.tiff"))
    if len(references) != 1:
        raise ValueError(f"Expected one focused reference TIFF in {directory}; found {len(references)}")
    reference = references[0]
    metadata = json.loads(reference.with_suffix(".json").read_text())
    focus_stage_mm = float(metadata["focus_state"]["result"]["final_z_mm"])
    if not np.isfinite(focus_stage_mm):
        raise ValueError(f"Invalid focused stage position in {reference.with_suffix('.json')}")

    scan_dir = directory / "height_scan"
    frames = pd.read_csv(scan_dir / "exposure_log.csv")
    required = {"image_index", "stage_position_mm", "exposure_ms", "filename"}
    if not required.issubset(frames.columns) or frames.empty:
        raise ValueError(f"Empty or incomplete exposure log in {scan_dir}")
    for column in ("image_index", "stage_position_mm", "exposure_ms"):
        frames[column] = pd.to_numeric(frames[column], errors="raise")
        if not np.isfinite(frames[column]).all():
            raise ValueError(f"Nonfinite {column} in {scan_dir}")
    if (frames.image_index < 0).any() or (frames.image_index % 1 != 0).any():
        raise ValueError(f"Invalid image_index in {scan_dir}")
    frames["image_index"] = frames.image_index.astype(int)
    if frames.image_index.duplicated().any() or frames.filename.duplicated().any():
        raise ValueError(f"Duplicate frame indices or filenames in {scan_dir}")
    if (frames.exposure_ms <= 0).any():
        raise ValueError(f"Nonpositive exposure in {scan_dir}")
    files = {p.name: p for p in scan_dir.glob("*.tiff")}
    logged = set(frames.filename)
    if logged != set(files):
        raise ValueError(
            f"TIFF/log mismatch in {scan_dir}: "
            f"missing TIFFs={sorted(logged - set(files))}, "
            f"unlogged TIFFs={sorted(set(files) - logged)}"
        )
    frames = frames.sort_values("image_index").reset_index(drop=True)
    frames["path"] = frames.filename.map(files)
    frames["height_um"] = (
        (frames.stage_position_mm - focus_stage_mm) * 1000 * stage_to_height_ratio
    )
    return CaHeightScan(
        ion_index, wavelength_nm, directory, reference, metadata,
        focus_stage_mm, stage_to_height_ratio, frames, device_name, flip_vertical,
    )


def discover_ca_scans(data_root, ion_index, *, stage_to_height_ratio=0.5,
                      wavelengths=WAVELENGTHS_NM):
    """Import metadata for one ion, with None slots for unmeasured wavelengths.

    Zero is focus_state.result.final_z_mm in each reference JSON. Positive
    stage motion increases height. The 2:1 stage ratio means 2 um of stage
    travel gives 1 um of height: height_um = (stage_mm - focus_mm)*1000*0.5.
    Preserve readback precision instead of rounding to a nominal 1 um grid.
    Reject incomplete/ambiguous datasets rather than silently guessing files.
    """
    if ion_index not in ION_INDICES:
        raise ValueError(f"ion_index must be one of {ION_INDICES}")
    if not np.isfinite(stage_to_height_ratio) or stage_to_height_ratio <= 0:
        raise ValueError("stage_to_height_ratio must be finite and positive")
    ion_dir = Path(data_root).expanduser() / f"ion_{ion_index}"
    if not ion_dir.is_dir():
        raise FileNotFoundError(ion_dir)

    scans = dict.fromkeys(WAVELENGTHS_NM)
    for wavelength in wavelengths:
        if wavelength not in WAVELENGTHS_NM:
            raise ValueError(f"Unsupported wavelength: {wavelength}")
        directory = ion_dir / f"{wavelength}_nm"
        if not directory.exists():
            continue
        scans[wavelength] = load_height_scan(
            directory, wavelength_nm=wavelength, ion_index=ion_index,
            stage_to_height_ratio=stage_to_height_ratio,
        )
    return scans


def discover_broadband_scans(calcium_root, barium_root, ion_index, *,
                             barium_ion_index=None, stage_to_height_ratio=0.5,
                             wavelength_overrides=None):
    """Combine R2C2 Ca (405/729/880) with vertically flipped R3C6 Ba (450/515/635).

    Ion labels can be selected independently for the two devices. References,
    raw frames, and raw stacks retain their original camera orientation; the
    plotting loader applies flip_vertical after the manual lens rotation.
    wavelength_overrides maps selected wavelengths to already loaded scans
    from another device; the original source is not read for those slots.
    """
    overrides = {} if wavelength_overrides is None else dict(wavelength_overrides)
    for wl, scan in overrides.items():
        if wl not in WAVELENGTHS_NM or scan is None or scan.wavelength_nm != wl:
            raise ValueError("Each wavelength override must match a supported scan wavelength")
    scans = dict.fromkeys(WAVELENGTHS_NM)
    for root, ion, wavelengths, device, flip in (
        (calcium_root, ion_index, (405, 729, 880), "R2C2", False),
        (barium_root, barium_ion_index if barium_ion_index is not None else ion_index,
         (450, 515, 635), "R3C6", True),
    ):
        wavelengths = tuple(wl for wl in wavelengths if wl not in overrides)
        if not wavelengths:
            continue
        discovered = discover_ca_scans(
            root, ion, stage_to_height_ratio=stage_to_height_ratio, wavelengths=wavelengths,
        )
        for wl in wavelengths:
            if discovered[wl] is not None:
                scans[wl] = replace(discovered[wl], device_name=device, flip_vertical=flip)
    scans.update(overrides)
    return scans


def summarize_ca_scans(scans):
    """One inventory row per wavelength, including empty measurement slots."""
    rows = []
    for wavelength, scan in scans.items():
        row = {"wavelength_nm": wavelength, "status": "not measured", "frames": 0}
        if scan is not None:
            frames = scan.frames
            row.update(
                status="available", frames=len(frames),
                height_min_um=frames.height_um.min(), height_max_um=frames.height_um.max(),
                median_height_step_um=frames.height_um.diff().median(),
                exposure_min_ms=frames.exposure_ms.min(), exposure_max_ms=frames.exposure_ms.max(),
                focus_stage_mm=scan.focus_stage_mm, reference=scan.reference_path.name,
                ion_index=scan.ion_index, device_name=scan.device_name,
                flip_vertical=scan.flip_vertical, directory=str(scan.directory),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def peak_count_rate_vs_height(scan):
    """Return each raw frame's maximum RGB-sum count divided by exposure seconds.

    Read one TIFF at a time. Evaluate the original full frame so rotations and
    plotting crops cannot remove the maximum. For grayscale use its single
    channel. No background subtraction, smoothing, or 0–1 normalization.
    Preserve the recorded frame order, calibrated heights, and exposure times.
    These are camera counts per second at the brightest pixel, not summed
    counts over the whole image or a calibrated photon rate.
    """
    records = []
    for row in scan.frames.itertuples(index=False):
        raw = read_tiff(row.path)
        if raw.ndim == 3 and raw.shape[2] in (3, 4):
            image = raw[..., :3].sum(axis=2, dtype=np.float32)
        elif raw.ndim == 2:
            image = raw.astype(np.float32)
        else:
            raise ValueError(f"Expected grayscale or RGB(A) TIFF: {row.path}")
        exposure_ms = float(row.exposure_ms)
        if not np.isfinite(exposure_ms) or exposure_ms <= 0:
            raise ValueError(f"Exposure must be positive and finite: {row.path}")
        iy, ix = np.unravel_index(np.argmax(image), image.shape)
        peak = float(image[iy, ix])
        records.append(dict(
            ion_index=scan.ion_index, wavelength_nm=scan.wavelength_nm,
            device_name=scan.device_name, flip_vertical=scan.flip_vertical,
            image_index=int(row.image_index), height_um=float(row.height_um),
            stage_position_mm=float(row.stage_position_mm),
            focus_stage_mm=scan.focus_stage_mm, stage_to_height_ratio=scan.stage_to_height_ratio,
            exposure_ms=exposure_ms, peak_counts=peak,
            peak_count_rate_per_s=peak * 1000.0 / exposure_ms,
            peak_column_px=int(ix), peak_row_px=int(iy), source_path=str(row.path),
        ))
    return pd.DataFrame(records)
