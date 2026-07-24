#!/usr/bin/env python3
"""Plot VIA I-V sweeps and extract resistance from their ohmic region.

The script accepts either one text file or a directory.  Directory inputs are
searched recursively and produce one plot per file plus a CSV summary.

Example
-------
python analyze_via_iv.py /path/to/probe.txt --output-dir iv_results
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FitResult:
    source: str
    voltage_column: str
    current_column: str
    points_total: int
    points_fit: int
    fit_min_v: float
    fit_max_v: float
    conductance_s: float
    conductance_se_s: float
    resistance_ohm: float
    resistance_se_ohm: float
    current_intercept_a: float
    r_squared: float
    rmse_a: float
    compliance_start_v: float
    compliance_current_a: float
    fit_selection: str


def _normalized(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def _choose_column(
    names: Sequence[str], requested: Optional[str], candidates: Sequence[str], kind: str
) -> str:
    """Resolve a column name by explicit name/index or common aliases."""
    if requested is not None:
        if requested.isdigit():
            index = int(requested)
            if 0 <= index < len(names):
                return names[index]
            raise ValueError(f"{kind} column index {index} is out of range")
        lookup = {_normalized(name): name for name in names}
        key = _normalized(requested)
        if key in lookup:
            return lookup[key]
        raise ValueError(f"{kind} column {requested!r} not found in {list(names)}")

    lookup = {_normalized(name): name for name in names}
    for candidate in candidates:
        if _normalized(candidate) in lookup:
            return lookup[_normalized(candidate)]
    raise ValueError(
        f"Could not identify the {kind} column in {list(names)}; "
        f"pass --{kind}-column NAME_OR_INDEX"
    )


def load_iv(
    path: Path,
    voltage_column: Optional[str] = None,
    current_column: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Load voltage and current arrays from a headered text file."""
    last_error: Optional[Exception] = None
    data = None
    for delimiter in (None, ",", ";"):
        try:
            trial = np.genfromtxt(
                path,
                names=True,
                delimiter=delimiter,
                comments="#",
                autostrip=True,
                encoding=None,
            )
            if trial.dtype.names and len(trial.dtype.names) >= 2:
                data = np.atleast_1d(trial)
                break
        except (OSError, TypeError, ValueError) as error:
            last_error = error
    if data is None or data.dtype.names is None:
        detail = f": {last_error}" if last_error else ""
        raise ValueError(f"Could not parse a header and at least two columns{detail}")

    names = list(data.dtype.names)
    v_name = _choose_column(
        names, voltage_column, ("VA", "Voltage", "VoltageA", "V"), "voltage"
    )
    i_name = _choose_column(
        names, current_column, ("IA", "Current", "CurrentA", "I"), "current"
    )
    voltage = np.asarray(data[v_name], dtype=float)
    current = np.asarray(data[i_name], dtype=float)
    finite = np.isfinite(voltage) & np.isfinite(current)
    voltage, current = voltage[finite], current[finite]
    if voltage.size < 3:
        raise ValueError("Fewer than three finite I-V points were found")
    return voltage, current, v_name, i_name


def detect_compliance_start(
    current: np.ndarray,
    flat_run: int = 3,
    flat_relative_tolerance: float = 1e-4,
    minimum_level_fraction: float = 0.5,
) -> Optional[int]:
    """Return the first point on a sustained, high-current flat plateau."""
    max_abs_current = float(np.max(np.abs(current)))
    if max_abs_current == 0.0 or current.size < flat_run + 2:
        return None
    tolerance = max(flat_relative_tolerance * max_abs_current, 10 * np.finfo(float).eps)
    flat_steps = np.abs(np.diff(current)) <= tolerance
    for start in range(0, len(flat_steps) - flat_run + 1):
        if np.all(flat_steps[start : start + flat_run]):
            plateau = current[start : start + flat_run + 1]
            if np.median(np.abs(plateau)) >= minimum_level_fraction * max_abs_current:
                return start
    return None


def _linear_fit(voltage: np.ndarray, current: np.ndarray):
    coefficients, covariance = np.polyfit(voltage, current, 1, cov=True)
    conductance, intercept = (float(value) for value in coefficients)
    prediction = conductance * voltage + intercept
    residual = current - prediction
    residual_sum = float(np.sum(residual**2))
    total_sum = float(np.sum((current - np.mean(current)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0
    rmse = math.sqrt(residual_sum / len(voltage))
    conductance_se = math.sqrt(float(covariance[0, 0]))
    return conductance, intercept, conductance_se, r_squared, rmse


def select_fit_indices(
    voltage: np.ndarray,
    current: np.ndarray,
    fit_min_v: Optional[float] = None,
    fit_max_v: Optional[float] = None,
    minimum_points: int = 6,
) -> Tuple[np.ndarray, Optional[int], str]:
    """Select a manual range or every point before current compliance."""
    if minimum_points < 3:
        raise ValueError("minimum_points must be at least 3")

    compliance_start = detect_compliance_start(current)
    if fit_min_v is not None or fit_max_v is not None:
        mask = np.ones(voltage.size, dtype=bool)
        if fit_min_v is not None:
            mask &= voltage >= fit_min_v
        if fit_max_v is not None:
            mask &= voltage <= fit_max_v
        indices = np.flatnonzero(mask)
        if indices.size < minimum_points:
            raise ValueError(
                f"Manual fit range contains {indices.size} points; "
                f"at least {minimum_points} are required"
            )
        return indices, compliance_start, "manual voltage range"

    candidate_stop = compliance_start if compliance_start is not None else voltage.size
    candidates = np.arange(candidate_stop)
    if candidates.size < minimum_points:
        candidates = np.arange(voltage.size)
    if candidates.size < minimum_points:
        raise ValueError(
            f"Only {candidates.size} usable points; at least {minimum_points} are required"
        )

    return candidates, compliance_start, "all pre-compliance data"


def fit_resistance(
    source: Path,
    voltage: np.ndarray,
    current: np.ndarray,
    voltage_column: str,
    current_column: str,
    fit_indices: np.ndarray,
    compliance_start: Optional[int],
    fit_selection: str,
) -> FitResult:
    """Fit I = G V + I0 and propagate slope error to R = 1/G."""
    fit_v, fit_i = voltage[fit_indices], current[fit_indices]
    conductance, intercept, conductance_se, r_squared, rmse = _linear_fit(fit_v, fit_i)
    if conductance == 0.0:
        raise ValueError("The fitted conductance is zero; resistance is undefined")
    resistance = 1.0 / conductance
    resistance_se = abs(conductance_se / conductance**2)
    if compliance_start is None:
        compliance_start_v = math.nan
        compliance_current_a = math.nan
    else:
        compliance_start_v = float(voltage[compliance_start])
        compliance_current_a = float(np.median(current[compliance_start:]))

    return FitResult(
        source=str(source.resolve()),
        voltage_column=voltage_column,
        current_column=current_column,
        points_total=int(voltage.size),
        points_fit=int(fit_indices.size),
        fit_min_v=float(np.min(fit_v)),
        fit_max_v=float(np.max(fit_v)),
        conductance_s=conductance,
        conductance_se_s=conductance_se,
        resistance_ohm=resistance,
        resistance_se_ohm=resistance_se,
        current_intercept_a=intercept,
        r_squared=r_squared,
        rmse_a=rmse,
        compliance_start_v=compliance_start_v,
        compliance_current_a=compliance_current_a,
        fit_selection=fit_selection,
    )


def _current_scale(current: np.ndarray) -> Tuple[float, str]:
    magnitude = float(np.nanmax(np.abs(current)))
    for scale, unit in ((1.0, "A"), (1e-3, "mA"), (1e-6, "µA"), (1e-9, "nA")):
        if magnitude >= scale:
            return scale, unit
    return 1e-12, "pA"


def _format_resistance(value: float, uncertainty: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1e6:
        scale, unit = 1e6, "MΩ"
    elif magnitude >= 1e3:
        scale, unit = 1e3, "kΩ"
    elif magnitude < 1.0:
        scale, unit = 1e-3, "mΩ"
    else:
        scale, unit = 1.0, "Ω"
    return f"{value / scale:.4g} ± {uncertainty / scale:.2g} {unit}"


def set_publication_style() -> None:
    """Set consistent journal-figure defaults."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 18,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "axes.linewidth": 1.3,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.major.width": 1.3,
            "ytick.major.width": 1.3,
            "legend.fontsize": 15,
            "lines.linewidth": 2.2,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_iv(
    source: Path,
    voltage: np.ndarray,
    current: np.ndarray,
    fit_indices: np.ndarray,
    result: FitResult,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create the figure in memory without displaying or exporting it."""
    set_publication_style()
    scale, current_unit = _current_scale(current)
    plotted_current = current / scale
    fit_v = voltage[fit_indices]
    fit_i = current[fit_indices]
    order = np.argsort(fit_v)
    line_v = np.linspace(float(np.min(fit_v)), float(np.max(fit_v)), 250)
    line_i = (result.conductance_s * line_v + result.current_intercept_a) / scale

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.5), constrained_layout=True)
    color_data, color_fit = "#1f4e79", "#b22222"

    axes[0].plot(
        voltage,
        plotted_current,
        color=color_data,
        marker="o",
        markersize=4.8,
        markeredgewidth=0,
        label="Measured",
    )
    axes[0].set_title(f"Measured IV Curve: {title or source.stem}")
    axes[0].set_xlabel("Voltage (V)")
    axes[0].set_ylabel(f"Current ({current_unit})")
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].plot(
        fit_v[order],
        fit_i[order] / scale,
        linestyle="none",
        marker="o",
        markersize=6.5,
        color=color_data,
        label="All pre-compliance data",
    )
    axes[1].plot(line_v, line_i, color=color_fit, label="Linear fit")
    axes[1].set_title("Pre-compliance Fit")
    axes[1].set_xlabel("Voltage (V)")
    axes[1].set_ylabel(f"Current ({current_unit})")
    axes[1].legend(frameon=False, loc="lower right")
    fit_text = (
        f"$R$ = {_format_resistance(result.resistance_ohm, result.resistance_se_ohm)}\n"
        f"$R^2$ = {result.r_squared:.5f}\n"
        f"Fit: {result.fit_min_v:g}–{result.fit_max_v:g} V"
    )
    axes[1].text(0.05, 0.95, fit_text, transform=axes[1].transAxes, va="top", fontsize=16)

    for label, axis in zip(("(a)", "(b)"), axes):
        axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontweight="bold")
        axis.tick_params(top=True, right=True)
        axis.margins(x=0.04, y=0.08)

    return figure, axes


def _input_files(path: Path, pattern: str) -> Sequence[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob(pattern) if item.is_file())
    raise FileNotFoundError(path)


def _write_summary(path: Path, results: Sequence[FitResult]) -> None:
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="I-V text file or directory")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("iv_analysis"), help="output directory"
    )
    parser.add_argument("--pattern", default="*.txt", help="recursive directory glob")
    parser.add_argument("--voltage-column", help="voltage column name or zero-based index")
    parser.add_argument("--current-column", help="current column name or zero-based index")
    parser.add_argument("--fit-min-v", type=float, help="manual lower fit limit in volts")
    parser.add_argument("--fit-max-v", type=float, help="manual upper fit limit in volts")
    parser.add_argument("--min-fit-points", type=int, default=6)
    parser.add_argument(
        "--formats", nargs="+", default=("png", "pdf", "svg"), choices=("png", "pdf", "svg")
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--title", help="optional figure title (single-file input only)")
    parser.add_argument("--show", action="store_true", help="open the plot window")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    files = _input_files(input_path, args.pattern)
    if not files:
        raise SystemExit(f"No files matching {args.pattern!r} under {input_path}")

    results = []
    for source in files:
        try:
            voltage, current, v_name, i_name = load_iv(
                source, args.voltage_column, args.current_column
            )
            fit_indices, compliance_start, selection = select_fit_indices(
                voltage,
                current,
                fit_min_v=args.fit_min_v,
                fit_max_v=args.fit_max_v,
                minimum_points=args.min_fit_points,
            )
            result = fit_resistance(
                source,
                voltage,
                current,
                v_name,
                i_name,
                fit_indices,
                compliance_start,
                selection,
            )
            if input_path.is_dir():
                relative = source.relative_to(input_path).with_suffix("")
                output_stem = output_dir / relative.parent / f"{relative.name}_iv"
            else:
                output_stem = output_dir / f"{source.stem}_iv"
            figure, _ = plot_iv(
                source,
                voltage,
                current,
                fit_indices,
                result,
                title=args.title,
            )
            output_stem.parent.mkdir(parents=True, exist_ok=True)
            for file_format in args.formats:
                figure.savefig(
                    output_stem.with_suffix(f".{file_format.lower()}"), dpi=args.dpi
                )
            if args.show and len(files) == 1:
                plt.show()
            plt.close(figure)
            results.append(result)
            print(
                f"{source}: R = {_format_resistance(result.resistance_ohm, result.resistance_se_ohm)}, "
                f"R^2 = {result.r_squared:.6f}, "
                f"fit = [{result.fit_min_v:g}, {result.fit_max_v:g}] V"
            )
        except Exception as error:  # Continue a batch when one file is malformed.
            print(f"WARNING: skipped {source}: {error}", file=sys.stderr)

    _write_summary(output_dir / "iv_fit_summary.csv", results)
    print(f"Analyzed {len(results)} of {len(files)} file(s); outputs: {output_dir}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
