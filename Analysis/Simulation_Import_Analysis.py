"""Read Lumerical MAT exports and analyze their XY power profiles."""

from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit


LUMERICAL_DATA_DIR = Path(
    "/Users/yiyangzhi/Library/CloudStorage/GoogleDrive-yiyang_zhi3@berkeley.edu/"
    "My Drive/Ming Wu Integrated Photonics Group/Simulations/Lumerical/"
    "Integrated_Trapped_Ions_Library/Gen2_pub"
)


def import_lumerical_mat(filename, data_dir=LUMERICAL_DATA_DIR, squeeze=True):
    """Load a flat numeric MAT export, preserving complex fields and units.

    Accept an absolute filename or a path relative to data_dir. Restore MATLAB
    dimension order for HDF5 exports. squeeze=True removes singleton dimensions
    so coordinate vectors become 1D. Older MAT files use scipy.io.loadmat.
    """
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = Path(data_dir).expanduser() / path
    if not path.is_file():
        raise FileNotFoundError(f"MAT file not found: {path}")

    if not h5py.is_hdf5(path):
        return {
            name: value
            for name, value in loadmat(path, squeeze_me=squeeze).items()
            if not name.startswith("__")
        }

    data = {}
    with h5py.File(path, "r") as mat:
        for name, dataset in mat.items():
            if name.startswith("#"):
                continue
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"{name!r} is not a flat numeric dataset in {path}")
            values = dataset[()]
            if values.dtype.names and {"real", "imag"} <= set(values.dtype.names):
                values = values["real"] + 1j * values["imag"]
            # MATLAB stores HDF5 dimensions in reverse order.
            values = values.T
            data[name] = values.squeeze() if squeeze else values
    return data


class Simulation_Import_Analysis:
    """Load one simulation and store its normalized image, peak, and profiles.

    .data retains every imported array in its original units; .P is the raw
    power map. .norm_P is normalized and flipped left-to-right. Coordinates
    follow the comparison convention: x_um = -x * 1e6, y_um = y * 1e6.
    Both .axial_slice and .radial_slice initially pass through the maximum.
    Call fit_gaussian_slices() to store their fitted 1/e² radii in micrometers.
    """

    def __init__(self, filename, data_dir=LUMERICAL_DATA_DIR):
        self.data = import_lumerical_mat(filename, data_dir=data_dir)
        self.P = self.data["P"]
        self.x_um = -self.data["x"] * 1e6
        self.y_um = self.data["y"] * 1e6

        # Power must be a real XY image, indexed as [y, x].
        if self.x_um.ndim != 1 or self.y_um.ndim != 1 or self.P.shape != (self.y_um.size, self.x_um.size):
            raise ValueError("Expected 1D x/y coordinates and P with shape (len(y), len(x)).")
        if np.iscomplexobj(self.P):
            raise ValueError("P must be real power; complex fields are preserved by import_lumerical_mat().")

        # Normalize and mirror the power exactly as in the comparison notebook.
        peak = self.P.max()
        self.norm_P = np.fliplr(self.P / peak if peak != 0 else self.P.copy())

        # Locate the maximum in the processed image.
        self.iy, self.ix = np.unravel_index(np.argmax(self.norm_P), self.norm_P.shape)
        self.x0 = self.x_um[self.ix]
        self.y0 = self.y_um[self.iy]
        self.get_axial_slice()
        self.get_radial_slice()

    def get_axial_slice(self, y_cut_um=None):
        """Store .axial_slice along x, at the nearest y or at the maximum."""
        if y_cut_um is None:
            row = self.iy
        else:
            row = np.argmin(np.abs(self.y_um - y_cut_um))
        self.axial_y_um = self.y_um[row]
        self.axial_slice = self.norm_P[row, :]
        self.axial_waist_um = self.axial_fit = self.axial_fit_params = None
        return self.x_um, self.axial_slice

    def get_radial_slice(self, x_cut_um=None):
        """Store .radial_slice along y, at the nearest x or at the maximum."""
        if x_cut_um is None:
            column = self.ix
        else:
            column = np.argmin(np.abs(self.x_um - x_cut_um))
        self.radial_x_um = self.x_um[column]
        self.radial_slice = self.norm_P[:, column]
        self.radial_waist_um = self.radial_fit = self.radial_fit_params = None
        return self.y_um, self.radial_slice

    @staticmethod
    def gaussian(position_um, amplitude, center_um, waist_um, background):
        """Intensity Gaussian; waist_um is the 1/e² radius above background."""
        return background + amplitude * np.exp(-2 * ((position_um - center_um) / waist_um) ** 2)

    @classmethod
    def _fit_gaussian(cls, position_um, intensity):
        """Fit one complete slice, keeping the original coordinate order."""
        position_um = np.asarray(position_um, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        if intensity.size < 5 or not np.all(np.isfinite(intensity)) or not np.all(np.isfinite(position_um)):
            raise ValueError("A Gaussian fit needs at least five finite samples.")
        amplitude = np.ptp(intensity)
        if amplitude == 0 or np.ptp(position_um) == 0:
            raise ValueError("A constant slice or coordinate axis has no measurable Gaussian waist.")

        # Estimate the initial radius from the width at half maximum.
        background = intensity.min()
        center = position_um[np.argmax(intensity)]
        half_max_positions = position_um[intensity >= background + amplitude / 2]
        waist = np.ptp(half_max_positions) / np.sqrt(2 * np.log(2))
        if waist == 0:
            waist = np.ptp(position_um) / 10

        parameters, _ = curve_fit(
            cls.gaussian, position_um, intensity,
            p0=(amplitude, center, waist, background),
            bounds=(
                [0, position_um.min(), np.finfo(float).eps, -np.inf],
                [np.inf, position_um.max(), np.inf, np.inf],
            ),
            maxfev=10000,
        )
        fit_params = dict(zip(
            ("amplitude", "center_um", "waist_um", "background"), parameters
        ))
        return fit_params, cls.gaussian(position_um, *parameters)

    def fit_gaussian_slices(self):
        """Fit the current axial and radial slices and return this object.

        Model: background + amplitude * exp(-2 * ((position - center) / w)**2).
        Store .axial_waist_um and .radial_waist_um (radii, not diameters),
        .axial_fit and .radial_fit (curves on the original axes), and
        .axial_fit_params and .radial_fit_params (parameter dictionaries).
        These are widths at this simulation plane; a propagation waist is
        only obtained if this plane is at the beam focus. Changing a slice
        clears its fit; call this method again to refit.
        """
        axial_params, axial_fit = self._fit_gaussian(self.x_um, self.axial_slice)
        radial_params, radial_fit = self._fit_gaussian(self.y_um, self.radial_slice)
        self.axial_fit_params, self.axial_fit = axial_params, axial_fit
        self.radial_fit_params, self.radial_fit = radial_params, radial_fit
        self.axial_waist_um = axial_params["waist_um"]
        self.radial_waist_um = radial_params["waist_um"]
        return self
