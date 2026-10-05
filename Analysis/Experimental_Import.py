"""Import the experimental images and perform processing for analysis."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


DARK_FRAME_FOLDER = Path(
    "/Users/yiyangzhi/Library/CloudStorage/GoogleDrive-yiyang_zhi3@berkeley.edu/"
    "My Drive/Ming Wu Integrated Photonics Group/Experiments/Measurements/"
    "Dark_Frames/dark_20261002_113902_15MHz"
)


class Experimental_Import:

    def __init__(self, pixel_size_um=5.3, magnification=50):
        self.spacing = pixel_size_um / magnification  # µm per pixel, in x and y.

    def align_cache_zone(self, file_path):
        """Return (x, y, image, angle, x_old, y_old, image_old) for a TIFF.

        x and y are in µm, centered on the leftmost trench edge midpoint.
        x increases rightward; y increases upward. The angle is in degrees,
        positive counterclockwise. Rotation keeps the original image size.
        x_old and y_old start at zero and increase along columns and rows;
        image_old is the original TIFF before rotation.
        """
        # 1. Read the TIFF. Array dimensions are rows (y), then columns (x).
        with Image.open(file_path) as tiff:
            image = np.array(tiff)
        Ny, Nx = image.shape[:2]

        # 2. Build physical axes starting at zero: 0, dx, 2*dx, ...
        x = np.arange(Nx) * self.spacing
        y = np.arange(Ny) * self.spacing
        x_old = x.copy()
        y_old = y.copy()
        image_old = image.copy()

        # 3. Find the leftmost trench edge and its correction angle.
        top, bottom = _find_left_edge(image)
        delta_x, delta_y = bottom - top
        angle = -np.degrees(np.arctan2(delta_x, delta_y))

        # 4. Rotate only the image, with MATLAB-style nearest-neighbor cropping.
        image = ndimage.rotate(image, angle, axes=(1, 0), reshape=False, order=0)

        # Exclude black corners introduced by rotation when finding the edge again.
        valid = ndimage.rotate(np.ones((Ny, Nx)), angle, reshape=False, order=0)
        valid = ndimage.binary_erosion(valid, iterations=6)

        # 5. Find the midpoint in the rotated image and shift the fixed axes.
        top, bottom = _find_left_edge(image, valid)
        x0, y0 = (top + bottom) / 2
        x = x - x0 * self.spacing
        y = -(y - y0 * self.spacing)

        return x, y, image, angle, x_old, y_old, image_old

    def import_height_scan(self, folder_path, angle, height_spacing_um=1.0):
        """Return (height_um, intensity, exposure_ms) in sorted filename order.

        Reads .tiff files and exposure_log.csv from the folder. The log must
        contain filename and exposure_ms columns. Zero-padded filenames sort
        in acquisition order, e.g. _000.tiff, _001.tiff, ...

        Heights start at height_spacing_um: 1, 2, 3, ... µm by default.
        intensity[frame, row, column] contains R + G + B after rotation with
        the white-light angle. Values are raw channel sums, not divided by
        exposure. Grayscale images retain their single-channel values.
        """
        folder = Path(folder_path)
        files = sorted(folder.glob("*.tiff"))
        if not files:
            raise FileNotFoundError(f"No .tiff files found in {folder}")

        # Match each image to its exposure time using the filename in the log.
        with (folder / "exposure_log.csv").open(newline="") as log:
            exposures = {row["filename"]: float(row["exposure_ms"])
                         for row in csv.DictReader(log)}

        height_um = np.arange(1, len(files) + 1) * height_spacing_um
        exposure_ms = np.array([exposures[file.name] for file in files])

        for i, file in enumerate(files):
            with Image.open(file) as tiff:
                image = np.array(tiff)

            # Apply the same rotation and cropping as the white-light image.
            image = ndimage.rotate(image, angle, axes=(1, 0), reshape=False, order=0)

            # Float values prevent overflow when adding the RGB channels.
            image = image.astype(np.float32)
            if image.ndim == 3:
                image = image[..., :3].sum(axis=2)

            # Allocate the stack once, then fill one height plane at a time.
            if i == 0:
                intensity = np.empty((len(files), *image.shape), dtype=np.float32)
            intensity[i] = image

        return height_um, intensity, exposure_ms

    def subtract_background(self, intensity, exposure_ms, folder_path=DARK_FRAME_FOLDER):
        """Subtract nearest-exposure dark counts from intensity in place.

        Call once on the raw floating-point stack from import_height_scan,
        before normalizing or dividing by exposure. Read dark_calibration.npz
        from folder_path; actual_exposures_ms and dark_counts contain the
        calibration exposures and mean R+G+B counts per pixel, respectively.

        Subtract one scalar background from each frame using its nearest
        calibration exposure (no interpolation). Out-of-range exposures use
        the nearest endpoint. Clip negative results to zero and return the same
        intensity array. Calling again subtracts the background again.
        """
        with np.load(Path(folder_path).expanduser() / "dark_calibration.npz",
                     allow_pickle=False) as dark:
            dark_exposure_ms = dark["actual_exposures_ms"]
            dark_counts = dark["dark_counts"]

        exposure_ms = np.asarray(exposure_ms).reshape(-1)
        if len(exposure_ms) != len(intensity):
            raise ValueError("Provide one exposure time for each intensity frame.")
        if not np.issubdtype(intensity.dtype, np.floating):
            raise TypeError("intensity must be a floating-point array, as returned by import_height_scan.")

        for i, exposure in enumerate(exposure_ms):
            nearest = np.argmin(np.abs(dark_exposure_ms - exposure))
            intensity[i] -= dark_counts[nearest]
            np.maximum(intensity[i], 0, out=intensity[i])

        return intensity


def _find_left_edge(image, valid=None):
    """Find endpoints of the leftmost long dark boundary, in (column, row) pixels.

    This detector assumes a dark, approximately vertical trench in a white-light
    image. Kept separate so the alignment method reads as five simple steps.
    """
    # Convert RGB to grayscale and scale the contrast.
    gray = image.astype(float)
    if gray.ndim == 3:
        gray = gray[..., :3] @ np.array([0.299, 0.587, 0.114])
    low, high = np.percentile(gray, [1, 99])
    gray = ndimage.gaussian_filter((gray - low) / (high - low), 1.5)

    # A bright-to-dark transition from left to right marks the outside boundary.
    response = np.maximum(-np.gradient(gray, axis=1), 0)
    if valid is not None:
        response = np.where(valid, response, 0)

    # Retain long vertical features and join short gaps in them.
    length = max(9, round(gray.shape[0] * 0.08) | 1)
    vertical = ndimage.grey_opening(response, size=(length, 1))
    mask = vertical > max(0.005, 0.25 * vertical.max())
    mask = ndimage.grey_closing(mask, size=(max(3, length // 2) | 1, 1))
    labels, _ = ndimage.label(mask)

    # Select the leftmost component that is long and narrow.
    candidates = []
    for label, (rows, cols) in enumerate(ndimage.find_objects(labels), start=1):
        height = rows.stop - rows.start
        width = cols.stop - cols.start
        if height >= 2 * length and height >= 3 * width:
            candidates.append((cols.start, label, rows, cols))
    if not candidates:
        raise ValueError("No long left trench edge found in this image.")
    _, label, rows, cols = min(candidates, key=lambda item: item[0])

    # Locate the strongest boundary in each row of the selected component.
    row = np.arange(rows.start, rows.stop)
    strength = np.where(labels[rows, cols] == label, response[rows, cols], -np.inf)
    col = np.argmax(strength, axis=1) + cols.start

    # Refine the boundary position between pixels using its three-point peak.
    left = response[row, np.maximum(col - 1, 0)]
    center = response[row, col]
    right = response[row, np.minimum(col + 1, gray.shape[1] - 1)]
    curvature = left - 2 * center + right
    offset = np.zeros_like(center)
    np.divide(0.5 * (left - right), curvature, out=offset, where=curvature < -1e-12)
    col = col + np.clip(offset, -0.5, 0.5)

    # Fit a straight line, excluding rounded corners and isolated outliers.
    margin = max(2, int(0.1 * len(row)))
    keep = np.zeros(len(row), dtype=bool)
    keep[margin:-margin] = True
    for _ in range(3):
        line = np.polyfit(row[keep], col[keep], 1)
        residual = col - np.polyval(line, row)
        median = np.median(residual[keep])
        deviation = np.median(np.abs(residual[keep] - median))
        keep &= np.abs(residual - median) <= max(1, 3 * 1.4826 * deviation)
    line = np.polyfit(row[keep], col[keep], 1)
    end_rows = np.array([row[0], row[-1]])
    return np.column_stack((np.polyval(line, end_rows), end_rows))
