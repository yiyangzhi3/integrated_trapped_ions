"""Interactive image alignment and import for Gate-zone profiling."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcsetup
from PIL import Image
from scipy import ndimage


ImageInput = Union[str, Path, np.ndarray]


@dataclass(frozen=True)
class WhiteLightAlignment:
    """White-light image and its calibrated coordinate system."""

    image: np.ndarray
    angle_deg: float
    x_um: np.ndarray
    y_trench_um: np.ndarray
    y_ion_um: np.ndarray
    trench_center_px: Tuple[float, float]
    rotation_roi_px: np.ndarray
    rotation_points_px: np.ndarray
    center_roi_px: np.ndarray
    trench_edges_px: np.ndarray


@dataclass(frozen=True)
class GateFrameSet:
    """Rotated grayscale frames on the white-light physical coordinates."""

    labels: Tuple[str, ...]
    raw: np.ndarray  # [row, column, frame]
    normalized: np.ndarray
    x_um: np.ndarray
    y_trench_um: np.ndarray
    y_ion_um: np.ndarray
    angle_deg: float
    normalization: str
    normalization_scales: np.ndarray


class GateImageProfiler:
    """Stateful white-light alignment and frame importer.

    Call ``align_white_light`` first. It asks the user for the angle and
    coordinate-origin points. ``import_frames`` then applies that same transform
    to every measurement frame.
    """

    def __init__(
        self,
        *,
        pixel_pitch_um: Sequence[float] = (5.3, 5.3),
        magnification: float = 50.0,
        ion_relative_to_wg_lens_um: float = 3.0,
        wg_lens_relative_to_trench_um: float = 43.372,
        interpolation_order: int = 0,
    ) -> None:
        pitch = np.asarray(pixel_pitch_um, dtype=float)
        if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
            raise ValueError("pixel_pitch_um must contain positive (x, y) values")
        if not np.isfinite(magnification) or magnification <= 0:
            raise ValueError("magnification must be positive and finite")
        if not np.isfinite(ion_relative_to_wg_lens_um) or not np.isfinite(
            wg_lens_relative_to_trench_um
        ):
            raise ValueError("ion/trench offsets must be finite")
        if isinstance(interpolation_order, (bool, np.bool_)) or not isinstance(
            interpolation_order, (int, np.integer)
        ):
            raise TypeError("interpolation_order must be an integer from 0 through 5")
        if not 0 <= int(interpolation_order) <= 5:
            raise ValueError("interpolation_order must be from 0 through 5")

        self.pixel_pitch_um = pitch
        self.magnification = float(magnification)
        self.ion_offset_um = float(ion_relative_to_wg_lens_um) + float(
            wg_lens_relative_to_trench_um
        )
        self.interpolation_order = int(interpolation_order)

        self.alignment: Optional[WhiteLightAlignment] = None
        self.frames: Optional[GateFrameSet] = None

    @property
    def sample_spacing_um(self) -> Tuple[float, float]:
        """Return calibrated (dx, dy) at the sample plane."""
        spacing = self.pixel_pitch_um / self.magnification
        return float(spacing[0]), float(spacing[1])

    def align_white_light(
        self, image: ImageInput, *, brightness_offset: float = 50.0
    ) -> WhiteLightAlignment:
        """Interactively rotate the white-light image and define its origin.

        The method performs four two-click selections:
        1. NW and SE corners of an ROI containing the trench edge.
        2. LEFT and RIGHT points on that edge to determine its angle.
        3. NW and SE corners of a post-rotation ROI.
        4. LEFT and RIGHT trench edges; their midpoint is the physical origin.
        """
        self._require_interactive_backend()
        source = self._load_image(image)
        display_image = self._add_brightness(source, brightness_offset)

        rotation_roi = self._pick_two_points(
            display_image, "1/4  Select angle ROI: NW corner, then SE corner"
        )
        rotation_points = self._pick_two_points(
            display_image,
            "2/4  Select trench angle: LEFT point, then RIGHT point",
            roi=rotation_roi,
        )
        angle_deg = self._angle_from_points(rotation_points)
        rotated = self._rotate_crop(display_image, angle_deg)

        center_roi = self._pick_two_points(
            rotated, "3/4  Select origin ROI: NW corner, then SE corner"
        )
        trench_edges = self._pick_two_points(
            rotated,
            "4/4  Select trench edges: LEFT edge, then RIGHT edge",
            roi=center_roi,
        )

        self.alignment = self._build_alignment(
            rotated,
            angle_deg,
            rotation_roi,
            rotation_points,
            center_roi,
            trench_edges,
        )
        self.frames = None  # A new alignment invalidates previously imported frames.
        return self.alignment

    def import_frames(
        self,
        frame_paths: Mapping[str, ImageInput],
        *,
        normalization: str = "global",
    ) -> GateFrameSet:
        """Import frames using the current alignment.

        ``global`` matches ``ITI_Optics_Profiling_Gate``: one maximum over the
        entire stack. ``per_frame`` is an explicit visualization alternative.
        """
        if self.alignment is None:
            raise RuntimeError("Call align_white_light() before import_frames()")
        if not isinstance(frame_paths, Mapping) or not frame_paths:
            raise ValueError("frame_paths must be a non-empty label-to-image mapping")
        if normalization not in {"global", "per_frame"}:
            raise ValueError("normalization must be 'global' or 'per_frame'")

        labels = tuple(str(label) for label in frame_paths)
        expected_shape = self.alignment.image.shape[:2]
        imported = []
        for label, source in frame_paths.items():
            image = self._load_image(source)
            if image.shape[:2] != expected_shape:
                raise ValueError(
                    f"Frame {label!r} has shape {image.shape[:2]}; expected {expected_shape}"
                )
            rotated = self._rotate_crop(image, self.alignment.angle_deg)
            imported.append(self._matlab_rgb2gray(rotated))

        raw = np.stack(imported, axis=-1)
        if normalization == "global":
            scales = np.array([float(np.max(raw))])
            if scales[0] <= 0:
                raise ValueError("Cannot normalize a stack with a non-positive maximum")
            normalized = raw / scales[0]
        else:
            scales = np.max(raw, axis=(0, 1)).astype(float)
            if np.any(scales <= 0):
                bad = [labels[index] for index in np.flatnonzero(scales <= 0)]
                raise ValueError(f"Cannot normalize non-positive frame(s): {bad}")
            normalized = raw / scales[None, None, :]

        self.frames = GateFrameSet(
            labels=labels,
            raw=raw,
            normalized=normalized,
            x_um=self.alignment.x_um.copy(),
            y_trench_um=self.alignment.y_trench_um.copy(),
            y_ion_um=self.alignment.y_ion_um.copy(),
            angle_deg=self.alignment.angle_deg,
            normalization=normalization,
            normalization_scales=scales,
        )
        return self.frames

    def _build_alignment(
        self,
        rotated_image: np.ndarray,
        angle_deg: float,
        rotation_roi: Sequence[Sequence[float]],
        rotation_points: Sequence[Sequence[float]],
        center_roi: Sequence[Sequence[float]],
        trench_edges: Sequence[Sequence[float]],
    ) -> WhiteLightAlignment:
        """Construct calibrated axes from already selected points."""
        height, width = rotated_image.shape[:2]
        selections = {
            "rotation ROI": self._validate_points(rotation_roi, width, height),
            "rotation points": self._validate_points(rotation_points, width, height),
            "center ROI": self._validate_points(center_roi, width, height),
            "trench edges": self._validate_points(trench_edges, width, height),
        }
        center_x, center_y = np.mean(selections["trench edges"], axis=0)
        dx_um, dy_um = self.sample_spacing_um
        x_um = (np.arange(width, dtype=float) - center_x) * dx_um
        y_trench_um = (np.arange(height, dtype=float) - center_y) * dy_um
        return WhiteLightAlignment(
            image=np.array(rotated_image, copy=True),
            angle_deg=float(angle_deg),
            x_um=x_um,
            y_trench_um=y_trench_um,
            y_ion_um=y_trench_um - self.ion_offset_um,
            trench_center_px=(float(center_x), float(center_y)),
            rotation_roi_px=selections["rotation ROI"].copy(),
            rotation_points_px=selections["rotation points"].copy(),
            center_roi_px=selections["center ROI"].copy(),
            trench_edges_px=selections["trench edges"].copy(),
        )

    @staticmethod
    def _load_image(image: ImageInput) -> np.ndarray:
        if isinstance(image, (str, Path)):
            path = Path(image).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            with Image.open(path) as opened:
                if opened.mode in {"1", "P"}:
                    opened = opened.convert("RGB")
                array = np.array(opened)
        elif isinstance(image, np.ndarray):
            array = np.array(image, copy=True)
        else:
            raise TypeError("image must be a path or NumPy array")

        if array.ndim not in (2, 3):
            raise ValueError(f"Expected a 2-D or 3-D image; received {array.shape}")
        if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
            raise ValueError("A color image must have 1, 3, or 4 channels")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("Image height and width must be nonzero")
        if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)):
            raise TypeError(f"Unsupported image dtype: {array.dtype}")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError("Floating-point image contains NaN or infinity")
        return array

    @staticmethod
    def _add_brightness(image: np.ndarray, offset: float) -> np.ndarray:
        if not np.isscalar(offset) or not np.isfinite(offset):
            raise ValueError("brightness_offset must be a finite scalar")
        result = np.array(image, copy=True)
        if float(offset) == 0:
            return result
        target = result[..., :3] if result.ndim == 3 and result.shape[2] == 4 else result
        bright = target.astype(np.float64) + float(offset)
        if np.issubdtype(result.dtype, np.integer):
            limits = np.iinfo(result.dtype)
            target[...] = np.rint(np.clip(bright, limits.min, limits.max)).astype(result.dtype)
        else:
            if not np.isfinite(bright).all():
                raise OverflowError("Brightness addition overflowed")
            target[...] = bright.astype(result.dtype)
        return result

    def _rotate_crop(self, image: np.ndarray, angle_deg: float) -> np.ndarray:
        rotated = ndimage.rotate(
            image.astype(np.float64),
            angle=float(angle_deg),
            axes=(1, 0),
            reshape=False,
            order=self.interpolation_order,
            mode="constant",
            cval=0.0,
            prefilter=self.interpolation_order > 1,
        )
        if np.issubdtype(image.dtype, np.integer):
            limits = np.iinfo(image.dtype)
            return np.rint(np.clip(rotated, limits.min, limits.max)).astype(image.dtype)
        return rotated.astype(image.dtype, copy=False)

    @staticmethod
    def _matlab_rgb2gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.astype(np.float64)
        if image.shape[2] == 1:
            return image[..., 0].astype(np.float64)
        weights = np.array([0.2989360213, 0.5870430745, 0.1140209043])
        gray = np.tensordot(image[..., :3].astype(np.float64), weights, axes=([-1], [0]))
        if np.issubdtype(image.dtype, np.integer):
            limits = np.iinfo(image.dtype)
            gray = np.rint(np.clip(gray, limits.min, limits.max))
        return gray.astype(np.float64, copy=False)

    @staticmethod
    def _angle_from_points(points: Sequence[Sequence[float]]) -> float:
        points = np.asarray(points, dtype=float)
        delta_x, delta_y = points[1] - points[0]
        if delta_x <= 0:
            raise ValueError("Angle points must be selected LEFT then RIGHT")
        return float(np.degrees(np.arctan2(delta_y, delta_x)))

    @staticmethod
    def _validate_points(points: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.shape != (2, 2) or not np.isfinite(points).all():
            raise ValueError("Each selection must contain exactly two finite (x, y) points")
        if (
            np.any(points[:, 0] < 0)
            or np.any(points[:, 0] > width - 1)
            or np.any(points[:, 1] < 0)
            or np.any(points[:, 1] > height - 1)
        ):
            raise ValueError("Selected points must lie inside the image")
        return points

    @classmethod
    def _pick_two_points(
        cls,
        image: np.ndarray,
        title: str,
        *,
        roi: Optional[Sequence[Sequence[float]]] = None,
    ) -> np.ndarray:
        height, width = image.shape[:2]
        fig, axis = plt.subplots(figsize=(12, 8))
        display = image[..., 0] if image.ndim == 3 and image.shape[2] == 1 else image
        axis.imshow(display, origin="upper")
        axis.set_title(title)
        axis.set_xlabel("x / column [px]")
        axis.set_ylabel("y / row [px]")

        if roi is not None:
            roi = cls._validate_points(roi, width, height)
            x_limits = sorted(roi[:, 0])
            y_limits = sorted(roi[:, 1], reverse=True)
            if np.isclose(x_limits[0], x_limits[1]) or np.isclose(y_limits[0], y_limits[1]):
                plt.close(fig)
                raise ValueError("ROI corners must define a nonzero area")
            axis.set_xlim(x_limits)
            axis.set_ylim(y_limits)

        plt.show(block=False)
        points = np.asarray(plt.ginput(2, timeout=-1, show_clicks=True), dtype=float)
        plt.close(fig)
        if points.shape != (2, 2):
            raise RuntimeError("Selection was cancelled before two points were chosen")
        return cls._validate_points(points, width, height)

    @staticmethod
    def _require_interactive_backend() -> None:
        backend = plt.get_backend()
        backend_lower = backend.lower()
        interactive_backends = {name.lower() for name in rcsetup.interactive_bk}
        is_interactive = backend_lower in interactive_backends or any(
            name in backend_lower for name in ("ipympl", "widget")
        )
        if not is_interactive:
            raise RuntimeError(
                f"Matplotlib backend {backend!r} is not interactive. "
                "Run `%matplotlib qt` before align_white_light()."
            )
