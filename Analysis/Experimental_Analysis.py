"""Load saved height scans and extract intensity profiles."""

import numpy as np


class Experimental_Analysis:
    """Load a scan, then call select_height() to store its image and profiles.

    Results are available as .height_slice, .axial_slice, and .radial_slice.
    """

    def __init__(self, file_path):
        # Load the arrays before closing the archive.
        with np.load(file_path, allow_pickle=False) as scan:
            self.height_um = scan["height_um"]
            self.intensity = scan["intensity"]
            self.exposure_ms = scan["exposure_ms"]
            self.x_um = scan["x_um"]
            self.y_um = scan["y_um"]
            self.angle_deg = scan["angle_deg"].item()

        self.height_slice = None
        self.raw_height_slice = None
        self.axial_slice = None
        self.radial_slice = None
        self.background = None
        self.background_region = None
        self._background_selector = None
        self._background_close_id = None

    @classmethod
    def from_image(cls, image, x_um, y_um, *, height_um=0.0, exposure_ms=1.0):
        """Analyze an already aligned height plane without writing an NPZ file.

        Axes are physical pixel coordinates in µm, including any device flip.
        The same maximum and axial/radial slice methods as saved scans apply.
        """
        image = np.asarray(image)
        x, y = np.asarray(x_um, dtype=float), np.asarray(y_um, dtype=float)
        if (image.ndim != 2 or x.ndim != 1 or y.ndim != 1
                or image.shape != (len(y), len(x)) or min(image.shape) < 2
                or not np.isfinite(image).all() or not np.isfinite(x).all()
                or not np.isfinite(y).all()):
            raise ValueError("Image must be finite 2-D data matching finite x/y axes.")
        if any(not (np.all(np.diff(a) > 0) or np.all(np.diff(a) < 0)) for a in (x, y)):
            raise ValueError("Image axes must be strictly monotonic.")
        if not np.isfinite([height_um, exposure_ms]).all() or exposure_ms <= 0:
            raise ValueError("Height must be finite and exposure positive and finite.")
        analysis = cls.__new__(cls)
        analysis.height_um = np.array([height_um], dtype=float)
        analysis.intensity = image[None, ...]
        analysis.exposure_ms = np.array([exposure_ms], dtype=float)
        analysis.x_um, analysis.y_um = x.copy(), y.copy()
        analysis.angle_deg = 0.0  # Rotation is already included in the supplied plane.
        analysis._background_selector = None
        analysis._background_close_id = None
        return analysis.select_height(height_um, normalize=False)

    @staticmethod
    def fit_gaussian_profile(coordinates_um, counts, *, half_width_um=30.0):
        """Fit I(u) = B + A exp(-2 (u-u0)^2 / w^2) near a profile's maximum.

        w is the 1/e² intensity radius; FWHM = sqrt(2 ln 2) * w. Fit an
        independent nonnegative constant background without smoothing or
        resampling. Return measured samples, fitted values and fit diagnostics.
        """
        from scipy.optimize import curve_fit

        x, values = np.asarray(coordinates_um, dtype=float), np.asarray(counts, dtype=float)
        if x.ndim != 1 or values.shape != x.shape:
            raise ValueError("Profile coordinates and counts must be matching 1-D arrays.")
        if not np.isfinite(half_width_um) or half_width_um <= 0:
            raise ValueError("Fit half-width must be positive and finite.")
        valid = np.isfinite(x) & np.isfinite(values)
        x, values = x[valid], values[valid]
        if len(x) < 6 or not np.any(values > 0):
            raise ValueError("At least six finite samples and positive signal are required.")
        order = np.argsort(x)
        x, values = x[order], values[order]
        if np.any(np.diff(x) <= 0) or np.any(values < 0):
            raise ValueError("Profile coordinates must be unique and counts nonnegative.")
        peak_x = x[np.argmax(values)]
        window = np.abs(x-peak_x) <= half_width_um
        x, values = x[window], values[window]
        if len(x) < 6 or np.ptp(values) <= 1e-10 * max(1.0, float(values.max())):
            raise ValueError("Fit window needs at least six samples with nonconstant signal.")
        scale = float(values.max())
        normalized = values / scale
        baseline = float(np.percentile(normalized, 10))
        signal = np.maximum(normalized-baseline, 0)
        # Shift the fit coordinate near zero for numerical stability.
        u = x-peak_x
        spacing, span = float(np.median(np.diff(x))), float(np.ptp(x))
        radius0 = 2*np.sqrt(np.sum(signal*u*u)/np.sum(signal))
        radius0 = np.clip(radius0, spacing, span)

        def gaussian(u, amplitude, center, radius, background):
            return background + amplitude*np.exp(-2*((u-center)/radius)**2)

        parameters, _ = curve_fit(
            gaussian, u, normalized, p0=(1-baseline, 0, radius0, baseline),
            bounds=([0, u.min(), spacing/4, 0], [np.inf, u.max(), 2*span, np.inf]),
            maxfev=20000,
        )
        amplitude, center, radius, background = parameters
        fitted = gaussian(u, *parameters)*scale
        residual = np.sum((values-fitted)**2)
        total = np.sum((values-values.mean())**2)
        fwhm = float(np.sqrt(2*np.log(2))*radius)
        return dict(
            radius_um=float(radius), fwhm_um=fwhm,
            center_um=float(center+peak_x), amplitude_counts=float(amplitude*scale),
            background_counts=float(background*scale), r_squared=float(1-residual/total),
            half_max_covered=bool(x.min() <= center+peak_x-fwhm/2
                                  and x.max() >= center+peak_x+fwhm/2),
            coordinates_um=x, counts=values, fitted_counts=fitted,
        )

    def fit_beam_profiles(self, *, half_width_um=30.0):
        """Fit axial x and radial y single-pixel cuts through the image maximum."""
        if self.height_slice is None:
            raise RuntimeError("Call select_height() first.")
        x, axial = self.get_axial_slice()
        y, radial = self.get_radial_slice()
        return {
            "axial": self.fit_gaussian_profile(x, axial, half_width_um=half_width_um),
            "radial": self.fit_gaussian_profile(y, radial, half_width_um=half_width_um),
        }

    def select_height(self, height_um, normalize=True, y_cut_um=None, x_cut_um=None):
        """Store the nearest height image, its maximum, and both profiles.

        Heights and cut positions are in µm. By default, both profiles pass
        through the maximum; specify y_cut_um for the axial profile or x_cut_um
        for the radial profile. Selecting another height refreshes all results.

        Normalize by this image's maximum; an all-zero image stays zero.
        Keep the source stack unchanged and apply no exposure correction.
        For saved heights 1, 2, ..., a request for 90 µm selects index 89.
        """
        index = np.argmin(np.abs(self.height_um - height_um))
        self.selected_height_um = self.height_um[index]
        self.selected_exposure_ms = self.exposure_ms[index]
        image = self.intensity[index].copy()
        self.raw_height_slice = image.copy()
        self.background = None
        self.background_region = None
        self._normalize = normalize
        self._y_cut_um = y_cut_um
        self._x_cut_um = x_cut_um

        if normalize and image.max() > 0:
            image = image / image.max()

        self.height_slice = image
        self._update_profiles()
        return self

    def _update_profiles(self):
        """Refresh the peak and profiles after changing the selected image."""
        self.iy, self.ix = self.find_maximum(self.height_slice)
        self.x0 = self.x_um[self.ix]
        self.y0 = self.y_um[self.iy]
        self.get_axial_slice(self._y_cut_um)
        self.get_radial_slice(self._x_cut_um)

    def get_height_slice(self, height_um, normalize=True):
        """Select a height and return .height_slice, also refreshing profiles."""
        self.select_height(height_um, normalize=normalize)
        return self.height_slice

    def remove_background_interactive(self, clip=True, cmap="magma", block=True):
        """Draw a background rectangle, then close the window to subtract it.

        In a notebook, run `%matplotlib qt` first (or `%matplotlib widget` if
        ipympl is installed). Desktop windows run their GUI event loop until
        closed, so later notebook code cannot stall mouse interaction. Use
        block=False to return immediately; widget backends always return
        immediately. Drawing or resizing only selects a region.
        Closing applies the median subtraction once,
        then refreshes .height_slice and its profiles. Normalization follows
        select_height(); negative values are clipped unless clip=False.

        Closing without a rectangle, changing height, or replacing this window
        with another selector cancels the pending selection without subtracting.
        Subtraction always starts from .raw_height_slice, not a corrected image.
        .background_region stores (xmin, xmax, ymin, ymax) in aligned µm.
        Closing prints the subtracted median in stored intensity units, before
        normalization. Read this value afterward with .background. The method
        returns the figure; subtraction happens on close.
        """
        import matplotlib.pyplot as plt
        from matplotlib import rcsetup
        from matplotlib.widgets import RectangleSelector

        if self.raw_height_slice is None:
            raise RuntimeError("Call select_height() first.")
        backend = plt.get_backend().lower()
        widget_backend = any(s in backend for s in ("widget", "ipympl"))
        interactive = {name.lower() for name in rcsetup.interactive_bk}
        if backend not in interactive and not widget_backend:
            raise RuntimeError("Run %matplotlib qt (or %matplotlib widget) before selecting a region.")

        # Cancel a previous window without applying its pending selection.
        if self._background_selector is not None:
            previous = self._background_selector.ax.figure
            previous.canvas.mpl_disconnect(self._background_close_id)
            self._background_selector.disconnect_events()
            plt.close(previous)

        selected_raw = self.raw_height_slice
        selection = None
        finished = False
        x, y = self.x_um, self.y_um
        dx, dy = x[1] - x[0], y[1] - y[0]
        extent = [x[0] - dx/2, x[-1] + dx/2, y[-1] + dy/2, y[0] - dy/2]
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=False)
        fig.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.90)
        original = ax.imshow(selected_raw, origin="upper", extent=extent, cmap=cmap)
        ax.set(title="Drag over background; close window to apply",
               xlabel="x (µm)", ylabel="y (µm)")
        fig.colorbar(original, ax=ax, label="Raw intensity")

        def on_select(press, release):
            nonlocal selection
            # An old figure must not modify a newly selected height.
            if self.raw_height_slice is not selected_raw:
                selection = None
                selector.set_active(False)
                ax.set_title("Height changed: open a new selector")
                fig.canvas.draw_idle()
                return

            xmin, xmax = sorted((press.xdata, release.xdata))
            ymin, ymax = sorted((press.ydata, release.ydata))
            columns = np.flatnonzero((x >= xmin) & (x <= xmax))
            rows = np.flatnonzero((y >= ymin) & (y <= ymax))
            if rows.size and columns.size:
                selection = (rows, columns, (xmin, xmax, ymin, ymax))
                ax.set_title("Region selected; close window to subtract background")
            else:
                selection = None
                ax.set_title("No pixels selected: draw a larger rectangle")
            fig.canvas.draw_idle()

        def on_close(event):
            nonlocal finished
            if finished:
                return
            finished = True
            fig.canvas.stop_event_loop()
            selector.disconnect_events()
            self._background_selector = None
            self._background_close_id = None
            visible = any(artist.get_visible() for artist in selector.artists)
            if selection is None or not visible or self.raw_height_slice is not selected_raw:
                return

            # Apply the final rectangle only now, when the window closes.
            rows, columns, region = selection
            self.background = float(np.median(selected_raw[np.ix_(rows, columns)]))
            self.background_region = region
            image = selected_raw.astype(float) - self.background
            if clip:
                image = np.maximum(image, 0)
            if self._normalize and image.max() > 0:
                image = image / image.max()
            self.height_slice = image
            self._update_profiles()
            print(
                f"Subtracted background: {self.background:.6g} "
                f"(intensity units before normalization; "
                f"height = {self.selected_height_um:g} µm, "
                f"exposure = {self.selected_exposure_ms:.6g} ms)",
                flush=True,
            )

        # Keep a reference so the selector stays responsive after this returns.
        selector = RectangleSelector(
            ax, on_select, button=[1], interactive=True,
            useblit=fig.canvas.supports_blit,
            minspanx=5, minspany=5, spancoords="pixels",
            props={"facecolor": "none", "edgecolor": "cyan", "linewidth": 2},
        )
        self._background_selector = selector
        self._background_close_id = fig.canvas.mpl_connect("close_event", on_close)
        plt.show(block=False)
        fig.canvas.draw()
        if block and not widget_backend and not finished:
            # Process GUI events until this selector closes, even while a
            # notebook cell is running. Other open figures need not close.
            fig.canvas.start_event_loop(timeout=0)
        return fig

    @staticmethod
    def find_maximum(image):
        """Return (iy, ix), the row and column of the image maximum.

        Its physical location is (self.x_um[ix], self.y_um[iy]).
        If several pixels share the maximum, return the first in row order.
        """
        return np.unravel_index(np.argmax(image), image.shape)

    def get_axial_slice(self, y_cut_um=None):
        """Store .axial_slice and return (x_um, axial_slice) at a fixed y.

        Use the nearest y pixel, or the maximum's row when y_cut_um is None.
        Call select_height() first; .axial_y_um records the actual cut position.
        """
        if self.height_slice is None:
            raise RuntimeError("Call select_height() first.")
        self._y_cut_um = y_cut_um
        if y_cut_um is None:
            row = self.iy
        else:
            row = np.argmin(np.abs(self.y_um - y_cut_um))

        self.axial_y_um = self.y_um[row]
        self.axial_slice = self.height_slice[row, :]
        return self.x_um, self.axial_slice

    def get_radial_slice(self, x_cut_um=None):
        """Store .radial_slice and return (y_um, radial_slice) at a fixed x.

        Use the nearest x pixel, or the maximum's column when x_cut_um is None.
        Call select_height() first; .radial_x_um records the actual cut position.
        """
        if self.height_slice is None:
            raise RuntimeError("Call select_height() first.")
        self._x_cut_um = x_cut_um
        if x_cut_um is None:
            column = self.ix
        else:
            column = np.argmin(np.abs(self.x_um - x_cut_um))

        self.radial_x_um = self.x_um[column]
        self.radial_slice = self.height_slice[:, column]
        return self.y_um, self.radial_slice
