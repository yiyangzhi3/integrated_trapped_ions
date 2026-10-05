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
