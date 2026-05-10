"""Buffon's needle Monte Carlo estimator of pi with a Tk GUI.

The experiment drops needles of length L onto a floor ruled by parallel
stripes separated by distance d. For an isotropic needle the expected
number of stripe crossings is E = 2*L / (pi * d), so

    pi_hat = 2 * L * N / (d * C)

where N is the number of tosses and C is the total crossing count.
This holds for any L and d (including L > d, where one needle can cross
more than one stripe), which is why we count *total* crossings rather
than just "did it cross at least one stripe".
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class BuffonApp:
    N_STRIPES = 8
    MAX_VISIBLE_NEEDLES = 600

    def __init__(self, root):
        self.root = root
        root.title("Buffon's Needle — Monte Carlo estimate of pi")

        self.needle_length = tk.DoubleVar(value=1.0)
        self.stripe_spacing = tk.DoubleVar(value=1.5)
        self.batch_size = tk.IntVar(value=500)
        self.target_tosses = tk.IntVar(value=50000)

        self.total_tosses = 0
        self.total_crossings = 0
        self.history_n = []
        self.history_pi = []
        self.needle_segments = []
        self.needle_colors = []
        self.running = False

        self._build_ui()

        for var in (self.needle_length, self.stripe_spacing):
            var.trace_add("write", lambda *_: self._on_geometry_changed())

        self._reset()

    def _build_ui(self):
        controls = ttk.Frame(self.root, padding=8)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="Needle length L:").grid(row=0, column=0, padx=(0, 4))
        ttk.Spinbox(
            controls, from_=0.05, to=10.0, increment=0.05,
            textvariable=self.needle_length, width=7,
        ).grid(row=0, column=1, padx=(0, 12))

        ttk.Label(controls, text="Stripe spacing d:").grid(row=0, column=2, padx=(0, 4))
        ttk.Spinbox(
            controls, from_=0.05, to=10.0, increment=0.05,
            textvariable=self.stripe_spacing, width=7,
        ).grid(row=0, column=3, padx=(0, 12))

        ttk.Label(controls, text="Batch size:").grid(row=0, column=4, padx=(0, 4))
        ttk.Spinbox(
            controls, from_=1, to=100000, increment=100,
            textvariable=self.batch_size, width=8,
        ).grid(row=0, column=5, padx=(0, 12))

        ttk.Label(controls, text="Total tosses:").grid(row=0, column=6, padx=(0, 4))
        ttk.Spinbox(
            controls, from_=10, to=10_000_000, increment=1000,
            textvariable=self.target_tosses, width=10,
        ).grid(row=0, column=7, padx=(0, 12))

        self.run_btn = ttk.Button(controls, text="Run", command=self._toggle_run)
        self.run_btn.grid(row=0, column=8, padx=4)
        ttk.Button(controls, text="Reset", command=self._reset).grid(row=0, column=9, padx=4)

        self.status_var = tk.StringVar(value="")
        ttk.Label(
            self.root, textvariable=self.status_var, padding=(10, 2),
            font=("TkDefaultFont", 10),
        ).pack(side=tk.TOP, anchor="w")

        self.fig = Figure(figsize=(11, 5.2), dpi=100)
        self.ax_needles = self.fig.add_subplot(1, 2, 1)
        self.ax_conv = self.fig.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _on_geometry_changed(self):
        # Changing L or d invalidates accumulated statistics.
        if self.total_tosses > 0 or self.needle_segments:
            self._reset()

    def _reset(self):
        self.running = False
        self.run_btn.config(text="Run")
        self.total_tosses = 0
        self.total_crossings = 0
        self.history_n = []
        self.history_pi = []
        self.needle_segments = []
        self.needle_colors = []
        self._draw()
        self._update_status(float("nan"))

    def _toggle_run(self):
        if self.running:
            self.running = False
            self.run_btn.config(text="Run")
            return
        if self.total_tosses >= self._safe_int(self.target_tosses, 0):
            self._reset()
        self.running = True
        self.run_btn.config(text="Pause")
        self.root.after(10, self._step)

    def _safe_float(self, var, default):
        try:
            v = float(var.get())
            return v if v > 0 else default
        except (tk.TclError, ValueError):
            return default

    def _safe_int(self, var, default):
        try:
            return max(1, int(var.get()))
        except (tk.TclError, ValueError):
            return default

    def _step(self):
        if not self.running:
            return

        L = self._safe_float(self.needle_length, 1.0)
        d = self._safe_float(self.stripe_spacing, 1.5)
        batch = self._safe_int(self.batch_size, 500)
        target = self._safe_int(self.target_tosses, 50000)

        remaining = target - self.total_tosses
        if remaining <= 0:
            self.running = False
            self.run_btn.config(text="Run")
            return

        n = min(batch, remaining)

        # Sample needle midpoint y-coordinate and orientation uniformly.
        yc = np.random.uniform(0.0, d, n)
        theta = np.random.uniform(0.0, np.pi, n)
        half = 0.5 * L
        dy = half * np.sin(theta)
        dx = half * np.cos(theta)
        y1 = yc - dy
        y2 = yc + dy

        # Number of stripe lines (integer multiples of d) the needle crosses.
        ymin = np.minimum(y1, y2)
        ymax = np.maximum(y1, y2)
        crossings_per = (
            np.floor(ymax / d).astype(np.int64) - np.floor(ymin / d).astype(np.int64)
        )
        crossings = int(crossings_per.sum())

        # Build display coordinates: spread needles over the visible band
        # by shifting each in y by an integer multiple of d (does not affect
        # the crossing count).
        x_extent = 6.0 * d
        xc = np.random.uniform(0.0, x_extent, n)
        x1 = xc - dx
        x2 = xc + dx
        k_off = np.random.randint(1, max(2, self.N_STRIPES - 1), n)
        y1_disp = y1 + k_off * d
        y2_disp = y2 + k_off * d

        crossed = crossings_per > 0
        red = "#d62728"
        blue = "#1f77b4"
        for i in range(n):
            self.needle_segments.append(
                [(x1[i], y1_disp[i]), (x2[i], y2_disp[i])]
            )
            self.needle_colors.append(red if crossed[i] else blue)
        if len(self.needle_segments) > self.MAX_VISIBLE_NEEDLES:
            cut = len(self.needle_segments) - self.MAX_VISIBLE_NEEDLES
            self.needle_segments = self.needle_segments[cut:]
            self.needle_colors = self.needle_colors[cut:]

        self.total_tosses += n
        self.total_crossings += crossings

        if self.total_crossings > 0:
            est = (2.0 * L * self.total_tosses) / (d * self.total_crossings)
        else:
            est = float("nan")
        self.history_n.append(self.total_tosses)
        self.history_pi.append(est)

        self._draw()
        self._update_status(est)

        if self.total_tosses < target:
            self.root.after(1, self._step)
        else:
            self.running = False
            self.run_btn.config(text="Run")

    def _update_status(self, est):
        L = self._safe_float(self.needle_length, 1.0)
        d = self._safe_float(self.stripe_spacing, 1.5)
        if np.isnan(est):
            est_str = "—"
            err_str = "—"
        else:
            est_str = f"{est:.6f}"
            err_str = f"{abs(est - np.pi):.5f}"
        regime = "short needle (L ≤ d)" if L <= d else "long needle (L > d)"
        self.status_var.set(
            f"L = {L:g}   d = {d:g}   [{regime}]   "
            f"tosses = {self.total_tosses:,}   crossings = {self.total_crossings:,}   "
            f"π̂ = {est_str}   |π̂ − π| = {err_str}"
        )

    def _draw(self):
        from matplotlib.collections import LineCollection

        d = self._safe_float(self.stripe_spacing, 1.5)
        x_extent = 6.0 * d
        band_height = self.N_STRIPES * d

        ax = self.ax_needles
        ax.clear()
        ax.set_title("Recent needles on the striped floor")
        for k in range(self.N_STRIPES + 1):
            ax.axhline(k * d, color="0.55", lw=0.8)
        if self.needle_segments:
            lc = LineCollection(
                self.needle_segments, colors=self.needle_colors,
                linewidths=0.9, alpha=0.75,
            )
            ax.add_collection(lc)
        ax.set_xlim(-0.02 * x_extent, 1.02 * x_extent)
        ax.set_ylim(0, band_height)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

        ax2 = self.ax_conv
        ax2.clear()
        ax2.set_title("Monte Carlo estimate of π vs. number of tosses")
        if self.history_n:
            ax2.plot(
                self.history_n, self.history_pi,
                color="#1f77b4", lw=1.3, label="estimate",
            )
        ax2.axhline(np.pi, color="#d62728", lw=1.0, ls="--", label="π")
        if self.history_n and self.history_n[-1] > 10:
            ax2.set_xscale("log")
        ax2.set_xlabel("tosses")
        ax2.set_ylabel("π estimate")
        ax2.set_ylim(2.6, 3.7)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")

        self.fig.tight_layout()
        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    BuffonApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
