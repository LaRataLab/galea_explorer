"""GUIs that analyse channels over the time range selected in a SignalBrowser.

    b = SignalBrowser(s)
    PairGUI(b)                         # relationship between two channels

PairGUI picks two channels, a common preprocessing / resampling, and one of the
analyses in ANALYSES (cross-correlation, coherence, ...), each with its own options.
"run" recomputes on the browser's current clip (or visible window); "export figure"
saves the result as exports/<dataset>_<A>_vs_<B>_<analysis>_<t0>-<t1>s.png.

Adding an analysis: subclass PairAnalysis (name, controls(), compute(), plot()) and
append it to ANALYSES. Other kinds of GUI can subclass AnalysisGUI.
"""
from __future__ import annotations

import os
import re

import ipywidgets as W
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

from . import preprocess as pp
from .style import ACCENT, BAD, BRAND, GOOD, GROUP_COLORS, MUTED, TEMPLATE, WARN

EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")
PAIR_GROUPS = ("EEG", "EMG", "EOG", "Aux", "Derived")   # no signal-check


# ----------------------------------------------------------------------------- data helpers
def channel_fs(session, ch) -> float:
    if ch in session.exg.columns:
        return 250.0
    if ch in session.aux.columns:
        return 50.0
    return 1.0 / np.median(np.diff(session.live["t"].to_numpy()))


def get_segment(session, ch, t0, t1, prep="demean", hp_cutoff=0.1, notch=True):
    """(t, y, fs) of one channel inside [t0, t1], preprocessed, railed samples NaN.

    Filtering runs per contiguous run (dropouts and railed stretches split runs).
    prep: "none" | "demean" | "linear detrend" | "high-pass".
    """
    df = session.frame_of(ch)
    t = df["t"].to_numpy()
    m = (t >= t0) & (t <= t1)
    t, x = t[m], df[ch].to_numpy(float)[m]
    fs = channel_fs(session, ch)
    is_exg = ch in session.exg_channels   # raw ExG only (not derived envelopes)
    spec = ("highpass", float(hp_cutoff)) if prep == "high-pass" else None
    y = pp.process(ch, t, x, fs, spec, 60.0 if (notch and is_exg) else None, mask_rails=True,
                   max_dt=3.0 / fs if fs > 25 else 1.5 / fs)
    ok = np.isfinite(y)
    if prep == "demean" and ok.any():
        y = y - np.nanmean(y)
    elif prep == "linear detrend" and ok.sum() > 2:
        c = np.polyfit(t[ok], y[ok], 1)
        y = y - np.polyval(c, t)
    return t, y, fs


def to_grid(t, y, fs_src, grid, fs_dst):
    """Resample onto a uniform grid. Anti-alias if downsampling; NaN across gaps/NaNs."""
    y = y.copy()
    if fs_src > 1.01 * fs_dst:
        sos = signal.butter(4, 0.4 * fs_dst / (fs_src / 2), output="sos")
        for sl in pp.segments(t, 1.5 / fs_src):
            seg = y[sl]
            ok = np.isfinite(seg)
            edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))   # each finite run separately
            for a, b in zip(edges[::2], edges[1::2]):
                if b - a > 30:
                    seg[a:b] = signal.sosfiltfilt(sos, seg[a:b], padlen=min(b - a - 1, int(3 * fs_src)))
                else:
                    seg[a:b] = np.nan
            y[sl] = seg
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.full(len(grid), np.nan)
    out = np.interp(grid, t[ok], y[ok])
    # invalidate grid points whose nearest source sample is NaN or too far away (gap)
    i = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    near = np.where(grid - t[i - 1] < t[i] - grid, i - 1, i)
    out[~ok[near] | (np.abs(t[near] - grid) > 1.5 / fs_src)] = np.nan
    return out


def pair_on_grid(session, a, b, t0, t1, fs=0.0, prep="demean", hp_cutoff=0.1, notch=True):
    """Both channels preprocessed and resampled to a common grid (fs=0: the lower native rate)."""
    ta, ya, fa = get_segment(session, a, t0, t1, prep, hp_cutoff, notch)
    tb, yb, fb = get_segment(session, b, t0, t1, prep, hp_cutoff, notch)
    fs = fs or min(fa, fb)
    grid = np.arange(t0, t1, 1.0 / fs)
    return grid, to_grid(ta, ya, fa, grid, fs), to_grid(tb, yb, fb, grid, fs), fs


def xcorr(x, y, maxlag_n, norm="pearson"):
    """Cross-correlation r(k) = corr(x[n], y[n+k]) for |k| ≤ maxlag_n, NaN-aware.

    norm: "pearson"  – Pearson r over the overlapping valid samples at each lag (default)
          "coeff"    – global normalisation: Σ x·y / sqrt(Σx² Σy²) (MATLAB 'coeff'), after demeaning
          "cov"      – unbiased covariance: mean over overlapping pairs of (x−x̄)(y−ȳ)
          "none"     – raw sum Σ x[n] y[n+k]
    Returns lags (samples), values, n_pairs.
    """
    mx, my = np.isfinite(x), np.isfinite(y)
    x0, y0 = np.where(mx, x, 0.0), np.where(my, y, 0.0)
    fx, fy = mx.astype(float), my.astype(float)
    n = len(x)

    def c(a, b):  # Σ_n a[n] b[n+k] for all k, sliced to |k| ≤ maxlag
        full = signal.correlate(b, a, mode="full", method="fft")   # index n-1 is lag 0
        mid = n - 1
        return full[mid - maxlag_n: mid + maxlag_n + 1]

    lags = np.arange(-maxlag_n, maxlag_n + 1)
    N = np.round(c(fx, fy))
    Sxy = c(x0, y0)
    with np.errstate(invalid="ignore", divide="ignore"):
        if norm == "none":
            v = Sxy
        elif norm == "coeff":
            xm = np.where(mx, x - np.nanmean(x), 0.0)
            ym = np.where(my, y - np.nanmean(y), 0.0)
            v = c(xm, ym) / np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
        else:
            Sx, Sy = c(x0, fy), c(fx, y0)
            if norm == "cov":
                v = (Sxy - Sx * Sy / N) / (N - 1)
            else:
                Sxx, Syy = c(x0 ** 2, fy), c(fx, y0 ** 2)
                v = (N * Sxy - Sx * Sy) / np.sqrt((N * Sxx - Sx ** 2) * (N * Syy - Sy ** 2))
        v = np.where(N >= 10, v, np.nan)
    return lags, v, N


def _finite_runs(ok):
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def _n_segments(run_lengths, nperseg, step):
    return sum((L - nperseg) // step + 1 for L in run_lengths if L >= nperseg)


def coherence(x, y, fs, n_windows=8, overlap=0.5, window="hann", alpha=0.05, min_nperseg=16):
    """Welch magnitude-squared coherence and cross-spectral phase, NaN-aware.

    Segments are taken only inside runs where both x and y are finite; the segment length is the
    longest that still yields >= n_windows segments in total. Spectra are averaged over all segments.
    Phase = angle(Pxy) with Pxy = E[conj(X)·Y]: phase > 0 means B (y) leads A (x).
    Significance: MSC > 1 − α^(1/(n_eff−1)), n_eff = equivalent independent segments (Welch's
    overlap correction for the chosen window).
    Returns dict(f, msc, phase_deg, Pxx, Pyy, nperseg, n_segments, n_eff, threshold, df).
    """
    runs = _finite_runs(np.isfinite(x) & np.isfinite(y))
    lengths = [b - a for a, b in runs]
    if not lengths or max(lengths) < min_nperseg:
        raise ValueError("not enough overlapping valid samples")
    nperseg = None
    for L in range(max(lengths), min_nperseg - 1, -1):          # longest segment giving >= n_windows
        if _n_segments(lengths, L, max(1, int(round(L * (1 - overlap))))) >= n_windows:
            nperseg = L
            break
    if nperseg is None:
        nperseg = min_nperseg
    step = max(1, int(round(nperseg * (1 - overlap))))
    noverlap = nperseg - step
    Pxy = Pxx = Pyy = 0.0
    n_seg = 0
    for a, b in runs:
        if b - a < nperseg:
            continue
        k = (b - a - nperseg) // step + 1
        kw = dict(fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, detrend="constant")
        f, sxy = signal.csd(x[a:b], y[a:b], **kw)
        _, sxx = signal.welch(x[a:b], **kw)
        _, syy = signal.welch(y[a:b], **kw)
        Pxy, Pxx, Pyy, n_seg = Pxy + k * sxy, Pxx + k * sxx, Pyy + k * syy, n_seg + k
    Pxy, Pxx, Pyy = Pxy / n_seg, Pxx / n_seg, Pyy / n_seg
    with np.errstate(invalid="ignore", divide="ignore"):
        msc = np.abs(Pxy) ** 2 / (Pxx * Pyy)
    # Welch: equivalent number of independent segments for overlapping windows
    w = signal.get_window(window, nperseg)
    rho = [np.sum(w[: nperseg - j * step] * w[j * step:]) / np.sum(w ** 2) for j in range(1, n_seg) if j * step < nperseg]
    n_eff = n_seg / (1 + 2 * sum((1 - (j + 1) / n_seg) * r ** 2 for j, r in enumerate(rho)))
    thr = 1 - alpha ** (1 / (n_eff - 1)) if n_eff > 1 else np.nan
    return dict(f=f, msc=msc, phase_deg=np.degrees(np.angle(Pxy)), Pxx=Pxx, Pyy=Pyy, nperseg=nperseg,
                n_segments=n_seg, n_eff=n_eff, threshold=thr, df=fs / nperseg)


# ----------------------------------------------------------------------------- pair analyses
class PairAnalysis:
    """One analysis of a channel pair. Subclasses define name, n_rows, controls(), compute(), plot()."""
    name = "analysis"
    n_rows = 1                      # result panels below the signals panel
    row_heights = (1.0,)

    def __init__(self):
        self.st = {"description_width": "90px"}

    def controls(self) -> W.Widget:
        return W.HBox([])

    def compute(self, xa, xb, fs) -> dict:
        raise NotImplementedError

    def plot(self, fig, row, res, a, b) -> str:
        """Add result traces to fig from `row` on; return an HTML summary."""
        raise NotImplementedError


class CrossCorrelation(PairAnalysis):
    """r(τ) = corr(A(t), B(t+τ)); a peak at τ > 0 means B lags A."""
    name = "cross-correlation"

    def __init__(self):
        super().__init__()
        self.norm = W.Dropdown(options=[("Pearson r per lag", "pearson"), ("global coeff", "coeff"),
                                        ("covariance", "cov"), ("none (raw sum)", "none")],
                               value="pearson", description="normalise", style=self.st)
        self.maxlag = W.BoundedFloatText(value=5.0, min=0.01, max=600, step=0.5, description="max lag (s)",
                                         style=self.st, layout=W.Layout(width="200px"))

    def controls(self):
        return W.HBox([self.norm, self.maxlag])

    def compute(self, xa, xb, fs):
        maxlag_n = int(min(round(self.maxlag.value * fs), len(xa) - 1))
        lags, r, n = xcorr(xa, xb, maxlag_n, self.norm.value)
        return dict(tau=lags / fs, r=r, n=n, norm=self.norm.value)

    def plot(self, fig, row, res, a, b):
        tau, r, n = res["tau"], res["r"], res["n"]
        fig.add_trace(go.Scatter(x=tau, y=r, name="xcorr", line=dict(width=1.5, color=BRAND), customdata=n,
                                 hovertemplate="τ=%{x:.3f} s<br>value=%{y:.4g}<br>pairs=%{customdata}<extra></extra>"),
                      row, 1)
        fig.add_vline(x=0, line=dict(color="#999", width=1, dash="dot"), row=row, col=1)
        fig.update_xaxes(title_text=f"lag τ (s)  ·  τ > 0: {b} lags {a}", row=row, col=1)
        fig.update_yaxes(title_text={"pearson": "Pearson r", "coeff": "coeff", "cov": "covariance",
                                     "none": "Σ x·y"}[res["norm"]], row=row, col=1)
        if res["norm"] == "pearson":
            fig.update_yaxes(range=[-1.05, 1.05], row=row, col=1)
        if not np.isfinite(r).any():
            return "no valid overlap"
        k = int(np.nanargmax(np.abs(r)))
        k0 = int(np.argmin(np.abs(tau)))
        fig.add_trace(go.Scatter(x=[tau[k]], y=[r[k]], mode="markers+text", marker=dict(size=9, color=ACCENT),
                                 text=[f" peak {r[k]:.3g} at τ={tau[k]:.3f} s"], textposition="middle right",
                                 showlegend=False, hoverinfo="skip"), row, 1)
        res.update(peak_r=float(r[k]), peak_tau=float(tau[k]))
        return f"<b>peak {r[k]:.3g} at τ = {tau[k]:+.3f} s</b> · value at τ=0: {r[k0]:.3g}"


class Coherence(PairAnalysis):
    """Magnitude-squared coherence and phase spectrum (Welch)."""
    name = "coherence"
    n_rows = 2
    row_heights = (0.55, 0.45)

    def __init__(self):
        super().__init__()
        st = self.st
        self.fmin = W.BoundedFloatText(value=0.0, min=0, max=1000, step=0.05, description="f min (Hz)",
                                       style=st, layout=W.Layout(width="190px"))
        self.fmax = W.BoundedFloatText(value=2.0, min=0.01, max=1000, step=0.5, description="f max (Hz)",
                                       style=st, layout=W.Layout(width="190px"))
        self.nwin = W.BoundedIntText(value=8, min=2, max=1000, description="# windows", style=st,
                                     layout=W.Layout(width="170px"))
        self.overlap = W.BoundedIntText(value=50, min=0, max=90, step=10, description="overlap %", style=st,
                                        layout=W.Layout(width="170px"))
        self.window = W.Dropdown(options=["hann", "hamming", "blackman", "boxcar"], value="hann",
                                 description="window", style=st, layout=W.Layout(width="200px"))
        self.alpha = W.Dropdown(options=[0.05, 0.01, 0.001], value=0.05, description="α", style=st,
                                layout=W.Layout(width="170px"))

    def controls(self):
        return W.VBox([W.HBox([self.fmin, self.fmax, self.nwin]), W.HBox([self.overlap, self.window, self.alpha])])

    def compute(self, xa, xb, fs):
        return coherence(xa, xb, fs, n_windows=self.nwin.value, overlap=self.overlap.value / 100,
                         window=self.window.value, alpha=self.alpha.value)

    def plot(self, fig, row, res, a, b):
        f, msc, ph, thr = res["f"], res["msc"], res["phase_deg"], res["threshold"]
        band = (f >= self.fmin.value) & (f <= self.fmax.value)
        f, msc, ph = f[band], msc[band], ph[band]
        if not len(f):
            return "no frequencies in range (Δf = %.3g Hz)" % res["df"]
        with np.errstate(divide="ignore", invalid="ignore"):
            lag = np.where(f > 0, -ph / (360 * f), np.nan)          # equivalent lag; > 0: B lags A
        fig.add_trace(go.Scatter(x=f, y=msc, name="coherence", line=dict(color=BRAND, width=1.5),
                                 hovertemplate="f=%{x:.3f} Hz<br>MSC=%{y:.3f}<extra></extra>"), row, 1)
        if np.isfinite(thr):
            fig.add_hline(y=thr, line=dict(color=ACCENT, dash="dash", width=1), row=row, col=1,
                          annotation_text=f"α={self.alpha.value} ({thr:.2f})", annotation_position="top right")
        sig = msc > thr
        fig.add_trace(go.Scatter(x=f[~sig], y=ph[~sig], mode="markers", name="phase (n.s.)",
                                 marker=dict(color="#ccc", size=4), hoverinfo="skip"), row + 1, 1)
        fig.add_trace(go.Scatter(x=f[sig], y=ph[sig], mode="markers", name="phase (sig.)",
                                 marker=dict(color=ACCENT, size=6), customdata=np.c_[lag[sig], msc[sig]],
                                 hovertemplate="f=%{x:.3f} Hz<br>phase=%{y:.0f}°<br>lag=%{customdata[0]:.3f} s"
                                               "<br>MSC=%{customdata[1]:.2f}<extra></extra>"), row + 1, 1)
        fig.add_hline(y=0, line=dict(color="#999", width=1, dash="dot"), row=row + 1, col=1)
        fig.update_yaxes(title_text="MSC", range=[0, 1.02], row=row, col=1)
        fig.update_yaxes(title_text=f"phase (°), > 0: {b} leads", range=[-185, 185], tickvals=[-180, -90, 0, 90, 180],
                         row=row + 1, col=1)
        fig.update_xaxes(title_text="frequency (Hz)", row=row + 1, col=1)
        fig.update_xaxes(matches=f"x{row}", row=row + 1, col=1)
        k = int(np.nanargmax(msc))
        res.update(peak_f=float(f[k]), peak_msc=float(msc[k]), peak_phase_deg=float(ph[k]), peak_lag_s=float(lag[k]))
        return (f"<b>peak MSC {msc[k]:.2f} at {f[k]:.3f} Hz</b> (phase {ph[k]:+.0f}°, lag {lag[k]:+.3f} s) · "
                f"{res['n_segments']} windows of {1 / res['df']:.1f} s → "
                f"Δf {res['df']:.3g} Hz, n_eff {res['n_eff']:.1f}, threshold {thr:.2f}")


ANALYSES = [CrossCorrelation, Coherence]


# ----------------------------------------------------------------------------- base GUI
class AnalysisGUI:
    """Base: range source + run button + output figure. Subclasses add controls/compute."""
    title = "analysis"

    def __init__(self, browser):
        self.b = browser
        self.s = browser.s
        self.source = W.Dropdown(options=["browser clip (else view)", "browser view", "whole session"],
                                 value="browser clip (else view)", description="range",
                                 style={"description_width": "50px"}, layout=W.Layout(width="290px"))
        self.run_btn = W.Button(description="run", button_style="primary", icon="play",
                                layout=W.Layout(width="90px"))
        self.run_btn.on_click(lambda _: self.run())
        self.info = W.HTML()
        self.fig = go.FigureWidget()
        self.fig.update_layout(template=TEMPLATE, margin=dict(l=70, r=20, t=40, b=40), height=560,
                               width=browser.width.value)
        header = W.HTML(f"<h3 style='margin:4px 0'>{self.title}</h3>")
        self.widget = W.VBox([header, self.controls(), W.HBox([self.source, self.run_btn, *self.extra_buttons(),
                                                              self.info]), self.fig])
        self.run()

    def _ipython_display_(self):
        from IPython.display import display
        display(self.widget)

    def extra_buttons(self) -> list:
        return []

    def time_range(self) -> tuple[float, float]:
        src = self.source.value
        if src == "whole session":
            return 0.0, float(self.s.exg["t"].iloc[-1])
        if src.startswith("browser clip") and self.b.clip is not None:
            return self.b.clip
        return self.b._range

    def run(self):
        t0, t1 = self.time_range()
        self.t_range = (t0, t1)
        try:
            msg = self.compute(t0, t1)
        except Exception as e:  # keep the GUI alive
            msg = f"<span style='color:{BAD}'>error: {type(e).__name__}: {e}</span>"
        self.info.value = (f"<span style='color:#555'>&nbsp; {t0:.2f}–{t1:.2f} s ({t1 - t0:.1f} s)"
                           f"{' · clipped' if self.b.clip else ''}</span>&nbsp; {msg or ''}")

    def controls(self) -> W.Widget:
        raise NotImplementedError

    def compute(self, t0, t1) -> str | None:
        raise NotImplementedError


def _safe(name: str) -> str:
    name = name.replace("|ω|", "w").replace("|a|", "a").replace("ω", "w")
    return re.sub(r"[^A-Za-z0-9.\-]+", "-", name).strip("-")


# ----------------------------------------------------------------------------- pair GUI
class PairGUI(AnalysisGUI):
    """Relationship between two channels: common preprocessing + a selectable analysis."""
    title = "Pair analysis"

    def __init__(self, browser, a: str | None = None, b: str | None = None, analysis: str = "cross-correlation"):
        groups = browser.s.channels
        self._chs = [c for g in PAIR_GROUPS for c in groups.get(g, [])]
        a = a if a in self._chs else self._chs[0]
        b = b if b in self._chs else self._chs[min(1, len(self._chs) - 1)]
        self._defaults = (a, b)
        self.analyses = {cls.name: cls() for cls in ANALYSES}
        self._analysis0 = analysis if analysis in self.analyses else ANALYSES[0].name
        self.result: dict = {}
        super().__init__(browser)

    def controls(self):
        st = {"description_width": "90px"}
        self.ch_a = W.Dropdown(options=self._chs, value=self._defaults[0], description="channel A", style=st)
        self.ch_b = W.Dropdown(options=self._chs, value=self._defaults[1], description="channel B", style=st)
        self.fs = W.Dropdown(options=[("auto (lower fs)", 0.0), ("250 Hz", 250.0), ("50 Hz", 50.0),
                                      ("25 Hz", 25.0), ("10 Hz", 10.0)], value=0.0, description="resample", style=st)
        self.prep = W.Dropdown(options=["none", "demean", "linear detrend", "high-pass"], value="high-pass",
                               description="preprocess", style=st)
        self.cutoff = W.BoundedFloatText(value=0.05, min=0.001, max=100, step=0.05, description="cutoff (Hz)",
                                         style=st, layout=W.Layout(width="190px"))
        self.notch = W.Checkbox(value=True, description="notch 60/120 Hz (ExG)", indent=False)
        self.cutoff.layout.visibility = "visible"
        self.prep.observe(lambda c: setattr(self.cutoff.layout, "visibility",
                                            "visible" if c["new"] == "high-pass" else "hidden"), "value")
        self.analysis = W.ToggleButtons(options=list(self.analyses), value=self._analysis0, description="analysis",
                                        style={"description_width": "90px", "button_width": "140px"})
        self.opt_box = W.VBox([self.analyses[self._analysis0].controls()],
                              layout=W.Layout(border="1px solid #ddd", padding="4px", margin="2px 0"))
        self.analysis.observe(self._on_analysis, "value")
        return W.VBox([W.HBox([self.ch_a, self.ch_b, self.fs]),
                       W.HBox([self.prep, self.cutoff, self.notch]),
                       self.analysis, self.opt_box])

    def extra_buttons(self):
        self.export_btn = W.Button(description="export figure", icon="download", layout=W.Layout(width="130px"),
                                   tooltip=f"save to {EXPORT_DIR}")
        self.export_btn.on_click(lambda _: self.export())
        return [self.export_btn]

    def _on_analysis(self, change):
        self.opt_box.children = [self.analyses[change["new"]].controls()]
        self.run()

    @property
    def current(self) -> PairAnalysis:
        return self.analyses[self.analysis.value]

    def compute(self, t0, t1):
        a, b, an = self.ch_a.value, self.ch_b.value, self.current
        grid, xa, xb, fs = pair_on_grid(self.s, a, b, t0, t1, self.fs.value, self.prep.value,
                                        self.cutoff.value, self.notch.value)
        res = an.compute(xa, xb, fs)
        heights = [0.3] + [0.7 * h for h in an.row_heights]
        fig = make_subplots(rows=1 + an.n_rows, cols=1, row_heights=heights, vertical_spacing=0.09)
        z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) or 1)
        col = lambda ch: GROUP_COLORS.get(self.b._group(ch), "#444")
        step = max(1, len(grid) // 4000)
        fig.add_trace(go.Scattergl(x=grid[::step], y=z(xa)[::step], name=a, line=dict(width=1, color=col(a))), 1, 1)
        fig.add_trace(go.Scattergl(x=grid[::step], y=z(xb)[::step] - 4, name=b,
                                   line=dict(width=1, color="#444" if col(a) == col(b) else col(b))), 1, 1)
        fig.update_yaxes(showticklabels=False, title_text=f"z ({fs:g} Hz)", row=1, col=1)
        fig.update_xaxes(title_text="t (s)", row=1, col=1)
        msg = an.plot(fig, 2, res, a, b)
        nvalid = int(np.sum(np.isfinite(xa) & np.isfinite(xb)))
        self.result = dict(res, analysis=an.name, a=a, b=b, fs=fs, t0=t0, t1=t1, valid_s=nvalid / fs)
        fig.update_layout(template=TEMPLATE, margin=dict(l=70, r=20, t=95, b=40),
                          height=400 + 220 * an.n_rows, width=self.b.width.value,
                          legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
                          title=dict(text=self.figure_title(), x=0.01, y=0.985, yanchor="top"))
        with self.fig.batch_update():
            self.fig.data = []
            self.fig.layout = fig.layout
            self.fig.add_traces(list(fig.data))
        return f"{msg} · {nvalid / fs:.1f} s of valid pairs"

    def figure_title(self) -> str:
        r = self.result
        return (f"{self.s.name} · {r.get('a')} vs {r.get('b')} · {r.get('analysis')} · "
                f"{r.get('t0', 0):.1f}–{r.get('t1', 0):.1f} s")

    def export_name(self) -> str:
        r = self.result
        return "_".join([_safe(self.s.name), _safe(r["a"]), "vs", _safe(r["b"]), _safe(r["analysis"]),
                         f"{r['t0']:.1f}-{r['t1']:.1f}s"])

    def export(self, folder: str | None = None, fmt: str = "png") -> str:
        """Save the current result figure; falls back to .html if static export is unavailable."""
        folder = folder or EXPORT_DIR
        os.makedirs(folder, exist_ok=True)
        fig = go.Figure(self.fig)
        path = os.path.join(folder, f"{self.export_name()}.{fmt}")
        try:
            fig.write_image(path, scale=2)
        except Exception as e:
            path = os.path.splitext(path)[0] + ".html"
            fig.write_html(path, include_plotlyjs="cdn")
            self.info.value += f" <span style='color:{WARN}'>(PNG export unavailable: {type(e).__name__}; saved HTML)</span>"
        self.info.value += f" <span style='color:{GOOD}'>saved {os.path.basename(path)}</span>"
        return path
