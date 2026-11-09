"""Display transforms. Filters run per contiguous segment (between dropouts) with filtfilt.

Railed samples become NaN and split the signal like dropouts (each good run is filtered
separately). Optionally (mask_rails=False) they are linearly bridged instead, for display.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .io import RAIL_FRACTION, RAIL_UV, channel_group

FILTERS = {
    "raw": None,
    "remove median": "median",
    "high-pass 0.5 Hz": ("highpass", 0.5),
    "band-pass 1–40 Hz (EEG)": ("bandpass", (1.0, 40.0)),
    "band-pass 0.1–15 Hz (EOG)": ("bandpass", (0.1, 15.0)),
    "band-pass 20–120 Hz (EMG)": ("bandpass", (20.0, 120.0)),
    "low-pass 5 Hz": ("lowpass", 5.0),
    "linear detrend": "detrend",
}


def segments(t: np.ndarray, max_dt: float) -> list[slice]:
    cut = np.where(np.diff(t) > max_dt)[0] + 1
    edges = np.r_[0, cut, len(t)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]


def rail_mask(name: str, x: np.ndarray) -> np.ndarray:
    g = channel_group(name)
    if g not in RAIL_UV:
        return np.zeros(len(x), bool)
    return np.abs(x) >= RAIL_FRACTION * RAIL_UV[g]


def _bridge(x: np.ndarray, bad: np.ndarray) -> np.ndarray:
    if not bad.any():
        return x
    if bad.all():
        return np.zeros_like(x)
    i = np.arange(len(x))
    y = x.copy()
    y[bad] = np.interp(i[bad], i[~bad], x[~bad])
    return y


def _filt(x: np.ndarray, fs: float, spec, notch: float | None) -> np.ndarray:
    if spec == "median":
        x = x - np.nanmedian(x)
    elif spec == "detrend":
        x = signal.detrend(x)
    elif spec is not None:
        kind, f = spec
        nyq = fs / 2
        if kind == "bandpass":
            f = (f[0], min(f[1], 0.95 * nyq))
        elif kind == "lowpass":
            f = min(f, 0.95 * nyq)
        sos = signal.butter(4, np.asarray(f) / nyq, btype=kind, output="sos")
        padlen = min(len(x) - 1, 3 * fs)
        if len(x) > 30:
            x = signal.sosfiltfilt(sos, x, padlen=int(padlen))
    if notch and fs > 2 * notch and len(x) > 30:
        for h in (notch, 2 * notch):
            if h < fs / 2:
                b, a = signal.iirnotch(h, 30, fs)
                x = signal.filtfilt(b, a, x, padlen=min(len(x) - 1, 3 * int(fs)))
    return x


def process(name: str, t: np.ndarray, x: np.ndarray, fs: float, filt: str = "raw",
            notch: float | None = None, mask_rails: bool = True, max_dt: float | None = None) -> np.ndarray:
    """Return the transformed signal (same length as x).

    mask_rails=True (default): railed samples are NaN and act like dropouts — each run of
    good samples between gaps/rails is filtered on its own, so nothing is interpolated.
    Good runs too short to filter (≤ 30 samples) are NaN as well when a filter is applied.
    mask_rails=False: railed stretches are linearly bridged before filtering (display only).
    """
    x = np.asarray(x, float)
    missing = ~np.isfinite(x)                 # e.g. gaps in derived signals: always split on and keep NaN
    bad = rail_mask(name, x) | missing
    spec = FILTERS[filt] if isinstance(filt, str) else filt   # or a spec, e.g. ("highpass", 0.1)
    filtering = spec is not None or bool(notch)
    out = np.full_like(x, np.nan)
    max_dt = max_dt if max_dt is not None else 3.0 / fs
    for sl in segments(t, max_dt):
        if not mask_rails:
            seg = _bridge(x[sl], bad[sl])
            out[sl] = _filt(seg, fs, spec, notch) if filtering else seg
            continue
        good = ~bad[sl]
        # runs of consecutive good samples inside this segment
        edges = np.flatnonzero(np.diff(np.r_[0, good.astype(int), 0]))
        for a, b in zip(edges[::2] + sl.start, edges[1::2] + sl.start):
            if filtering and b - a <= 30:
                continue
            out[a:b] = _filt(x[a:b], fs, spec, notch) if filtering else x[a:b]
    out[missing] = np.nan
    return out


def robust_scale(y: np.ndarray) -> float:
    """Spread used for stacking: 1st–99th percentile range of the finite samples."""
    f = y[np.isfinite(y)]
    if len(f) == 0:
        return 1.0
    lo, hi = np.percentile(f, [1, 99])
    s = hi - lo
    return s if s > 0 else (np.abs(f).max() or 1.0)


def minmax_decimate(t: np.ndarray, y: np.ndarray, n_out: int) -> np.ndarray:
    """Indices of a min/max-preserving decimation of y to about n_out points."""
    n = len(t)
    if n <= n_out:
        return np.arange(n)
    nb = max(n_out // 2, 1)
    edges = np.linspace(0, n, nb + 1).astype(int)
    idx = []
    yy = np.where(np.isfinite(y), y, np.nan)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = yy[a:b]
        if np.isnan(seg).all():
            idx.append(a)
            continue
        i1, i2 = a + np.nanargmin(seg), a + np.nanargmax(seg)
        idx.extend((i1, i2))
        nan = np.flatnonzero(np.isnan(seg))
        if len(nan):                       # keep a NaN so masked stretches still break the line
            idx.append(a + nan[0])
    return np.unique(np.asarray(idx))
