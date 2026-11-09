"""EMG linear envelope ("muscle activity")."""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..preprocess import process


def emg_envelope(name: str, t: np.ndarray, x: np.ndarray, fs: float = 250.0, band=(20.0, 120.0),
                 lp_hz: float = 5.0, notch: float | None = 60.0) -> np.ndarray:
    """Linear envelope ("muscle activity"): band-pass → full-wave rectify → low-pass.

    Railed samples and dropouts are NaN; every finite run is processed separately.
    """
    y = np.abs(process(name, t, x, fs, ("bandpass", band), notch, mask_rails=True))
    out = np.full_like(y, np.nan)
    sos = signal.butter(4, lp_hz / (fs / 2), output="sos")
    ok = np.isfinite(y)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a > 30:
            out[a:b] = signal.sosfiltfilt(sos, y[a:b], padlen=min(b - a - 1, int(3 * fs)))
    return np.maximum(out, 0.0)
