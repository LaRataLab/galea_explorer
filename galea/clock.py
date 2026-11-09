"""Common clock for all streams.

ExG and Aux share the headset's device clock (`dev_ts`, seconds). The PC arrival
time (`pc_ts`) of each sample is device time + transmission latency, with
latency >= some floor and heavily jittered (batching). We therefore fit

    pc ≈ a + b * dev

on the *lower envelope* of (pc - dev) — the per-window minimum latency — and
map every device timestamp through that line. Signal-check Live is already in
PC time. All streams get a column `t` = seconds since the first valid ExG
sample (`t0_epoch`, UTC epoch seconds).

Note: the envelope includes the constant minimum transmission latency, so
absolute wall time is late by that floor (typically a few ms to tens of ms);
relative timing between streams is unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ClockFit:
    a: float                 # intercept (epoch s)
    b: float                 # slope: PC seconds per device second
    t0_epoch: float          # epoch of t = 0
    env_dev: np.ndarray      # envelope points used in the fit (device s)
    env_off: np.ndarray      # pc - dev at those points
    env_used: np.ndarray     # bool, kept after outlier rejection
    env_resid_ms: np.ndarray # residual of envelope points (ms)
    latency_ms: np.ndarray   # per-ExG-sample pc - fitted, ms (>= ~0)
    sync_err_ms: np.ndarray  # per-ExG-sample (sync_ts - fitted), ms — quality of the vendor `Timestamp` column

    @property
    def drift_pct(self) -> float:
        """How much faster the device clock runs than the PC clock (%)."""
        return (1.0 / self.b - 1.0) * 100

    def dev_to_epoch(self, dev):
        return self.a + self.b * np.asarray(dev)

    def dev_to_t(self, dev):
        return self.dev_to_epoch(dev) - self.t0_epoch

    def summary(self) -> dict:
        u = self.env_resid_ms[self.env_used]
        return {
            "slope b (pc s / dev s)": self.b,
            "device clock fast by (%)": self.drift_pct,
            "envelope points used": f"{self.env_used.sum()} / {len(self.env_used)}",
            "envelope residual RMS (ms)": float(np.sqrt(np.mean(u**2))),
            "envelope residual max |.| (ms)": float(np.abs(u).max()),
            "arrival latency median (ms)": float(np.median(self.latency_ms)),
            "arrival latency p99 (ms)": float(np.percentile(self.latency_ms, 99)),
            "vendor Timestamp error range (ms)": (float(self.sync_err_ms.min()), float(self.sync_err_ms.max())),
        }


def fit_envelope(dev: np.ndarray, pc: np.ndarray, window_s: float = 2.0, n_iter: int = 5,
                 k_mad: float = 3.0):
    off = pc - dev
    bins = np.floor((dev - dev[0]) / window_s).astype(int)
    order = np.lexsort((off, bins))
    first = np.r_[True, np.diff(bins[order]) != 0]
    sel = order[first]                      # argmin of offset in every window
    x, y = dev[sel], off[sel]
    used = np.ones(len(x), bool)
    for _ in range(n_iter):
        c1, c0 = np.polyfit(x[used], y[used], 1)
        r = y - (c0 + c1 * x)
        mad = np.median(np.abs(r[used] - np.median(r[used]))) * 1.4826 + 1e-6
        new = r < np.median(r[used]) + k_mad * mad   # reject windows with no low-latency packet (upper outliers only)
        if (new == used).all():
            break
        used = new
    c1, c0 = np.polyfit(x[used], y[used], 1)
    r = y - (c0 + c1 * x)
    # pc = dev + c0 + c1*dev  ->  a = c0, b = 1 + c1
    return c0, 1.0 + c1, x, y, r, used


def find_gaps(dev: np.ndarray, factor: float = 1.5) -> pd.DataFrame:
    """Gaps in a device-timestamp series: steps larger than factor x median step."""
    dd = np.diff(dev)
    med = np.median(dd)
    i = np.where(dd > factor * med)[0]
    return pd.DataFrame({
        "row_before": i,
        "dev_start": dev[i],
        "dev_end": dev[i + 1],
        "missing_samples": np.round(dd[i] / med).astype(int) - 1,
    })


def apply(session) -> ClockFit:
    exg, aux, live = session.exg, session.aux, session.live
    dev = exg["dev_ts"].to_numpy()
    pc = exg["pc_ts"].to_numpy()
    a, b, ex, ey, er, used = fit_envelope(dev, pc)
    fitted = a + b * dev
    t0 = fitted[0]
    fit = ClockFit(a=a, b=b, t0_epoch=t0, env_dev=ex, env_off=ey, env_used=used,
                   env_resid_ms=er * 1e3, latency_ms=(pc - fitted) * 1e3,
                   sync_err_ms=(exg["sync_ts"].to_numpy() - fitted) * 1e3)

    exg["t"] = fitted - t0
    if len(aux):
        aux["t"] = fit.dev_to_t(aux["dev_ts"].to_numpy())
    if len(live):
        live["t"] = live["pc_ts"].to_numpy() - t0

    gaps = []
    for name, df in (("ExG", exg), ("Aux", aux)):
        if len(df) < 3:
            continue
        g = find_gaps(df["dev_ts"].to_numpy())
        g.insert(0, "stream", name)
        g["t_start"] = fit.dev_to_t(g["dev_start"])
        g["t_end"] = fit.dev_to_t(g["dev_end"])
        g["duration_s"] = g["t_end"] - g["t_start"]
        gaps.append(g)
    session.gaps = pd.concat(gaps, ignore_index=True) if gaps else pd.DataFrame()
    session.clock = fit
    return fit


def with_breaks(t: np.ndarray, y: np.ndarray, max_dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN points where consecutive t differ by more than max_dt (so plotted lines break)."""
    dt = np.diff(t)
    i = np.where(dt > max_dt)[0]
    if len(i) == 0:
        return t, y
    t2 = np.insert(t.astype(float), i + 1, t[i] + dt[i] / 2)
    y2 = np.insert(y.astype(float), i + 1, np.nan)
    return t2, y2
