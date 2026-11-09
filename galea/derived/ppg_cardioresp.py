"""Heart beats and breathing cycle from the PPG.

The PPG's dominant oscillation is the cardiac pulse. Breathing modulates it:
heart rate (respiratory sinus arrhythmia) and pulse amplitude oscillate at the
breathing rate. A breathing signal is built from these two beat-to-beat series:

    B = z(HR_band) + sign · z(pulse amplitude_band, aligned to HR)

with both band-passed to RESP_BAND, and the amplitude series shifted by the lag
and sign of its cross-correlation with HR (pulse amplitude typically leads HR by
~0.5 s and is anti-correlated). Phase (Hilbert): 0° = maximum of breathing-linked
heart rate, which by RSA physiology falls near the end of inspiration; ±180° near
the end of expiration (inferred, not measured — there is no airflow reference).

Computed per contiguous run with stable skin contact (PPG level within 10 % of its median),
split at dropouts; NaN elsewhere.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator

RESP_BAND = (0.15, 0.5)   # Hz (9–30 breaths/min)
FS_R = 10.0               # internal rate for beat series, Hz
MIN_RUN_S = 20.0


def ppg_beats(t, x, fs=50.0):
    """Beat times (max upslope of the pulse) and per-beat pulse amplitude.

    The PPG polarity is chosen so the systolic upstroke is the short edge.
    Returns dict(t_beat, amp, polarity, y) with implausible intervals removed.
    """
    sos = signal.butter(3, [0.5 / (fs / 2), 4.0 / (fs / 2)], "band", output="sos")
    best = None
    for pol in (1, -1):
        y = signal.sosfiltfilt(sos, pol * (x - np.median(x)))
        sd = 1.4826 * np.median(np.abs(y - np.median(y)))     # robust to contact transients
        pk, _ = signal.find_peaks(y, distance=int(0.33 * fs), prominence=0.5 * sd)
        tr, _ = signal.find_peaks(-y, distance=int(0.33 * fs), prominence=0.5 * sd)
        if len(pk) < 5 or len(tr) < 5:
            continue
        rise = [p - tr[tr < p][-1] for p in pk if np.any(tr < p)]
        score = np.median(rise) / np.median(np.diff(pk))   # short upstroke → small score
        if best is None or score < best[0]:
            best = (score, pol, y, pk, tr)
    if best is None:
        return dict(t_beat=np.array([]), amp=np.array([]), polarity=0, y=None)
    _, pol, y, pk, tr = best
    dy = np.gradient(y)
    tb, amp = [], []
    for p in pk:
        prev = tr[tr < p]
        if not len(prev) or p - prev[-1] > 0.6 * fs:
            continue
        a0 = prev[-1]
        i = a0 + int(np.argmax(dy[a0:p + 1]))
        off = 0.0
        if 0 < i < len(dy) - 1:                       # sub-sample refinement (parabola on derivative)
            d0, d1, d2 = dy[i - 1], dy[i], dy[i + 1]
            den = d0 - 2 * d1 + d2
            off = 0.5 * (d0 - d2) / den if den != 0 else 0.0
        tb.append(np.interp(i + off, np.arange(len(t)), t))
        amp.append(y[p] - y[a0])
    tb, amp = np.array(tb), np.array(amp)
    if len(tb) < 3:
        return dict(t_beat=tb, amp=amp, polarity=pol, y=y)
    ibi = np.diff(tb)
    keep = np.r_[True, (np.abs(ibi / np.median(ibi) - 1) < 0.35) & (ibi > 0.33) & (ibi < 2.0)]
    return dict(t_beat=tb[keep], amp=amp[keep], polarity=pol, y=y)


def beat_series_on_grid(tb, values, grid, max_gap=2.5):
    """Interpolate a per-beat series onto grid (monotone cubic); NaN across long gaps / outside."""
    if len(tb) < 4:
        return np.full(len(grid), np.nan)
    out = PchipInterpolator(tb, values, extrapolate=False)(grid)
    i = np.clip(np.searchsorted(tb, grid), 1, len(tb) - 1)
    out[(tb[i] - tb[i - 1]) > max_gap] = np.nan
    return out


def _bandpass(y, lo, hi, fs):
    sos = signal.butter(2, [lo / (fs / 2), hi / (fs / 2)], "band", output="sos")
    return signal.sosfiltfilt(sos, y)


def _z(y):
    return (y - np.mean(y)) / np.std(y)


def _xcorr_lag(x, y, maxlag_s, fs):
    """Lag (s) and r at max |r| of corr(x(t), y(t+τ))."""
    x, y = _z(x), _z(y)
    n, L = len(x), int(maxlag_s * fs)
    best = (0.0, 0.0)
    for k in range(-L, L + 1):
        r = np.corrcoef(x[max(0, -k): n - max(0, k)], y[max(0, k): n - max(0, -k)])[0, 1]
        if abs(r) > abs(best[1]):
            best = (k / fs, r)
    return best


def contact_runs(t, ppg, fs=50.0, min_len_s=MIN_RUN_S, gap_s=0.1):
    """(t0, t1) runs with stable PPG skin contact (2-s mean within 10 % of the median level), split at gaps."""
    from scipy.ndimage import uniform_filter1d
    lvl = uniform_filter1d(ppg.astype(float), int(2 * fs))
    ok = np.abs(lvl / np.median(ppg) - 1) <= 0.10
    ok &= np.r_[True, np.diff(t) < gap_s]           # a gap starts a new run
    runs, start = [], None
    for i in range(len(t)):
        if ok[i] and start is None:
            start = i
        brk = (i + 1 == len(t)) or (not ok[i + 1]) or (t[i + 1] - t[i] >= gap_s)
        if start is not None and brk:
            if t[i] - t[start] >= min_len_s:
                runs.append((t[start] + 2.0, t[i] - 0.5))   # trim contact transitions
            start = None
    return runs


def breathing_from_ppg(aux, ppg_col="PPG IR"):
    """Per-sample (aux rate) heart rate, breathing signal and breathing phase; NaN where unavailable.

    Returns dict(hr, breath, phase_deg, runs, info) — info lists per run: beats, lag, r, rate.
    """
    t = aux["t"].to_numpy()
    x = aux[ppg_col].to_numpy(float)
    hr_out = np.full(len(t), np.nan)
    b_out = np.full(len(t), np.nan)
    ph_out = np.full(len(t), np.nan)
    info = []
    for t0, t1 in contact_runs(t, x):
        m = (t >= t0 - 1) & (t <= t1 + 0.5)
        bt = ppg_beats(t[m], x[m])
        tb = bt["t_beat"]
        if len(tb) < 10:
            continue
        g = np.arange(t0, t1, 1.0 / FS_R)
        ibi = np.diff(tb)
        good = (ibi > 0.33) & (ibi < 2.0) & (np.abs(ibi / np.median(ibi) - 1) < 0.35)   # skips intervals over dropped beats
        hr = beat_series_on_grid(tb[1:][good], 60.0 / ibi[good], g)
        amp = beat_series_on_grid(tb, bt["amp"], g)
        ok = np.isfinite(hr) & np.isfinite(amp)
        if ok.sum() < MIN_RUN_S * FS_R:
            continue
        hr_f, amp_f = np.interp(g, g[ok], hr[ok]), np.interp(g, g[ok], amp[ok])
        hb, ab = _bandpass(hr_f, *RESP_BAND, FS_R), _bandpass(amp_f, *RESP_BAND, FS_R)
        lag, r = _xcorr_lag(hb, ab, 3.0, FS_R)
        B = _z(hb) + np.sign(r) * _z(np.interp(g + lag, g, ab))
        ph = np.angle(signal.hilbert(B))
        B[~ok], ph[~ok] = np.nan, np.nan
        sel = (t >= t0) & (t <= t1)
        hr_out[sel] = np.interp(t[sel], g, hr)               # NaN propagates across beat gaps
        b_out[sel] = np.interp(t[sel], g, B)
        # interpolate phase via unit vector (avoid wrap artefacts)
        c = np.interp(t[sel], g, np.cos(ph))
        s_ = np.interp(t[sel], g, np.sin(ph))
        ph_out[sel] = np.degrees(np.arctan2(s_, c))
        ph_out[sel & ~np.isfinite(b_out)] = np.nan
        pk, _ = signal.find_peaks(np.nan_to_num(B), distance=int(2.0 * FS_R), prominence=0.5)
        rate = 60.0 / np.median(np.diff(g[pk])) if len(pk) > 2 else np.nan
        info.append(dict(t0=t0, t1=t1, beats=len(tb), hr_amp_lag_s=lag, hr_amp_r=r, breaths_per_min=rate))
    return dict(hr=hr_out, breath=b_out, phase_deg=ph_out, info=info)
