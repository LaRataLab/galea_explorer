"""Automated data-quality checks for a loaded Session."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .annotations import analysis_range, valid_range, valid_range_info
from .io import EXG_CHANNELS, RAIL_FRACTION, RAIL_UV, channel_group, read_exg


def railed_mask(session, ch: str) -> np.ndarray:
    x = session.exg[ch].to_numpy()
    return np.abs(x) >= RAIL_FRACTION * RAIL_UV[channel_group(ch)]


def channel_table(session) -> pd.DataFrame:
    """Per-channel stats: ExG railing / flatness, Aux constancy."""
    rows = []
    for ch in session.exg_channels:
        x = session.exg[ch].to_numpy()
        rail = RAIL_UV[channel_group(ch)]
        r = railed_mask(session, ch)
        good = x[~r]
        rows.append({
            "channel": ch, "group": channel_group(ch), "unit": "uV",
            "rail (+/-)": rail,
            "% railed": 100 * r.mean(),
            "% at exact rail": 100 * (np.abs(np.abs(x) - rail) < 1).mean(),
            "% zero": 100 * (x == 0).mean(),
            "min": x.min(), "max": x.max(),
            "median (non-railed)": np.median(good) if len(good) else np.nan,
            "std (non-railed)": good.std() if len(good) else np.nan,
            "n unique": len(np.unique(x)),
        })
    for ch in session.aux_channels:
        x = session.aux[ch].to_numpy()
        rows.append({
            "channel": ch, "group": "Aux", "unit": session.units.get(ch, ""),
            "% zero": 100 * (x == 0).mean(), "min": x.min(), "max": x.max(),
            "median (non-railed)": np.median(x), "std (non-railed)": x.std(),
            "n unique": len(np.unique(x)),
        })
    return pd.DataFrame(rows).set_index("channel")


SETTLE_MARGIN_S = 15.0   # after the last railed sample


def settle_info(session) -> dict:
    """Heuristic estimate of when headset setup ends (input to suggest_valid_range only).

    Returns dict(rails_end, ppg_contact, eda_flat_end, settle) in s: last railed ExG sample,
    start of stable PPG skin contact, end of the initial flat EDA (each None if not applicable),
    and settle = max(rails_end + SETTLE_MARGIN_S, ppg_contact), 0 if neither applies.
    """
    t = session.exg["t"].to_numpy()
    rails = np.zeros(len(t), bool)
    for ch in session.exg_channels:
        rails |= railed_mask(session, ch)
    out = dict(rails_end=float(t[rails].max()) if rails.any() else None, ppg_contact=None, eda_flat_end=None)
    aux = session.aux
    if len(aux) > 100 and "PPG IR" in aux:
        from .derived.ppg_cardioresp import contact_runs
        runs = contact_runs(aux["t"].to_numpy(), aux["PPG IR"].to_numpy(float))
        if runs:
            out["ppg_contact"] = float(runs[0][0])
    if len(aux) > 100 and "EDA" in aux:
        x = aux["EDA"].to_numpy(float)
        k = int(np.argmax(np.abs(x - x[0]) > 0.005))
        if k > 50 * 5:
            out["eda_flat_end"] = float(aux["t"].iloc[k])
    cands = [0.0]
    if out["rails_end"] is not None:
        cands.append(out["rails_end"] + SETTLE_MARGIN_S)
    if out["ppg_contact"] is not None:
        cands.append(out["ppg_contact"])
    out["settle"] = min(max(cands), float(t[-1]))
    return out


def suggest_valid_range(session) -> dict:
    """A *suggested* valid range, for the user to confirm or adjust in the browser.

    Only a suggestion: the confirmed range is stored by galea.annotations. Returns dict(t0, t1, reasons, info): t0 = settle_info's estimate
    (later of last railing + 15 s and stable PPG contact), t1 = end of recording.
    """
    si = settle_info(session)
    why = []
    if si["rails_end"] is not None:
        why.append(f"last railed sample at {si['rails_end']:.1f} s (+{SETTLE_MARGIN_S:.0f} s)")
    if si["ppg_contact"] is not None:
        why.append(f"stable PPG contact from {si['ppg_contact']:.1f} s")
    return dict(t0=si["settle"], t1=float(session.exg["t"].iloc[-1]), reasons=why, info=si)


def channel_quality(session, t0: float | None = None, t1: float | None = None) -> pd.DataFrame:
    """Per-ExG-channel quality over [t0, t1] (default: the confirmed valid range, else the whole recording).

    rms_1_40_uV: SD of the 1–40 Hz band. line_ratio: mean PSD 58–62 Hz / mean PSD 40–56 Hz.
    kurtosis: excess kurtosis of the 1–40 Hz signal (spikes / pops; Gaussian = 0).
    drift_uV: SD of the 0.1–1 Hz band. grade (heuristic, relative to the channel's group):
    poor if kurtosis > 150, rms > 2× group median, or (EEG) rms > 50 µV;
    ok if kurtosis > 50, line ratio > 10 (notchable) or rms > 1.3× group median; good otherwise.
    """
    from scipy import signal as sg
    from scipy.stats import kurtosis
    from . import preprocess as pp
    exg = session.exg
    t = exg["t"].to_numpy()
    a0, a1, _ = analysis_range(session)
    t0 = a0 if t0 is None else t0
    t1 = a1 if t1 is None else t1
    sel = (t >= t0) & (t <= t1)
    fs = 250.0
    rows = []
    for ch in session.exg_channels:
        x = exg[ch].to_numpy(float)
        bp = pp.process(ch, t, x, fs, "band-pass 1–40 Hz (EEG)", None, True)[sel]
        ok = np.isfinite(bp)
        row = dict(channel=ch, group=channel_group(ch), valid_s=ok.sum() / fs)
        if ok.sum() >= fs * 5:
            hp = pp.process(ch, t, x, fs, "high-pass 0.5 Hz", None, True)[sel]
            lf = pp.process(ch, t, x, fs, ("bandpass", (0.1, 1.0)), None, True)[sel]
            b = bp[ok]
            f, P = sg.welch(np.nan_to_num(hp), fs=fs, nperseg=512)
            row.update(rms_1_40_uV=np.std(b),
                       line_ratio=P[(f >= 58) & (f <= 62)].mean() / P[(f >= 40) & (f <= 56)].mean(),
                       kurtosis=kurtosis(b), drift_uV=np.nanstd(lf))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("channel")
    if "rms_1_40_uV" not in df:
        df["grade"] = "n/a"
        return df
    med = df.groupby("group")["rms_1_40_uV"].transform("median")
    okish = (df["line_ratio"] > 10) | (df["kurtosis"] > 50) | (df["rms_1_40_uV"] > 1.3 * med)
    grade = np.where(okish, "ok", "good").astype(object)
    poor = (df["rms_1_40_uV"] > 2 * med) | (df["kurtosis"] > 150) | ((df["group"] == "EEG") & (df["rms_1_40_uV"] > 50))
    grade[poor.to_numpy()] = "poor"
    grade[df["rms_1_40_uV"].isna().to_numpy()] = "n/a"
    df["grade"] = grade
    return df


def qc_report(session) -> pd.DataFrame:
    """List of findings: one row per check, with severity and measured values."""
    out = []

    def add(check, severity, detail):
        out.append({"check": check, "severity": severity, "detail": detail})

    exg, aux, live, fit = session.exg, session.aux, session.live, session.clock

    # --- files
    if session.files.get("openbci_raw") is None:
        add("files", "info", "OpenBCI-RAW-<date>.txt missing (the BrainFlow CSV has the same ExG data).")
    for key, label in (("aux", "OpenBCI-RAW-Aux (EDA/PPG/IMU)"), ("live", "SignalCheck-Live"),
                       ("sync", "Timestamp_*.txt"), ("impedance", "SignalCheck-Impedance")):
        if session.files.get(key) is None:
            add("files", "warn" if key == "aux" else "info", f"{label} file missing.")
    if session.files.get("impedance") and len(session.impedance) == 0:
        add("files", "warn", "SignalCheck-Impedance file has a header only — no impedance measurements.")

    # --- durations / rates
    dur = exg["t"].iloc[-1]
    add("duration", "info", f"ExG {len(exg)} samples over {dur:.2f} s "
        f"({session.t0_wall():%Y-%m-%d %H:%M:%S.%f}"[:-3] + f" local → +{dur:.1f} s).")
    med_dev = np.median(np.diff(exg["dev_ts"]))
    med_t = np.median(np.diff(exg["t"]))
    add("sample rate", "info", f"ExG: {1/med_dev:.2f} Hz in device seconds, {1/med_t:.2f} Hz on the common clock (nominal 250).")
    if len(aux):
        add("sample rate", "info", f"Aux: {1/np.median(np.diff(aux['t'])):.2f} Hz on the common clock (nominal 50).")
        # Aux shares the device clock with ExG?
        on_grid = np.isin(aux["dev_ts"].to_numpy(), exg["dev_ts"].to_numpy()).mean()
        add("clock", "info", f"{100*on_grid:.1f} % of Aux device timestamps coincide exactly with an ExG sample → one shared device clock.")

    # --- startup glitch
    if session.startup_rows_dropped:
        raw = read_exg(session.files["exg"]).iloc[: session.startup_rows_dropped + 1]
        steps = np.diff(raw["dev_ts"]) * 1e3
        zero_rows = int((raw[[c for c in EXG_CHANNELS]].iloc[:-1] == 0).all(axis=1).sum())
        add("startup", "warn", f"First {session.startup_rows_dropped} ExG rows dropped: device-ts steps "
            f"{steps.min():.3f}–{np.median(steps):.3f} ms instead of {med_dev*1e3:.1f} ms (buffer flush); "
            f"{zero_rows} of them all-zero.")

    # --- gaps
    g = session.gaps
    if g is not None and len(g):
        for _, r in g.iterrows():
            add("gap", "error", f"{r['stream']}: {r['duration_s']:.3f} s dropout at t={r['t_start']:.2f}–{r['t_end']:.2f} s "
                f"(~{r['missing_samples']} samples lost).")
    else:
        add("gap", "ok", "No dropouts.")
    # sample-counter continuity (idx wraps at 256)
    di = np.diff(exg["idx"].to_numpy().astype(int)) % 256
    n_bad = int((di != 1).sum())
    add("sample counter", "ok" if n_bad == 0 else "warn",
        f"ExG idx (mod 256) discontinuities: {n_bad}" + (" (match the dropouts)." if n_bad == len(g[g.stream == 'ExG']) else "."))
    nonmono = int((np.diff(exg["dev_ts"]) <= 0).sum())
    add("device ts", "ok" if nonmono == 0 else "error", f"Non-increasing device timestamps: {nonmono}.")

    # --- clock
    s = fit.summary()
    add("clock drift", "warn", f"Device clock runs {s['device clock fast by (%)']:.3f} % fast vs PC "
        f"(slope {fit.b:.5f}) → {abs(dur*s['device clock fast by (%)']/100):.2f} s over this session if uncorrected.")
    add("clock fit", "info", f"Lower-envelope fit residual RMS {s['envelope residual RMS (ms)']:.1f} ms "
        f"({s['envelope points used']} windows); arrival latency median {s['arrival latency median (ms)']:.0f} ms, "
        f"p99 {s['arrival latency p99 (ms)']:.0f} ms above the envelope.")
    st = exg["sync_ts"].to_numpy()
    back = int((np.diff(st) < 0).sum())
    jumps = np.abs(np.diff(st) - np.diff(exg["t"].to_numpy()))
    add("vendor Timestamp", "error", f"'Timestamp' column is not usable: goes backwards {back}×, "
        f"steps deviate up to {jumps.max()*1e3:.0f} ms, error vs fitted clock spans "
        f"{fit.sync_err_ms.min():.0f}…{fit.sync_err_ms.max():.0f} ms.")
    if len(session.sync_pings):
        pdev = session.sync_pings["timestamp_device"].median()
        rtt = ", ".join(f"{r * 1e3:.0f}" for r in session.sync_pings["rtt"]) if "rtt" in session.sync_pings else "?"
        add("sync file", "info", f"Timestamp_*.txt = BrainFlow's 3 round-trip pings before streaming (RTT {rtt} ms). "
            f"Its device value ≈ {pdev:.1f} does not advance and is not on the data's device clock "
            f"(starts at {exg['dev_ts'].iloc[0]:.3f} s) — cannot anchor.")
    m = re.search(r"(\d{2}-\d{2}-\d{2})_\d+\.csv$", session.files["exg"])
    if m and m.group(1) != f"{session.t0_wall():%H-%M-%S}":
        add("file names", "info", f"BrainFlow file name time ({m.group(1)}) ≠ stream start "
            f"({session.t0_wall():%H-%M-%S}); formatted timestamps are local time, raw ones UTC epoch.")

    # --- channels
    ct = channel_table(session)
    if session.dropped_channels:
        add("channels", "warn", f"All-zero channels dropped: {', '.join(session.dropped_channels)} (only "
            f"{sum(c.startswith('EEG') for c in session.exg_channels)} EEG channels active).")
    exg_ct = ct[ct.group != "Aux"]
    worst = exg_ct["% railed"].sort_values(ascending=False)
    if worst.iloc[0] > 1:
        add("railing", "error" if worst.iloc[0] > 10 else "warn",
            "% samples ≥90 % of full scale: " + ", ".join(f"{c} {v:.1f}" for c, v in worst.items() if v > 1) + ".")
    else:
        add("railing", "ok", f"Railed samples ≤ 1 % on every channel (max {worst.index[0]} {worst.iloc[0]:.2f} %).")
    vr = valid_range(session)
    if vr is not None:
        info = valid_range_info(session)
        add("valid range", "ok", f"Confirmed t={vr[0]:.1f}–{vr[1]:.1f} s ({vr[1] - vr[0]:.0f} s; "
            f"confirmed {info.get('confirmed_at', '?')[:16].replace('T', ' ')}). Quality metrics use this range.")
    else:
        sug = suggest_valid_range(session)
        add("valid range", "warn", "Not confirmed: quality metrics use the whole recording. "
            + (f"Suggested t > {sug['t0']:.0f} s ({'; '.join(sug['reasons'])}). " if sug["t0"] > 0 else "")
            + "Confirm or adjust it in the browser (quality report, section 6).")
    a0, a1, label = analysis_range(session)
    cq = channel_quality(session, a0, a1)
    if "rms_1_40_uV" in cq:
        eeg = cq[cq.group == "EEG"]
        noisy = eeg.index[eeg["rms_1_40_uV"] > 50].tolist()
        add("noise", "warn" if noisy else "ok",
            f"1–40 Hz RMS over t={a0:.0f}–{a1:.0f} s ({label}): "
            + ", ".join(f"{c} {v:.0f}" for c, v in cq["rms_1_40_uV"].items())
            + " µV." + (f" EEG above 50 µV (suspect contact): {', '.join(noisy)}." if noisy else ""))
        poor = cq.index[cq.grade == "poor"].tolist()
        add("channel grades", "warn" if poor else "ok",
            "; ".join(f"{g}: " + ", ".join(f"{c} {cq.grade[c]}" for c in cq.index[cq.group == g])
                      for g in cq.group.unique()) + " (see channel_quality).")
    add("units", "info", "ExG values are µV: EEG rails at ±375 000 µV (4.5 V / gain 12), EMG/EOG at ±1 125 000 µV "
        "(gain 4). Large DC offsets → high-pass before viewing.")
    markers = int((exg["marker"] != 0).sum())
    add("markers", "info" if markers else "warn", f"Non-zero marker samples: {markers}.")

    # --- aux
    for ch in session.aux_channels:
        r = ct.loc[ch]
        x = aux[ch].to_numpy()
        if r["n unique"] <= 2:
            vals = ", ".join(f"{v:g}" for v in np.unique(x))
            add("aux constant", "warn", f"{ch} is constant (values: {vals}).")
        elif (x[:5] == 0).any() and r["% zero"] < 50:
            nz = int(np.argmax(x != 0))
            add("aux startup", "info", f"{ch} is 0 for the first {nz} samples ({nz/50:.2f} s), then live.")
    for ch in ("PPG Red", "PPG IR", "EDA"):
        if ch not in aux:
            continue
        # level shifts: compare 2 s medians before/after each point
        sec = aux.groupby(np.floor(aux["t"] / 2))[ch].median()
        ratio = (sec / sec.shift(1)).abs()
        k = ratio.idxmax()
        if ratio.max() > 3:
            add("aux level shift", "warn", f"{ch} level jumps ×{ratio.max():.1f} ({sec.shift(1)[k]:.4g} → {sec[k]:.4g}) "
                f"near t≈{2*k:.0f} s (sensor contact / LED current change?).")
    if "EDA" in aux:
        x = aux["EDA"].to_numpy()
        flat = int(np.argmax(np.abs(x - x[0]) > 0.005))
        if flat > 50 * 5:
            add("aux EDA", "info", f"EDA flat at {x[0]:.3f} for the first {aux['t'].iloc[flat]:.0f} s, then live "
                f"(ends at {x[-1]:.3f}; electrode settling?).")

    # --- live signal check
    if len(live):
        dtl = np.diff(live["t"])
        add("signal-check", "info", f"Live: {live.attrs.get('rows_raw', len(live))} rows written but only "
            f"{len(live)} distinct updates (rows repeated 1–4×); update interval median {np.median(dtl)*1e3:.0f} ms, "
            f"max {dtl.max():.2f} s; covers t={live['t'].iloc[0]:.2f}–{live['t'].iloc[-1]:.2f} s.")
        cmp = []
        for ch in session.exg_channels:
            col = f"railed_{ch}"
            if col in live:
                cmp.append((ch, live[col].mean(), 100 * railed_mask(session, ch).mean()))
        if cmp:
            c = pd.DataFrame(cmp, columns=["ch", "gui", "ours"])
            add("signal-check", "warn", "GUI 'railed %' disagrees with raw data (GUI mean vs ≥90 %FS fraction): "
                + ", ".join(f"{r.ch} {r.gui:.0f}/{r.ours:.0f}" for r in c.itertuples()) + ".")
    return pd.DataFrame(out)
