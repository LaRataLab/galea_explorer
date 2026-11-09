"""Plotly figures and summary text for the interactive quality report (quality_report.ipynb).

Every figure is built from raw (non-derived) data and measured at run time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

from . import preprocess as pp
from .io import channel_group
from .annotations import analysis_range, valid_range, valid_range_info
from .qc import channel_table, railed_mask, suggest_valid_range

GROUP_COLORS = {"EEG": "#2a6fdb", "EMG": "#d1495b", "EOG": "#8a5cc2", "Aux": "#2e8b57"}
SEV_COLORS = {"error": "#f8d7da", "warn": "#fff3cd", "info": "", "ok": "#d4edda"}
GRADE_COLORS = {"good": "#d4edda", "ok": "#fff3cd", "poor": "#f8d7da", "n/a": "#eee"}
TEMPLATE = "plotly_white"


def _gap_bands(fig, s, rows=None, stream="ExG"):
    if s.gaps is None or not len(s.gaps):
        return
    for _, g in s.gaps[s.gaps.stream == stream].iterrows():
        kw = dict(fillcolor="#999", opacity=0.3, line_width=0, layer="below")
        if rows is None:
            fig.add_vrect(x0=g.t_start, x1=g.t_end, **kw)
        else:
            for r in rows:
                fig.add_vrect(x0=g.t_start, x1=g.t_end, row=r, col=1, **kw)


def _range_marks(fig, s, rows=None):
    """Confirmed valid range (solid green lines) or, if none, the suggested start (dashed grey)."""
    vr = valid_range(s)
    if vr is not None:
        marks = [("valid from", vr[0], "#2e7d32", "solid", "top right"), ("valid to", vr[1], "#2e7d32", "solid", "top left")]
    else:
        sug = suggest_valid_range(s)
        marks = [("suggested start (not confirmed)", sug["t0"], "#777", "dash", "top right")]
        if sug["info"].get("ppg_contact") is not None:
            marks.append(("PPG contact", sug["info"]["ppg_contact"], "#b36b00", "dot", "top left"))
    dur = float(s.exg["t"].iloc[-1])
    for name, x, c, dash, pos in marks:
        if not x or x >= dur:
            continue
        kw = dict(x=x, line=dict(color=c, width=1.5, dash=dash))
        lab = dict(annotation_text=name, annotation_position=pos, annotation_font_color=c, annotation_font_size=10)
        if rows is None:
            fig.add_vline(**kw, **lab)
        else:
            for i, r in enumerate(rows):
                fig.add_vline(row=r, col=1, **kw, **(lab if i == 0 else {}))


def valid_range_text(s) -> str:
    """One line: the confirmed valid range, or 'not confirmed' plus the suggestion."""
    vr = valid_range(s)
    if vr is not None:
        info = valid_range_info(s)
        return (f"t = {vr[0]:.1f}–{vr[1]:.1f} s ({vr[1] - vr[0]:.0f} s), confirmed "
                f"{info.get('confirmed_at', '?')[:16].replace('T', ' ')}")
    sug = suggest_valid_range(s)
    return ("not confirmed" + (f" (suggested t > {sug['t0']:.0f} s: {'; '.join(sug['reasons'])})" if sug["t0"] > 0 else ""))


# ----------------------------------------------------------------------------- session
def session_summary(s) -> pd.DataFrame:
    exg, aux = s.exg, s.aux
    rows = [
        ("session", s.name),
        ("folder", s.folder),
        ("start (recording local time)", f"{s.t0_wall():%Y-%m-%d %H:%M:%S %z}"),
        ("duration", f"{exg['t'].iloc[-1]:.1f} s"),
        ("ExG", f"{len(s.exg_channels)} channels × {len(exg)} samples, "
                f"{1 / np.median(np.diff(exg['t'])):.1f} Hz"
                + (f" (dropped all-zero: {', '.join(s.dropped_channels)})" if s.dropped_channels else "")),
        ("Aux", f"{len(s.aux_channels)} channels × {len(aux)} samples, {1 / np.median(np.diff(aux['t'])):.1f} Hz"
         if len(aux) else "missing"),
        ("Signal-check (GUI)", f"{len(s.live)} updates" if len(s.live) else "missing"),
        ("valid range", valid_range_text(s)),
    ]
    rows += [(f"file: {k}", v.split("\\")[-1].split("/")[-1] if v else "— missing —") for k, v in s.files.items()]
    return pd.DataFrame(rows, columns=["item", "value"]).set_index("item")


# ----------------------------------------------------------------------------- clock
def fig_clock(s, step=5) -> go.Figure:
    fit, e = s.clock, s.exg
    k = slice(None, None, step)
    dev = e.dev_ts.to_numpy()
    off = lambda v: (v - dev - fit.a) * 1e3
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.65, .35], vertical_spacing=.05)
    f.add_scattergl(x=e.t[k], y=off(e.pc_ts.to_numpy())[k], mode="markers", marker=dict(size=2, color="#9bb7e0"),
                    name="PC arrival ts − device ts", row=1, col=1)
    f.add_scattergl(x=e.t[k], y=off(e.sync_ts.to_numpy())[k], mode="lines", line=dict(color="#d1495b", width=1),
                    name="vendor Timestamp − device ts", row=1, col=1)
    f.add_scatter(x=fit.dev_to_t(fit.env_dev), y=(fit.env_off - fit.a) * 1e3, mode="markers",
                  marker=dict(color="black", size=5), name="per-window minimum (envelope)", row=1, col=1)
    f.add_scatter(x=e.t[k], y=off(fit.dev_to_epoch(dev))[k], line=dict(color="#2e8b57"),
                  name=f"fit: pc = a + {fit.b:.5f}·dev", row=1, col=1)
    f.add_scattergl(x=e.t[k], y=np.maximum(fit.latency_ms[k], 0.1), mode="markers",
                    marker=dict(size=2, color="#9bb7e0"), showlegend=False, row=2, col=1)
    _gap_bands(f, s, rows=[1, 2])
    lo, hi = np.percentile(off(e.pc_ts.to_numpy()), [0.1, 99.5])
    f.update_yaxes(title="offset (ms, rel. to a)", range=[lo - 200, hi + 100], row=1, col=1)
    f.update_yaxes(title="latency above fit (ms)", type="log", row=2, col=1)
    f.update_xaxes(title="t (s)", row=2, col=1)
    f.update_layout(height=560, template=TEMPLATE, legend=dict(orientation="h", y=1.08))
    return f


# ----------------------------------------------------------------------------- railing
def railed_fraction_map(s, bin_s=1.0) -> pd.DataFrame:
    """% railed samples per ExG channel per time bin (rows: channels, columns: bin start s)."""
    t = s.exg["t"].to_numpy()
    b = np.floor(t / bin_s) * bin_s
    data = {ch: railed_mask(s, ch) for ch in s.exg_channels}
    df = pd.DataFrame(data).groupby(b).mean() * 100
    return df.T


def fig_railing_map(s, bin_s=1.0) -> go.Figure:
    m = railed_fraction_map(s, bin_s)
    f = go.Figure(go.Heatmap(z=m.to_numpy(), x=m.columns + bin_s / 2, y=m.index, colorscale="Reds", zmin=0, zmax=100,
                             colorbar=dict(title="% railed"),
                             hovertemplate="%{y} t=%{x:.0f} s: %{z:.0f} % railed<extra></extra>"))
    _gap_bands(f, s)
    _range_marks(f, s)
    f.update_layout(template=TEMPLATE, height=120 + 18 * len(m), margin=dict(t=40),
                    xaxis_title="t (s)", yaxis=dict(autorange="reversed"),
                    title=f"Railed samples (|x| ≥ 90 % of full scale) per {bin_s:g} s")
    return f


def fig_railing_bars(s) -> go.Figure:
    ct = channel_table(s)
    ct = ct[ct.group != "Aux"]
    f = go.Figure()
    f.add_bar(x=ct.index, y=ct["% railed"], name="raw data (≥ 90 % full scale)",
              marker_color=[GROUP_COLORS.get(g, "#444") for g in ct.group])
    if len(s.live):
        gui = [s.live[f"railed_{c}"].mean() if f"railed_{c}" in s.live else np.nan for c in ct.index]
        f.add_bar(x=ct.index, y=gui, name="GUI signal-check 'railed %' (mean)", marker_color="#ccc",
                  marker_line_color="#888", marker_line_width=1)
    f.update_layout(template=TEMPLATE, height=330, barmode="group", yaxis_title="% of session",
                    legend=dict(orientation="h", y=1.12), margin=dict(t=40), title="Railing per channel, whole session")
    return f


# ----------------------------------------------------------------------------- channel quality
def style_grades(cq: pd.DataFrame):
    fmt = {"valid_s": "{:.1f}", "rms_1_40_uV": "{:.1f}", "line_ratio": "{:.1f}", "kurtosis": "{:.0f}",
           "drift_uV": "{:.0f}"}
    return (cq.style.format({k: v for k, v in fmt.items() if k in cq}, na_rep="—")
            .apply(lambda col: [f"background-color: {GRADE_COLORS.get(v, '')}" for v in col], subset=["grade"]))


def fig_psd(s, t0=None, t1=None, fmax=125.0, nperseg=1024) -> go.Figure:
    """Welch PSD per ExG channel, high-pass 0.5 Hz, no notch (line noise visible), one panel per group."""
    t = s.exg["t"].to_numpy()
    a0, a1, _ = analysis_range(s)
    t0 = a0 if t0 is None else t0
    t1 = a1 if t1 is None else t1
    sel = (t >= t0) & (t <= t1)
    groups = [g for g in ("EEG", "EMG", "EOG") if any(channel_group(c) == g for c in s.exg_channels)]
    f = make_subplots(rows=1, cols=len(groups), subplot_titles=groups, shared_yaxes=True, horizontal_spacing=0.04)
    for j, g in enumerate(groups, 1):
        for ch in [c for c in s.exg_channels if channel_group(c) == g]:
            y = pp.process(ch, t, s.exg[ch].to_numpy(float), 250.0, "high-pass 0.5 Hz", None, True)[sel]
            y = y[np.isfinite(y)]
            if len(y) < nperseg:
                continue
            fr, P = signal.welch(y, fs=250.0, nperseg=nperseg)
            m = fr <= fmax
            f.add_scatter(x=fr[m], y=P[m], name=ch, mode="lines", line=dict(width=1), row=1, col=j,
                          hovertemplate=ch + " %{x:.1f} Hz: %{y:.3g}<extra></extra>")
        f.update_xaxes(title_text="Hz", row=1, col=j)
    f.update_yaxes(type="log", title_text="µV²/Hz", row=1, col=1)
    f.update_layout(template=TEMPLATE, height=420, margin=dict(t=50),
                    title=f"PSD, t = {t0:.0f}–{t1:.0f} s (high-pass 0.5 Hz, no notch)")
    return f


# ----------------------------------------------------------------------------- aux
def fig_aux(s, step=1) -> go.Figure | None:
    aux = s.aux
    if not len(aux):
        return None
    t = aux["t"].to_numpy()
    panels = [(c, aux[c].to_numpy(float)) for c in ("EDA", "PPG Red", "PPG IR") if c in aux]
    acc = [c for c in aux if c.startswith("Accelerometer")]
    gyr = [c for c in aux if c.startswith("Gyroscope")]
    if acc:
        panels.append(("accel |raw| (file units)", np.linalg.norm(aux[acc].to_numpy(float), axis=1)))
    if gyr:
        panels.append(("gyro |raw| (file units)", np.linalg.norm(aux[gyr].to_numpy(float), axis=1)))
    if not panels:
        return None
    f = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.025,
                      subplot_titles=[p[0] for p in panels])
    for i, (name, y) in enumerate(panels, 1):
        f.add_scattergl(x=t[::step], y=y[::step], mode="lines", line=dict(width=1, color="#2e8b57"), name=name,
                        showlegend=False, row=i, col=1)
    rows = list(range(1, len(panels) + 1))
    _gap_bands(f, s, rows=rows, stream="Aux")
    _range_marks(f, s, rows=rows)
    f.update_xaxes(title_text="t (s)", row=len(panels), col=1)
    f.update_layout(template=TEMPLATE, height=150 * len(panels) + 60, margin=dict(t=40))
    f.update_annotations(font_size=11)
    return f


def constant_channels(s) -> pd.DataFrame:
    ct = channel_table(s)
    c = ct[(ct.group == "Aux") & (ct["n unique"] <= 2)]
    return c[["unit", "min", "max", "n unique"]]


# ----------------------------------------------------------------------------- signal check
def fig_signalcheck(s) -> go.Figure | None:
    """GUI signal-check uVrms per channel over time (log colour)."""
    live = s.live
    cols = [c for c in live.columns if c.startswith("uVrms_")]
    if not len(live) or not cols:
        return None
    z = np.log10(np.clip(live[cols].to_numpy(float).T, 1e-1, None))
    f = go.Figure(go.Heatmap(z=z, x=live["t"], y=[c[6:] for c in cols], colorscale="Viridis",
                             colorbar=dict(title="log10 µVrms"), customdata=live[cols].to_numpy().T,
                             hovertemplate="%{y} t=%{x:.1f} s: %{customdata:.0f} µVrms<extra></extra>"))
    _range_marks(f, s)
    f.update_layout(template=TEMPLATE, height=120 + 18 * len(cols), margin=dict(t=40), xaxis_title="t (s)",
                    yaxis=dict(autorange="reversed"), title="GUI signal-check µVrms over time")
    return f


# ----------------------------------------------------------------------------- summary
def recommendations(s, q: pd.DataFrame, cq: pd.DataFrame) -> str:
    """Markdown bullet list derived from the QC results."""
    dur = float(s.exg["t"].iloc[-1])
    out = []
    vr = valid_range(s)
    if vr is not None:
        out.append(f"**Valid range: t = {vr[0]:.1f}–{vr[1]:.1f} s** ({vr[1] - vr[0]:.0f} s of {dur:.0f} s, confirmed). "
                   "Channel grades above are computed over it.")
    else:
        sug = suggest_valid_range(s)
        out.append("**Valid range not confirmed**: channel grades above use the whole recording, including any setup "
                   "period. " + (f"Suggested t > {sug['t0']:.0f} s ({'; '.join(sug['reasons'])}). " if sug["t0"] > 0 else "")
                   + "Confirm or adjust it in the browser (section 6).")
    if s.gaps is not None and len(s.gaps):
        g = s.gaps[s.gaps.stream == "ExG"]
        if len(g):
            out.append("Exclude / split at dropouts: " + ", ".join(f"{r.t_start:.1f}–{r.t_end:.1f} s"
                                                                  for r in g.itertuples()) + ".")
    if "grade" in cq:
        for grp in cq.group.unique():
            sub = cq[cq.group == grp]
            order = sub.assign(_g=sub.grade.map({"good": 0, "ok": 1, "poor": 2, "n/a": 3})).sort_values(
                ["_g", "rms_1_40_uV"])
            best = [c for c in order.index if order.grade[c] in ("good", "ok")][:3]
            poor = list(order.index[order.grade == "poor"])
            line = [c for c in sub.index if sub.line_ratio[c] > 10]
            txt = f"**{grp}**: best {', '.join(best) or '—'}"
            if poor:
                txt += f"; avoid {', '.join(poor)}"
            if line:
                txt += f"; strong 60 Hz on {', '.join(line)} (notch)"
            out.append(txt + ".")
    missing = [k for k in ("aux", "live", "impedance") if not s.files.get(k)]
    if missing:
        out.append("Missing files: " + ", ".join(missing) + ".")
    if s.files.get("impedance") and not len(s.impedance):
        out.append("No impedance measurements were saved: run the GUI impedance check before recording.")
    if "marker" in s.exg and not (s.exg["marker"] != 0).any():
        out.append("No event markers: epoching needs markers (or an external log aligned via the PC clock).")
    out.append("Timestamps: use the common clock `t` (fitted), not the vendor `Timestamp` column.")
    return "\n".join(f"- {o}" for o in out)
