"""Interactive stacked-signal browser (ipywidgets + plotly FigureWidget).

    from galea.browser import SignalBrowser
    SignalBrowser(session)          # displays itself in a notebook cell
    SignalBrowser(session, groups=["EEG", "EOG"])   # restrict the channel groups offered

Groups offered by default: every group the session has. Derived channels exist only if the
session was loaded with derived=True (the quality report loads without them).

Every selected channel is drawn on the common clock `t`, offset vertically and
normalised so traces don't overlap; the y tick label carries the channel name
and its scale (units per division). Hover shows the real (filtered) value.

Scaling: by default each channel is centred (median) and scaled (1st–99th percentile
range) on the *visible window*, recomputed on every zoom/pan, so slow drifts (EDA)
stay in their lane. The "whole session" modes use one fixed scale per channel.

Clip: "clip to window" restricts the session to the plotted time range. Filters,
whole-session scales and the "full" button then use only that range (e.g. to leave
out the setup period); "unclip" restores the full recording. The current clip is
available as `browser.clip` (None or (t0, t1) in s).

Valid range (annotate=True): the part of the recording the user confirms as usable,
stored with the dataset (galea.annotations, `<session>/galea_session.json`). It is set by the
user: "view suggestion" zooms to the suggested range (galea.qc.suggest_valid_range,
dashed outline); adjust the view by zooming / start+window, then "mark view as valid" saves
it (solid green edges; the excluded part is tinted). "clear" makes it undefined again.
`browser.on_valid_change(fn)` registers fn(range_or_None), e.g. to refresh dependent outputs.

Zooming re-fetches data for the visible range with min/max decimation, so full
250 Hz detail appears when you zoom in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ipywidgets as W
import plotly.graph_objects as go

from . import preprocess as pp
from . import annotations as ann
from .io import channel_group
from .style import BAD, GOOD, GROUP_COLORS, MUTED, TEMPLATE, WARN

POINTS_PER_TRACE = 2500


class SignalBrowser:
    def __init__(self, session, channels: list[str] | None = None, height_per_channel: int = 70,
                 width: int = 1300, groups: list[str] | None = None, annotate: bool = True):
        self.s = session
        self._cache: dict = {}
        self._busy = False
        self.clip: tuple[float, float] | None = None
        self.annotate = annotate
        self._valid_cbs: list = []
        self.valid = ann.valid_range(session) if annotate else None
        self._suggest = None
        groups = {g: c for g, c in session.channels.items() if c and (groups is None or g in groups)}
        self.groups = groups
        default = channels or self._default_channels(groups)

        # --- channel pickers, one per group
        self.pickers = {}
        for g, chs in groups.items():
            if not chs:
                continue
            self.pickers[g] = W.SelectMultiple(
                options=chs, value=[c for c in default if c in chs], rows=min(10, len(chs)),
                description="", layout=W.Layout(width="150px" if g not in ("SignalCheck", "Derived") else "170px"))
        pick_box = W.HBox([W.VBox([W.HTML(f"<b style='color:{GROUP_COLORS.get(g, '#444')}'>{g}</b>"), p])
                           for g, p in self.pickers.items()])
        clear = W.Button(description="clear", layout=W.Layout(width="70px"))
        clear.on_click(lambda _: self._set_selection([]))
        presets = W.Dropdown(options=["— preset —"] + [k for k, v in self._presets().items() if v],
                             layout=W.Layout(width="170px"))
        presets.observe(self._on_preset, "value")
        self._preset = presets

        # --- processing / display controls
        st = {"description_width": "90px"}
        self.exg_filter = W.Dropdown(options=list(pp.FILTERS), value="high-pass 0.5 Hz", description="ExG filter", style=st)
        self.aux_filter = W.Dropdown(options=list(pp.FILTERS), value="raw", description="Aux/derived", style=st)
        self.notch = W.Checkbox(value=True, description="notch 60/120 Hz (ExG)", indent=False)
        self.mask_rails = W.Checkbox(value=True, description="blank railed samples", indent=False)
        self.scaling = W.Dropdown(options=["per channel · window", "per group · window",
                                           "per channel · whole session", "per group · whole session"],
                                  value="per channel · window", description="scaling", style=st,
                                  layout=W.Layout(width="320px"))
        self.gain = W.FloatLogSlider(value=1.0, base=10, min=-2, max=2, step=0.05, description="amplitude ×",
                                     style=st, readout_format=".2f")
        self.xmode = W.Dropdown(options=["t (s)", "wall clock"], value="t (s)", description="x axis", style=st)
        self.shade = W.Checkbox(value=True, description="shade dropouts", indent=False)
        self.hover = W.Dropdown(options=["closest", "x unified"], value="closest", description="hover", style=st)
        self.width = W.IntSlider(value=width, min=600, max=2600, step=50, description="width (px)", style=st)
        self.hpc = W.IntSlider(value=height_per_channel, min=20, max=200, step=5, description="height/ch", style=st)

        # --- time navigation
        self._dur = float(session.exg["t"].iloc[-1])
        self.t_start = W.BoundedFloatText(value=0.0, min=-5, max=self._dur, step=1, description="start (s)", style=st,
                                          layout=W.Layout(width="190px"))
        self.t_win = W.BoundedFloatText(value=self._dur, min=0.05, max=self._dur + 10, step=1, description="window (s)",
                                        style=st, layout=W.Layout(width="190px"))
        go_btn = W.Button(description="go", layout=W.Layout(width="50px"))
        prev_btn = W.Button(description="◀", layout=W.Layout(width="40px"))
        next_btn = W.Button(description="▶", layout=W.Layout(width="40px"))
        full_btn = W.Button(description="full", layout=W.Layout(width="50px"), tooltip="whole session (or clip)")
        clip_btn = W.Button(description="clip to window", layout=W.Layout(width="115px"), button_style="warning",
                            tooltip="restrict the session to the plotted time range")
        unclip_btn = W.Button(description="unclip", layout=W.Layout(width="70px"), tooltip="restore full recording")
        go_btn.on_click(lambda _: self._goto(self.t_start.value, self.t_win.value))
        prev_btn.on_click(lambda _: self._goto(self.t_start.value - self.t_win.value, self.t_win.value))
        next_btn.on_click(lambda _: self._goto(self.t_start.value + self.t_win.value, self.t_win.value))
        full_btn.on_click(lambda _: self._goto(*self._bounds_start_width()))
        clip_btn.on_click(lambda _: self.set_clip(*self._range))
        unclip_btn.on_click(lambda _: self.set_clip(None))
        self.status = W.HTML()
        self.clip_label = W.HTML()
        self.valid_label = W.HTML()
        view_sug_btn = W.Button(description="view suggestion", layout=W.Layout(width="125px"),
                                tooltip="zoom to the suggested valid range (or to the confirmed one)")
        mark_btn = W.Button(description="mark view as valid", layout=W.Layout(width="140px"), button_style="success",
                            tooltip="confirm the plotted time range as the valid range; saved with the dataset")
        clear_valid_btn = W.Button(description="clear", layout=W.Layout(width="60px"),
                                   tooltip="remove the confirmed valid range (undefined again)")
        view_sug_btn.on_click(lambda _: self._view_suggestion())
        mark_btn.on_click(lambda _: self.mark_valid(*self._range))
        clear_valid_btn.on_click(lambda _: self.clear_valid())

        self.fig = go.FigureWidget()
        self.fig.update_layout(template=TEMPLATE, margin=dict(l=150, r=20, t=30, b=40),
                               showlegend=False, dragmode="zoom")
        self.fig.layout.on_change(self._on_relayout, "xaxis.range")

        for w in [*self.pickers.values(), self.exg_filter, self.aux_filter, self.notch, self.mask_rails,
                  self.scaling, self.gain, self.xmode, self.shade, self.hover, self.width, self.hpc]:
            w.observe(lambda ch: self.redraw(), "value")

        controls = W.VBox([
            W.HBox([self.exg_filter, self.aux_filter, self.scaling]),
            W.HBox([self.notch, self.mask_rails, self.shade, self.gain]),
            W.HBox([self.xmode, self.hover, self.width, self.hpc]),
            W.HBox([self.t_start, self.t_win, go_btn, prev_btn, next_btn, full_btn, self.status]),
            W.HBox([clip_btn, unclip_btn, self.clip_label]),
        ] + ([W.HBox([W.HTML(f"<b style='color:{GOOD}'>valid range</b>&nbsp;"), view_sug_btn, mark_btn,
                      clear_valid_btn, self.valid_label])] if annotate else []))
        self.widget = W.VBox([W.HBox([pick_box, W.VBox([presets, clear])]), controls, self.fig])
        self._range = (0.0, self._dur)
        self._update_clip_label()
        self._update_valid_label()
        self.redraw()

    # ------------------------------------------------------------------ helpers
    def _ipython_display_(self):
        from IPython.display import display
        display(self.widget)

    def _set_selection(self, chs):
        self._busy = True
        for g, p in self.pickers.items():
            p.value = [c for c in p.options if c in chs]
        self._busy = False
        self.redraw()

    @staticmethod
    def _default_channels(groups) -> list[str]:
        """First 2 EEG, first EMG and EOG, EDA and PPG IR — whatever exists."""
        out = groups.get("EEG", [])[:2] + groups.get("EMG", [])[:1] + groups.get("EOG", [])[:1]
        return out + [c for c in ("EDA", "PPG IR") if c in groups.get("Aux", [])]

    def _presets(self) -> dict[str, list[str]]:
        g = self.groups
        aux, der = g.get("Aux", []), g.get("Derived", [])
        motion = ("Linear accel |a|", "Gyro |ω|", "Pitch", "Roll")
        return {
            "all EEG": g.get("EEG", []),
            "EMG + EOG": g.get("EMG", []) + g.get("EOG", []),
            "all ExG": [c for k in ("EMG", "EOG", "EEG") for c in g.get(k, [])],
            "physio (EDA, PPG)": [c for c in ("EDA", "PPG Red", "PPG IR") if c in aux],
            "IMU (raw)": [c for c in aux if c.startswith(("Accel", "Gyro", "Magnet"))],
            "board (temp, battery)": [c for c in aux if c in ("Board Temperature", "Battery")],
            "EEG signal-check": [c for c in g.get("SignalCheck", []) if c.startswith("uVrms_EEG")],
            "railed % (signal-check)": [c for c in g.get("SignalCheck", []) if c.startswith("railed_")],
            "heart & breathing": [c for c in ("PPG IR", "Heart rate", "Breathing (PPG)", "Breathing phase")
                                  if c in der or (c in aux and "Heart rate" in der)],
            "head motion": [c for c in der if c in motion],
            "EMG envelopes": [c for c in der if c.endswith(" env")],
        }

    def _on_preset(self, change):
        sel = self._presets().get(change["new"])
        if sel is not None:
            self._set_selection(sel)
        self._preset.value = "— preset —"

    def selected(self) -> list[str]:
        return [c for p in self.pickers.values() for c in p.value]

    def _kind(self, ch):
        if ch in self.s.derived_channels:
            return "derived"
        if ch in self.s.exg.columns:
            return "exg"
        if ch in self.s.aux.columns:
            return "aux"
        return "live"

    # ------------------------------------------------------------------ clip
    def _bounds(self) -> tuple[float, float]:
        return self.clip if self.clip is not None else (0.0, self._dur)

    def _bounds_start_width(self) -> tuple[float, float]:
        a, b = self._bounds()
        return a, b - a

    def set_clip(self, t0: float | None, t1: float | None = None):
        """Restrict the session to [t0, t1] s (None: full recording) and redraw."""
        if t0 is None:
            self.clip = None
        else:
            a, b = max(float(t0), 0.0), min(float(t1), self._dur)
            if b - a < 0.1:
                return
            self.clip = (a, b)
        self._cache.clear()
        self._update_clip_label()
        self._goto(*self._bounds_start_width())

    def _update_clip_label(self):
        if self.clip is None:
            self.clip_label.value = f"<span style='color:{MUTED}'>session: full recording</span>"
        else:
            a, b = self.clip
            self.clip_label.value = (f"<span style='color:{WARN}'><b>clipped</b> to {a:.2f}–{b:.2f} s "
                                     f"({b - a:.1f} s): filters, whole-session scales and 'full' use this range</span>")

    # ------------------------------------------------------------------ valid range
    @property
    def suggestion(self) -> dict:
        if self._suggest is None:
            from .qc import suggest_valid_range
            self._suggest = suggest_valid_range(self.s)
        return self._suggest

    def on_valid_change(self, fn):
        """Call fn(valid_range_or_None) whenever the valid range is marked or cleared."""
        self._valid_cbs.append(fn)

    def _notify_valid(self):
        self._update_valid_label()
        self.redraw()
        for fn in self._valid_cbs:
            fn(self.valid)

    def mark_valid(self, t0: float, t1: float):
        """Confirm [t0, t1] as the valid range and save it with the dataset."""
        try:
            self.valid = ann.set_valid_range(self.s, t0, t1)
        except ValueError as e:
            self.valid_label.value = f"<span style='color:{BAD}'>{e}</span>"
            return
        self._notify_valid()

    def clear_valid(self):
        ann.clear_valid_range(self.s)
        self.valid = None
        self._notify_valid()

    def _view_suggestion(self):
        a, b = self.valid if self.valid is not None else (self.suggestion["t0"], self.suggestion["t1"])
        self._goto(a, b - a)

    def _update_valid_label(self):
        if not self.annotate:
            return
        if self.valid is not None:
            a, b = self.valid
            self.valid_label.value = (f"<span style='color:{GOOD}'><b>confirmed</b> {a:.2f}–{b:.2f} s ({b - a:.1f} s), "
                                      f"saved in {ann.ANNOT_FILE}</span>")
        else:
            sg = self.suggestion
            hint = (f"suggested {sg['t0']:.1f}–{sg['t1']:.1f} s ({'; '.join(sg['reasons'])})" if sg["t0"] > 0
                    else "no setup period detected; suggested: whole recording")
            self.valid_label.value = f"<span style='color:{WARN}'><b>not confirmed</b> · {hint}</span>"

    def _valid_shapes(self) -> list[dict]:
        if not self.annotate:
            return []
        x = lambda v: self._to_x(np.array([v]))[0]
        if self.valid is not None:
            a, b = self.valid
            out = [dict(type="line", xref="x", yref="paper", x0=x(v), x1=x(v), y0=0, y1=1,
                        line=dict(color=GOOD, width=2)) for v in (a, b)]
            for lo, hi in ((min(0.0, a), a), (b, max(self._dur, b))):
                if hi > lo:
                    out.append(dict(type="rect", xref="x", yref="paper", x0=x(lo), x1=x(hi), y0=0, y1=1,
                                    fillcolor=BAD, opacity=0.07, line_width=0, layer="below"))
            return out
        sg = self.suggestion
        return [dict(type="rect", xref="x", yref="paper", x0=x(sg["t0"]), x1=x(sg["t1"]), y0=0, y1=1,
                     fillcolor="rgba(0,0,0,0)", line=dict(color=GOOD, width=1.5, dash="dash"), layer="above")]

    # ------------------------------------------------------------------ data
    def _data(self, ch):
        """(t, y_processed, fs, unit, scale, centre), cached on processing settings and clip."""
        kind = self._kind(ch)
        if kind == "exg":
            key = (ch, self.exg_filter.value, self.notch.value, self.mask_rails.value)
        elif kind in ("aux", "derived"):
            key = (ch, self.aux_filter.value)
        else:
            key = (ch,)
        if key not in self._cache:
            df = self.s.frame_of(ch)
            t = df["t"].to_numpy()
            x = df[ch].to_numpy(float)
            if self.clip is not None:
                keep = (t >= self.clip[0]) & (t <= self.clip[1])
                t, x = t[keep], x[keep]
            if kind == "exg":
                y = pp.process(ch, t, x, 250.0, self.exg_filter.value, 60.0 if self.notch.value else None,
                               self.mask_rails.value)
                fs = 250.0
            elif kind == "aux":
                y = pp.process(ch, t, x, 50.0, self.aux_filter.value, None, False)
                fs = 50.0
            elif kind == "derived":               # already computed: only the Aux/derived filter applies
                fs = 250.0 if ch in self.s.exg.columns else 50.0
                y = pp.process(ch, t, x, fs, self.aux_filter.value, None, False)
            else:
                y, fs = x, 20.0
            cen = float(np.nanmedian(y)) if np.isfinite(y).any() else 0.0
            self._cache[key] = (t, y, fs, self.s.units.get(ch, ""), pp.robust_scale(y), cen)
        return self._cache[key]

    def _stats(self, ch, t0, t1):
        """(scale, centre) of a channel over the visible window, or the whole (clipped) session."""
        t, y, fs, unit, sc_all, cen_all = self._data(ch)
        if "whole" in self.scaling.value:
            return sc_all, cen_all
        a, b = np.searchsorted(t, [t0, t1])
        yy = y[a:b]
        if not np.isfinite(yy).any():
            return sc_all, cen_all
        return pp.robust_scale(yy), float(np.nanmedian(yy))

    def _to_x(self, t):
        if self.xmode.value == "t (s)":
            return t
        t0 = self.s.t0_wall().tz_localize(None)
        return (t0 + pd.to_timedelta(t, unit="s")).to_numpy()

    def _from_x(self, v):
        if self.xmode.value == "t (s)":
            return float(v)
        t0 = self.s.t0_wall().tz_localize(None)
        return (pd.Timestamp(v) - t0).total_seconds()

    def _window(self, t, y, fs, t0, t1):
        a, b = np.searchsorted(t, [t0, t1])
        a, b = max(a - 1, 0), min(b + 1, len(t))
        tt, yy = t[a:b], y[a:b]
        outs_t, outs_y = [], []
        for sl in pp.segments(tt, 3.0 / fs if fs > 25 else 1.0):
            i = pp.minmax_decimate(tt[sl], yy[sl], POINTS_PER_TRACE)
            outs_t += [tt[sl][i], [np.nan]]
            outs_y += [yy[sl][i], [np.nan]]
        if not outs_t:
            return np.array([]), np.array([])
        return np.concatenate(outs_t[:-1]), np.concatenate(outs_y[:-1])

    def _group(self, ch):
        if ch in self.s.live.columns:
            return "SignalCheck"
        return channel_group(ch)

    def _compute_layout(self, chs, t0, t1):
        stats = {ch: self._stats(ch, t0, t1) for ch in chs}
        scales = {ch: stats[ch][0] for ch in chs}
        if self.scaling.value.startswith("per group"):
            gmax = {}
            for ch in chs:
                g = self._group(ch)
                gmax[g] = max(gmax.get(g, 0), scales[ch])
            scales = {ch: gmax[self._group(ch)] for ch in chs}
        gain = self.gain.value
        n = len(chs)
        self._layout = [(ch, float(n - 1 - i), scales[ch] / gain, stats[ch][1])  # (ch, offset, scale, centre)
                        for i, ch in enumerate(chs)]

    def _ticktext(self):
        return [f"<b>{ch}</b><br><span style='font-size:9px;color:{MUTED}'>{sc:.3g} {self.s.units.get(ch, '')}/div</span>"
                for ch, off, sc, cen in self._layout]

    # ------------------------------------------------------------------ drawing
    def redraw(self):
        if self._busy:
            return
        chs = self.selected()
        n = len(chs)
        t0, t1 = self._range
        self._compute_layout(chs, t0, t1)

        traces = []
        for ch, off, sc, cen in self._layout:
            t, y, fs, unit, *_ = self._data(ch)
            tw, yw = self._window(t, y, fs, t0, t1)
            g = self._group(ch)
            traces.append(go.Scattergl(
                x=self._to_x(tw), y=(yw - cen) / sc + off, customdata=yw, name=ch, mode="lines",
                line=dict(width=1, color=GROUP_COLORS.get(g, "#444"),
                          shape="hv" if g == "SignalCheck" else "linear"),
                hovertemplate=f"<b>{ch}</b> %{{customdata:.4g}} {unit}<br>%{{x}}<extra></extra>"))
        self._busy = True
        self.fig.data = []
        if traces:
            self.fig.add_traces(traces)
        with self.fig.batch_update():
            self.fig.update_layout(
                width=self.width.value, height=max(320, self.hpc.value * n + 110), hovermode=self.hover.value,
                yaxis=dict(tickvals=[l[1] for l in self._layout], ticktext=self._ticktext(), range=[-1, max(n, 1)],
                           zeroline=False, showgrid=True, gridcolor="#eee", fixedrange=False),
                xaxis=dict(title="time since start (s)" if self.xmode.value == "t (s)" else
                           f"local time ({self.s.t0_wall():%Y-%m-%d}, UTC{self.s.t0_wall():%z})",
                           range=[self._to_x(np.array([t0]))[0], self._to_x(np.array([t1]))[0]],
                           showspikes=True, spikemode="across", spikethickness=1),
            )
            shapes = []
            if self.shade.value and self.s.gaps is not None:
                for _, r in self.s.gaps[self.s.gaps.stream == "ExG"].iterrows():
                    xs = self._to_x(np.array([r.t_start, r.t_end]))
                    shapes.append(dict(type="rect", xref="x", yref="paper", x0=xs[0], x1=xs[1], y0=0, y1=1,
                                       fillcolor="#999", opacity=0.25, line_width=0, layer="below"))
            self.fig.layout.shapes = shapes + self._valid_shapes()
        self._busy = False
        self._set_status()

    def _set_status(self):
        t0, t1 = self._range
        self.status.value = f"<span style='color:{MUTED}'>{len(self._layout)} channels · {t0:.2f}–{t1:.2f} s</span>"

    def _refresh_data(self):
        """Re-fetch decimated data for the current x range; in window-scaling mode also rescale."""
        t0, t1 = self._range
        windowed = "window" in self.scaling.value
        if windowed:
            self._compute_layout([l[0] for l in self._layout], t0, t1)
        with self.fig.batch_update():
            if windowed:
                self.fig.layout.yaxis.ticktext = self._ticktext()
            for tr, (ch, off, sc, cen) in zip(self.fig.data, self._layout):
                t, y, fs, *_ = self._data(ch)
                tw, yw = self._window(t, y, fs, t0, t1)
                tr.x, tr.y, tr.customdata = self._to_x(tw), (yw - cen) / sc + off, yw
        self._set_status()

    def _on_relayout(self, layout, xrange):
        if xrange is None or self._busy:
            return
        try:
            t0, t1 = self._from_x(xrange[0]), self._from_x(xrange[1])
        except Exception:
            return
        if (t0, t1) == self._range:
            return
        self._range = (t0, t1)
        self._busy = True
        self.t_start.value = round(min(max(t0, self.t_start.min), self.t_start.max), 3)
        self.t_win.value = round(min(max(t1 - t0, 0.05), self.t_win.max), 3)
        self._busy = False
        self._refresh_data()

    def _goto(self, start, width):
        self._range = (float(start), float(start + width))
        self._busy = True
        self.t_start.value = round(min(max(start, self.t_start.min), self.t_start.max), 3)
        self.t_win.value = round(min(max(width, 0.05), self.t_win.max), 3)
        self._busy = False
        self.redraw()
