"""Visual identity (LaRata LAB / AKCIT purple) shared by all figures and widgets.

Importing this module registers the plotly template "galea" (plotly_white + brand colours).

Channel-group colours were checked for colour-vision deficiency (OKLab ΔE ×100 under
protan/deutan/tritan simulation): every pair ≥ 8 except EMG–Aux (7.1), which is acceptable
because channels are always labelled by name as well.
Status colours (findings severity, channel grades, valid range) stay conventional
green / yellow / red and are not part of the brand palette.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

BRAND = "#330585"        # AKCIT purple (banner background)
BRAND_ALT = "#392d7b"    # LaRata LAB purple
ACCENT = "#e0609a"       # pink, for highlights (peaks, thresholds)
INK = "#2e2b36"          # text and neutral marks
MUTED = "#8a8496"        # secondary text, reference lines
LIGHT = "#cfc8e6"        # light lavender, for dense background points

GROUP_COLORS = {
    "EEG": "#4b2bb8",          # violet
    "EMG": "#e0609a",          # pink
    "EOG": "#0b6a62",          # dark teal
    "Aux": "#c98a00",          # amber
    "Derived": "#6aa8e6",      # light blue
    "SignalCheck": INK,        # near-black
}
COLORWAY = [GROUP_COLORS[g] for g in ("EEG", "EMG", "EOG", "Aux", "Derived")] + ["#8f7ad8", "#8a5a2b", INK]

# sequential, one hue (white → brand purple), for magnitude heatmaps
SEQUENTIAL = [[0.0, "#f7f5fc"], [0.35, "#c3b4ec"], [0.7, "#7550cf"], [1.0, BRAND]]

# status (reserved; not brand colours)
GOOD = "#2e7d32"
WARN = "#b36b00"
BAD = "#b01e2e"

TEMPLATE = "galea"
_t = go.layout.Template(pio.templates["plotly_white"])
_t.layout.colorway = COLORWAY
_t.layout.font = dict(color=INK)
_t.layout.title = dict(font=dict(color=BRAND))
_t.layout.colorscale = dict(sequential=SEQUENTIAL)
pio.templates[TEMPLATE] = _t
