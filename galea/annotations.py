"""User annotations stored with the dataset, in `<session folder>/galea_session.json`.

Currently one annotation: the **valid range**, the part of the recording the user confirmed
as usable (e.g. after headset setup). It is never set automatically: the quality report
suggests one (`galea.qc.suggest_valid_range`), the user confirms or adjusts it in the
browser, and until then it is undefined (None).

Times are stored both as `t` (s on the common clock) and as PC epoch seconds; on reading,
the epoch is preferred, so the range stays put even if the loader's t = 0 changes.
"""
from __future__ import annotations

import datetime as dt
import json
import os

ANNOT_FILE = "galea_session.json"


def _path(session) -> str:
    return os.path.join(session.folder, ANNOT_FILE)


def load_annotations(session) -> dict:
    p = _path(session)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _save(session, data: dict):
    p = _path(session)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


def valid_range(session) -> tuple[float, float] | None:
    """Confirmed valid range (t0, t1) in s on the common clock, or None if not confirmed."""
    vr = load_annotations(session).get("valid_range")
    if not vr:
        return None
    t0e = getattr(session.clock, "t0_epoch", None)
    if t0e is not None and "epoch_start" in vr:
        a, b = vr["epoch_start"] - t0e, vr["epoch_end"] - t0e
    else:
        a, b = vr["t_start"], vr["t_end"]
    dur = float(session.exg["t"].iloc[-1])
    return max(0.0, float(a)), min(dur, float(b))


def valid_range_info(session) -> dict | None:
    """The stored valid-range record (with confirmation time and source), or None."""
    return load_annotations(session).get("valid_range") or None


def set_valid_range(session, t0: float, t1: float, source: str = "browser", note: str = "") -> tuple[float, float]:
    """Confirm [t0, t1] (s) as the valid range and write it to the session folder."""
    dur = float(session.exg["t"].iloc[-1])
    a, b = max(0.0, float(t0)), min(dur, float(t1))
    if b - a < 1.0:
        raise ValueError(f"valid range {a:.2f}–{b:.2f} s is shorter than 1 s")
    t0e = session.clock.t0_epoch
    wall = session.t0_wall()
    data = load_annotations(session)
    data["valid_range"] = {
        "t_start": round(a, 3), "t_end": round(b, 3),
        "epoch_start": round(t0e + a, 4), "epoch_end": round(t0e + b, 4),
        "wall_start": str(wall + dt.timedelta(seconds=a)), "wall_end": str(wall + dt.timedelta(seconds=b)),
        "confirmed_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source, "note": note,
    }
    _save(session, data)
    return a, b


def clear_valid_range(session):
    """Remove the confirmed valid range (back to undefined)."""
    data = load_annotations(session)
    if data.pop("valid_range", None) is not None:
        _save(session, data)


def analysis_range(session) -> tuple[float, float, str]:
    """(t0, t1, label) to compute quality metrics over: the confirmed valid range, else the whole recording."""
    vr = valid_range(session)
    if vr is not None:
        return vr[0], vr[1], "confirmed valid range"
    return 0.0, float(session.exg["t"].iloc[-1]), "whole recording (no valid range confirmed)"
