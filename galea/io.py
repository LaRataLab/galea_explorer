"""Loading of a Galea / OpenBCI GUI recording folder into a Session object.

Files handled (found by prefix glob in the session folder):
  BrainFlow-RAW_*.csv                 ExG stream, 250 Hz, tab-separated, no header
  OpenBCI-RAW-Aux-*.txt               Aux stream (EDA, PPG, IMU...), 50 Hz
  OpenBCI-SignalCheck-Live-*.txt      uVrms / railed % per channel, variable rate, PC ms clock
  OpenBCI-SignalCheck-Impedance-*.txt impedance table (often empty)
  Timestamp_*.txt                     BrainFlow's RTT pings before streaming (kept for reference)

Derived signals (EMG envelopes, head motion, heart rate / breathing) are optional:
load_session(folder, derived=True) or galea.derived.add_derived(s).
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Column map of the BrainFlow CSV. The PDF guide omits the `Timestamp` column (index 27).
EXG_CHANNELS = (
    ["EMG0", "EMG1", "EMG2", "EMG3", "EOG0", "EOG1", "EMG4", "EMG5"]
    + [f"EEG{i}" for i in range(16)]
)
BRAINFLOW_COLUMNS = ["idx", *EXG_CHANNELS, "pc_ts", "dev_ts", "sync_ts", "marker"]

# ADC full scale in uV: 4.5 V / gain. EEG gain 12, EMG/EOG gain 4.
RAIL_UV = {"EEG": 4.5 / 12 * 1e6, "EMG": 4.5 / 4 * 1e6, "EOG": 4.5 / 4 * 1e6}
RAIL_FRACTION = 0.90  # OpenBCI GUI convention: |x| >= 90 % of full scale counts as railed

AUX_RENAME = {
    "Sample Index": "idx",
    "Raw PC Timestamp": "pc_ts",
    "Raw Device Timestamp": "dev_ts",
    "Timestamp": "sync_ts",
    "Marker": "marker",
    "Timestamp (Formatted)": "ts_formatted",
}
AUX_UNITS = {
    "EDA": "uS?", "PPG Red": "a.u.", "PPG IR": "a.u.", "Board Temperature": "degC",
    "Battery": "%", "Accelerometer X": "g", "Accelerometer Y": "g", "Accelerometer Z": "g",
    "Gyroscope X": "deg/s", "Gyroscope Y": "deg/s", "Gyroscope Z": "deg/s",
    "Magnetometer X": "uT?", "Magnetometer Y": "uT?", "Magnetometer Z": "uT?",
}

# Names of derived channels (computed in galea.derived), recognised here so groups/colours work.
DERIVED_SUFFIX = " env"   # EMG envelopes: "EMG0 env"
DERIVED_AUX = ("Linear accel |a|", "Gyro |ω|", "Pitch", "Roll", "Heart rate", "Breathing (PPG)", "Breathing phase")


def channel_group(name: str) -> str:
    if name.endswith(DERIVED_SUFFIX) or name in DERIVED_AUX:
        return "Derived"
    for g in ("EEG", "EMG", "EOG"):
        if name.startswith(g):
            return g
    return "Aux"


@dataclass
class Session:
    folder: str
    exg: pd.DataFrame            # ExG samples: idx, channels, pc_ts, dev_ts, sync_ts, marker (+ t after clock)
    aux: pd.DataFrame            # Aux samples, same time columns
    live: pd.DataFrame           # signal-check live: uVrms_<ch>, railed_<ch>, pc_ts
    impedance: pd.DataFrame
    sync_pings: pd.DataFrame
    exg_channels: list[str]
    aux_channels: list[str]
    dropped_channels: list[str] = field(default_factory=list)
    startup_rows_dropped: int = 0
    files: dict[str, str] = field(default_factory=dict)
    clock: object = None         # galea.clock.ClockFit, set by load_session
    gaps: pd.DataFrame = None    # gap table in t (s)
    units: dict[str, str] = field(default_factory=dict)
    imu_cal: object = None       # galea.derived.head_motion.ImuCalibration, set by load_session
    derived_channels: list[str] = field(default_factory=list)  # computed signals (columns in exg or aux)
    resp_info: list = field(default_factory=list)              # per-run breathing extraction summary
    tz: str | None = None        # timezone for wall-clock display (None: recording's UTC offset, else system local)
    utc_offset_h: float | None = None   # recording PC's UTC offset, inferred from the formatted timestamps

    @property
    def name(self) -> str:
        """Session name = folder name."""
        return os.path.basename(os.path.normpath(self.folder))

    @property
    def channels(self) -> dict[str, list[str]]:
        """Channels grouped for the browser."""
        groups: dict[str, list[str]] = {}
        for c in self.exg_channels:
            groups.setdefault(channel_group(c), []).append(c)
        groups["Aux"] = list(self.aux_channels)
        groups["Derived"] = list(self.derived_channels)
        groups["SignalCheck"] = [c for c in self.live.columns if c.startswith(("uVrms_", "railed_"))]
        return groups

    def frame_of(self, ch: str) -> pd.DataFrame:
        if ch in self.exg.columns:
            return self.exg
        if ch in self.aux.columns:
            return self.aux
        if ch in self.live.columns:
            return self.live
        raise KeyError(ch)

    def t0_wall(self) -> pd.Timestamp:
        """Wall-clock time of t = 0: in self.tz if set, else in the recording PC's local time
        (UTC offset inferred from the files), else in this machine's local timezone."""
        import datetime as dt
        ts = pd.Timestamp(self.clock.t0_epoch, unit="s", tz="UTC")
        if self.tz:
            return ts.tz_convert(self.tz)
        if self.utc_offset_h is not None:
            return ts.tz_convert(dt.timezone(dt.timedelta(hours=self.utc_offset_h)))
        return ts.tz_convert(dt.datetime.now().astimezone().tzinfo)


def find_sessions(root: str) -> list[str]:
    """Session folders (containing a BrainFlow-RAW_*.csv) under root, recursively, sorted by name."""
    hits = glob.glob(os.path.join(root, "**", "BrainFlow-RAW_*.csv"), recursive=True)
    return sorted({os.path.dirname(h) for h in hits})


def resolve_data(data: str | None = None, root: str | None = None) -> str:
    """Session folder to use: `data` if given, else $GALEA_DATA, else the first session under root (default ./data)."""
    data = data or os.environ.get("GALEA_DATA")
    if data:
        if not os.path.isdir(data):
            raise FileNotFoundError(data)
        sessions = find_sessions(data)
        if not sessions:
            raise FileNotFoundError(f"No BrainFlow-RAW_*.csv under {data}")
        if os.path.normpath(sessions[0]) != os.path.normpath(data) and len(sessions) > 1:
            print(f"{len(sessions)} sessions under {data}; using {sessions[0]}")
        return sessions[0]
    root = root or "data"
    sessions = find_sessions(root)
    if not sessions:
        raise FileNotFoundError(f"No session folder under {os.path.abspath(root)}; set DATA or $GALEA_DATA")
    return sessions[0]


def _utc_offset_h(df: pd.DataFrame) -> float | None:
    """Recording PC's UTC offset (h, rounded to 15 min) from 'ts_formatted' (local) vs 'pc_ts' (epoch)."""
    if df is None or not len(df) or "ts_formatted" not in df or "pc_ts" not in df:
        return None
    try:
        loc = pd.to_datetime(df["ts_formatted"].iloc[:50].str.strip())
        utc = pd.to_datetime(df["pc_ts"].iloc[:50], unit="s")
        h = float(np.median((loc - utc).dt.total_seconds())) / 3600
        return round(h * 4) / 4
    except Exception:
        return None


def _find(folder: str, pattern: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(folder, pattern)))
    return hits[0] if hits else None


def _read_openbci_txt(path: str) -> pd.DataFrame:
    # 4 '%' header lines. Do NOT use comment='%' — the Live header contains "(railed %)".
    df = pd.read_csv(path, skiprows=4, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def _startup_glitch_rows(dev: np.ndarray, n_check: int = 10, max_scan: int = 2000) -> int:
    """Number of leading rows whose device-timestamp steps are not ~1 sample period."""
    dd = np.diff(dev[: max_scan + n_check + 1])
    med = np.median(np.diff(dev))
    ok = (dd > 0.5 * med) & (dd < 1.5 * med)
    for i in range(len(ok) - n_check):
        if ok[i : i + n_check].all():
            return i
    return 0


def read_exg(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=BRAINFLOW_COLUMNS, dtype=np.float64)
    df["idx"] = df["idx"].astype(np.int16)
    return df


def load_session(folder: str, drop_zero_channels: bool = True, drop_startup: bool = True,
                 fit_clock: bool = True, derived: bool = False, tz: str | None = None) -> Session:
    """Load a session folder. derived=True also computes the derived channels (galea.derived);
    tz sets the display timezone (default: system local)."""
    files = {
        "exg": _find(folder, "BrainFlow-RAW_*.csv"),
        "aux": _find(folder, "OpenBCI-RAW-Aux-*.txt"),
        "live": _find(folder, "OpenBCI-SignalCheck-Live-*.txt"),
        "impedance": _find(folder, "OpenBCI-SignalCheck-Impedance-*.txt"),
        "sync": _find(folder, "Timestamp_*.txt"),
        "openbci_raw": _find(folder, "OpenBCI-RAW-2*.txt"),
    }
    if files["exg"] is None:
        raise FileNotFoundError(f"No BrainFlow-RAW_*.csv in {folder}")

    exg = read_exg(files["exg"])
    n_start = _startup_glitch_rows(exg["dev_ts"].to_numpy()) if drop_startup else 0
    exg = exg.iloc[n_start:].reset_index(drop=True)

    exg_channels = list(EXG_CHANNELS)
    dropped = []
    if drop_zero_channels:
        dropped = [c for c in exg_channels if (exg[c] == 0).all()]
        exg = exg.drop(columns=dropped)
        exg_channels = [c for c in exg_channels if c not in dropped]

    aux = pd.DataFrame()
    aux_channels: list[str] = []
    if files["aux"]:
        aux = _read_openbci_txt(files["aux"]).rename(columns=AUX_RENAME)
        aux_channels = [c for c in aux.columns if c not in AUX_RENAME.values()]
        if drop_startup:
            aux = aux[aux["dev_ts"] >= exg["dev_ts"].iloc[0]].reset_index(drop=True)

    live = pd.DataFrame()
    if files["live"]:
        raw = _read_openbci_txt(files["live"])
        ren = {}
        for c in raw.columns:
            if c.endswith("(uVrms)"):
                ren[c] = "uVrms_" + c.replace(" Channel ", "").replace(" (uVrms)", "")
            elif c.endswith("(railed %)"):
                ren[c] = "railed_" + c.replace(" Channel ", "").replace(" (railed %)", "")
            elif c == "Timestamp":
                ren[c] = "pc_ms"
            elif c == "Timestamp (Formatted)":
                ren[c] = "ts_formatted"
        live = raw.rename(columns=ren)
        # The GUI writes a row every ~16 ms but values only update every ~48 ms: keep value changes only.
        vals = live[[c for c in live.columns if c.startswith(("uVrms_", "railed_"))]].to_numpy()
        changed = np.r_[True, (np.diff(vals, axis=0) != 0).any(axis=1)]
        live.attrs["rows_raw"] = len(live)
        live = live[changed].reset_index(drop=True)
        live["pc_ts"] = live["pc_ms"] / 1000.0

    impedance = _read_openbci_txt(files["impedance"]) if files["impedance"] else pd.DataFrame()
    pings = pd.DataFrame()
    if files["sync"]:
        with open(files["sync"]) as f:
            pings = pd.DataFrame([json.loads(l) for l in f if l.strip()])

    units = {c: "uV" for c in exg_channels}
    units.update({c: AUX_UNITS.get(c, "") for c in aux_channels})
    units.update({c: ("uVrms" if c.startswith("uVrms") else "%") for c in live.columns
                  if c.startswith(("uVrms_", "railed_"))})

    s = Session(folder=folder, exg=exg, aux=aux, live=live, impedance=impedance, sync_pings=pings,
                exg_channels=exg_channels, aux_channels=aux_channels, dropped_channels=dropped,
                startup_rows_dropped=n_start, files=files, units=units, tz=tz,
                utc_offset_h=_utc_offset_h(aux) if len(aux) else _utc_offset_h(live))
    if fit_clock:
        from . import clock
        clock.apply(s)
        if derived:
            from .derived import add_derived
            add_derived(s)
    return s
