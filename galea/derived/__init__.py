"""Derived signals, computed on demand (not used by the quality report).

* emg_envelope   – EMG linear envelope ("EMG<n> env", µV, 250 Hz, columns of s.exg)
* head_motion    – IMU fusion: "Linear accel |a|" (g), "Gyro |ω|" (deg/s), "Pitch", "Roll" (deg)
* ppg_cardioresp – "Heart rate" (bpm), "Breathing (PPG)" (z), "Breathing phase" (deg) from the PPG

All are listed in s.derived_channels (browser group "Derived").
"""
from __future__ import annotations

from ..io import DERIVED_SUFFIX, channel_group


def add_derived(s) -> None:
    """Compute every derived channel the session's streams allow; idempotent."""
    if s.derived_channels:
        return
    from . import emg_envelope, head_motion, ppg_cardioresp
    t = s.exg["t"].to_numpy()
    for ch in s.exg_channels:
        if channel_group(ch) == "EMG":
            name = ch + DERIVED_SUFFIX
            s.exg[name] = emg_envelope.emg_envelope(ch, t, s.exg[ch].to_numpy(float))
            s.derived_channels.append(name)
            s.units[name] = "uV"
    aux = s.aux
    if all(c in aux for c in head_motion.ACC + head_motion.GYR) and len(aux) > 100:
        hm = head_motion.head_motion(aux)
        s.imu_cal = hm.pop("_cal")
        for name, v in hm.items():
            aux[name] = v
            s.derived_channels.append(name)
        s.units.update({"Linear accel |a|": "g", "Gyro |ω|": "deg/s", "Pitch": "deg", "Roll": "deg"})
    if "PPG IR" in aux and len(aux) > 100:
        r = ppg_cardioresp.breathing_from_ppg(aux)
        aux["Heart rate"], aux["Breathing (PPG)"], aux["Breathing phase"] = r["hr"], r["breath"], r["phase_deg"]
        s.derived_channels += ["Heart rate", "Breathing (PPG)", "Breathing phase"]
        s.units.update({"Heart rate": "bpm", "Breathing (PPG)": "z", "Breathing phase": "deg"})
        s.resp_info = r["info"]
