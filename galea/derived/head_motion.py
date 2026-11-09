"""IMU sensor fusion: separate gravity from head movement (6-axis Madgwick filter).

Standard approach for head/body-worn IMUs: a Madgwick (2010) gradient-descent
orientation filter integrates the gyroscope and corrects its drift towards the
accelerometer's gravity direction. Rotating gravity into the sensor frame and
subtracting it leaves the *linear* (movement) acceleration; its norm is an
orientation-independent movement signal (0 at rest).

The Galea sample data are not on correct physical scales (accelerometer norm at
rest ≈ 0.51 "g"; gyroscope ≈ 4× under-scaled vs deg/s — likely range-setting
mismatches), so each session is calibrated from its own data:

* gyro bias  – median gyro over the stillest 30 % of samples;
* g units    – median accelerometer norm over still samples (→ output in g);
* gyro scale – least-squares factor between the rate of change of the low-passed
               gravity direction and the rotation predicted by the gyro
               (d u/dt = −ω × u), on samples with movement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

ACC = [f"Accelerometer {a}" for a in "XYZ"]
GYR = [f"Gyroscope {a}" for a in "XYZ"]


@dataclass
class ImuCalibration:
    g_units: float            # accelerometer norm at rest (file units per g)
    gyro_bias: np.ndarray     # file units
    gyro_scale: float         # multiply file gyro (after bias) by this to get deg/s
    gyro_scale_r: float       # correlation of the scale fit (quality)


def calibrate(t, acc, gyr, fs=50.0, lp_hz=0.5) -> ImuCalibration:
    gn = np.linalg.norm(gyr, axis=1)
    still = gn < np.percentile(gn, 30)
    bias = np.median(gyr[still], axis=0)
    g_units = float(np.median(np.linalg.norm(acc[still], axis=1)))

    sos = signal.butter(2, lp_hz / (fs / 2), output="sos")
    al = signal.sosfiltfilt(sos, acc, axis=0)
    u = al / np.linalg.norm(al, axis=1)[:, None]
    du = np.gradient(u, 1 / fs, axis=0)
    w = signal.sosfiltfilt(sos, np.radians(gyr - bias), axis=0)
    pred = -np.cross(w, u)
    contig = np.r_[True, np.diff(t) < 2.0 / fs] & np.r_[np.diff(t) < 2.0 / fs, True]
    ok = (gn > np.percentile(gn, 60)) & contig
    k = float((pred[ok] * du[ok]).sum() / (pred[ok] ** 2).sum())
    r = float(np.corrcoef(pred[ok].ravel(), du[ok].ravel())[0, 1])
    return ImuCalibration(g_units=g_units, gyro_bias=bias, gyro_scale=k, gyro_scale_r=r)


def madgwick(t, acc, gyr_rad, beta=0.1, beta_init=2.5, init_s=1.0, gap_s=0.1):
    """Quaternion (w, x, y, z) per sample, sensor→earth. Re-initialises after gaps."""
    n = len(t)
    Q = np.empty((n, 4))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    t_init = t[0]
    for i in range(n):
        dt = t[i] - t[i - 1] if i else 0.02
        if dt > gap_s:                      # dropout: re-converge from the accelerometer
            t_init, dt = t[i], 0.02
        b = beta_init if t[i] - t_init < init_s else beta
        q0, q1, q2, q3 = q
        gx, gy, gz = gyr_rad[i]
        qd = 0.5 * np.array([-q1 * gx - q2 * gy - q3 * gz,
                             q0 * gx + q2 * gz - q3 * gy,
                             q0 * gy - q1 * gz + q3 * gx,
                             q0 * gz + q1 * gy - q2 * gx])
        a = acc[i]
        an = np.linalg.norm(a)
        if an > 0:
            ax, ay, az = a / an
            f = np.array([2 * (q1 * q3 - q0 * q2) - ax,
                          2 * (q0 * q1 + q2 * q3) - ay,
                          2 * (0.5 - q1 * q1 - q2 * q2) - az])
            J = np.array([[-2 * q2, 2 * q3, -2 * q0, 2 * q1],
                          [2 * q1, 2 * q0, 2 * q3, 2 * q2],
                          [0.0, -4 * q1, -4 * q2, 0.0]])
            s = J.T @ f
            sn = np.linalg.norm(s)
            if sn > 0:
                qd = qd - b * s / sn
        q = q + qd * dt
        q /= np.linalg.norm(q)
        Q[i] = q
    return Q


def gravity_in_sensor(Q):
    q0, q1, q2, q3 = Q.T
    return np.c_[2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2]


def pitch_roll(grav):
    """Head tilt (deg) from the fused gravity direction in the sensor frame.

    Assumed mounting (not documented, verify with a nod/tilt recording): sensor X → forward,
    Y → left, Z → up (at rest gravity lies along +Z in the sample data).
    pitch > 0: nose up (rotation about the ear-to-ear axis);
    roll  > 0: right ear down (rotation about the front-back axis).
    Yaw (left/right turn) is not available: there is no heading reference without the magnetometer.
    """
    gx, gy, gz = grav.T
    pitch = np.degrees(np.arctan2(gx, np.hypot(gy, gz)))
    roll = np.degrees(np.arctan2(gy, gz))
    return pitch, roll


def head_motion(aux: "pd.DataFrame", beta=0.1, cal: ImuCalibration | None = None) -> dict:
    """Derived head-motion channels: linear accel |a| (g), gyro |ω| (deg/s), pitch, roll (deg)."""
    lin, grav, cal = linear_acceleration(aux, beta=beta, cal=cal)
    w = (aux[GYR].to_numpy(float) - cal.gyro_bias) * cal.gyro_scale
    pitch, roll = pitch_roll(grav)
    return {"Linear accel |a|": np.linalg.norm(lin, axis=1), "Gyro |ω|": np.linalg.norm(w, axis=1),
            "Pitch": pitch, "Roll": roll, "_cal": cal}


def linear_acceleration(aux: "pd.DataFrame", beta=0.1, cal: ImuCalibration | None = None):
    """(linear acc (n×3, g), gravity (n×3, unit), calibration). aux needs t + accel/gyro columns."""
    t = aux["t"].to_numpy()
    acc = aux[ACC].to_numpy(float)
    gyr = aux[GYR].to_numpy(float)
    cal = cal or calibrate(t, acc, gyr)
    a_g = acc / cal.g_units
    w = np.radians((gyr - cal.gyro_bias) * cal.gyro_scale)
    Q = madgwick(t, a_g, w, beta=beta)
    grav = gravity_in_sensor(Q)
    return a_g - grav, grav, cal
