# galea_explorer

This package loads, quality-checks and browses recordings from the **Galea** headset (OpenBCI GUI / BrainFlow). It works in Jupyter (VS Code or JupyterLab) with ipywidgets and plotly.

## Install

```
conda env create -f environment.yml      # env "galea"; also pip-installs this package (editable)
conda activate galea
```

In VS Code, pick the `galea` kernel. If the env already exists, run `pip install -e .` from the repo root.

## Data

Point the notebooks at one **session folder**: the folder an OpenBCI GUI recording writes. It contains:

| file | content |
|---|---|
| `BrainFlow-RAW_*.csv` | ExG: EMG0–5, EOG0–1, EEG0–15 at 250 Hz. Tab-separated, no header, 29 columns. **Required.** |
| `OpenBCI-RAW-Aux-*.txt` | Aux: EDA, PPG Red/IR, IMU, board temperature and battery, at 50 Hz |
| `OpenBCI-SignalCheck-Live-*.txt` | GUI signal-check: µVrms and railed % per channel |
| `OpenBCI-SignalCheck-Impedance-*.txt` | impedance table (often empty) |
| `Timestamp_*.txt` | BrainFlow's round-trip pings, sent before streaming |

Put sessions under `data/`, which git ignores, e.g. `data/2024-09-27_16-46-03/`. In each notebook, set `DATA = "<session folder>"`. Alternatively, leave it as `None` and the notebook uses `$GALEA_DATA`, or else the first session under `./data`.

## Notebooks

- **`quality_report.ipynb`**: the interactive data-quality report. It covers:
  - session summary;
  - automated findings;
  - clock fit and latency;
  - gaps;
  - a railing map with the suggested setup period;
  - per-channel quality grades and PSDs;
  - Aux sensors;
  - GUI signal-check;
  - a raw-signal browser, where you confirm the **valid range**;
  - recommendations.

  It uses raw signals only.

  **Valid range.** This is the part of the recording you confirm as usable, for example after headset setup. It is never set automatically:
  - the report suggests one;
  - in the browser, **view suggestion**, adjust the zoom, then **mark view as valid**;
  - it is saved with the dataset in `<session>/galea_session.json`.

  Until you confirm one, it stays undefined, and the quality metrics use the whole recording. `analysis.ipynb` starts clipped to it when it exists.
- **`analysis.ipynb`**: loads the session with derived signals and opens the browser and the **pair-analysis GUI**.
  - The GUI analyses two channels over the browser's clip or view, using cross-correlation, or coherence with the MSC and phase spectra.
  - Each analysis has its own options.
  - **export figure** writes `exports/<dataset>_<A>_vs_<B>_<analysis>_<t0>-<t1>s.png`.

## Package `galea`

| module | role |
|---|---|
| `io` | `load_session(folder, derived=False, tz=None)`, `find_sessions`, `resolve_data` |
| `annotations` | user annotations stored in the session folder (`galea_session.json`): `valid_range`, `set_valid_range`, `clear_valid_range`, `analysis_range` |
| `clock` | common clock: device time → PC time by a lower-envelope linear fit; `t` = s since the first valid ExG sample |
| `qc` | `qc_report`, `channel_table`, `channel_quality`, `suggest_valid_range` |
| `qc_plots` | figures and recommendations for the quality report |
| `preprocess` | per-segment filters (railed samples → NaN), min/max decimation |
| `browser` | `SignalBrowser`: stacked, zoom-refetching signal browser with clip and presets |
| `pair_gui` | `PairGUI` and pluggable `PairAnalysis` classes (`CrossCorrelation`, `Coherence`) |
| `derived.emg_envelope` | EMG linear envelope (`EMG<n> env`) |
| `derived.head_motion` | IMU calibration and Madgwick fusion: `Linear accel \|a\|`, `Gyro \|ω\|`, `Pitch`, `Roll` |
| `derived.ppg_cardioresp` | PPG beats, `Heart rate`, `Breathing (PPG)`, `Breathing phase` |

To add a pair analysis, subclass `PairAnalysis` in `pair_gui.py` (`controls`, `compute`, `plot`) and append it to `ANALYSES`.

## Timing notes

- **Shared clock:** the headset stamps every sample with its own µs clock, and Aux shares that clock with ExG.
- **Arrival stamps:** samples reach the PC over UDP in chunks of about 48 ms. BrainFlow stamps each chunk on arrival.
- **Vendor `Timestamp` column:** this is a running-mean sync with no drift model, and it is not reliable.
- **Common clock:** `galea.clock` fits `pc = a + b·dev` to the per-window minima of `pc − dev`. Because arrival latency is always positive, those minima trace the true relation. The fit is therefore late by the unknown minimum one-way latency (0 to about 50 ms).
- **When that offset matters:** it doesn't matter within the headset. It does matter when aligning to other devices, which needs hardware markers.
