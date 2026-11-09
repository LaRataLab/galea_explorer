<p align="center"><img src="assets/banner.png" alt="AKCIT (Centro de Competência EMBRAPII em Tecnologias Imersivas) · LaRata LAB (Neurofisiologia do Comportamento)" width="720"></p>

# galea_explorer

This package loads, quality-checks and browses recordings from the **Galea** headset (OpenBCI GUI / BrainFlow). It works in Jupyter (VS Code or JupyterLab) with ipywidgets and plotly.

## Install

```
git clone https://github.com/LaRataLab/galea_explorer.git
cd galea_explorer
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

Put sessions under `data/`, which git ignores, e.g. `data/2024-09-27_16-46-03/`, or anywhere else. Each notebook picks the session from `DATA` at the top:
- `None` (default): an OS folder dialog opens, starting at the last folder chosen (remembered in `~/.galea_explorer.json`).
- a path: that session, no dialog.
- `$GALEA_DATA`, if set, is used when `DATA` is `None`, also without a dialog (e.g. for headless runs).

If no dialog can be shown (no display), the first session under `./data` is used.

## Notebooks

- **`dataset_quality.ipynb`**: the interactive data-quality report. It covers:
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

  **Valid range.** This is the part of the recording you confirm as usable, for example after headset setup. It is set by the user:
  - the report suggests one;
  - in the browser, **view suggestion**, adjust the zoom, then **mark view as valid**;
  - it is saved with the dataset in `<session>/galea_session.json`.

  While no range is confirmed, the quality metrics use the whole recording. `dataset_explorer.ipynb` starts clipped to it when it exists.
- **`dataset_explorer.ipynb`**: loads the session with derived signals and opens the browser and the **pair-analysis GUI**.
  - The GUI analyses two channels over the browser's clip or view, using cross-correlation, or coherence with the MSC and phase spectra.
  - Each analysis has its own options.
  - **export figure** writes `exports/<dataset>_<A>_vs_<B>_<analysis>_<t0>-<t1>s.png`.

## Package `galea`

| module | role |
|---|---|
| `io` | `load_session(folder, derived=False, tz=None)`, `find_sessions`, `resolve_data` (path, `$GALEA_DATA` or folder dialog), `choose_folder` |
| `annotations` | user annotations stored in the session folder (`galea_session.json`): `valid_range`, `set_valid_range`, `clear_valid_range`, `analysis_range` |
| `clock` | common clock: device time → PC time by a lower-envelope linear fit; `t` = s since the first valid ExG sample |
| `qc` | `qc_report`, `channel_table`, `channel_quality`, `suggest_valid_range` |
| `qc_plots` | figures and recommendations for the quality report (`dataset_quality.ipynb`) |
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

## Colours

Figures and widgets use the LaRata LAB / AKCIT purple identity, defined in `galea/style.py`, which also registers the plotly template `galea`. Each channel group has a fixed colour: EEG violet, EMG pink, EOG dark teal, Aux amber, Derived light blue, and signal-check near-black. These colours were checked for colour-vision deficiency. Status colours (green / yellow / red for findings, grades and the valid range) follow the usual convention and are kept separate from the brand palette.

## License

MIT; see [LICENSE](LICENSE). Developed at LaRata LAB (Neurofisiologia do Comportamento), with AKCIT (Centro de Competência EMBRAPII em Tecnologias Imersivas).
