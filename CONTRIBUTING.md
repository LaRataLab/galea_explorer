# Contributing to galea_explorer

Contributions are welcome: bug fixes, new quality checks, new derived signals, new pair analyses, and support for more recordings.

## Getting access

- **LaRata LAB members:** ask a maintainer for write access. Then clone the repo and work on a branch (below).
- **Everyone else:** fork the repo on GitHub, work in your fork, and open a pull request against `LaRataLab/galea_explorer`.

## Workflow

`main` is protected:
- every change arrives through a pull request, approved by a maintainer;
- force pushes to `main` and deleting it are blocked.

```
git clone https://github.com/LaRataLab/galea_explorer.git   # or your fork
cd galea_explorer
conda env create -f environment.yml                         # env "galea", installs the package editable
conda activate galea
nbstripout --install                                        # once per clone (see "Notebooks")

git switch -c fix/short-description                         # one branch per change
# …edit, commit…
git push -u origin fix/short-description
gh pr create --fill                                          # or "Compare & pull request" on GitHub
```

- **Keep pull requests small and focused.** One fix or feature per pull request.
- **Describe the change.** Say what changed and how you checked it: which session you ran it on and what you looked at.
- **When `main` has moved on**, update your branch with `git pull --rebase origin main`. Force-pushing your *own* branch afterwards is fine.

## Data: never commit recordings

This repository is **public**: anything pushed is published.
- Put recordings in `data/` (git-ignored) or anywhere outside the repo, and point the notebooks at them with `DATA` or the folder dialog.
- The same applies to:
  - `galea_session.json`, which the valid-range annotation writes into each session folder;
  - exported figures (`exports/`, git-ignored);
  - anything that identifies participants.
- Before committing, check `git status` for files you didn't mean to add.

## Notebooks

- **Commit notebooks without outputs.** Outputs hold plots of the data and make diffs unreadable. `nbstripout --install` sets up git to strip them automatically when you commit. Your local copy keeps its outputs.
- **Edit the notebooks themselves.** No script generates them.
- **Don't hard-code a session path.** Leave `DATA = None`.

## Code

The package is `galea/`; the README lists what each module does. Where things go:

- **Quality checks:** add a finding in `qc.qc_report` (with a severity: `ok`, `info`, `warn` or `error`). Add any figure to `qc_plots.py` and a cell to `dataset_quality.ipynb`.
- **A derived signal:** add a module under `galea/derived/` and register its channels in `derived.add_derived`, with units. The browser shows it in the "Derived" group.
- **A pair analysis:** subclass `PairAnalysis` in `pair_gui.py` (`controls`, `compute`, `plot`) and append it to `ANALYSES`.
- **Colours:** use the constants in `galea/style.py` (group colours, brand, status) and the plotly template `TEMPLATE`, not ad-hoc hex values.

A few conventions:
- **Missing files and streams:** a session may lack the Aux, signal-check, impedance or timestamp files. Every check and figure has to cope with that, by skipping or saying so, and never crash.
- **Time:** use the common clock `t` (s since the first valid ExG sample). Never use the vendor `Timestamp` column.
- **Invalid samples:** railed samples and dropouts are NaN. Filter each finite run separately (see `preprocess`).
- **Interpretation:** state the method and its limits in the docstring, and in the notebook text if users see the result. Derived signals are often indirect: for example, breathing is inferred from the PPG.

## Checking your change

There is no test suite yet. Before opening a pull request:

1. Run both notebooks top to bottom on at least one session. This script runs them without the folder dialog and reports any cell that fails:

   ```
   python tools/run_notebooks.py /path/to/session
   ```
2. If you touched loading or quality checks, also run on a session folder containing only the `BrainFlow-RAW_*.csv` file. This checks that missing files are handled.
3. For interactive parts (browser, valid range, pair GUI), say in the pull request what you clicked through.

## Questions

Open an issue on GitHub.
