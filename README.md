Hydrogen PAW Present Constraints
================================

> I built a tiny parser + dashboard for PAW pseudopotential metadata (Hydrogen, PBE/JTH) that turns the XML into tidy tables, visualizes the radial grid and energy decomposition, and quantifies how **present modeling constraints**—augmentation radius (`r_paw`) and planewave cutoffs (`E_cut`)—shape accuracy–cost tradeoffs (`Δ` metrics, basis size estimates) in plane-wave DFT.

Project Layout
--------------

```
paw-present/
  README.md
  data/
    H.GGA_PBE-JTH.xml
    H.LDA_PW-JTH.xml
  src/
    pawparse.py
    features.py
    plots.py
  notebooks/
    01_parse_and_summarize.ipynb
    02_radial_grid_and_states.ipynb
    03_cost_vs_accuracy.ipynb
    04_compare_paw_files.ipynb
  tests/
    test_parse_minimals.py
    test_features_units.py
    test_compare_datasets.py
```

Setup
-----

- Python 3.10+
- `pip install pandas numpy lxml matplotlib`
- When using the notebooks, run the bootstrap cell at the top to add the project root to `sys.path` before importing from `src`.

Quickstart
----------

```python
from pathlib import Path

from src.pawparse import parse_multiple
from src.features import dataset_features, compare_datasets

paths = [
    Path("data/H.GGA_PBE-JTH.xml"),
    Path("data/H.LDA_PW-JTH.xml"),
]
datasets = parse_multiple(paths)

gga = dataset_features(datasets[paths[0].resolve()], label="PBE/JTH")
lda = dataset_features(datasets[paths[1].resolve()], label="LDA/JTH")
comparison = compare_datasets({"PBE/JTH": datasets[paths[0].resolve()], "LDA/JTH": datasets[paths[1].resolve()]})
```

What The Parser Extracts
------------------------

- Metadata: symbol, Z, valence/core counts, XC functional, generator, `r_paw`.
- All-electron energy budget: kinetic, exchange-correlation, electrostatic, total (Hartree).
- Recommended plane-wave cutoffs for tiers (low/medium/high).
- Valence state table (`n`, `l`, occupation, eigenvalue, `r_c`, id).
- Radial grid values and resolution profile for the augmentation sphere.

Derived Features
----------------

- `G_max` (`sqrt(2 * E_cut)`) and real-space grid spacing (`Δx = π / G_max`).
- Estimated plane-wave count for a configurable cubic supercell (`N_G = L^3 G_max^3 / (6π^2)`).
- Heuristic hardness index (`E_cut / r_paw`) to rank how “hard” the PAW is.
- Optional Δ/Δ1 annotations if available in the XML header.
- Multi-file comparison helper that stacks tier-wise metrics per dataset (`compare_datasets`).

Planned Plots
-------------

- Stacked AE energy bar chart with total overlay.
- Radial grid density (`r_i` vs. index) and resolution (`Δr` vs. `r` on a log scale).
- Tier table showing cutoff → `G_max` → `Δx` → estimated plane-wave count.
- Hardness dashboard summarising the present trade-off between augmentation radius and cutoff.
- Dual-axis cutoff vs. cost chart comparing PBE vs LDA recommendations.

Present-Centered Lens
---------------------

We treat the PAW XML as a *constraint specification* for predictions **now**: the augmentation radius (`r_paw`) and recommended cutoffs (`E_cut`) directly determine resolution (`Δx`), basis size (`N_G`), and expected accuracy labels (`Δ`, `Δ1`). Rather than narrating electrons as moving beads, this project reads the file as operational limits that link the chosen model setup to the resources and accuracy you can access today.

Stretch Goals
-------------

1. Unit conversions to eV/Å in the features table (already scaffolded).
2. Compare multiple PAWs (different `r_paw`) to visualize hardness vs. cutoffs.
3. Streamlit mini-app with sliders for `L` and tier selection to refresh `Δx`/`N_G`/hardness live.
