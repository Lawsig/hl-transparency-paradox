# HL Transparency Paradox — Replication Code

[![DOI: Replication Data](https://img.shields.io/badge/DOI%20Data-10.5281%2Fzenodo.20328478-blue)](https://doi.org/10.5281/zenodo.20328478)
[![DOI: Raw Archive](https://img.shields.io/badge/DOI%20Raw-10.5281%2Fzenodo.18759046-blue)](https://doi.org/10.5281/zenodo.18759046)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

Replication code for the SUSS DBA dissertation **"Liquidation-Aware Market Making Under On-Chain Transparency: Evidence from the Hyperliquid October 10, 2025 Liquidation Cascade"**.

This repository contains all Python scripts needed to reproduce the empirical findings (Chapter 4-5, Appendix A-F). Data is archived separately on Zenodo (links above).

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Lawsig/hl-transparency-paradox.git
cd hl-transparency-paradox

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download pre-built replication data from Zenodo (~45 MB)
python scripts/setup/download_zenodo_data.py

# 4. Reproduce all figures + regression tables
bash scripts/reproduction/reproduce_all.sh     # Linux / macOS
# or
pwsh scripts/reproduction/reproduce_all.ps1    # Windows PowerShell
```

After running, all reproduced outputs land in `./outputs/`.

---

## What This Repository Reproduces

The dissertation reports six core empirical findings, all of which can be reproduced from this code + Zenodo data:

| Finding | Headline number | Script |
|---|---|---|
| **H1**: Spread widening from liquidation cascade | CAS = +129.6 bp (15-min mean) | `scripts/02_event_study/` |
| **H2**: Depth collapse | CAD = −28.4% mean | `scripts/02_event_study/` |
| **H3**: Transparency amplification | amp = **3.94×** (5 specs: 3.29~4.24×) | `scripts/03_twfe/` |
| **H3 sign reversal**: τ=+3 minute reversal | β crosses zero at τ=3, VPIN reverse-shrinks | `scripts/03_twfe/` |
| **H4**: Leverage moderation | p=0.037 overall (partial support) | `scripts/03_twfe/` |
| **H5**: LINK 53h asymmetric recovery | β₂=−0.00116, p=0.015 | `scripts/03_twfe/`, `scripts/09_robustness/` |
| **Placebo**: 0/51 false positives | All 51 placebo tests fail at 5% left-tail | `scripts/04_placebo/` |
| **Theory match**: GM calibration | V*=0.464, γ*=0.600 → 6/6 findings match | `scripts/06_calibration/` |
| **Cross-exchange**: HL vs Binance | HL amp = 5.5× of BIN | `scripts/07_cross_exchange/` |
| **External validity**: 3-event replication | All 3 events show amp > 1 | `scripts/08_multi_event/` |

---

## Repository Structure

```
.
├── README.md                         # This file
├── LICENSE                           # MIT
├── .gitignore                        # Excludes data, IDE files, etc.
├── requirements.txt                  # Python dependencies
│
├── scripts/                          # All reproduction scripts (25 total)
│   ├── 01_panel_build/               # Build analytical panels from raw HL data
│   │   ├── hl_s3_downloader_full.py
│   │   ├── hl_node_fills_downloader.py
│   │   ├── hl_parse_l2.py
│   │   ├── jsonl_l2book_to_csv_fixed.py
│   │   └── chapter4_l2book_v2.py
│   │
│   ├── 02_event_study/               # Chapter 4: L1+L2 event study
│   │   └── ch4_volume_analysis.py
│   │
│   ├── 03_twfe/                      # Chapter 5: panel regressions (H3 amplification)
│   │   ├── ch5_Step1to6_run_analysis.py
│   │   ├── Ch5_Step5A.py
│   │   ├── ch5_h31_volume_spread.py
│   │   ├── fig5_1_generate.py
│   │   ├── fig5_2_generate.py
│   │   └── fig5_3_spread_vshape.py
│   │
│   ├── 04_placebo/                   # Appendix B: 51 three-layer placebo tests
│   │   └── placebo_test.py
│   │
│   ├── 05_vpin/                      # Appendix C: VPIN computation
│   │   ├── vpin_extract_takers.py
│   │   └── vpin_calculation.py
│   │
│   ├── 06_calibration/               # Appendix D: GM model V × γ grid
│   │   └── model_simulation.py
│   │
│   ├── 07_cross_exchange/            # Appendix E: HL vs Binance
│   │   ├── binance_vision_downloader.py
│   │   ├── binance_panel_build.py
│   │   ├── cross_exchange_ddd.py
│   │   └── vpin_binance.py
│   │
│   ├── 08_multi_event/               # Appendix F: 3-event replication
│   │   ├── hl_all_coin_hourly_liq.py
│   │   ├── hl_replication_panel_build.py
│   │   └── replication_twfe_compare.py
│   │
│   ├── 09_robustness/                # Appendix A: H5 threshold robustness
│   │   ├── run_robustness_grid.py
│   │   └── recompute_w0_from_ch4.py
│   │
│   ├── setup/                        # One-time setup helpers
│   │   └── download_zenodo_data.py   # Pull data from Zenodo
│   │
│   └── reproduction/                 # One-click reproducers
│       ├── reproduce_all.sh
│       └── reproduce_all.ps1
│
├── docs/
│   ├── DATA_DOWNLOAD.md              # How to fetch data from Zenodo
│   ├── METHODOLOGY.md                # Methodology notes (Driscoll-Kraay, dual filter, etc.)
│   └── CHANGELOG.md                  # Version history
│
└── .github/
    ├── workflows/
    │   └── lint.yml                  # Optional CI: ruff lint
    └── ISSUE_TEMPLATE/
        ├── bug_report.md
        └── question.md
```

---

## Data Sources

This code depends on data hosted on Zenodo (not in this repository):

### Replication Data Package (recommended — ~45 MB)

**DOI**: [10.5281/zenodo.20328478](https://doi.org/10.5281/zenodo.20328478)

Pre-built analytical panels + appendix intermediate results. Sufficient to reproduce all empirical findings.

Use this if you want to **verify the paper's results quickly**.

### Raw On-Chain Archive (~14 GB)

**DOI**: [10.5281/zenodo.18759046](https://doi.org/10.5281/zenodo.18759046)

Original Hyperliquid JSONL fills and L2 order book snapshots for 2025-10-07 to 2025-10-15.

Use this only if you want to **re-derive panels from scratch**, or perform new analyses on the raw event data.

### Binance USDⓂ-M Perpetual Data (Appendix E)

The cross-exchange comparison uses Binance Vision public archive:
- aggTrades: `https://data.binance.vision/data/futures/um/daily/aggTrades/`
- bookDepth: `https://data.binance.vision/data/futures/um/daily/bookDepth/`
- metrics: `https://data.binance.vision/data/futures/um/daily/metrics/`

The downloader `scripts/07_cross_exchange/binance_vision_downloader.py` automates this.

---

## Configuration

Most scripts read data location from environment variables:

```bash
# Path to downloaded replication data (Zenodo 20328478 contents)
export REPL_DATA_DIR="$HOME/hl-data/replication"

# Path to raw HL data (Zenodo 18759046 contents) — only if rebuilding panels
export HL_RAW_DATA_DIR="$HOME/hl-data/raw"
```

Default (if env var unset): `./data` and `./raw_data` respectively.

---

## Dependencies

Python ≥ 3.10. Install via:

```bash
pip install -r requirements.txt
```

Core dependencies:
- `pandas` ≥ 2.0
- `pyarrow` ≥ 12.0
- `numpy` ≥ 1.24
- `statsmodels` ≥ 0.14
- `linearmodels` ≥ 5.3 (for Driscoll-Kraay SE)
- `scipy` ≥ 1.11
- `matplotlib` ≥ 3.7
- `requests` ≥ 2.31 (for Zenodo/Binance Vision downloaders)
- `tqdm` ≥ 4.66

See `requirements.txt` for pinned versions used in the paper.

---

## Reproduction Workflow

### Verifying the headline H3 = 3.94× result

```python
import pandas as pd
import statsmodels.formula.api as smf

# Load main panel (downloaded from Zenodo 20328478)
df = pd.read_parquet("data/panels/panel_minute_HL_main_event.parquet")
df = df.dropna(subset=["log_spread", "liq_vol_std", "liq_x_event"])

model = smf.ols(
    "log_spread ~ liq_vol_std + liq_x_event + ret + log_vol + C(coin) + C(minute)",
    data=df
).fit(cov_type="HC3")

baseline = model.params["liq_vol_std"]
event_adj = model.params["liq_x_event"]
amp_ratio = abs(baseline + event_adj) / abs(baseline)

print(f"H3 amplification ratio: {amp_ratio:.2f}× (expected ≈ 3.94×)")
```

For the exact paper specification (Driscoll-Kraay standard errors with AR(1) bandwidth), use `scripts/03_twfe/ch5_Step1to6_run_analysis.py`.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@dataset{luo_2026_replication_data,
  author    = {Luo, Yan},
  title     = {Replication Data Package — Liquidation-Aware Market Making
               Under On-Chain Transparency},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20328478},
  url       = {https://doi.org/10.5281/zenodo.20328478}
}

@dataset{luo_2025_hyperliquid_raw,
  author    = {Luo, Yan},
  title     = {Hyperliquid 2025-10-07 to 2025-10-15 TradesFillsAndL2 Archive},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18759046},
  url       = {https://doi.org/10.5281/zenodo.18759046}
}

@software{luo_2026_replication_code,
  author    = {Luo, Yan},
  title     = {HL Transparency Paradox: Replication Code},
  year      = {2026},
  url       = {https://github.com/Lawsig/hl-transparency-paradox}
}
```

---

## Issues & Contributions

- 🐛 Found a bug? Open an [issue](../../issues/new?template=bug_report.md).
- ❓ Methodology question? Use [question template](../../issues/new?template=question.md).
- 🤝 Want to contribute (extend to other events, alternative estimators)? PRs welcome — please open an issue first to discuss scope.

---

## License

MIT License — see [LICENSE](LICENSE).

Note: **Data files** archived on Zenodo are licensed under CC-BY 4.0. This MIT license applies only to the code in this repository.

---

## Acknowledgements

- **Hyperliquid Foundation** for the open S3 archive
- **Binance Vision** for public perpetual futures data archive
- **Zenodo / CERN** for permanent academic data archival

For full acknowledgements, see the dissertation.

---

*Last updated: 2026-05-22*
