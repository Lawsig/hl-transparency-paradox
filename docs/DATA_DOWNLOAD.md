# Data Download Guide

This repository does not include any data. All data is hosted on Zenodo.

## Option 1: Automated (recommended)

Run the included downloader:

```bash
python scripts/setup/download_zenodo_data.py
```

This will:
1. Fetch the replication data package from `https://doi.org/10.5281/zenodo.20328478` (~45 MB)
2. Extract files into `./data/`
3. Set up the expected directory structure for reproduction scripts

## Option 2: Manual download (replication panels only — recommended for most users)

1. Visit https://zenodo.org/records/20328478
2. Click "Download all" (or download individual files)
3. Extract `replication_data_v1.0.zip` into `./data/`
4. Final structure should look like:

```
./data/
├── panels/
│   ├── panel_minute_HL_main_event.parquet
│   ├── vpin_panel_HL_main_event.parquet
│   ├── vpin_takers_minute_HL_main_event.parquet
│   ├── vpin_panel_BIN_main_event.parquet
│   ├── panel_minute_HL_event2_2025-11-21.parquet
│   └── panel_minute_HL_event3_2026-01-30.parquet
├── platform_aggregates/
├── chapter4_results/
├── chapter5_results/
├── appendix_A_robustness/
├── appendix_B_placebo/
├── appendix_C_volume_vpin/
├── appendix_D_calibration/
├── appendix_E_cross_exchange/
├── appendix_F_multi_event/
├── README.md
└── CODEBOOK.md
```

## Option 3: Raw on-chain data (only if re-deriving panels from scratch)

If you want to re-build the analytical panels from the raw Hyperliquid JSONL/L2 data:

1. Visit https://zenodo.org/records/18759046 (~14 GB)
2. Download all files
3. Extract into `./raw_data/`
4. Final structure should look like:

```
./raw_data/
├── raw_jsonl/
│   ├── 20251007/
│   ├── 20251008/
│   ├── ...
│   └── 20251015/
└── l2book/
    ├── 20251007/
    ├── ...
    └── 20251015/
```

5. Run `scripts/01_panel_build/` scripts in order to rebuild panels.

> ⚠️ **Note**: The raw archive is ~14 GB. Most users should use the pre-built panels (Option 1 or 2) instead.

## Option 4: Binance USDⓂ-M Perpetual data (Appendix E only)

For the cross-exchange comparison, raw Binance data is fetched from the public Binance Vision archive. Use the included downloader:

```bash
python scripts/07_cross_exchange/binance_vision_downloader.py
```

This pulls:
- `aggTrades/` (~10 GB across 9 coins × event window)
- `bookDepth/`
- `metrics/`

into `./raw_data/binance/`.

## Verifying integrity

After download, you can verify file integrity using the manifest from the Zenodo record:

```bash
# Linux / macOS
md5sum -c <(awk '{print $3"  "$1}' data/manifest.txt | grep -v '^#')

# Windows PowerShell
Get-Content data/manifest.txt | Where-Object { $_ -notmatch '^#' -and $_ -ne '' } |
  ForEach-Object {
    $parts = $_ -split '\s+'
    $expected = $parts[2]
    $actual = (Get-FileHash "data/$($parts[0])" -Algorithm MD5).Hash.ToLower()
    if ($expected -eq $actual) { "OK  $($parts[0])" } else { "BAD $($parts[0])" }
  }
```

## Citations

When you use any data, please cite both records (see main README.md for BibTeX entries).
