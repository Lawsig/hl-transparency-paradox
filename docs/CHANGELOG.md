# Changelog

All notable changes to this replication code repository are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-05-22

Initial public release accompanying the SUSS DBA dissertation submission.

### Added
- 25 reproduction scripts organized by chapter/appendix (`scripts/01_panel_build/` through `scripts/09_robustness/`)
- One-click setup script `scripts/setup/download_zenodo_data.py` to fetch data from Zenodo
- Reproduction wrappers: `scripts/reproduction/reproduce_all.sh` and `.ps1`
- Documentation: `docs/DATA_DOWNLOAD.md`, `docs/METHODOLOGY.md`
- Dependencies: `requirements.txt`
- Issue templates and lint CI workflow

### Data references
- Replication data package: [10.5281/zenodo.20328478](https://doi.org/10.5281/zenodo.20328478)
- Raw on-chain archive: [10.5281/zenodo.18759046](https://doi.org/10.5281/zenodo.18759046)

### Known limitations
- The raw data downloader for Binance Vision relies on Binance's archive policy (subject to upstream changes)
- Scripts use hardcoded sample paths in environment variables; users must configure `REPL_DATA_DIR` and `HL_RAW_DATA_DIR`
