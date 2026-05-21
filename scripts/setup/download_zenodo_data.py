"""Download the replication data package from Zenodo (DOI: 10.5281/zenodo.20328478).

Usage:
    python scripts/setup/download_zenodo_data.py [--target ./data]

This downloads the ~45 MB replication package and unzips it into ./data/
(or the path specified via --target).
"""
import os
import sys
import argparse
import zipfile
import urllib.request
from pathlib import Path

# Zenodo record metadata
ZENODO_RECORD = "20328478"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"


def fetch_record_metadata():
    """Fetch file list and download URLs from Zenodo API."""
    print(f"Fetching record metadata from {ZENODO_API}")
    with urllib.request.urlopen(ZENODO_API) as response:
        import json
        data = json.loads(response.read().decode("utf-8"))
    return data


def download_with_progress(url, dest_path):
    """Stream-download a file with progress reporting."""
    print(f"Downloading: {url}")
    print(f"      → {dest_path}")

    def progress_hook(blocknum, block_size, total_size):
        if total_size > 0:
            percent = min(100.0, (blocknum * block_size) / total_size * 100)
            mb_done = (blocknum * block_size) / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            sys.stdout.write(f"\r  {percent:5.1f}%  {mb_done:.1f} / {mb_total:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
    print()  # newline after progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", "-t", default="./data",
        help="Target directory to extract data into (default: ./data)"
    )
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="Keep the downloaded zip file after extraction"
    )
    args = parser.parse_args()

    target = Path(args.target).resolve()
    target.mkdir(parents=True, exist_ok=True)

    print(f"Target directory: {target}")
    print()

    # Fetch metadata
    try:
        record = fetch_record_metadata()
    except Exception as e:
        print(f"ERROR: Failed to fetch record metadata: {e}")
        print()
        print("Manual download fallback:")
        print(f"  Visit https://zenodo.org/records/{ZENODO_RECORD}")
        print("  Download all files manually into your target directory.")
        return 1

    print(f"Found Zenodo record: {record['metadata']['title']}")
    print(f"Files in record: {len(record['files'])}")
    print()

    # Download each file
    download_dir = target.parent / f".zenodo_download_{ZENODO_RECORD}"
    download_dir.mkdir(exist_ok=True)

    for f in record["files"]:
        filename = f["key"]
        size_mb = f["size"] / 1024 / 1024
        download_url = f["links"]["self"]
        dest = download_dir / filename

        if dest.exists() and dest.stat().st_size == f["size"]:
            print(f"Skipping (already downloaded, size match): {filename}")
            continue

        print(f"\n[{filename}] {size_mb:.2f} MB")
        download_with_progress(download_url, dest)

    # Extract any zip files
    print()
    print("Extracting archives...")
    for zip_file in download_dir.glob("*.zip"):
        print(f"  Extracting {zip_file.name}")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(target)

    # Copy non-zip files (e.g., standalone README.md) into target
    for f in download_dir.iterdir():
        if f.suffix == ".zip":
            continue
        import shutil
        dst = target / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
            print(f"  Copied {f.name} to {target.name}/")

    # Cleanup
    if not args.keep_zip:
        import shutil
        shutil.rmtree(download_dir)
        print(f"\nCleanup: removed {download_dir.name}")

    print()
    print("=" * 60)
    print(f"✅ Data downloaded to: {target}")
    print(f"   You can now run reproduction scripts. Example:")
    print(f"     python scripts/03_twfe/ch5_Step1to6_run_analysis.py")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
