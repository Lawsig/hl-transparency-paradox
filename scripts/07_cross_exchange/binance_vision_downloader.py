"""R1.3 Step 1: Binance Vision 历史数据下载器

数据范围：9 coin × 9 天 × 3 类 = 243 文件
  Coins: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, BNBUSDT, DOGEUSDT, AVAXUSDT, LINKUSDT, HYPEUSDT
  Dates: 2025-10-07 至 2025-10-15 (与 HL 同窗口)
  Types: aggTrades / bookDepth / metrics

存储：E:\\data2\\binance\\raw\\<type>\\<coin>\\<file>.zip （保留 zip 原文件）
"""
import os, sys, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = r"E:\data2\binance\raw"
BASE_URL = "https://data.binance.vision/data/futures/um/daily"

COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'HYPEUSDT']
DATES = ["2025-10-07", "2025-10-08", "2025-10-09", "2025-10-10", "2025-10-11",
         "2025-10-12", "2025-10-13", "2025-10-14", "2025-10-15"]
TYPES = ['aggTrades', 'bookDepth', 'metrics']

MAX_WORKERS = 6  # concurrent downloads


def build_url_and_path(type_, coin, date):
    fname = f"{coin}-{type_}-{date}.zip"
    url = f"{BASE_URL}/{type_}/{coin}/{fname}"
    out_path = os.path.join(OUT_DIR, type_, coin, fname)
    return url, out_path


def download_one(type_, coin, date, retry=3):
    url, out_path = build_url_and_path(type_, coin, date)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return ('SKIP', type_, coin, date, os.path.getsize(out_path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for attempt in range(retry):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=180) as resp, open(out_path, 'wb') as f:
                data = resp.read()
                f.write(data)
            sz = len(data)
            if sz < 100:
                return ('TINY', type_, coin, date, sz)
            return ('OK', type_, coin, date, sz)
        except Exception as e:
            if attempt == retry - 1:
                return ('FAIL', type_, coin, date, str(e)[:80])
            time.sleep(2 ** attempt)
    return ('FAIL', type_, coin, date, 'max retry')


def main():
    t0 = time.time()
    jobs = [(t, c, d) for t in TYPES for c in COINS for d in DATES]
    print(f"Total jobs: {len(jobs)} ({len(TYPES)} types × {len(COINS)} coins × {len(DATES)} dates)")
    print(f"Concurrent workers: {MAX_WORKERS}")
    print(f"Output: {OUT_DIR}")
    print()

    ok = 0; skip = 0; fail = 0; tiny = 0; total_bytes = 0
    fail_list = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futs = {exe.submit(download_one, t, c, d): (t, c, d) for t, c, d in jobs}
        completed = 0
        for fut in as_completed(futs):
            status, t, c, d, info = fut.result()
            completed += 1
            if status == 'OK':
                ok += 1; total_bytes += info
                if completed % 20 == 0:
                    elapsed = time.time() - t0
                    rate = total_bytes / 1024 / 1024 / elapsed if elapsed > 0 else 0
                    print(f"  [{completed}/{len(jobs)}] OK so far: {ok}, skip: {skip}, fail: {fail}, tiny: {tiny} | DL {total_bytes/1024/1024:.1f} MB @ {rate:.1f} MB/s | elapsed {elapsed:.0f}s")
            elif status == 'SKIP':
                skip += 1; total_bytes += info
            elif status == 'TINY':
                tiny += 1
                fail_list.append((t, c, d, f"tiny {info} bytes"))
            else:
                fail += 1
                fail_list.append((t, c, d, info))

    elapsed = time.time() - t0
    print()
    print("=" * 70); print("Download summary"); print("=" * 70)
    print(f"  Total jobs:    {len(jobs)}")
    print(f"  OK:            {ok}")
    print(f"  Skipped (exist): {skip}")
    print(f"  Tiny:          {tiny}")
    print(f"  Failed:        {fail}")
    print(f"  Total bytes:   {total_bytes/1024/1024:.1f} MB ({total_bytes/1024/1024/1024:.2f} GB)")
    print(f"  Elapsed:       {elapsed:.1f}s")
    if fail_list:
        print()
        print("Failures detail:")
        for t, c, d, info in fail_list[:30]:
            print(f"  {t}/{c}/{d}: {info}")


if __name__ == "__main__":
    main()
