"""Phase 5C-1: 从 raw node_fills jsonl 抽取 taker 买卖量

输入：E:\\data2\\hyperliquid\\hyperliquid_s3_data\\raw_jsonl\\YYYYMMDD\\HH.jsonl
  每行 = 一个 block envelope，含 events: [[wallet, fill_data]] 数组
  fill_data 含: coin, px, sz, side(B/A), crossed(true/false), time, tid

规则：
  - 只统计 crossed=true 的 fill（taker side），避免双重计数
  - taker side == "B": 买方主动（向上吃单）→ buy_vol
  - taker side == "A": 卖方主动（向下吃单）→ sell_vol
  - 9 个目标品种：BTC ETH SOL XRP BNB DOGE AVAX LINK HYPE
  - 按 (coin, minute_floor) 聚合

输出：E:\\data2\\hyperliquid\\ch5_output\\vpin_takers_minute.parquet
"""
import os, sys, json
import gzip
from collections import defaultdict
from datetime import datetime, timezone
import pandas as pd
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAW_DIR = r"E:\data2\hyperliquid\hyperliquid_s3_data\raw_jsonl"
OUT_PARQUET = r"E:\data2\hyperliquid\ch5_output\vpin_takers_minute.parquet"

COINS = {'BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'AVAX', 'LINK', 'HYPE'}
DATES = ["20251007", "20251008", "20251009", "20251010", "20251011",
         "20251012", "20251013", "20251014", "20251015"]


def process_file(path, agg):
    """agg = dict[(coin, minute_iso)] -> [buy_vol_usd, sell_vol_usd, n_taker, n_buy, n_sell]"""
    n_lines = 0
    n_events = 0
    n_takers = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            n_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            events = obj.get('events') if isinstance(obj, dict) else None
            if not events: continue
            for ev in events:
                n_events += 1
                if not isinstance(ev, list) or len(ev) < 2: continue
                fd = ev[1]
                if not isinstance(fd, dict): continue
                coin = fd.get('coin')
                if coin not in COINS: continue
                if not fd.get('crossed'): continue  # only takers
                n_takers += 1
                try:
                    px = float(fd['px'])
                    sz = float(fd['sz'])
                except (KeyError, ValueError, TypeError):
                    continue
                vol_usd = px * sz
                ms = fd.get('time')
                if ms is None: continue
                # Floor to minute
                minute_ts = (ms // 60000) * 60000
                key = (coin, minute_ts)
                side = fd.get('side')
                row = agg.get(key)
                if row is None:
                    row = [0.0, 0.0, 0, 0, 0]
                    agg[key] = row
                if side == 'B':  # taker bought
                    row[0] += vol_usd
                    row[3] += 1
                elif side == 'A':  # taker sold
                    row[1] += vol_usd
                    row[4] += 1
                row[2] += 1
    return n_lines, n_events, n_takers


def main():
    t0 = time.time()
    print("=" * 70); print("VPIN extraction: taker buy/sell vol from raw jsonl"); print("=" * 70)
    print(f"Output: {OUT_PARQUET}")
    print(f"Coins: {COINS}")

    agg = {}  # (coin, minute_ts_ms) -> [buy_vol, sell_vol, n_taker, n_buy, n_sell]
    total_lines = total_events = total_takers = 0

    for date in DATES:
        d_path = os.path.join(RAW_DIR, date)
        if not os.path.isdir(d_path):
            print(f"  MISSING: {d_path}")
            continue
        files = sorted(os.listdir(d_path), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 999)
        d_lines = d_events = d_takers = 0
        for f in files:
            if not f.endswith('.jsonl'): continue
            fp = os.path.join(d_path, f)
            try:
                nl, ne, nt = process_file(fp, agg)
            except Exception as e:
                print(f"    {f}: ERROR {e}")
                continue
            d_lines += nl; d_events += ne; d_takers += nt
        elapsed = time.time() - t0
        print(f"  {date}: lines={d_lines:,}  events={d_events:,}  takers={d_takers:,}  agg_size={len(agg):,}  elapsed={elapsed:.1f}s")
        total_lines += d_lines; total_events += d_events; total_takers += d_takers

    print(f"\nTotal: lines={total_lines:,}  events={total_events:,}  takers={total_takers:,}")
    print(f"Aggregated keys: {len(agg):,}")

    # ── Convert to DataFrame ──
    print("\nConverting to DataFrame...")
    rows = []
    for (coin, ms), row in agg.items():
        rows.append({
            'coin': coin,
            'minute': pd.Timestamp(ms, unit='ms', tz='UTC'),
            'buy_vol_usd': row[0],
            'sell_vol_usd': row[1],
            'n_taker': row[2],
            'n_buy_taker': row[3],
            'n_sell_taker': row[4],
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(['coin', 'minute']).reset_index(drop=True)
    print(f"DataFrame: {len(df):,} rows × {df.shape[1]} cols")
    print(f"  Per coin counts: {df['coin'].value_counts().to_dict()}")
    print(f"  Time range: {df['minute'].min()} ~ {df['minute'].max()}")

    # Save parquet
    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nSaved: {OUT_PARQUET}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
