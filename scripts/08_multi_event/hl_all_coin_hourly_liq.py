"""提取 HL 全平台（所有币种）小时级清算总额，用于事件零点重定位

输入：E:\\data\\hyperliquid\\hyperliquid_s3_data\\raw_jsonl\\YYYYMMDD\\HH.jsonl
方法：identical to vpin_extract_takers.py + hl_replication_panel_build.py
  - 解析每个 fill_data
  - 仅保留 is_liq = (hash == ZERO_HASH) 的清算 fills
  - tid 去重避免双重计数
  - 按 (hour, coin) 聚合 liq_vol_usd + n_liq + ...

输出：data/hl_all_coin_hourly_liq.csv
"""
import os, sys, json, time
import pandas as pd
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")

RAW = r"E:\data\hyperliquid\hyperliquid_s3_data\raw_jsonl"
OUT_DIR = os.environ.get("REPL_DATA_DIR", "./data")
os.makedirs(OUT_DIR, exist_ok=True)

ZERO_HASH = "0x" + "0" * 64

# Combine Nov 17-25 + Jan 26-Feb 03 = 18 days × 24h = 432 jsonl files
DATES = [
    # Nov 17-25
    "20251117","20251118","20251119","20251120","20251121","20251122","20251123","20251124","20251125",
    # Jan 26-Feb 03
    "20260126","20260127","20260128","20260129","20260130","20260131","20260201","20260202","20260203",
]


def main():
    t0 = time.time()
    print(f"Processing {len(DATES)} dates")

    # (hour_ms, coin) -> [liq_vol_usd, n_liq_trades]
    agg = {}
    total_liq = 0; total_events = 0
    seen_tids = set()

    for date in DATES:
        d_path = os.path.join(RAW, date)
        if not os.path.isdir(d_path):
            print(f"  MISSING: {date}")
            continue
        files = sorted(os.listdir(d_path), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 999)
        d_liq = 0
        for fn in files:
            if not fn.endswith('.jsonl'): continue
            fp = os.path.join(d_path, fn)
            try:
                with open(fp, encoding='utf-8') as f:
                    for line in f:
                        try: obj = json.loads(line)
                        except: continue
                        events_arr = obj.get('events') if isinstance(obj, dict) else None
                        if not events_arr: continue
                        for ev in events_arr:
                            if not isinstance(ev, list) or len(ev) < 2: continue
                            fd = ev[1]
                            if not isinstance(fd, dict): continue
                            total_events += 1
                            if fd.get('hash', '') != ZERO_HASH: continue
                            tid = fd.get('tid')
                            if tid is not None:
                                if tid in seen_tids: continue
                                seen_tids.add(tid)
                            try:
                                px = float(fd['px']); sz = float(fd['sz'])
                            except: continue
                            coin = fd.get('coin', 'UNKNOWN')
                            ms = fd.get('time')
                            if ms is None: continue
                            hour_ms = (ms // 3600000) * 3600000
                            vol_usd = px * sz
                            key = (coin, hour_ms)
                            row = agg.get(key)
                            if row is None:
                                row = [0.0, 0]
                                agg[key] = row
                            row[0] += vol_usd
                            row[1] += 1
                            d_liq += 1
            except Exception as e:
                print(f"    {fn}: ERR {e}")
                continue
        elapsed = time.time() - t0
        print(f"  {date}: liq_count={d_liq:,} (total {sum(r[0] for r in agg.values())/1e6:.1f}M agg_keys={len(agg):,}) elapsed={elapsed:.0f}s")
        total_liq += d_liq

    print(f"\nTotal events processed: {total_events:,}")
    print(f"Total liquidations: {total_liq:,}")
    print(f"Agg keys: {len(agg):,}")

    # Convert to DataFrame
    rows = []
    for (coin, hour_ms), v in agg.items():
        rows.append({
            'hour_utc': pd.Timestamp(hour_ms, unit='ms', tz='UTC'),
            'coin': coin,
            'liq_vol_usd': v[0],
            'liq_count': v[1],
        })
    df = pd.DataFrame(rows).sort_values(['hour_utc','coin']).reset_index(drop=True)
    df.to_csv(os.path.join(OUT_DIR, 'hl_all_coin_hourly_liq.csv'), index=False, encoding='utf-8-sig')
    print(f"\nSaved CSV: {len(df):,} rows × {df.shape[1]} cols")

    # Hourly total across ALL coins
    hourly_total = df.groupby('hour_utc').agg(
        liq_vol_total=('liq_vol_usd', 'sum'),
        liq_count_total=('liq_count', 'sum'),
        n_coins=('coin', 'nunique'),
    ).reset_index().sort_values('hour_utc')
    hourly_total.to_csv(os.path.join(OUT_DIR, 'hl_all_platform_hourly.csv'), index=False, encoding='utf-8-sig')

    # Daily total
    hourly_total['date'] = hourly_total['hour_utc'].dt.date
    daily = hourly_total.groupby('date').agg(
        liq_vol_total=('liq_vol_total', 'sum'),
        liq_count_total=('liq_count_total', 'sum'),
    ).reset_index()
    daily.to_csv(os.path.join(OUT_DIR, 'hl_all_platform_daily.csv'), index=False, encoding='utf-8-sig')

    # ── Print key results ──
    print(f"\n{'='*70}\nFULL HL PLATFORM DAILY TOTALS\n{'='*70}")
    print(f"{'date':<14}{'liq_vol_total ($M)':>22}{'liq_count':>15}")
    for _, r in daily.iterrows():
        print(f"{str(r['date']):<14}{r['liq_vol_total']/1e6:>20.1f}M{int(r['liq_count_total']):>15,}")

    print(f"\n{'='*70}\nNov 17-25 TOP 20 HOURS (all coins)\n{'='*70}")
    nov_top = hourly_total[(hourly_total['hour_utc'] >= '2025-11-17') & (hourly_total['hour_utc'] < '2025-11-26')].nlargest(20, 'liq_vol_total')
    print(f"{'hour_utc':<28}{'liq_vol ($M)':>14}{'liq_count':>14}{'n_coins':>10}")
    for _, r in nov_top.iterrows():
        print(f"{str(r['hour_utc']):<28}{r['liq_vol_total']/1e6:>12.2f}M{int(r['liq_count_total']):>14,}{int(r['n_coins']):>10}")

    print(f"\n{'='*70}\nJan 26-Feb 03 TOP 20 HOURS (all coins)\n{'='*70}")
    jan_top = hourly_total[(hourly_total['hour_utc'] >= '2026-01-26') & (hourly_total['hour_utc'] < '2026-02-04')].nlargest(20, 'liq_vol_total')
    for _, r in jan_top.iterrows():
        print(f"{str(r['hour_utc']):<28}{r['liq_vol_total']/1e6:>12.2f}M{int(r['liq_count_total']):>14,}{int(r['n_coins']):>10}")

    # Top coins on key dates
    print(f"\n{'='*70}\nTop coins on Nov 20-21 + Jan 29-30 (where 9-coin missed long-tail)\n{'='*70}")
    for date_str in ['2025-11-20', '2025-11-21', '2026-01-29', '2026-01-30']:
        sub = df[df['hour_utc'].dt.date.astype(str) == date_str]
        if len(sub) == 0: continue
        coin_total = sub.groupby('coin')['liq_vol_usd'].sum().sort_values(ascending=False).head(15)
        print(f"\n[{date_str}] top 15 coins by liq_vol:")
        for coin, v in coin_total.items():
            print(f"  {coin:<10}: ${v/1e6:>10.2f}M")

    print(f"\nTotal elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
