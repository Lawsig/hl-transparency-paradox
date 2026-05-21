"""R1.5 多事件外部有效性 — HL 复制 panel 构建

事件零点方法学（详见附录 F §F.2.1 + §F.6）：
  - Oct 10 主事件: 21:00 UTC = MacKinlay 1997 announcement-time (Trump 关税 20:50 UTC + 整点对齐)
  - Nov 21 复制 1: 11:00 UTC = peak-liq-hour（事件日峰值 60-min 清算窗口的整点 floor，因无 external news）
  - Jan 30 复制 2: 18:00 UTC = peak-liq-hour（同上）

数据源：E:\\data\\hyperliquid\\hyperliquid_s3_data\\raw_jsonl\\YYYYMMDD\\HH.jsonl
方法学（与主分析一致）：
  - 清算识别：hash == "0x" + "0"*64 (ZERO_HASH，HL 官方约定)
  - taker 端：crossed=true + side(B/A)
  - liq_vol_std 标准化：除以该 event 子样本中 liq_vol_usd 非零值的标准差
  - Roll's spread：2·sqrt(max(0, -cov(Δp_t, Δp_{t-1}))) 作为 spread 估计

输出：data/panel_minute_<event_name>.parquet
"""
import os, sys, json, time
import pandas as pd
import numpy as np
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAW = r"E:\data\hyperliquid\hyperliquid_s3_data\raw_jsonl"
OUT_DIR = os.environ.get("REPL_DATA_DIR", "./data")
os.makedirs(OUT_DIR, exist_ok=True)

COINS = {'BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'AVAX', 'LINK', 'HYPE'}
ZERO_HASH = "0x" + "0" * 64

# Define 2 replication events
EVENTS = [
    {
        'name': '2025-11-21',
        # M2 methodology (daily peak day + intraday peak hour):
        # 全 HL 平台 daily peak: Nov 21 = $343.1M (vs Nov 20 = $251.2M)
        # Nov 21 intraday peak hour (全平台): 12:00 UTC = $46.07M (9-coin 同样 $40.70M)
        # 新闻触发: BLS 11-20 宣布 Oct 就业报告作废 → Fed 12月降息概率 100%→33% 暴跌 → 累积至 11-21 12:00 UTC cascade peak
        'event_start': pd.Timestamp('2025-11-21 12:00:00', tz='UTC'),
        'event_end':   pd.Timestamp('2025-11-22 12:00:00', tz='UTC'),
        'dates': ['20251117','20251118','20251119','20251120','20251121','20251122','20251123','20251124','20251125'],
    },
    {
        'name': '2026-01-30',
        # M2 methodology (daily peak day + intraday peak hour):
        # 全 HL 平台 daily peak: Jan 30 = $357.6M (vs Jan 29 = $345.5M, very close)
        # Jan 30 intraday peak hour (全平台): 18:00 UTC = $42.60M
        # 新闻触发: Trump 提名 Kevin Warsh 为新 Fed 主席 + 美国预算截止日
        # 注：Jan 29 17:00 UTC = $71.62M 是窗口内单 hour 全局最大，但属于 cascade pre-peak phase
        'event_start': pd.Timestamp('2026-01-30 18:00:00', tz='UTC'),
        'event_end':   pd.Timestamp('2026-01-31 18:00:00', tz='UTC'),
        'dates': ['20260126','20260127','20260128','20260129','20260130','20260131','20260201','20260202','20260203'],
    },
]


def process_event(event):
    name = event['name']
    print(f"\n{'='*70}\nProcessing event: {name}\n{'='*70}")
    print(f"  event_start: {event['event_start']}, event_end: {event['event_end']}")
    print(f"  dates: {event['dates']}")

    # Aggregator: (coin, minute_ms) -> dict
    agg = {}  # key: (coin, minute_ms), val: dict with sums

    total_events = 0; total_takers = 0; total_liq = 0; total_lines = 0
    seen_tids = set()  # dedup across days

    for date in event['dates']:
        d_path = os.path.join(RAW, date)
        if not os.path.isdir(d_path):
            print(f"  MISSING date dir: {date}")
            continue
        files = sorted(os.listdir(d_path), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 999)
        d_lines = d_events = d_takers = d_liq = 0
        for fn in files:
            if not fn.endswith('.jsonl'): continue
            fp = os.path.join(d_path, fn)
            try:
                with open(fp, encoding='utf-8') as f:
                    for line in f:
                        d_lines += 1
                        try:
                            obj = json.loads(line)
                        except: continue
                        events_arr = obj.get('events') if isinstance(obj, dict) else None
                        if not events_arr: continue
                        for ev in events_arr:
                            d_events += 1
                            if not isinstance(ev, list) or len(ev) < 2: continue
                            wallet, fd = ev[0], ev[1]
                            if not isinstance(fd, dict): continue
                            coin = fd.get('coin')
                            if coin not in COINS: continue
                            tid = fd.get('tid')
                            # tid dedup
                            if tid is not None:
                                if tid in seen_tids: continue
                                seen_tids.add(tid)
                            try:
                                px = float(fd['px']); sz = float(fd['sz'])
                            except: continue
                            ms = fd.get('time')
                            if ms is None: continue
                            minute_ms = (ms // 60000) * 60000
                            vol_usd = px * sz
                            is_liq = (fd.get('hash', '') == ZERO_HASH)
                            crossed = bool(fd.get('crossed'))
                            side = fd.get('side')
                            key = (coin, minute_ms)
                            row = agg.get(key)
                            if row is None:
                                row = {
                                    'vol_usd': 0.0, 'n_trades': 0,
                                    'buy_vol_usd': 0.0, 'sell_vol_usd': 0.0,
                                    'n_taker': 0, 'liq_vol_usd': 0.0, 'n_liq_trades': 0,
                                    'last_price': px, 'first_price': px,
                                    'prices': [],
                                }
                                agg[key] = row
                            row['vol_usd'] += vol_usd
                            row['n_trades'] += 1
                            row['last_price'] = px
                            row['prices'].append(px)
                            if crossed:
                                row['n_taker'] += 1
                                d_takers += 1
                                if side == 'B': row['buy_vol_usd'] += vol_usd
                                elif side == 'A': row['sell_vol_usd'] += vol_usd
                            if is_liq:
                                row['liq_vol_usd'] += vol_usd
                                row['n_liq_trades'] += 1
                                d_liq += 1
            except Exception as e:
                print(f"    {fn}: ERR {e}")
                continue
        print(f"  {date}: lines={d_lines:,} events={d_events:,} takers={d_takers:,} liq={d_liq:,} agg_size={len(agg):,}")
        total_lines += d_lines; total_events += d_events; total_takers += d_takers; total_liq += d_liq

    print(f"\n  TOTAL: lines={total_lines:,} events={total_events:,} takers={total_takers:,} liq={total_liq:,}")

    # Build DataFrame
    rows = []
    for (coin, ms), v in agg.items():
        prices = v['prices']
        # Roll's spread
        roll_spread_abs = 0.0
        if len(prices) >= 3:
            dp = np.diff(prices)
            if len(dp) >= 2:
                c = np.cov(dp[:-1], dp[1:])[0, 1]
                roll_spread_abs = 2 * np.sqrt(-c) if c < 0 else 0.0
        rows.append({
            'minute': pd.Timestamp(ms, unit='ms', tz='UTC'),
            'coin': coin,
            'vol_usd': v['vol_usd'],
            'n_trades': v['n_trades'],
            'buy_vol_usd': v['buy_vol_usd'],
            'sell_vol_usd': v['sell_vol_usd'],
            'n_taker': v['n_taker'],
            'liq_vol_usd': v['liq_vol_usd'],
            'n_liq_trades': v['n_liq_trades'],
            'last_price': v['last_price'],
            'roll_spread_abs': roll_spread_abs,
        })
    df = pd.DataFrame(rows).sort_values(['coin', 'minute']).reset_index(drop=True)
    print(f"\n  DataFrame: {len(df):,} rows × {df.shape[1]} cols")
    print(f"  per-coin counts: {df['coin'].value_counts().to_dict()}")

    # Compute derived fields
    df['spread_bps'] = (df['roll_spread_abs'] / df['last_price']) * 10000
    df['log_spread'] = np.log(df['spread_bps'].replace(0, np.nan))
    df['log_vol'] = np.log(df['vol_usd'].replace(0, np.nan))
    df['log_liq_vol'] = np.log(df['liq_vol_usd'].replace(0, np.nan))
    df['liq_ratio'] = df['liq_vol_usd'] / df['vol_usd'].replace(0, np.nan)
    # 1-min return
    df['ret'] = df.groupby('coin')['last_price'].pct_change()
    # D_event indicator
    df['D_event'] = ((df['minute'] >= event['event_start']) & (df['minute'] < event['event_end'])).astype(int)
    df['is_pre'] = df['minute'] < event['event_start']
    df['is_event'] = df['D_event'] == 1
    df['is_recovery'] = df['minute'] >= event['event_end']
    # liq_vol_std (normalize by sample std of nonzero liq_vol)
    nz = df['liq_vol_usd'][df['liq_vol_usd'] > 0]
    sd = nz.std() if len(nz) > 0 else 1.0
    df['liq_vol_std'] = df['liq_vol_usd'] / sd
    df['liq_x_event'] = df['liq_vol_std'] * df['D_event']
    # event metadata
    df['event_name'] = name

    # Save
    out_path = os.path.join(OUT_DIR, f'panel_minute_{name}.parquet')
    df.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")
    # Per-window summary
    print(f"\n  Window counts:")
    print(f"    pre:      {int(df['is_pre'].sum()):,}")
    print(f"    event:    {int(df['is_event'].sum()):,}")
    print(f"    recovery: {int(df['is_recovery'].sum()):,}")
    print(f"  liq_vol_usd nonzero count: {int((df['liq_vol_usd']>0).sum()):,}")
    print(f"  liq_vol_usd in event window: ${df[df['D_event']==1]['liq_vol_usd'].sum()/1e6:.1f} M")
    return df


def main():
    t0 = time.time()
    for event in EVENTS:
        process_event(event)
    print(f"\n=== Total elapsed: {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
