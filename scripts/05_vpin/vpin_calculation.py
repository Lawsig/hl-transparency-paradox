"""Phase 5C-2: VPIN 计算（Easley/de Prado/O'Hara 2012）

输入：vpin_takers_minute.parquet (from vpin_extract_takers.py)
  columns: coin, minute, buy_vol_usd, sell_vol_usd, n_taker, ...

方法（Easley et al. 2012, RFS）：
  1. 按等成交量构造 bucket，每 bucket size = (品种 9 天均 daily vol) / 50
  2. 每 bucket 内累计 V_B (taker buy) 和 V_S (taker sell)
  3. 滚动 n=50 窗口计算 VPIN_τ = (1/n) × Σ |V_B − V_S| / (V_B + V_S)
  4. 输出每品种 VPIN 时序，三窗口对比

输出：
  - vpin_panel.parquet: (coin, bucket_end_time, vpin)
  - vpin_summary.csv: 9 品种 × 三窗口 VPIN 均值/最大值/异常倍数
  - vpin_timeseries.png: 9 品种 VPIN 时序图
"""
import os, sys, json, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

INPUT_PARQUET = r"E:\data2\hyperliquid\ch5_output\vpin_takers_minute.parquet"
OUT_DIR = r"E:\data2\hyperliquid\ch5_output"
LOCAL_OUT_DIR = os.environ.get("REPL_DATA_DIR", "./data")

COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'AVAX', 'LINK', 'HYPE']
EVENT_START = pd.Timestamp("2025-10-10 21:00:00", tz="UTC")
EVENT_END = pd.Timestamp("2025-10-11 21:00:00", tz="UTC")

# VPIN parameters (Easley 2012 defaults)
BUCKETS_PER_DAY = 50  # 每日划分 50 个等量 bucket
VPIN_WINDOW = 50      # n=50 滚动窗口


def compute_vpin_for_coin(df_coin, coin):
    """Compute VPIN time series for one coin"""
    df = df_coin.sort_values('minute').reset_index(drop=True).copy()
    df['vol_usd'] = df['buy_vol_usd'] + df['sell_vol_usd']
    total_vol = df['vol_usd'].sum()
    n_days = (df['minute'].max() - df['minute'].min()).days + 1
    avg_daily_vol = total_vol / n_days
    bucket_size = avg_daily_vol / BUCKETS_PER_DAY
    print(f"  {coin}: total_vol={total_vol:,.0f}, n_days={n_days}, avg_daily_vol={avg_daily_vol:,.0f}, bucket_size={bucket_size:,.0f}")

    # Walk through minutes, accumulate vol until bucket full
    buckets = []  # list of dicts
    cur_buy = 0.0
    cur_sell = 0.0
    cur_start_minute = None
    cur_end_minute = None

    for _, row in df.iterrows():
        if cur_start_minute is None:
            cur_start_minute = row['minute']
        cur_end_minute = row['minute']
        # Pro-rated split: if this minute fills > remaining bucket capacity,
        # we use atomic minute-level (no intra-minute splitting). Simple version.
        minute_vol = row['vol_usd']
        if cur_buy + cur_sell + minute_vol >= bucket_size:
            # Close bucket: add full minute to current, start new
            cur_buy += row['buy_vol_usd']
            cur_sell += row['sell_vol_usd']
            buckets.append({
                'coin': coin,
                'bucket_start': cur_start_minute,
                'bucket_end': cur_end_minute,
                'V_B': cur_buy,
                'V_S': cur_sell,
                'V_total': cur_buy + cur_sell,
            })
            cur_buy = 0.0; cur_sell = 0.0; cur_start_minute = None
        else:
            cur_buy += row['buy_vol_usd']
            cur_sell += row['sell_vol_usd']

    # Don't forget last partial bucket (could include if vol > bucket_size/2)
    if cur_start_minute is not None and (cur_buy + cur_sell) >= bucket_size / 2:
        buckets.append({
            'coin': coin,
            'bucket_start': cur_start_minute,
            'bucket_end': cur_end_minute,
            'V_B': cur_buy,
            'V_S': cur_sell,
            'V_total': cur_buy + cur_sell,
        })

    bdf = pd.DataFrame(buckets)
    if len(bdf) == 0:
        return bdf
    # Compute per-bucket trade imbalance
    bdf['imbalance'] = (bdf['V_B'] - bdf['V_S']).abs() / bdf['V_total']
    # Rolling VPIN over n buckets
    bdf['vpin'] = bdf['imbalance'].rolling(VPIN_WINDOW, min_periods=VPIN_WINDOW).mean()
    bdf['bucket_size_used'] = bucket_size
    return bdf


def window_label(t):
    if t < EVENT_START: return 'pre'
    if t < EVENT_END: return 'event'
    return 'recovery'


def main():
    t0 = time.time()
    print("=" * 70); print("Phase 5C-2: VPIN computation"); print("=" * 70)
    if not os.path.exists(INPUT_PARQUET):
        print(f"ERROR: input not found: {INPUT_PARQUET}")
        print("Run vpin_extract_takers.py first.")
        sys.exit(1)
    df_all = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df_all):,} taker-minute rows from {INPUT_PARQUET}\n")

    # Compute per-coin VPIN
    all_buckets = []
    for coin in COINS:
        df_c = df_all[df_all['coin'] == coin]
        if len(df_c) == 0:
            print(f"  {coin}: no data, skip")
            continue
        bdf = compute_vpin_for_coin(df_c, coin)
        if len(bdf) == 0:
            print(f"  {coin}: no buckets, skip")
            continue
        all_buckets.append(bdf)
        valid_vpin = bdf['vpin'].dropna()
        print(f"     n_buckets={len(bdf)}, valid_vpin_obs={len(valid_vpin)}, vpin_mean={valid_vpin.mean():.4f}, vpin_max={valid_vpin.max():.4f}")

    bdf_all = pd.concat(all_buckets, ignore_index=True)
    bdf_all['window'] = bdf_all['bucket_end'].apply(window_label)

    # Save bucket-level VPIN
    out_parq = os.path.join(OUT_DIR, 'vpin_panel.parquet')
    bdf_all.to_parquet(out_parq, index=False)
    print(f"\nSaved bucket-level VPIN: {out_parq}")

    # ── Window-level summary ──
    print()
    print("=" * 70); print("VPIN by coin × window"); print("=" * 70)
    summary_rows = []
    for coin in COINS:
        coin_buckets = bdf_all[(bdf_all['coin'] == coin) & bdf_all['vpin'].notna()]
        if len(coin_buckets) == 0: continue
        row = {'coin': coin}
        for w in ['pre', 'event', 'recovery']:
            wb = coin_buckets[coin_buckets['window'] == w]
            if len(wb) > 0:
                row[f'{w}_mean'] = float(wb['vpin'].mean())
                row[f'{w}_max'] = float(wb['vpin'].max())
                row[f'{w}_n'] = int(len(wb))
            else:
                row[f'{w}_mean'] = None
                row[f'{w}_max'] = None
                row[f'{w}_n'] = 0
        # Anomaly multipliers
        if row['pre_mean'] and row['pre_mean'] > 0:
            row['event_x_pre'] = row['event_mean'] / row['pre_mean']
            row['recovery_x_pre'] = row['recovery_mean'] / row['pre_mean']
        else:
            row['event_x_pre'] = None
            row['recovery_x_pre'] = None
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_e = os.path.join(OUT_DIR, 'vpin_summary.csv')
    summary_csv_l = os.path.join(LOCAL_OUT_DIR, 'vpin_summary.csv')
    df_summary.to_csv(summary_csv_e, index=False)
    df_summary.to_csv(summary_csv_l, index=False)
    print(df_summary[['coin','pre_mean','event_mean','recovery_mean','event_x_pre']].to_string(index=False, float_format=lambda x: f'{x:.4f}' if pd.notna(x) else 'NA'))
    print(f"\nSaved summary: {summary_csv_e}")
    print(f"Saved summary (local): {summary_csv_l}")

    # Cross-section means
    cross_event_x_pre = df_summary['event_x_pre'].dropna().mean()
    cross_pre_mean = df_summary['pre_mean'].dropna().mean()
    cross_event_mean = df_summary['event_mean'].dropna().mean()
    print(f"\n  Cross-section mean (9 coins):")
    print(f"    pre mean VPIN     = {cross_pre_mean:.4f}")
    print(f"    event mean VPIN   = {cross_event_mean:.4f}")
    print(f"    event/pre x       = {cross_event_x_pre:.2f}×")

    # JSON summary
    json_summary = {
        'method': 'Easley et al. 2012 RFS - equal volume buckets + rolling n=50',
        'buckets_per_day': BUCKETS_PER_DAY,
        'rolling_window_n': VPIN_WINDOW,
        'event_start': str(EVENT_START),
        'event_end': str(EVENT_END),
        'cross_section_mean_event_x_pre': float(cross_event_x_pre),
        'cross_section_pre_mean_vpin': float(cross_pre_mean),
        'cross_section_event_mean_vpin': float(cross_event_mean),
        'per_coin_summary': df_summary.to_dict(orient='records'),
    }
    with open(os.path.join(LOCAL_OUT_DIR, 'vpin_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(json_summary, f, ensure_ascii=False, indent=2, default=str)

    # ── Figure ──
    print()
    print("=" * 70); print("Plot VPIN time series"); print("=" * 70)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    axes = axes.flatten()
    for idx, coin in enumerate(COINS):
        ax = axes[idx]
        cb = bdf_all[(bdf_all['coin'] == coin) & bdf_all['vpin'].notna()].sort_values('bucket_end')
        if len(cb) == 0: continue
        ax.plot(cb['bucket_end'], cb['vpin'], color='steelblue', linewidth=0.6)
        ax.axvline(EVENT_START, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(EVENT_END, color='red', linestyle=':', linewidth=1, alpha=0.4)
        ax.set_title(coin, fontsize=10)
        ax.set_ylim(0, max(0.5, cb['vpin'].max() * 1.1))
        ax.grid(alpha=0.3)
        if idx >= 6: ax.set_xlabel('UTC')
        if idx % 3 == 0: ax.set_ylabel('VPIN')
    fig.suptitle('图 5.6：9 品种 VPIN 时序（Easley 2012 等量 bucket，n=50 滚动；红实线 = 事件零点；红虚线 = 事件窗口终点）', fontsize=11, y=1.00)
    plt.tight_layout()
    fig_path = os.path.join(LOCAL_OUT_DIR, 'vpin_timeseries.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")

    print(f"\n=== Phase 5C-2 VPIN COMPLETE; elapsed {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
