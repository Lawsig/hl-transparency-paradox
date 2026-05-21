"""R1.3 Step 4: Binance VPIN 计算 + 跨交易所对比

复用 Phase 5C 的 Easley 2012 方法学，输入改为 Binance aggTrades 派生的 buy/sell taker volume。

输出：
  - vpin_binance_summary.csv: 9 品种 × 三窗口 VPIN
  - vpin_cross_exchange.csv: HL vs BIN 对比表
  - fig_e2_vpin_comparison.png: HL+BIN VPIN 时序对比
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

BIN_PANEL = r"E:\data2\binance\binance_panel_minute.parquet"
HL_VPIN_SUMMARY = os.environ.get("REPL_DATA_DIR", "./data")
OUT_DIR = os.environ.get("REPL_DATA_DIR", "./data")

COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'AVAX', 'LINK', 'HYPE']
EVENT_START = pd.Timestamp("2025-10-10 21:00:00", tz="UTC")
EVENT_END = pd.Timestamp("2025-10-11 21:00:00", tz="UTC")

BUCKETS_PER_DAY = 50
VPIN_WINDOW = 50


def compute_vpin_for_coin(df_coin, coin):
    df = df_coin.sort_values('minute').reset_index(drop=True).copy()
    df['vol_usd_total'] = df['buy_vol_usd'] + df['sell_vol_usd']
    # Drop zero-volume minutes
    df = df[df['vol_usd_total'] > 0]
    total_vol = df['vol_usd_total'].sum()
    n_days = (df['minute'].max() - df['minute'].min()).days + 1
    avg_daily_vol = total_vol / n_days
    bucket_size = avg_daily_vol / BUCKETS_PER_DAY
    print(f"  {coin}: total_vol={total_vol:,.0f}, n_days={n_days}, bucket_size={bucket_size:,.0f}")

    buckets = []
    cur_buy = 0.0; cur_sell = 0.0
    cur_start = None; cur_end = None
    for _, row in df.iterrows():
        if cur_start is None: cur_start = row['minute']
        cur_end = row['minute']
        if cur_buy + cur_sell + row['vol_usd_total'] >= bucket_size:
            cur_buy += row['buy_vol_usd']
            cur_sell += row['sell_vol_usd']
            buckets.append({'coin':coin,'bucket_start':cur_start,'bucket_end':cur_end,
                           'V_B':cur_buy,'V_S':cur_sell,'V_total':cur_buy+cur_sell})
            cur_buy = 0; cur_sell = 0; cur_start = None
        else:
            cur_buy += row['buy_vol_usd']; cur_sell += row['sell_vol_usd']
    if cur_start is not None and (cur_buy + cur_sell) >= bucket_size / 2:
        buckets.append({'coin':coin,'bucket_start':cur_start,'bucket_end':cur_end,
                       'V_B':cur_buy,'V_S':cur_sell,'V_total':cur_buy+cur_sell})
    bdf = pd.DataFrame(buckets)
    if len(bdf) == 0: return bdf
    bdf['imbalance'] = (bdf['V_B'] - bdf['V_S']).abs() / bdf['V_total']
    bdf['vpin'] = bdf['imbalance'].rolling(VPIN_WINDOW, min_periods=VPIN_WINDOW).mean()
    return bdf


def window_label(t):
    if t < EVENT_START: return 'pre'
    if t < EVENT_END: return 'event'
    return 'recovery'


def main():
    t0 = time.time()
    print("=" * 70); print("R1.3 Step 4: Binance VPIN + Cross-Exchange Compare"); print("=" * 70)
    bn = pd.read_parquet(BIN_PANEL)
    print(f"Binance panel: {len(bn):,} rows")

    all_buckets = []
    for coin in COINS:
        sub = bn[bn['coin'] == coin]
        if len(sub) == 0: continue
        bdf = compute_vpin_for_coin(sub, coin)
        if len(bdf) == 0: continue
        all_buckets.append(bdf)
        vv = bdf['vpin'].dropna()
        print(f"     n_buckets={len(bdf)}, valid={len(vv)}, vpin_mean={vv.mean():.4f}")
    bin_buckets = pd.concat(all_buckets, ignore_index=True)
    bin_buckets['window'] = bin_buckets['bucket_end'].apply(window_label)
    bin_buckets.to_parquet(os.path.join(OUT_DIR, 'vpin_binance_panel.parquet'), index=False)

    # Window summary
    bin_summary = []
    for coin in COINS:
        cb = bin_buckets[(bin_buckets['coin']==coin) & bin_buckets['vpin'].notna()]
        if len(cb) == 0: continue
        row = {'coin': coin}
        for w in ['pre','event','recovery']:
            wb = cb[cb['window']==w]
            row[f'{w}_mean'] = float(wb['vpin'].mean()) if len(wb)>0 else None
            row[f'{w}_max'] = float(wb['vpin'].max()) if len(wb)>0 else None
        if row['pre_mean'] and row['pre_mean'] > 0:
            row['event_x_pre'] = row['event_mean'] / row['pre_mean']
            row['recovery_x_pre'] = row['recovery_mean'] / row['pre_mean']
        bin_summary.append(row)
    df_bin = pd.DataFrame(bin_summary)
    df_bin.to_csv(os.path.join(OUT_DIR, 'vpin_binance_summary.csv'), index=False)
    print()
    print("Binance VPIN summary:")
    print(df_bin[['coin','pre_mean','event_mean','recovery_mean','event_x_pre']].to_string(index=False, float_format=lambda x: f'{x:.4f}' if pd.notna(x) else 'NA'))

    # Cross-exchange comparison
    hl = pd.read_csv(HL_VPIN_SUMMARY)
    cross = []
    for coin in COINS:
        hl_row = hl[hl['coin']==coin]
        bn_row = df_bin[df_bin['coin']==coin]
        if len(hl_row)==0 or len(bn_row)==0: continue
        hl_evx = float(hl_row['event_x_pre'].iloc[0]) if pd.notna(hl_row['event_x_pre'].iloc[0]) else None
        bn_evx = float(bn_row['event_x_pre'].iloc[0]) if pd.notna(bn_row['event_x_pre'].iloc[0]) else None
        cross.append({
            'coin': coin,
            'HL_pre': float(hl_row['pre_mean'].iloc[0]),
            'HL_event': float(hl_row['event_mean'].iloc[0]),
            'HL_event_x_pre': hl_evx,
            'BIN_pre': float(bn_row['pre_mean'].iloc[0]),
            'BIN_event': float(bn_row['event_mean'].iloc[0]),
            'BIN_event_x_pre': bn_evx,
            'transparency_gap (HL-BIN)': (hl_evx - bn_evx) if (hl_evx and bn_evx) else None,
        })
    df_cross = pd.DataFrame(cross)
    df_cross.to_csv(os.path.join(OUT_DIR, 'vpin_cross_exchange.csv'), index=False, encoding='utf-8-sig')
    print()
    print("Cross-exchange VPIN comparison:")
    print(df_cross.to_string(index=False, float_format=lambda x: f'{x:.4f}' if pd.notna(x) else 'NA'))

    # Cross-section means
    hl_cs = df_cross['HL_event_x_pre'].dropna().mean()
    bn_cs = df_cross['BIN_event_x_pre'].dropna().mean()
    print(f"\nCross-section mean event/pre ratio:")
    print(f"  HL : {hl_cs:.4f}× (event/pre, < 1 表示事件期 VPIN 下降)")
    print(f"  BIN: {bn_cs:.4f}× (event/pre)")
    print(f"  Gap (HL - BIN): {hl_cs - bn_cs:+.4f}")
    if hl_cs < bn_cs:
        print(f"  → HL VPIN 下降更显著（透明度引发对称投机平衡，与 §5.4.3 结论一致）")
    else:
        print(f"  → BIN VPIN 下降幅度大于等于 HL（透明度溢价 unclear）")

    # Figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    axes = axes.flatten()
    hl_panel = pd.read_parquet(r"E:\data2\hyperliquid\ch5_output\vpin_panel.parquet")
    for idx, coin in enumerate(COINS):
        ax = axes[idx]
        bc = bin_buckets[(bin_buckets['coin']==coin) & bin_buckets['vpin'].notna()].sort_values('bucket_end')
        hc = hl_panel[(hl_panel['coin']==coin) & hl_panel['vpin'].notna()].sort_values('bucket_end')
        if len(bc) > 0:
            ax.plot(bc['bucket_end'], bc['vpin'], color='orange', linewidth=0.7, label='BIN', alpha=0.8)
        if len(hc) > 0:
            ax.plot(hc['bucket_end'], hc['vpin'], color='steelblue', linewidth=0.7, label='HL', alpha=0.8)
        ax.axvline(EVENT_START, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(EVENT_END, color='red', linestyle=':', linewidth=1, alpha=0.4)
        ax.set_title(coin, fontsize=10)
        ax.set_ylim(0, 0.6)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)
        if idx >= 6: ax.set_xlabel('UTC')
        if idx % 3 == 0: ax.set_ylabel('VPIN')
    fig.suptitle('图 E.2：HL vs BIN VPIN 跨交易所对比（红实线=事件零点，红虚线=事件窗口终点）', fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_e2_vpin_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: fig_e2_vpin_comparison.png")
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
