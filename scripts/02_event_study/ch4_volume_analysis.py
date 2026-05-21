"""Phase 5A - ch4 §4.6 成交量动态：9 品种 × 三窗口描述统计

输出：
  - ch4_output/table4_6_volume_window.csv: 9 品种 × 三窗口 vol_usd 均值/标准差/异常倍数
  - ch4_output/table4_7_liq_ratio_window.csv: 9 品种 × 三窗口 liq_ratio 对比
  - fig4_3_volume_dynamics.png: 归一化成交量时序图

三窗口定义（与 H1/H2/H5 一致，按 panel 中 D_event/is_pre/is_recovery 列）：
  - pre:       Oct 7-9 (estimation window, is_pre=True)
  - event:     Oct 10 21:00 - Oct 11 21:00 UTC (D_event=1)
  - recovery:  Oct 11 21:00 - Oct 15 (post-event tracking, is_recovery=True)
"""
import os, sys, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

PANEL_PATH = r"E:\data2\hyperliquid\ch5_output\panel_minute.parquet"
OUTPUT_DIR = os.environ.get("REPL_DATA_DIR", "./data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70); print("Phase 5A - ch4 §4.6 成交量动态分析"); print("=" * 70)
panel = pd.read_parquet(PANEL_PATH)
print(f"Loaded {len(panel):,} rows, {panel['coin'].nunique()} coins")

# Verify window flags
print(f"\nWindow flags:")
print(f"  is_pre=True:      {panel['is_pre'].sum():,} rows")
print(f"  D_event=1:        {(panel['D_event']==1).sum():,} rows")
print(f"  is_recovery=True: {panel['is_recovery'].sum():,} rows")

# ───── Compute window labels ─────
def window_label(row):
    if row['is_pre']: return 'pre'
    if row['D_event'] == 1: return 'event'
    if row['is_recovery']: return 'recovery'
    return 'other'  # any minutes not in 3 windows

panel['window'] = panel.apply(window_label, axis=1)
print(f"\nWindow row counts: {panel['window'].value_counts().to_dict()}")

# ───── Table 4.6: vol_usd 9 coins × 3 windows ─────
print()
print("=" * 70); print("Table 4.6: vol_usd by coin × window"); print("=" * 70)
COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'AVAX', 'LINK', 'HYPE']
windows = ['pre', 'event', 'recovery']

table46 = []
for coin in COINS:
    coin_data = panel[panel['coin'] == coin]
    row = {'coin': coin}
    for w in windows:
        wd = coin_data[coin_data['window'] == w]['vol_usd']
        if len(wd) == 0:
            row[f'{w}_mean'] = None
            row[f'{w}_std'] = None
            row[f'{w}_n_min'] = 0
        else:
            row[f'{w}_mean'] = float(wd.mean())
            row[f'{w}_std'] = float(wd.std())
            row[f'{w}_n_min'] = int(len(wd))
    # 异常倍数 = event_mean / pre_mean
    if row['pre_mean'] and row['pre_mean'] > 0:
        row['event_x_pre'] = row['event_mean'] / row['pre_mean']
        row['recovery_x_pre'] = row['recovery_mean'] / row['pre_mean']
    else:
        row['event_x_pre'] = None
        row['recovery_x_pre'] = None
    table46.append(row)

df_46 = pd.DataFrame(table46)
df_46.to_csv(os.path.join(OUTPUT_DIR, 'table4_6_volume_window.csv'), index=False)
# Print
print(df_46[['coin', 'pre_mean', 'event_mean', 'recovery_mean', 'event_x_pre', 'recovery_x_pre']].to_string(index=False, float_format=lambda x: f'{x:,.0f}' if pd.notna(x) and x > 1 else f'{x:.3f}'))

# ───── Table 4.7: liq_ratio 9 coins × 3 windows ─────
print()
print("=" * 70); print("Table 4.7: liq_ratio by coin × window"); print("=" * 70)
table47 = []
for coin in COINS:
    coin_data = panel[panel['coin'] == coin]
    row = {'coin': coin}
    for w in windows:
        wd = coin_data[coin_data['window'] == w]['liq_ratio']
        if len(wd) == 0:
            row[f'{w}_mean'] = None
            row[f'{w}_max'] = None
        else:
            row[f'{w}_mean'] = float(wd.mean())
            row[f'{w}_max'] = float(wd.max())
    table47.append(row)

df_47 = pd.DataFrame(table47)
df_47.to_csv(os.path.join(OUTPUT_DIR, 'table4_7_liq_ratio_window.csv'), index=False)
print(df_47.to_string(index=False, float_format=lambda x: f'{x:.4f}' if pd.notna(x) else 'NA'))

# ───── Compute aggregated summary ─────
print()
print("=" * 70); print("Aggregated cross-section means"); print("=" * 70)
cross = {
    'window': windows,
}
for metric in ['vol_usd', 'liq_ratio', 'n_trades']:
    cross[f'{metric}_mean_x_coins'] = []
    for w in windows:
        wd = panel[panel['window'] == w][metric]
        cross[f'{metric}_mean_x_coins'].append(float(wd.mean()) if len(wd) > 0 else None)
df_cross = pd.DataFrame(cross)
print(df_cross.to_string(index=False, float_format=lambda x: f'{x:,.2f}' if pd.notna(x) else 'NA'))

# Average event_x_pre across coins (excluding None)
avg_event_x_pre = df_46['event_x_pre'].dropna().mean()
print(f"\n  9-coin mean event_x_pre vol multiple: {avg_event_x_pre:.2f}×")

# ───── Figure 4.3: normalized volume time series ─────
print()
print("=" * 70); print("Figure 4.3: normalized volume time series"); print("=" * 70)
EVENT_START = pd.Timestamp("2025-10-10 21:00:00", tz="UTC")

fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
axes = axes.flatten()
for idx, coin in enumerate(COINS):
    ax = axes[idx]
    coin_data = panel[panel['coin']==coin].sort_values('minute').copy()
    # Aggregate to hourly for readability
    coin_data['hour'] = coin_data['minute'].dt.floor('h')
    hourly = coin_data.groupby('hour')['vol_usd'].sum().reset_index()
    # Normalize by pre-event mean of hourly
    pre_hourly = hourly[hourly['hour'] < EVENT_START]['vol_usd'].mean()
    if pre_hourly > 0:
        hourly['norm_vol'] = hourly['vol_usd'] / pre_hourly
    else:
        hourly['norm_vol'] = hourly['vol_usd']
    ax.plot(hourly['hour'], hourly['norm_vol'], color='steelblue', linewidth=0.8)
    ax.axvline(EVENT_START, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_yscale('log')
    ax.set_title(coin, fontsize=10)
    ax.grid(alpha=0.3)
    if idx >= 6:
        ax.set_xlabel('UTC')
    if idx % 3 == 0:
        ax.set_ylabel('Vol / pre-mean')

fig.suptitle('图 4.3：9 品种归一化小时成交量轨迹（log scale，红线 = 事件零点 2025-10-10 21:00 UTC）',
             fontsize=12, y=1.00)
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'fig4_3_volume_dynamics.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {fig_path}")

# ───── JSON summary ─────
summary = {
    'panel_rows': len(panel),
    'coins': COINS,
    'windows': windows,
    'mean_event_x_pre_volume_multiple': float(avg_event_x_pre),
    'cross_section_means': df_cross.to_dict(orient='records'),
    'highlighted_coins': {
        'highest_event_vol_multiple': df_46.loc[df_46['event_x_pre'].idxmax(), 'coin'] if df_46['event_x_pre'].notna().any() else None,
        'highest_event_liq_ratio_max': df_47.loc[df_47['event_max'].idxmax(), 'coin'] if df_47['event_max'].notna().any() else None,
    }
}
with open(os.path.join(OUTPUT_DIR, 'phase5a_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n=== Phase 5A computation COMPLETE ===")
print(f"Outputs in: {OUTPUT_DIR}")
