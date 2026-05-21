"""
placebo_test.py

Part of the replication code for SUSS DBA thesis:
"Liquidation-Aware Market Making Under On-Chain Transparency"

Data DOIs:
  - Replication panels (recommended): https://doi.org/10.5281/zenodo.20328478
  - Raw on-chain archive: https://doi.org/10.5281/zenodo.18759046

Configuration:
  Set the following environment variables (or edit the constants below):
    REPL_DATA_DIR   - path to downloaded replication data folder
                      (default: ./data)
    HL_RAW_DATA_DIR - path to raw HL JSONL/L2 data
                      (only needed when re-deriving panels from scratch)
                      (default: ./raw_data)

License: MIT (see LICENSE in repository root)
"""

import os

r"""第五章 Phase 4 - Placebo Test（R1.2 导师必做反馈）

设计：三层安慰剂
  L1 - 固定时段对照：3 个非事件日同时段（21:00 UTC）安慰剂
  L2 - 随机时点：20 个随机非事件时间安慰剂（np.random.seed=20251018）
  L3 - 滚动 24h 窗口：4h 步长，全部有效非事件起点

判据：真实 β_event = −0.0716 是否位于安慰剂分布的左尾 < 5% 分位数

数据：E:\data2\hyperliquid\ch5_output\panel_minute.parquet（与 ch5_Step1to6 完全一致）
输出：
  - placebo_results.csv（每次回归 β + p）
  - placebo_distribution.png（分布直方图 + 实际 β 红线）
  - placebo_summary.json（汇总统计）
"""
import os, sys, json, time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Try to use a Chinese font if available; otherwise fallback to ASCII labels
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    USE_ZH = True
except Exception:
    USE_ZH = False

PANEL_PATH = r"E:\data2\hyperliquid\ch5_output\panel_minute.parquet"
OUTPUT_DIR = os.environ.get("REPL_DATA_DIR", "./data")

REAL_EVENT_START = pd.Timestamp("2025-10-10 21:00:00", tz="UTC")
REAL_EVENT_END = pd.Timestamp("2025-10-11 21:00:00", tz="UTC")
EVENT_DURATION = pd.Timedelta(hours=24)
DATA_START = pd.Timestamp("2025-10-07 00:00:00", tz="UTC")
DATA_END = pd.Timestamp("2025-10-15 23:59:00", tz="UTC")

# Real main result for comparison (确认值见 step2_amplification_summaries.txt 主规格 spec 3)
REAL_BETA_EVENT = -0.0716

L3_STEP_HOURS = 4  # rolling step for L3


def run_twfe_placebo(panel_full, placebo_start, drop_real_event=True):
    """Run main TWFE regression with a placebo event window.

    Excludes real event window from data (drop_real_event=True) for clean placebo.
    Returns dict with β_event_placebo, p, n_obs.
    """
    placebo_end = placebo_start + EVENT_DURATION

    if drop_real_event:
        mask = ~((panel_full['minute'] >= REAL_EVENT_START) & (panel_full['minute'] < REAL_EVENT_END))
        p = panel_full[mask].copy()
    else:
        p = panel_full.copy()

    p['D_event_placebo'] = ((p['minute'] >= placebo_start) & (p['minute'] < placebo_end)).astype(int)
    p['liq_x_event_placebo'] = p['liq_vol_std'] * p['D_event_placebo']

    if p['D_event_placebo'].sum() == 0:
        return None  # placebo window not in data

    p['time_id'] = pd.factorize(p['minute'])[0]
    p = p.set_index(['coin', 'time_id'])

    y = p['log_spread']
    X = p[['liq_vol_std', 'liq_x_event_placebo', 'ret', 'log_vol']]

    T = p.index.get_level_values('time_id').nunique()
    bw = max(1, int(T ** 0.25))

    m = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
        cov_type='kernel', kernel='bartlett', bandwidth=bw
    )

    return {
        'beta_event_placebo': float(m.params['liq_x_event_placebo']),
        'p_value': float(m.pvalues['liq_x_event_placebo']),
        'n_obs': int(m.nobs),
        'bw': int(bw),
        'placebo_n_minutes': int(p['D_event_placebo'].sum() // 9),  # per-coin minutes
    }


def valid_placebo_starts(step_minutes=60):
    """Yield placebo start times that don't overlap with real event window."""
    starts = []
    cur = DATA_START
    while cur + EVENT_DURATION <= DATA_END + pd.Timedelta(minutes=1):
        placebo_end = cur + EVENT_DURATION
        # Non-overlap: placebo_end <= real_event_start OR cur >= real_event_end
        if placebo_end <= REAL_EVENT_START or cur >= REAL_EVENT_END:
            starts.append(cur)
        cur += pd.Timedelta(minutes=step_minutes)
    return starts


def main():
    t0 = time.time()
    print("=" * 70); print("Placebo Test - Phase 4"); print("=" * 70)
    print(f"Loading panel data ...")
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  N = {len(panel):,}, coins = {panel['coin'].nunique()}, minutes = {panel['minute'].nunique()}")
    print(f"  real event window: {REAL_EVENT_START} ~ {REAL_EVENT_END}")
    print()

    # === Sanity check: re-estimate real β_event on full data ===
    print("--- Sanity check: re-estimate real β_event ---")
    sanity = run_twfe_placebo(panel, REAL_EVENT_START, drop_real_event=False)
    print(f"  Recomputed real β_event (drop_real=False, placebo=real window) = {sanity['beta_event_placebo']:+.4f}")
    print(f"  Reference REAL_BETA_EVENT = {REAL_BETA_EVENT}")
    print(f"  Match: {abs(sanity['beta_event_placebo'] - REAL_BETA_EVENT) < 0.005}")
    print()

    results = []

    # === L1: Fixed 21:00 UTC placebos on 3 non-event dates ===
    print("--- L1: Fixed same-hour (21:00 UTC) placebos ---")
    L1_dates = [
        pd.Timestamp("2025-10-08 21:00:00", tz="UTC"),
        pd.Timestamp("2025-10-13 21:00:00", tz="UTC"),
        pd.Timestamp("2025-10-14 21:00:00", tz="UTC"),
    ]
    for d in L1_dates:
        r = run_twfe_placebo(panel, d)
        if r:
            r['layer'] = 'L1_fixed_21UTC'
            r['placebo_start'] = str(d)
            results.append(r)
            sig = '***' if r['p_value']<0.001 else '**' if r['p_value']<0.01 else '*' if r['p_value']<0.05 else 'n.s.'
            print(f"  {d.strftime('%Y-%m-%d %H:%M UTC')}: β={r['beta_event_placebo']:+.4f}, p={r['p_value']:.4f} {sig}")
    print()

    # === L2: 20 random placebos ===
    print("--- L2: 20 random-time placebos (seed=20251018) ---")
    np.random.seed(20251018)
    all_valid = valid_placebo_starts(step_minutes=60)
    print(f"  pool size: {len(all_valid)} valid 1h start points")
    indices = np.random.choice(len(all_valid), size=20, replace=False)
    L2_starts = sorted([all_valid[i] for i in indices])
    for d in L2_starts:
        r = run_twfe_placebo(panel, d)
        if r:
            r['layer'] = 'L2_random'
            r['placebo_start'] = str(d)
            results.append(r)
            sig = '***' if r['p_value']<0.001 else '**' if r['p_value']<0.01 else '*' if r['p_value']<0.05 else 'n.s.'
            print(f"  {d.strftime('%Y-%m-%d %H:%M UTC')}: β={r['beta_event_placebo']:+.4f}, p={r['p_value']:.4f} {sig}")
    print()

    # === L3: Rolling 24h window with 4h step ===
    print(f"--- L3: Rolling 24h window, {L3_STEP_HOURS}h step ---")
    L3_pool = valid_placebo_starts(step_minutes=L3_STEP_HOURS * 60)
    # Exclude any duplicate with L1 or L2
    excluded = set(L1_dates + L2_starts)
    L3_starts = [d for d in L3_pool if d not in excluded]
    print(f"  L3 pool size: {len(L3_starts)} starts ({L3_STEP_HOURS}h step, excluding L1/L2 duplicates)")
    for i, d in enumerate(L3_starts):
        r = run_twfe_placebo(panel, d)
        if r:
            r['layer'] = 'L3_rolling'
            r['placebo_start'] = str(d)
            results.append(r)
        if (i + 1) % 5 == 0 or i == len(L3_starts) - 1:
            elapsed = time.time() - t0
            print(f"  L3 progress: {i+1}/{len(L3_starts)} (total={len(results)}, elapsed={elapsed:.1f}s)")
    print()

    # === Save results ===
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'placebo_results.csv'), index=False)
    print(f"Saved {len(df)} placebo runs to placebo_results.csv")
    print()

    # === Statistics ===
    all_betas = df['beta_event_placebo'].values
    sig_count_5pct = (df['p_value'] < 0.05).sum()
    pct_below_real = (all_betas <= REAL_BETA_EVENT).mean() * 100
    pct_abs_extreme = (np.abs(all_betas) >= abs(REAL_BETA_EVENT)).mean() * 100

    summary = {
        'L1_count': 3,
        'L2_count': len(L2_starts),
        'L3_count': len(L3_starts),
        'total_placebos': int(len(all_betas)),
        'real_beta_event_reference': REAL_BETA_EVENT,
        'real_beta_recomputed': float(sanity['beta_event_placebo']),
        'placebo_mean': float(np.mean(all_betas)),
        'placebo_median': float(np.median(all_betas)),
        'placebo_std': float(np.std(all_betas)),
        'placebo_min': float(np.min(all_betas)),
        'placebo_max': float(np.max(all_betas)),
        'placebo_p5': float(np.percentile(all_betas, 5)),
        'placebo_p95': float(np.percentile(all_betas, 95)),
        'pct_below_or_equal_real': float(pct_below_real),
        'pct_abs_more_extreme': float(pct_abs_extreme),
        'significant_at_5pct_count': int(sig_count_5pct),
        'pass_at_5pct': bool(pct_below_real <= 5),
        'l3_step_hours': L3_STEP_HOURS,
        'random_seed': 20251018,
    }
    with open(os.path.join(OUTPUT_DIR, 'placebo_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # === Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.hist(all_betas, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(REAL_BETA_EVENT, color='red', linestyle='--', linewidth=2,
               label=f'Real β_event = {REAL_BETA_EVENT}')
    ax.axvline(np.mean(all_betas), color='orange', linestyle=':', linewidth=2,
               label=f'Placebo mean = {np.mean(all_betas):+.4f}')
    ax.set_xlabel('β_event_placebo')
    ax.set_ylabel('频数' if USE_ZH else 'Frequency')
    title = ('安慰剂检验 β_event 经验分布' if USE_ZH else 'Placebo β_event Empirical Distribution') + \
            f' (N={len(all_betas)})'
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    # Layered scatter
    for layer, color, marker in [('L1_fixed_21UTC','red','D'),('L2_random','blue','o'),('L3_rolling','green','x')]:
        sub = df[df['layer']==layer]
        ax.scatter(range(len(sub)), sub['beta_event_placebo'], c=color, marker=marker, alpha=0.6,
                   label=f'{layer} (N={len(sub)})')
    ax.axhline(REAL_BETA_EVENT, color='red', linestyle='--', linewidth=2,
               label=f'Real β = {REAL_BETA_EVENT}')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Placebo index')
    ax.set_ylabel('β_event_placebo')
    ax.set_title('Placebo β by layer')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'placebo_distribution.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {fig_path}")
    print()

    # === Print summary ===
    print("=" * 70)
    print("PLACEBO TEST SUMMARY")
    print("=" * 70)
    print(f"  Total placebos: {summary['total_placebos']}")
    print(f"  L1 (fixed 21:00 UTC, 3): mean β = {df[df['layer']=='L1_fixed_21UTC']['beta_event_placebo'].mean():+.4f}")
    print(f"  L2 (random, 20):         mean β = {df[df['layer']=='L2_random']['beta_event_placebo'].mean():+.4f}")
    print(f"  L3 (rolling, {len(L3_starts)}):       mean β = {df[df['layer']=='L3_rolling']['beta_event_placebo'].mean():+.4f}")
    print(f"  ----")
    print(f"  Placebo mean   = {summary['placebo_mean']:+.4f}")
    print(f"  Placebo median = {summary['placebo_median']:+.4f}")
    print(f"  Placebo std    = {summary['placebo_std']:.4f}")
    print(f"  Placebo [p5, p95] = [{summary['placebo_p5']:+.4f}, {summary['placebo_p95']:+.4f}]")
    print(f"  Real β_event   = {REAL_BETA_EVENT}")
    print(f"  % placebos ≤ real β: {summary['pct_below_or_equal_real']:.1f}%")
    print(f"  % placebos |β| ≥ |real β|: {summary['pct_abs_more_extreme']:.1f}%")
    print(f"  Placebos significant at p<0.05: {summary['significant_at_5pct_count']}/{summary['total_placebos']}")
    print(f"  ----")
    print(f"  PASS at 5% (left-tail): {summary['pass_at_5pct']}")
    print()
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
