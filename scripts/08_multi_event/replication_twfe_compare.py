"""R1.5 Step 4: 3 事件 TWFE 回归对比

事件 1 (主分析): 2025-10-10 21:00 UTC — 关税冲击, $19B clearings
事件 2 (复制 1): 2025-11-21 07:00 UTC — Bitcoin 闪崩, ~$2B clearings
事件 3 (复制 2): 2026-01-30 09:00 UTC — Bitcoin 跌破支撑, ~$1.7B clearings

主回归规格（§5.3 与主分析一致）：
  log_spread = β1·liq_vol_std + β2·(liq_vol_std × D_event) + β3·ret + β4·log_vol + μ_i + λ_t + ε

输出:
  - replication_results.csv: 3 事件 × β1/β2/p/amplification 对比
  - replication_summary.json
  - fig_f1_amplification_3events.png
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

DATA = os.environ.get("REPL_DATA_DIR", "./data")
HL_MAIN_PANEL = r"E:\data2\hyperliquid\ch5_output\panel_minute.parquet"

# Reference (from §5.3 main result)
REAL_BETA_BASE = 0.0145
REAL_BETA_EVENT = -0.0716
REAL_AMP = 3.94


def run_twfe(df, label, dv='log_spread', x_vars=None, use_roll_spread=False):
    """Run §5.3 main spec TWFE regression"""
    if x_vars is None:
        x_vars = ['liq_vol_std', 'liq_x_event', 'ret', 'log_vol']
    df = df.dropna(subset=[dv] + x_vars).copy()
    df['time_id'] = pd.factorize(df['minute'])[0]
    df = df.set_index(['coin', 'time_id'])
    y = df[dv]
    X = df[x_vars]
    T = df.index.get_level_values('time_id').nunique()
    bw = max(1, int(T ** 0.25))
    m = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
        cov_type='kernel', kernel='bartlett', bandwidth=bw
    )
    res = {'event': label, 'n_obs': int(m.nobs), 'r2_within': float(m.rsquared_within), 'bw': int(bw)}
    for v in x_vars:
        res[f'beta_{v}'] = float(m.params[v])
        res[f'p_{v}'] = float(m.pvalues[v])
    # amplification
    b1 = res['beta_liq_vol_std']; b2 = res['beta_liq_x_event']
    if abs(b1) > 1e-10:
        res['amplification'] = abs(b1 + b2) / abs(b1)
        res['sign_reversed'] = ((b1 + b2) * b1 < 0)
    else:
        res['amplification'] = float('inf')
        res['sign_reversed'] = False
    return res


def main():
    print("=" * 70); print("R1.5 多事件 TWFE 对比"); print("=" * 70)

    results = []

    # Event 1: Oct 10 (main analysis, use existing panel)
    print("\n--- Event 1: 2025-10-10 (main) ---")
    main_panel = pd.read_parquet(HL_MAIN_PANEL)
    r1 = run_twfe(main_panel, '2025-10-10', dv='log_spread')
    results.append(r1)
    sig2 = '***' if r1['p_liq_x_event']<0.001 else '**' if r1['p_liq_x_event']<0.01 else '*' if r1['p_liq_x_event']<0.05 else 'n.s.'
    print(f"  N={r1['n_obs']:,}  β1={r1['beta_liq_vol_std']:+.4f}  β2={r1['beta_liq_x_event']:+.4f} {sig2}  amp={r1['amplification']:.3f}× ({'反向' if r1['sign_reversed'] else '同向'})")

    # Event 2 & 3: replications (use Roll's spread)
    for ev_name in ['2025-11-21', '2026-01-30']:
        print(f"\n--- Event: {ev_name} (replication, Roll's spread) ---")
        path = os.path.join(DATA, f'panel_minute_{ev_name}.parquet')
        if not os.path.exists(path):
            print(f"  ERROR: panel not found at {path}")
            continue
        df = pd.read_parquet(path)
        # Use spread_bps as DV (Roll's effective spread)
        r = run_twfe(df, ev_name, dv='log_spread', use_roll_spread=True)
        results.append(r)
        sig2 = '***' if r['p_liq_x_event']<0.001 else '**' if r['p_liq_x_event']<0.01 else '*' if r['p_liq_x_event']<0.05 else 'n.s.'
        print(f"  N={r['n_obs']:,}  β1={r['beta_liq_vol_std']:+.4f}  β2={r['beta_liq_x_event']:+.4f} {sig2}  amp={r['amplification']:.3f}× ({'反向' if r['sign_reversed'] else '同向'})")

    # Save CSV
    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(DATA, 'replication_results.csv'), index=False, encoding='utf-8-sig')

    # ───── Compare table ─────
    print()
    print("=" * 70); print("3 事件对比汇总"); print("=" * 70)
    print()
    print(f"{'事件':<14}{'量级':<14}{'N_obs':>10}{'β1':>10}{'β2':>10}{'p2':>10}{'amp':>10}{'反向?':>8}")
    print("-" * 86)
    event_sizes = {'2025-10-10':'$19B (#1 最大)', '2025-11-21':'$2.0B (#3)', '2026-01-30':'$1.7B (#2)'}
    for r in results:
        sz = event_sizes.get(r['event'], '')
        sign = '是' if r['sign_reversed'] else '否'
        print(f"{r['event']:<14}{sz:<14}{r['n_obs']:>10,}{r['beta_liq_vol_std']:>+10.4f}{r['beta_liq_x_event']:>+10.4f}{r['p_liq_x_event']:>10.4f}{r['amplification']:>10.3f}{sign:>8}")

    # Verdict
    print()
    print("=" * 70); print("VERDICT"); print("=" * 70)
    amps = [r['amplification'] for r in results]
    revs = [r['sign_reversed'] for r in results]
    sig_count = sum(1 for r in results if r['p_liq_x_event'] < 0.05)
    rev_count = sum(revs)
    print(f"  3 事件 amplification 区间: [{min(amps):.2f}×, {max(amps):.2f}×]")
    print(f"  3 事件 amplification 均值: {np.mean(amps):.2f}×")
    print(f"  反向放大事件数: {rev_count}/3")
    print(f"  事件期交互项 p<0.05 事件数: {sig_count}/3")
    if rev_count == 3 and sig_count == 3:
        print(f"  ★ H3 透明度悖论在 3 事件全部复制 — strong external validity")
    elif rev_count >= 2:
        print(f"  ⚠ H3 透明度悖论在 {rev_count}/3 事件复制 — partial external validity")
    else:
        print(f"  ! H3 透明度悖论可能仅适用于极端事件 — 需修订 scope condition (5a)")

    # Save JSON summary
    summary = {
        'events_count': len(results),
        'amplification_range': [float(min(amps)), float(max(amps))],
        'amplification_mean': float(np.mean(amps)),
        'sign_reversed_count': int(rev_count),
        'significant_p05_count': int(sig_count),
        'per_event_results': results,
    }
    with open(os.path.join(DATA, 'replication_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ───── Figure F.1: amplification bar chart with sign annotation ─────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    names = [r['event'] for r in results]
    sizes = [event_sizes.get(n,'') for n in names]
    amps_plot = [r['amplification'] for r in results]
    colors = ['darkred' if r['sign_reversed'] else 'steelblue' for r in results]
    bars = ax.bar(range(len(names)), amps_plot, color=colors, edgecolor='black', alpha=0.8)
    for i, (b, r) in enumerate(zip(bars, results)):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                f"{r['amplification']:.2f}×\n{'反向' if r['sign_reversed'] else '同向'}\np={r['p_liq_x_event']:.3f}",
                ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\n{s}" for n,s in zip(names, sizes)], fontsize=9)
    ax.set_ylabel('Amplification Ratio = |β1+β2| / |β1|')
    ax.set_title('图 F.1 (a)：3 事件 H3 透明度放大比率对比\n(深红=反向, 蓝=同向; 数值上方为 p 值)')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='amp=1 (无放大基准)')
    ax.axhline(REAL_AMP, color='red', linestyle=':', linewidth=1, alpha=0.6, label=f'主事件实证 {REAL_AMP}×')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(amps_plot) * 1.3)

    ax = axes[1]
    b2s = [r['beta_liq_x_event'] for r in results]
    p2s = [r['p_liq_x_event'] for r in results]
    bars2 = ax.bar(range(len(names)), b2s, color=['darkred' if b<0 else 'green' for b in b2s], edgecolor='black', alpha=0.8)
    for i, (b, beta, p) in enumerate(zip(bars2, b2s, p2s)):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h - 0.005 if h > 0 else h + 0.005,
                f"β={beta:+.4f}\np={p:.3f}",
                ha='center', va='bottom' if h < 0 else 'top', fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('β_event (事件期交互项)')
    ax.set_title('图 F.1 (b)：3 事件 β_event 估计值\n(深红=负向, 绿=正向)')
    ax.axhline(0, color='black', linewidth=0.6)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(DATA, 'fig_f1_amplification_3events.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure: {fig_path}")
    print("\n=== R1.5 多事件 TWFE 对比 COMPLETE ===")


if __name__ == "__main__":
    main()
