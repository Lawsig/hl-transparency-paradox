"""Phase 5B - ch5 §5.7 H3.1：成交量-价差关联的事件期放大检验

模型扩展（在主规格 §5.3 之上加 log_vol × D_event 交互项）：
  SPREAD_it = α + β₁·LIQ_VOL_it + β₂·(LIQ_VOL_it × D_event)         [原 H3]
            + γ₁·log_VOL_it + γ₂·(log_VOL_it × D_event)            [新 H3.1]
            + β₃·RET_it + μᵢ + λₜ + εᵢₜ

H3.1 假说：γ₂ 显著为正 → 事件期总成交量对价差具有独立放大效应（不只清算量驱动）

输出：
  - ch5_output/table5_7_h31_volume_spread.csv（5 规格 γ₂ 对比）
  - ch5_output/step7_h31.txt（详细 log）
"""
import os, sys, json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PANEL_PATH = r"E:\data2\hyperliquid\ch5_output\panel_minute.parquet"
OUTPUT_DIR = os.environ.get("REPL_DATA_DIR", "./data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70); print("Phase 5B - ch5 §5.7 H3.1 成交量-价差回归"); print("=" * 70)
panel = pd.read_parquet(PANEL_PATH)
print(f"Loaded {len(panel):,} rows; minutes = {panel['minute'].nunique():,}")

# Pre-compute log_vol × D_event
panel['log_vol_x_event'] = panel['log_vol'] * panel['D_event']

# Helpers
def run_h31_regression(df, x_vars, label):
    """Run TWFE H3.1 regression with given X vars"""
    df_t = df.copy()
    df_t['time_id'] = pd.factorize(df_t['minute'])[0]
    df_t = df_t.set_index(['coin', 'time_id'])
    y = df_t['log_spread']
    X = df_t[x_vars]
    T = df_t.index.get_level_values('time_id').nunique()
    bw = max(1, int(T ** 0.25))
    m = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
        cov_type='kernel', kernel='bartlett', bandwidth=bw
    )
    result = {
        'spec': label,
        'n_obs': int(m.nobs),
        'r2_within': float(m.rsquared_within),
        'bw': int(bw),
    }
    for v in x_vars:
        result[f'beta_{v}'] = float(m.params[v])
        result[f'p_{v}'] = float(m.pvalues[v])
    return result, m


# Spec 1: H3 base + log_vol_x_event (no other controls)
print("\n--- Spec 1: H3 base + log_vol_x_event (no other controls) ---")
spec1, m1 = run_h31_regression(
    panel,
    ['liq_vol_std', 'liq_x_event', 'log_vol', 'log_vol_x_event'],
    'spec1_base_h31'
)
print(json.dumps(spec1, ensure_ascii=False, indent=2))

# Spec 2: + ret
print("\n--- Spec 2: + ret control ---")
spec2, m2 = run_h31_regression(
    panel,
    ['liq_vol_std', 'liq_x_event', 'ret', 'log_vol', 'log_vol_x_event'],
    'spec2_with_ret_h31'
)
print(json.dumps(spec2, ensure_ascii=False, indent=2))

# Spec 3: main (with all controls) = 主规格
print("\n--- Spec 3: 主规格 (full H3 + H3.1 interaction) ---")
spec3, m3 = run_h31_regression(
    panel,
    ['liq_vol_std', 'liq_x_event', 'ret', 'log_vol', 'log_vol_x_event'],
    'spec3_main_h31'
)
# Same as spec 2 since we have no additional control beyond log_vol; mark as main
print("(spec3 == spec2 since no additional control; treat spec2 as 主规格)")

# Spec 4: short window (±6h around event start)
print("\n--- Spec 4: short window [event-6h, event+6h] ---")
EVENT_START = pd.Timestamp("2025-10-10 21:00:00", tz="UTC")
HOUR_DELTA = pd.Timedelta(hours=6)
short_panel = panel[
    (panel['minute'] >= EVENT_START - HOUR_DELTA) &
    (panel['minute'] < EVENT_START + HOUR_DELTA + pd.Timedelta(hours=24))  # event window + 6h after
].copy()
print(f"  Short window rows: {len(short_panel):,}")
spec4, m4 = run_h31_regression(
    short_panel,
    ['liq_vol_std', 'liq_x_event', 'ret', 'log_vol', 'log_vol_x_event'],
    'spec4_short_window_h31'
)
print(json.dumps(spec4, ensure_ascii=False, indent=2))

# Spec 5: ex-LINK
print("\n--- Spec 5: ex-LINK (剔除 LINK) ---")
ex_link = panel[panel['coin'] != 'LINK'].copy()
print(f"  ex-LINK rows: {len(ex_link):,} (8 coins)")
spec5, m5 = run_h31_regression(
    ex_link,
    ['liq_vol_std', 'liq_x_event', 'ret', 'log_vol', 'log_vol_x_event'],
    'spec5_ex_link_h31'
)
print(json.dumps(spec5, ensure_ascii=False, indent=2))


# ───── Build summary CSV ─────
print()
print("=" * 70); print("Summary table"); print("=" * 70)

specs = [spec1, spec2, spec4, spec5]  # spec3 == spec2, drop dup
labels = ['Spec 1\nbase + H3.1', 'Spec 2 (主规格)\n+ ret control', 'Spec 4\nshort window', 'Spec 5\nex-LINK']

rows = []
for sp, lbl in zip(specs, labels):
    rows.append({
        'specification': lbl,
        'N_obs': sp['n_obs'],
        'R2_within': f"{sp['r2_within']:.4f}",
        'beta_liq_vol_std (β1)': f"{sp['beta_liq_vol_std']:+.4f}",
        'p_liq_vol_std': f"{sp['p_liq_vol_std']:.4f}",
        'beta_liq_x_event (β2)': f"{sp['beta_liq_x_event']:+.4f}",
        'p_liq_x_event': f"{sp['p_liq_x_event']:.4f}",
        'beta_log_vol (γ1)': f"{sp['beta_log_vol']:+.4f}",
        'p_log_vol': f"{sp['p_log_vol']:.4f}",
        'beta_log_vol_x_event (γ2) ★': f"{sp['beta_log_vol_x_event']:+.4f}",
        'p_log_vol_x_event (γ2) ★': f"{sp['p_log_vol_x_event']:.4f}",
        'γ2 significance': '***' if sp['p_log_vol_x_event']<0.001 else '**' if sp['p_log_vol_x_event']<0.01 else '*' if sp['p_log_vol_x_event']<0.05 else 'n.s.',
    })

df_summary = pd.DataFrame(rows)
df_summary.to_csv(os.path.join(OUTPUT_DIR, 'table5_7_h31_volume_spread.csv'), index=False)
print(df_summary.to_string(index=False))

# Save text log
with open(os.path.join(OUTPUT_DIR, 'step7_h31.txt'), 'w', encoding='utf-8') as f:
    f.write("Phase 5B - H3.1 成交量-价差回归 结果\n")
    f.write("=" * 70 + "\n\n")
    f.write(df_summary.to_string(index=False) + "\n\n")
    f.write("Spec 1 details:\n" + str(m1.summary) + "\n\n")
    f.write("Spec 2 (主规格) details:\n" + str(m2.summary) + "\n\n")
    f.write("Spec 4 (short window):\n" + str(m4.summary) + "\n\n")
    f.write("Spec 5 (ex-LINK):\n" + str(m5.summary) + "\n\n")

print()
print("=" * 70); print("H3.1 判定"); print("=" * 70)
gamma2 = spec2['beta_log_vol_x_event']
p_gamma2 = spec2['p_log_vol_x_event']
print(f"  γ2 (log_vol × D_event) 主规格 = {gamma2:+.4f}")
print(f"  p-value = {p_gamma2:.4f}")
if p_gamma2 < 0.05 and gamma2 > 0:
    print(f"  → H3.1 SUPPORTED: γ2 > 0 且 p<0.05；事件期总成交量对价差具有独立放大效应")
elif p_gamma2 < 0.05 and gamma2 < 0:
    print(f"  → H3.1 REJECTED with reverse sign: γ2 < 0 且 p<0.05；事件期总成交量与价差负相关")
else:
    print(f"  → H3.1 NOT SUPPORTED: γ2 不显著；事件期总成交量无独立放大效应（仅清算量驱动）")

print("\n=== Phase 5B regression COMPLETE ===")
