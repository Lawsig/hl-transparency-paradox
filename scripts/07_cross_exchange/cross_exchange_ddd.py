"""R1.3 Step 3: 跨交易所 Triple-Difference (DDD) 回归 + separate TWFE

模型：
  Separate (per exchange):
    log_spread = α + β1·liq_vol_std + β2·(liq_vol_std × D_event) + β3·ret + β4·log_vol + μ_i + λ_t + ε

  DDD (pooled):
    log_spread = α + β1·liq_vol_std + β2·(liq_vol_std × D_event)
              + β3·(liq_vol_std × HL_dummy)
              + β4·(liq_vol_std × D_event × HL_dummy)    ← 透明度放大 渠道关键
              + β5·ret + β6·log_vol + μ_i + λ_t + ε

  H0: β4 = 0 (透明度不影响 amplification)
  H1: β4 < 0 (HL 透明度 引发 额外反向放大)
"""
import os, sys, json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HL_PANEL = r"E:\data2\hyperliquid\ch5_output\panel_minute.parquet"
BIN_PANEL = r"E:\data2\binance\binance_panel_minute.parquet"
OUT_DIR = os.environ.get("REPL_DATA_DIR", "./data")
os.makedirs(OUT_DIR, exist_ok=True)


def normalize_panel(df, exchange, keep_cols):
    """Standardize panel for cross-exchange merge"""
    df = df.copy()
    df['exchange'] = exchange
    # Match column names: use log_spread as DV
    if 'log_spread' not in df.columns and 'spread_bps' in df.columns:
        df['log_spread'] = np.log(df['spread_bps'].replace(0, np.nan))
    # Ensure key cols
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[keep_cols + ['exchange']]


def run_twfe(df, x_vars, label, dv='log_spread'):
    """Run TWFE regression with given X vars; returns dict"""
    df = df.dropna(subset=[dv] + x_vars).copy()
    df['time_id'] = pd.factorize(df['minute'])[0]
    df['entity_id'] = df['exchange'] + '_' + df['coin']
    df = df.set_index(['entity_id', 'time_id'])
    y = df[dv]
    X = df[x_vars]
    T = df.index.get_level_values('time_id').nunique()
    bw = max(1, int(T ** 0.25))
    m = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
        cov_type='kernel', kernel='bartlett', bandwidth=bw
    )
    res = {'spec': label, 'n_obs': int(m.nobs), 'r2_within': float(m.rsquared_within), 'bw': int(bw)}
    for v in x_vars:
        res[f'beta_{v}'] = float(m.params[v])
        res[f'p_{v}'] = float(m.pvalues[v])
    return res, m


def main():
    print("=" * 70); print("R1.3 Cross-Exchange DDD Regression"); print("=" * 70)
    # Load panels
    print(f"\nLoading HL panel: {HL_PANEL}")
    hl = pd.read_parquet(HL_PANEL)
    print(f"  rows={len(hl):,}, coins={hl['coin'].nunique()}, minutes={hl['minute'].nunique():,}")
    print(f"\nLoading BIN panel: {BIN_PANEL}")
    if not os.path.exists(BIN_PANEL):
        print("  ERROR: Binance panel not yet built. Run binance_panel_build.py first.")
        sys.exit(1)
    bn = pd.read_parquet(BIN_PANEL)
    print(f"  rows={len(bn):,}, coins={bn['coin'].nunique()}, minutes={bn['minute'].nunique():,}")

    keep_cols = ['minute','coin','log_spread','liq_vol_std','liq_x_event','D_event','ret','log_vol']
    hl_n = normalize_panel(hl, 'HL', keep_cols)
    bn_n = normalize_panel(bn, 'BIN', keep_cols)
    pooled = pd.concat([hl_n, bn_n], ignore_index=True)
    pooled['HL_dummy'] = (pooled['exchange'] == 'HL').astype(int)
    pooled['liq_x_HL'] = pooled['liq_vol_std'] * pooled['HL_dummy']
    pooled['liq_x_event_x_HL'] = pooled['liq_vol_std'] * pooled['D_event'] * pooled['HL_dummy']
    print(f"\nPooled panel: {len(pooled):,} rows; HL={int(pooled['HL_dummy'].sum()):,}, BIN={int((~pooled['HL_dummy'].astype(bool)).sum()):,}")

    # Separate TWFE
    print()
    print("=" * 70); print("Separate TWFE (per exchange)"); print("=" * 70)
    sep_results = []
    for ex, sub in [('HL', hl_n), ('BIN', bn_n)]:
        print(f"\n--- {ex} ---")
        r, _ = run_twfe(sub, ['liq_vol_std','liq_x_event','ret','log_vol'], f'separate_{ex}')
        sep_results.append(r)
        sig_b1 = '***' if r['p_liq_vol_std']<0.001 else '**' if r['p_liq_vol_std']<0.01 else '*' if r['p_liq_vol_std']<0.05 else 'n.s.'
        sig_b2 = '***' if r['p_liq_x_event']<0.001 else '**' if r['p_liq_x_event']<0.01 else '*' if r['p_liq_x_event']<0.05 else 'n.s.'
        print(f"  N={r['n_obs']:,}  R²_within={r['r2_within']:.4f}  bw={r['bw']}")
        print(f"  β1 liq_vol_std    = {r['beta_liq_vol_std']:+.4f} (p={r['p_liq_vol_std']:.4f}) {sig_b1}")
        print(f"  β2 liq_x_event    = {r['beta_liq_x_event']:+.4f} (p={r['p_liq_x_event']:.4f}) {sig_b2}")
        # Amplification ratio = |β1+β2| / |β1|
        if abs(r['beta_liq_vol_std']) > 1e-10:
            amp = abs(r['beta_liq_vol_std'] + r['beta_liq_x_event']) / abs(r['beta_liq_vol_std'])
            sign = '反向' if (r['beta_liq_vol_std'] + r['beta_liq_x_event']) * r['beta_liq_vol_std'] < 0 else '同向'
            print(f"  amplification = {amp:.3f}× ({sign})")

    # DDD (pooled)
    print()
    print("=" * 70); print("DDD pooled regression"); print("=" * 70)
    ddd_vars = ['liq_vol_std', 'liq_x_event', 'liq_x_HL', 'liq_x_event_x_HL', 'ret', 'log_vol']
    r_ddd, m_ddd = run_twfe(pooled, ddd_vars, 'DDD')
    print(f"\nDDD result:")
    print(f"  N={r_ddd['n_obs']:,}  R²_within={r_ddd['r2_within']:.4f}  bw={r_ddd['bw']}")
    for v in ['liq_vol_std','liq_x_event','liq_x_HL','liq_x_event_x_HL']:
        sig = '***' if r_ddd[f'p_{v}']<0.001 else '**' if r_ddd[f'p_{v}']<0.01 else '*' if r_ddd[f'p_{v}']<0.05 else 'n.s.'
        print(f"  β {v:25s} = {r_ddd[f'beta_{v}']:+.4f} (p={r_ddd[f'p_{v}']:.4f}) {sig}")

    # Verdict
    print()
    print("=" * 70); print("VERDICT"); print("=" * 70)
    beta_triple = r_ddd['beta_liq_x_event_x_HL']
    p_triple = r_ddd['p_liq_x_event_x_HL']
    if p_triple < 0.05 and beta_triple < 0:
        print(f"  ★ DDD β_4 (透明度放大渠道) = {beta_triple:+.4f}, p={p_triple:.4f} (显著)")
        print(f"  → R1.3 验证 H1：HL 透明度引发额外反向放大效应；")
        print(f"     透明度悖论的因果识别得到跨交易所支持。")
    elif p_triple < 0.05 and beta_triple > 0:
        print(f"  ! DDD β_4 = {beta_triple:+.4f} (p={p_triple:.4f}) 反向显著；")
        print(f"     与预期相反。可能 BIN 也呈反向，需深挖原因（7c）")
    else:
        print(f"  DDD β_4 = {beta_triple:+.4f} (p={p_triple:.4f}) 不显著；")
        print(f"  → 透明度渠道未能在 DDD 上被显著识别，需考虑其他原因（5a 诚实报告）")

    # Save results
    all_results = sep_results + [r_ddd]
    df_out = pd.DataFrame(all_results)
    df_out.to_csv(os.path.join(OUT_DIR, 'cross_exchange_results.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'cross_exchange_results.txt'), 'w', encoding='utf-8') as f:
        f.write("Separate HL summary:\n" + str(_) + "\n\n")
        f.write("DDD summary:\n" + str(m_ddd) + "\n")
    print(f"\nSaved: cross_exchange_results.csv + .txt")


if __name__ == "__main__":
    main()
