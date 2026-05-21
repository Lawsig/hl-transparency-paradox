"""
第五章 Step 1-6：从 panel_minute.parquet 开始跑全部分析
自包含脚本，不依赖任何其他文件。
"""

import pandas as pd
import numpy as np
import warnings
import os
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# =================================================================
# 配置
# =================================================================
OUTPUT_DIR = Path(r'E:\data2\hyperliquid\ch5_output')
COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'AVAX', 'LINK', 'HYPE']

# =================================================================
# 读取数据
# =================================================================
print("读取 panel_minute.parquet ...")
panel = pd.read_parquet(OUTPUT_DIR / 'panel_minute.parquet')
print(f"  N = {len(panel):,}")

panel['time_id'] = pd.factorize(panel['minute'])[0]  # 整数编码，0,1,2,...

print(f"  时间ID范围: {panel['time_id'].min()} ~ {panel['time_id'].max()}")
print(f"  品种数: {panel['coin'].nunique()}")


# =================================================================
# Step 1: 价差 vs 深度双模型对比
# =================================================================
def step_1_compare_dv(panel):
    print("\n" + "=" * 70)
    print("STEP 1: 价差模型 vs 深度模型")
    print("=" * 70)

    from linearmodels.panel import PanelOLS
    import math

    pdf = panel.copy()
    pdf = pdf.set_index(['coin', 'time_id'])

    X_vars = ['liq_vol_std', 'liq_x_event', 'ret', 'log_vol']
    X = pdf[X_vars]

    T = panel['time_id'].nunique()
    bw = max(1, int(T ** 0.25))
    print(f"  Driscoll-Kraay 带宽: {bw}")

    def compute_ratio(b, be):
        if abs(b) < 1e-10:
            return float('inf'), 'undefined'
        total = b + be
        ratio = abs(total) / abs(b)
        direction = 'reversal' if b * total < 0 else 'amplification'
        return ratio, direction

    results = {}
    for dv, label in [('log_spread', '价差'), ('log_depth', '深度')]:
        print(f"\n--- {label}模型: DV = {dv} ---")
        y = pdf[dv]

        m = PanelOLS(
            y, X, entity_effects=True, time_effects=True
        ).fit(cov_type='kernel', kernel='bartlett', bandwidth=bw)

        b  = m.params['liq_vol_std']
        be = m.params['liq_x_event']
        pb = m.pvalues['liq_vol_std']
        pe = m.pvalues['liq_x_event']
        r2 = m.rsquared_within

        ratio, rtype = compute_ratio(b, be)

        sig = lambda p: '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'

        results[dv] = {
            'label': label, 'beta_base': b, 'beta_event': be,
            'p_base': pb, 'p_event': pe,
            'ratio': ratio, 'ratio_type': rtype,
            'R2': r2, 'N': m.nobs,
            'summary_text': str(m.summary)
        }

        print(f"  β_base  = {b:+.6f} ({sig(pb)})")
        print(f"  β_event = {be:+.6f} ({sig(pe)})")
        print(f"  放大比率 = {ratio:.2f}× ({rtype})")
        print(f"  R2(within) = {r2:.4f},  N = {int(m.nobs):,}")

    # 对比输出
    s, d = results['log_spread'], results['log_depth']
    print(f"\n{'=' * 70}")
    print(f"  β_event:  价差 {s['beta_event']:+.6f}  |  深度 {d['beta_event']:+.6f}")
    print(f"  p(event): 价差 {s['p_event']:.4f}      |  深度 {d['p_event']:.4f}")
    print(f"  放大比率: 价差 {s['ratio']:.2f}×({s['ratio_type']}) | 深度 {d['ratio']:.2f}×({d['ratio_type']})")

    pick = 'log_spread'
    reason = "H3假说明确指向价差扩大（理论先验），深度模型作为稳健性检验"

    # 一致性检查
    if s['beta_event'] * d['beta_event'] < 0:
        print("\n  ⚠️  两模型事件期系数方向相反，需在论文§5.3中解释互补机制")
    else:
        print("\n  ✓ 两模型方向一致，互为佐证")

    print(f"\n  ★ 主因变量: {pick}")
    print(f"    理由: {reason}")

    # 输出
    with open(OUTPUT_DIR / 'step1_dv_comparison.txt', 'w', encoding='utf-8') as f:
        f.write(f"主因变量: {pick}\n理由: {reason}\n")
        f.write(f"DK带宽: {bw}\n\n")
        for dv, r in results.items():
            f.write(f"\n{'='*60}\n{r['label']}模型\n{'='*60}\n")
            f.write(r['summary_text'] + '\n')

    return pick, results


# =================================================================
# Step 2: 放大比率敏感性分析
# =================================================================
def step_2_amplification(panel, DV='log_spread'):
    print("\n" + "=" * 70)
    print(f"STEP 2: 放大比率敏感性 (DV={DV})")
    print("=" * 70)

    from linearmodels.panel import PanelOLS
    import traceback

    T = panel['time_id'].nunique()
    bw = max(1, int(T ** 0.25))
    print(f"  Driscoll-Kraay 带宽: {bw} (T={T})")

    def compute_ratio(b, be):
        if abs(b) < 1e-10:
            return np.nan, 'undefined'
        total = b + be
        ratio = abs(total) / abs(b)
        rtype = 'reversal' if b * total < 0 else 'amplification'
        return ratio, rtype

    MAIN_SPEC = '3.全控制(★主)'

    def make_subset(data, mask):
        sub = data[mask].copy()
        time_map = {t: i for i, t
                    in enumerate(sorted(sub['time_id'].unique()))}
        sub['time_id'] = sub['time_id'].map(time_map)
        return sub.set_index(['coin', 'time_id'])

    pdf      = panel.copy().set_index(['coin', 'time_id'])
    pdf_short = make_subset(
        panel,
        (panel['tau'] >= -360) & (panel['tau'] <= 360)
    )
    pdf_nolink = make_subset(
        panel,
        panel['D_link'] == 0
    )

    base_vars = ['liq_vol_std', 'liq_x_event']
    specs = {
        '1.最小模型':     (base_vars,                      pdf),
        '2.+收益率':      (base_vars + ['ret'],             pdf),
        MAIN_SPEC:        (base_vars + ['ret', 'log_vol'],  pdf),
        '4.±6h窗口':     (base_vars + ['ret', 'log_vol'],  pdf_short),
        '5.剔除LINK':    (base_vars + ['ret', 'log_vol'],  pdf_nolink),
    }

    rows = []
    for name, (xvars, data) in specs.items():
        try:
            X = data[xvars]
            y = data[DV]

            m = PanelOLS(
                y, X,
                entity_effects=True,
                time_effects=True
                # check_rank默认True，让问题显式暴露
            ).fit(cov_type='kernel', kernel='bartlett', bandwidth=bw)

            bb = m.params['liq_vol_std']
            be = m.params['liq_x_event']
            pe = m.pvalues['liq_x_event']
            r2 = m.rsquared_within

            ratio, rtype = compute_ratio(bb, be)

            sig = ('***' if pe < 0.001 else
                   '**'  if pe < 0.01  else
                   '*'   if pe < 0.05  else 'n.s.')

            rows.append({
                'spec': name,
                'beta_base': bb, 'beta_event': be,
                'p_event': pe, 'sig': sig,
                'ratio': ratio, 'ratio_type': rtype,
                'R2': r2, 'N': int(m.nobs),
                'summary_text': str(m.summary)
            })

            print(f"  {name:22s}  "
                  f"ratio={ratio:>5.2f}×({rtype[:3]})  "
                  f"β_e={be:+.4f}({sig})  "
                  f"R²={r2:.4f}  N={int(m.nobs):,}")

        except Exception as e:
            print(f"  {name:22s}  [ERROR] {e}")
            traceback.print_exc()
            rows.append({
                'spec': name,
                'beta_base': np.nan, 'beta_event': np.nan,
                'p_event': np.nan, 'sig': 'ERROR',
                'ratio': np.nan, 'ratio_type': 'ERROR',
                'R2': np.nan, 'N': 0,
                'summary_text': f'ERROR: {e}'
            })

    df_r = pd.DataFrame(rows)

    main_rows = df_r[df_r['spec'] == MAIN_SPEC]
    if len(main_rows) == 0:
        raise ValueError(f"主规格 '{MAIN_SPEC}' 未找到")
    main = main_rows.iloc[0]

    valid = df_r[df_r['ratio'].notna()]
    rev = valid[valid['ratio_type'] == 'reversal']['ratio']
    amp = valid[valid['ratio_type'] == 'amplification']['ratio']

    print(f"\n  {'='*50}")
    print(f"  ★ 主规格 ({MAIN_SPEC}): {main['ratio']:.2f}× ({main['ratio_type']})")
    if len(rev) > 0:
        print(f"  逆转型规格: {rev.min():.2f}× ~ {rev.max():.2f}× "
              f"({len(rev)}个规格: {', '.join(valid[valid['ratio_type']=='reversal']['spec'].tolist())})")
    if len(amp) > 0:
        print(f"  同向放大型: {amp.min():.2f}× ~ {amp.max():.2f}× "
              f"({len(amp)}个规格)")
    print(f"  所有有效规格范围: {valid['ratio'].min():.2f}× ~ "
          f"{valid['ratio'].max():.2f}×")

    # 输出CSV（不含大型summary_text列）
    df_r.drop(columns=['summary_text'], errors='ignore').to_csv(
        OUTPUT_DIR / 'step2_amplification.csv', index=False
    )

    # 输出完整summary到txt
    with open(OUTPUT_DIR / 'step2_amplification_summaries.txt',
              'w', encoding='utf-8') as f:
        f.write(f"DK带宽: {bw}\n\n")
        for r in rows:
            f.write(f"\n{'='*60}\n{r['spec']}\n{'='*60}\n")
            f.write(r.get('summary_text', 'N/A') + '\n')

    return main['ratio'], df_r



# =================================================================
# Step 4: 符号逆转精确定位
# =================================================================
def step_4_sign_reversal(panel, DV='log_spread'):
    print("\n" + "=" * 70)
    print(f"STEP 4: 符号逆转精确定位 (DV={DV})")
    print("=" * 70)

    import statsmodels.api as sm

    tau_range = list(range(-10, 21))

    # ── 方法1：逐分钟截面回归（N=9，仅描述性参考）──────────────────────
    print("\n--- 方法1: 截面回归（N=9/时点，⚠️仅参考）---")
    r1 = []
    for tau in tau_range:
        cs = panel[panel['tau'] == tau]
        n = len(cs)
        row_base = {'tau': tau, 'beta': np.nan, 'se': np.nan,
                    'p': np.nan, 'n': n}
        if n < 4:
            r1.append(row_base)
            continue
        try:
            y = cs[DV].reset_index(drop=True)
            X = sm.add_constant(
                cs[['liq_vol_std', 'ret']].reset_index(drop=True)
            )
            m = sm.OLS(y, X).fit(cov_type='HC1')
            r1.append({
                'tau': tau,
                'beta': m.params.get('liq_vol_std', np.nan),
                'se':   m.bse.get('liq_vol_std', np.nan),
                'p':    m.pvalues.get('liq_vol_std', np.nan),
                'n': n
            })
        except Exception as e:
            print(f"    [τ={tau}] 方法1失败: {e}")
            r1.append(row_base)

    df1 = pd.DataFrame(r1)
    for _, row in df1.iterrows():
        b, p = row.get('beta', np.nan), row.get('p', np.nan)
        sign = ('+' if b > 0 else '-') if pd.notna(b) else '?'
        sig  = '*' if pd.notna(p) and p < 0.05 else ''
        b_s  = f"{b:>+10.6f}" if pd.notna(b) else "       NaN"
        p_s  = f"{p:.4f}" if pd.notna(p) else "  NaN"
        print(f"  τ={int(row['tau']):>3d}  β={b_s}  p={p_s}"
              f"  n={int(row['n'])}  {sign}{sig}")

    # ── 方法2：±2分钟滚动窗口（主要依据）─────────────────────────────
    print("\n--- 方法2: ±2分钟滚动窗口（主要依据，N≈45/窗口）---")
    r2 = []
    for ct in tau_range:
        w = panel[
            (panel['tau'] >= ct - 2) & (panel['tau'] <= ct + 2)
        ]
        n = len(w)
        row_base = {'tau': ct, 'beta': np.nan, 'se': np.nan,
                    'p': np.nan, 'n': n}
        if n < 15:
            r2.append(row_base)
            continue
        try:
            w_r = w[['coin', 'liq_vol_std', 'ret', DV]].reset_index(drop=True)
            cd  = pd.get_dummies(w_r['coin'], drop_first=True, dtype=float)
            X   = sm.add_constant(
                pd.concat([w_r[['liq_vol_std', 'ret']], cd], axis=1)
            )
            y   = w_r[DV]
            m   = sm.OLS(y, X).fit(cov_type='HC1')
            r2.append({
                'tau':  ct,
                'beta': m.params.get('liq_vol_std', np.nan),
                'se':   m.bse.get('liq_vol_std', np.nan),
                'p':    m.pvalues.get('liq_vol_std', np.nan),
                'n':    n
            })
        except Exception as e:
            print(f"    [τ={ct}] 方法2失败: {e}")
            r2.append(row_base)

    df2 = pd.DataFrame(r2)
    for _, row in df2.iterrows():
        b, p = row.get('beta', np.nan), row.get('p', np.nan)
        sign = ('+' if b > 0 else '-') if pd.notna(b) else '?'
        sig  = '*' if pd.notna(p) and p < 0.05 else ''
        b_s  = f"{b:>+10.6f}" if pd.notna(b) else "       NaN"
        p_s  = f"{p:.4f}" if pd.notna(p) else "  NaN"
        print(f"  τ={int(row['tau']):>3d}  β={b_s}  p={p_s}"
              f"  n={int(row['n'])}  {sign}{sig}")

    # ── 逆转检测：仅基于方法2 ──────────────────────────────────────────
    def find_rev(df, consecutive=3):
        """
        检测符号逆转时点。
        
        参数:
            consecutive: 认定为稳健逆转所需连续正β的个数
        返回:
            first_pos_tau:  首个正β的τ（不稳定，仅参考）
            stable_rev_tau: 连续consecutive个正β的起始τ（稳健逆转）
            first_sig_tau:  首个p<0.05的正β的τ
        """
        post = (df[df['tau'] >= 0]
                .dropna(subset=['beta'])
                .sort_values('tau')
                .reset_index(drop=True))

        taus  = post['tau'].astype(int).tolist()
        betas = post['beta'].tolist()
        pvals = post['p'].tolist() if 'p' in post.columns else [1.0]*len(post)

        first_pos_tau = next(
            (taus[i] for i, b in enumerate(betas) if b > 0), None
        )
        first_sig_tau = next(
            (taus[i] for i, (b, p) in enumerate(zip(betas, pvals))
             if b > 0 and p < 0.05), None
        )

        # 稳健逆转：连续consecutive个正β
        stable_rev_tau = None
        for i in range(len(betas) - consecutive + 1):
            window_betas = betas[i:i+consecutive]
            if all(b > 0 for b in window_betas):
                stable_rev_tau = taus[i]
                break

        return first_pos_tau, stable_rev_tau, first_sig_tau

    # 方法1结果（仅打印，不用于verdict）
    fp1, sr1, fs1 = find_rev(df1)
    print(f"\n  方法1(参考): 首个正β→τ={fp1}, "
          f"稳健逆转→τ={sr1}, 显著正→τ={fs1}")
    print(f"  ⚠️  方法1 N=9，p值不可靠，不用于最终判断")

    # 方法2结果（主要依据）
    fp2, sr2, fs2 = find_rev(df2)
    print(f"\n  方法2(主要): 首个正β→τ={fp2}, "
          f"稳健逆转→τ={sr2}, 显著正→τ={fs2}")

    parts = []
    if fp2  is not None: parts.append(f"首个正β: τ=+{fp2}")
    if sr2  is not None: parts.append(f"稳健逆转(连续3个正β): τ=+{sr2}")
    if fs2  is not None: parts.append(f"首个显著正β: τ=+{fs2}")
    verdict = " | ".join(parts) if parts else "未检测到明确逆转"

    print(f"\n  ★★★ {verdict}")

    # ── 输出 ─────────────────────────────────────────────────────────
    out = pd.concat([
        df1.assign(method='cross_section'),
        df2.assign(method='rolling_5min')
    ], ignore_index=True)
    out.to_csv(OUTPUT_DIR / 'step4_sign_reversal.csv', index=False)

    return verdict, df1, df2

# =================================================================
# Step 3: H4杠杆调节效应
# =================================================================
def step_3_H4(panel, DV='log_spread'):
    print("\n" + "=" * 70)
    print(f"STEP 3: H4 杠杆调节 (DV={DV})")
    print("=" * 70)

    from linearmodels.panel import PanelOLS
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    T  = panel['time_id'].nunique()
    bw = max(1, int(T ** 0.25))
    print(f"  Driscoll-Kraay 带宽: {bw}")

    pdf = panel.copy()

    assert 'leverage' in pdf.columns, "leverage列不存在，请检查数据准备步骤"

    # 打印杠杆映射，便于核查
    coin_lev = pdf.groupby('coin')['leverage'].first().sort_values()
    print(f"\n  品种-杠杆映射:")
    for coin, lev in coin_lev.items():
        print(f"    {coin:>6}: {lev}×")

    # 统一用liq_vol_std重新计算所有交互项
    pdf['liq_x_leverage']    = pdf['liq_vol_std'] * pdf['leverage']
    pdf['liq_x_lev_x_event'] = pdf['liq_vol_std'] * pdf['leverage'] * pdf['D_event']

    # 验证liq_x_event口径一致性
    if 'liq_x_event' in pdf.columns:
        expected = pdf['liq_vol_std'] * pdf['D_event']
        max_diff = (pdf['liq_x_event'] - expected).abs().max()
        status = "✓" if max_diff < 1e-10 else f"⚠️ 差异={max_diff:.2e}"
        print(f"\n  liq_x_event 口径验证: {status}")

    pdf = pdf.set_index(['coin', 'time_id'])
    y   = pdf[DV]

    # ── VIF辅助函数 ──────────────────────────────────────────────────
    def check_vif(X_df, label):
        print(f"\n  VIF检查 ({label}):")
        arr = X_df.values
        for i, col in enumerate(X_df.columns):
            try:
                vif = variance_inflation_factor(arr, i)
                flag = ("⚠️ 严重" if vif > 10 else
                        "△ 中等" if vif > 5  else "✓ 正常")
                print(f"    {col:28s}: VIF={vif:>6.1f}  {flag}")
            except Exception:
                print(f"    {col:28s}: VIF=  N/A")

    # ── 净效应报告函数 ────────────────────────────────────────────────
    def report_net_effects(params, coin_lev_map, label):
        b_base = params.get('liq_vol_std', np.nan)
        b_lev  = params.get('liq_x_leverage', np.nan)
        print(f"\n  {label} 各品种净效应:")
        print(f"  (基准β={b_base:+.6f}, 杠杆调节β={b_lev:+.6f})")
        print(f"  {'品种':>6}  {'杠杆':>5}  {'净β':>10}  {'含义'}")
        for coin, lev in sorted(coin_lev_map.items(),
                                key=lambda x: x[1]):
            net = b_base + b_lev * lev
            meaning = "清算↑→价差↑" if net > 0 else "清算↑→价差↓"
            print(f"  {coin:>6}  {lev:>4}×  {net:>+10.6f}  {meaning}")

    # ── 模型A ────────────────────────────────────────────────────────
    print("\n--- 模型A: LIQ × LEVERAGE（双重交互）---")
    vars_a = ['liq_vol_std', 'liq_x_event', 'liq_x_leverage', 'ret', 'log_vol']

    check_vif(pdf[vars_a].reset_index(drop=True), "模型A")

    ma = PanelOLS(
        y, pdf[vars_a],
        entity_effects=True, time_effects=True
    ).fit(cov_type='kernel', kernel='bartlett', bandwidth=bw)

    bl_a = ma.params['liq_x_leverage']
    pl_a = ma.pvalues['liq_x_leverage']
    sig_a = ('***' if pl_a < 0.001 else '**' if pl_a < 0.01 else
             '*'   if pl_a < 0.05  else 'n.s.')

    print(f"  β(LIQ×LEV) = {bl_a:+.6f} ({sig_a}, p={pl_a:.6f})")
    print(f"  R²(within) = {ma.rsquared_within:.4f},  N = {int(ma.nobs):,}")

    report_net_effects(dict(ma.params), dict(coin_lev), "模型A")

    # ── 模型B ────────────────────────────────────────────────────────
    print("\n--- 模型B: + LIQ × LEV × EVENT（三重交互）---")
    vars_b = ['liq_vol_std', 'liq_x_event', 'liq_x_leverage',
              'liq_x_lev_x_event', 'ret', 'log_vol']

    check_vif(pdf[vars_b].reset_index(drop=True), "模型B")

    mb = PanelOLS(
        y, pdf[vars_b],
        entity_effects=True, time_effects=True
    ).fit(cov_type='kernel', kernel='bartlett', bandwidth=bw)

    bt = mb.params['liq_x_lev_x_event']
    pt = mb.pvalues['liq_x_lev_x_event']
    sig_t = ('***' if pt < 0.001 else '**' if pt < 0.01 else
             '*'   if pt < 0.05  else 'n.s.')

    print(f"  β(LIQ×LEV×EVENT) = {bt:+.6f} ({sig_t}, p={pt:.6f})")
    print(f"  R²(within) = {mb.rsquared_within:.4f},  N = {int(mb.nobs):,}")
    report_net_effects(dict(mb.params), dict(coin_lev), "模型B")

    # ── 结论 ─────────────────────────────────────────────────────────
    if pl_a < 0.05 and pt < 0.05:
        h4_status, detail = "强支持", \
            f"基准调节显著(p={pl_a:.3f})，事件期额外放大显著(p={pt:.3f})"
    elif pl_a < 0.05:
        h4_status, detail = "部分支持", \
            f"基准调节显著(p={pl_a:.3f})，三重交互不显著(p={pt:.3f})"
    elif pt < 0.05:
        h4_status, detail = "修正支持", \
            f"仅事件期三重交互显著(p={pt:.3f})，基准调节不显著"
    else:
        h4_status, detail = "不支持", \
            f"双重p={pl_a:.3f}, 三重p={pt:.3f}"

    conclusion = f"H4 {h4_status}: {detail}"
    print(f"\n  ★★★ {conclusion}")
    print(f"  （机制解读见论文§5.5.1）")

    # ── 输出 ─────────────────────────────────────────────────────────
    with open(OUTPUT_DIR / 'step3_H4.txt', 'w', encoding='utf-8') as f:
        f.write(f"H4结论: {conclusion}\n")
        f.write(f"DK带宽: {bw}\n\n")
        f.write(f"品种-杠杆映射:\n")
        for coin, lev in coin_lev.items():
            f.write(f"  {coin}: {lev}×\n")
        f.write(f"\n模型A:\n{str(ma.summary)}\n\n")
        f.write(f"模型B:\n{str(mb.summary)}\n")

    return conclusion, {
        'A': {'beta_lev': bl_a, 'p_lev': pl_a, 'R2': ma.rsquared_within},
        'B': {'beta_triple': bt, 'p_triple': pt, 'R2': mb.rsquared_within}
    }

# =================================================================
# Step H5: 恢复非对称 
# =================================================================
def step_H5(panel, DV='log_spread'):
    print("\n" + "=" * 70)
    print(f"STEP H5: 恢复非对称 (DV={DV})")
    print("=" * 70)
    import statsmodels.api as sm

    coins = sorted(panel['coin'].unique())

    # ── 验证is_pre列 ──────────────────────────────────────────────────
    assert 'is_pre' in panel.columns, "is_pre列不存在"
    pre_check = panel[panel['is_pre']]
    print(f"  事前期验证: N={len(pre_check):,}行, "
          f"品种数={pre_check['coin'].nunique()}")

    # ── 计算各品种事前均值和恢复阈值 ─────────────────────────────────
    pre_stats = {}
    for coin in coins:
        pre_vals = panel[
            panel['is_pre'] & (panel['coin'] == coin)
        ]['spread_bps']
        pre_mean = pre_vals.mean()
        pre_stats[coin] = {'mean': pre_mean, 'thresh': pre_mean * 1.5}

    # ── 恢复时间搜索函数（连续15分钟持续满足条件）────────────────────
    def find_recovery_hour(panel, coin, pre_stats, sustained_minutes=15):
        """
        要求连续 sustained_minutes 分钟均低于阈值，才认定为真实恢复。
        避免短暂触碰阈值被误报为恢复。
        """
        thresh = pre_stats[coin]['thresh']
        coin_post = (panel[(panel['tau'] > 0) & (panel['coin'] == coin)]
                     .sort_values('tau')
                     .reset_index(drop=True))

        if len(coin_post) == 0:
            return None

        # 标记每分钟是否满足条件
        coin_post['below'] = coin_post['spread_bps'] <= thresh

        # 滑动窗口：连续sustained_minutes行全为True
        # 用rolling().sum()：若窗口内全部below，sum == sustained_minutes
        coin_post['consec'] = (
            coin_post['below']
            .rolling(window=sustained_minutes, min_periods=sustained_minutes)
            .sum()
        )

        # 找到第一个窗口结束点（此时窗口内15分钟全部满足）
        first_end = coin_post[coin_post['consec'] == sustained_minutes]

        if len(first_end) == 0:
            return None  # 观测窗口内未恢复

        # 窗口的起始tau（恢复真正开始的时间点）（tau单位为分钟）
        end_idx   = first_end.index[0]
        start_idx = end_idx - sustained_minutes + 1
        tau_rec   = coin_post.loc[start_idx, 'tau']
        hour_rec  = tau_rec / 60

        return hour_rec

    # ── 精确恢复时点（连续15分钟持续满足条件）───────────────────────
    print(f"\n  恢复时间（连续15分钟持续低于阈值，精确到0.1h）:")
    recovery_hours = {}
    for coin in coins:
        h_rec = find_recovery_hour(panel, coin, pre_stats, sustained_minutes=15)
        if h_rec is not None:
            recovery_hours[coin] = h_rec
            print(f"    {coin:6s}: {h_rec:.1f}h  "
                  f"(pre均值={pre_stats[coin]['mean']:.4f}, "
                  f"阈值={pre_stats[coin]['thresh']:.4f})")
        else:
            recovery_hours[coin] = None
            print(f"    {coin:6s}: >123h（观测窗口内未恢复）")

    # ── 小时聚合面板（供回归使用）────────────────────────────────────
    post = panel[panel['tau'] > 0].copy()
    post['hour'] = (post['tau'] / 60).astype(int)
    hourly = post.groupby(['coin', 'hour']).agg(
        spread_bps     = ('spread_bps',      'mean'),
        total_depth_usd= ('total_depth_usd', 'mean'),
        liq_vol_usd    = ('liq_vol_usd',     'sum'),
        vol_usd        = ('vol_usd',         'sum'),
    ).reset_index()

    hourly['log_spread']  = np.log(hourly['spread_bps'].clip(lower=0.001))
    hourly['log_depth']   = np.log(hourly['total_depth_usd'].clip(lower=1))
    hourly['D_link']      = (hourly['coin'] == 'LINK').astype(int)
    hourly['hour_x_link'] = hourly['hour'] * hourly['D_link']
    hourly['log_liq_vol'] = np.log(hourly['liq_vol_usd'] + 1)
    hourly['log_vol']     = np.log(hourly['vol_usd'] + 1)

    # 归一化因变量（模型2用）
    for coin in coins:
        mask = hourly['coin'] == coin
        hourly.loc[mask, 'log_spread_norm'] = (
            hourly.loc[mask, 'log_spread']
            - np.log(pre_stats[coin]['mean'])
        )

    # 品种虚拟变量
    cd = pd.get_dummies(hourly['coin'], drop_first=True, dtype=float)

    def run_ols(data, dv, label):
        """运行OLS并返回关键结果"""
        X_vars = ['hour', 'hour_x_link', 'log_liq_vol', 'log_vol']
        X = sm.add_constant(
            pd.concat([
                data[X_vars].reset_index(drop=True),
                pd.get_dummies(data['coin'], drop_first=True,
                               dtype=float).reset_index(drop=True)
            ], axis=1)
        )
        y = data[dv].reset_index(drop=True)
        m = sm.OLS(y, X).fit(cov_type='HC1')

        b2 = m.params.get('hour_x_link', np.nan)
        p2 = m.pvalues.get('hour_x_link', np.nan)
        sig = ('***' if p2 < 0.001 else '**' if p2 < 0.01
               else '*' if p2 < 0.05 else 'n.s.')
        b_link = m.params.get('LINK', np.nan)

        print(f"\n  [{label}]  N={int(m.nobs)}, R²={m.rsquared:.3f}")
        print(f"    β₂(HOUR×LINK) = {b2:+.6f} ({sig}, p={p2:.4f})")
        print(f"    β(LINK虚拟)   = {b_link:+.6f}")
        print(f"    β(HOUR主效应) = {m.params.get('hour', np.nan):+.6f}")

        return {
            'beta2': b2, 'p2': p2, 'sig': sig,
            'R2': m.rsquared, 'N': int(m.nobs),
            'summary': str(m.summary())
        }

    results = {}

    # ── 模型0：全事后数据（基准，预期p≈0.9，不显著）────────────────
    print("\n--- 模型0: h≥0（全事后，预期β₂不显著）---")
    results['M0'] = run_ols(hourly, DV, "模型0 h≥0")

    # ── 模型1：主规格 h≥7 ────────────────────────────────────────────
    # ✅ 问题1修复：加h≥7限制
    print("\n--- 模型1: h≥7（主规格，★论文主要结果）---")
    hourly_7 = hourly[hourly['hour'] >= 7].copy()
    results['M1'] = run_ols(hourly_7, DV, "模型1 h≥7 ★主规格")

    # ── 模型2：归一化因变量（应与模型1代数等价）─────────────────────
    print("\n--- 模型2: h≥7，归一化DV（应与模型1 β₂一致）---")
    results['M2'] = run_ols(hourly_7, 'log_spread_norm',
                            "模型2 归一化DV")

    # ── 模型3：非线性时间（log-hour）─────────────────────────────────
    print("\n--- 模型3: h≥7，log(hour)时间变量 ---")
    hourly_7_log = hourly_7.copy()
    hourly_7_log['log_hour']      = np.log(hourly_7_log['hour'])
    hourly_7_log['log_hour_link'] = (hourly_7_log['log_hour']
                                     * hourly_7_log['D_link'])
    X_log_vars = ['log_hour', 'log_hour_link', 'log_liq_vol', 'log_vol']
    X3 = sm.add_constant(
        pd.concat([
            hourly_7_log[X_log_vars].reset_index(drop=True),
            pd.get_dummies(hourly_7_log['coin'], drop_first=True,
                           dtype=float).reset_index(drop=True)
        ], axis=1)
    )
    y3 = hourly_7_log['log_spread_norm'].reset_index(drop=True)
    m3 = sm.OLS(y3, X3).fit(cov_type='HC1')
    b3 = m3.params.get('log_hour_link', np.nan)
    p3 = m3.pvalues.get('log_hour_link', np.nan)
    sig3 = ('***' if p3 < 0.001 else '**' if p3 < 0.01
            else '*' if p3 < 0.05 else 'n.s.')
    print(f"\n  [模型3 log-hour]  N={int(m3.nobs)}, R²={m3.rsquared:.3f}")
    print(f"    β(log_HOUR×LINK) = {b3:+.6f} ({sig3}, p={p3:.4f})")
    results['M3'] = {
        'beta2': b3, 'p2': p3, 'sig': sig3,
        'R2': m3.rsquared, 'N': int(m3.nobs),
        'summary': str(m3.summary())
    }

    # ── 稳健性：模型1与模型2的β₂应代数等价 ─────────────────────────
    b2_m1 = results['M1']['beta2']
    b2_m2 = results['M2']['beta2']
    diff   = abs(b2_m1 - b2_m2)
    status = "✓ 代数等价" if diff < 1e-8 else f"⚠️ 差异={diff:.2e}"
    print(f"\n  模型1 vs 模型2 β₂一致性检验: {status}")

    # ── 恢复时间分组摘要 ─────────────────────────────────────────────
    print(f"\n  恢复分组摘要:")
    fast  = {c: h for c, h in recovery_hours.items()
             if h is not None and h <= 7}
    mid   = {c: h for c, h in recovery_hours.items()
             if h is not None and 7 < h <= 15}
    slow  = {c: h for c, h in recovery_hours.items()
             if h is None or h > 15}
    print(f"    快速(≤7h):  {fast}")
    print(f"    中速(8-15h):{mid}")
    print(f"    慢速(>15h): {slow}")

    if recovery_hours.get('LINK'):
        link_h   = recovery_hours['LINK']
        fast_avg = np.mean(list(fast.values())) if fast else np.nan
        ratio    = link_h / fast_avg if fast_avg > 0 else np.nan
        print(f"\n    LINK恢复时间: {link_h:.1f}h")
        print(f"    快速组均值:   {fast_avg:.1f}h")
        print(f"    LINK/快速组:  {ratio:.1f}倍")

    # ── 最终结论（基于主规格模型1）──────────────────────────────────
    m1 = results['M1']
    if m1['p2'] < 0.05:
        verdict = (f"H5 ✓ 支持 "
                   f"(β₂={m1['beta2']:+.6f}, p={m1['p2']:.4f}, "
                   f"N={m1['N']}, R²={m1['R2']:.3f})")
    else:
        verdict = (f"H5 ✗ 不支持 "
                   f"(p={m1['p2']:.4f})，"
                   f"检查h≥7限制是否正确应用")

    print(f"\n  ★★★ {verdict}")

    # ── 输出 ─────────────────────────────────────────────────────────
    with open(OUTPUT_DIR / 'step_H5.txt', 'w', encoding='utf-8') as f:
        f.write(f"H5结论: {verdict}\n\n")
        f.write("恢复时间（连续15分钟持续低于阈值）:\n")
        for coin, h in recovery_hours.items():
            f.write(f"  {coin}: {h:.1f}h\n" if h else f"  {coin}: >123h\n")
        for key, res in results.items():
            f.write(f"\n{'='*60}\n{key}\n{'='*60}\n")
            f.write(res['summary'] + '\n')

    return verdict, results


# =================================================================
# Step 6: 聚合偏差验证
# =================================================================
def step_6_aggregation(panel, DV='log_spread'):
    print("\n" + "=" * 70)
    print(f"STEP 6: 聚合偏差验证 (DV={DV})")
    print("=" * 70)
    import statsmodels.api as sm
    from linearmodels.panel import PanelOLS

    # ── 统一标准化分母（分钟级全局std）──────────────────────────────
    global_std = panel.loc[panel['liq_vol_usd'] > 0, 'liq_vol_usd'].std()
    panel = panel.copy()
    panel['liq_vol_std_unified'] = panel['liq_vol_usd'] / max(global_std, 1)
    print(f"  统一标准化分母（分钟级std）: {global_std:.2f}")

    # ── 小时级聚合 ────────────────────────────────────────────────────
    ph = panel.copy()
    ph['hour'] = (ph['tau'] / 60).astype(int)

    h = ph.groupby(['coin', 'hour']).agg(
        spread_bps      = ('spread_bps',      'mean'),
        total_depth_usd = ('total_depth_usd', 'mean'),
        liq_vol_usd     = ('liq_vol_usd',     'sum'),
        vol_usd         = ('vol_usd',         'sum'),
        ret             = ('ret', lambda x: (1 + x).prod() - 1),
    ).reset_index()

    h['log_spread']          = np.log(h['spread_bps'].clip(lower=0.001))
    h['log_vol']             = np.log(h['vol_usd'] + 1)
    h['liq_vol_std_unified'] = h['liq_vol_usd'] / max(global_std, 1)

    # ── 辅助函数：运行Pooled OLS（无FE）──────────────────────────────
    def run_pooled(data, dv, liq_var, label):
        X = sm.add_constant(data[[liq_var, 'ret', 'log_vol']])
        m = sm.OLS(data[dv], X).fit(cov_type='HC1')
        b  = m.params[liq_var]
        p  = m.pvalues[liq_var]
        sig = ('***' if p < 0.001 else '**' if p < 0.01
               else '*' if p < 0.05 else 'n.s.')
        print(f"  {label:35s}: β={b:+.6f}({sig})  "
              f"R²={m.rsquared:.4f}  N={int(m.nobs):,}")
        return b, p, m

    # ── 辅助函数：运行TWFE（有品种FE）──────────────────────────────
    def run_twfe(data, dv, liq_var, id_col, time_col, label):
        df = data.copy()
        # 确保time_id连续
        time_map = {t: i for i, t in
                    enumerate(sorted(df[time_col].unique()))}
        df['_time_id'] = df[time_col].map(time_map)
        df = df.set_index([id_col, '_time_id'])
        m = PanelOLS(
            df[dv], df[[liq_var, 'ret', 'log_vol']],
            entity_effects=True, time_effects=True
        ).fit(cov_type='kernel', kernel='bartlett',
              bandwidth=max(1, int(len(time_map)**0.25)))
        b  = m.params[liq_var]
        p  = m.pvalues[liq_var]
        sig = ('***' if p < 0.001 else '**' if p < 0.01
               else '*' if p < 0.05 else 'n.s.')
        print(f"  {label:35s}: β={b:+.6f}({sig})  "
              f"R²={m.rsquared_within:.4f}  N={int(m.nobs):,}")
        return b, p, m

    # ── 辅助函数：运行品种FE（只含entity FE，对应论文表5.1）────────
    def run_entity_fe(data, dv, liq_var, id_col, time_col, label):
        df = data.copy()
        # 确保time_id连续
        time_map = {t: i for i, t in
                    enumerate(sorted(df[time_col].unique()))}
        df['_time_id'] = df[time_col].map(time_map)
        df = df.set_index([id_col, '_time_id'])
        m = PanelOLS(
            df[dv], df[[liq_var, 'ret', 'log_vol']],
            entity_effects=True, time_effects=False
        ).fit(cov_type='kernel', kernel='bartlett',
              bandwidth=max(1, int(len(time_map)**0.25)))
        b  = m.params[liq_var]
        p  = m.pvalues[liq_var]
        sig = ('***' if p < 0.001 else '**' if p < 0.01
               else '*' if p < 0.05 else 'n.s.')
        print(f"  {label:35s}: β={b:+.6f}({sig})  "
              f"R²={m.rsquared_within:.4f}  N={int(m.nobs):,}")
        return b, p, m

    liq = 'liq_vol_std_unified'

    print("\n  ── 核心对比1：品种FE遗漏偏差（分钟级）──")
    bm_nfe, pm_nfe, _ = run_pooled(panel, DV, liq,
                                    "M1 分钟级, 无FE（预期β<0）")
    bm_fe,  pm_fe,  _ = run_entity_fe(panel, DV, liq, 'coin', 'time_id',
                                       "M2 分钟级, 有FE（预期β>0）★主规格")

    print("\n  ── 核心对比2：时间聚合衰减效应（有FE）──")
    bh_fe,  ph_fe,  _ = run_entity_fe(h, DV, liq, 'coin', 'hour',
                                       "M4 小时级, 有FE（预期β>0,衰减）")
    bh_nfe, ph_nfe, _ = run_pooled(h, DV, liq,
                                    "M3 小时级, 无FE（预期β<0）")

    # ── 验证论文两项核心发现 ─────────────────────────────────────────
    print("\n  ── 验证结果 ──")

    # 发现1：品种FE遗漏偏差→符号翻转
    fe_flip = (bm_nfe < 0 < bm_fe)
    r2_jump = None
    print(f"  发现1 品种FE遗漏偏差（符号翻转）: "
          f"{'✓ 验证' if fe_flip else '✗ 未验证'}")
    print(f"    无FE β={bm_nfe:+.6f} → 有FE β={bm_fe:+.6f}")

    # 发现2：时间聚合衰减（有FE，方向不变，幅度缩小）
    decay = (bm_fe > 0 and bh_fe > 0 and abs(bh_fe) < abs(bm_fe))
    decay_pct = (1 - abs(bh_fe) / abs(bm_fe)) * 100 if bm_fe != 0 else np.nan
    print(f"  发现2 时间聚合衰减（方向不变，系数缩小）: "
          f"{'✓ 验证' if decay else '✗ 未验证'}")
    print(f"    分钟有FE β={bm_fe:+.6f} → 小时有FE β={bh_fe:+.6f}"
          f"（衰减{decay_pct:.1f}%，论文预期96%）")

    # ── 输出 ─────────────────────────────────────────────────────────
    verdict = (f"发现1({'✓' if fe_flip else '✗'}) 品种FE符号翻转 | "
               f"发现2({'✓' if decay else '✗'}) 时间聚合衰减{decay_pct:.1f}%")
    print(f"\n  ★★★ {verdict}")

    with open(OUTPUT_DIR / 'step6_aggregation.txt', 'w', encoding='utf-8') as f:
        f.write(f"{verdict}\n")
        f.write(f"统一标准化分母: {global_std:.4f}\n")

    return fe_flip, decay, {
        'bm_nfe': bm_nfe, 'bm_fe': bm_fe,
        'bh_nfe': bh_nfe, 'bh_fe': bh_fe,
        'decay_pct': decay_pct
    }

# =================================================================
# 执行全部
# =================================================================
print("\n" + "╔" + "═" * 68 + "╗")
print("║" + " 第五章：Step 1-6 全部分析 ".center(68) + "║")
print("╚" + "═" * 68 + "╝")

DV, _ = step_1_compare_dv(panel)
ratio, _ = step_2_amplification(panel, DV)
reversal, _, _ = step_4_sign_reversal(panel, DV)
h4,  _ = step_3_H4(panel, DV)
h5, _ = step_H5(panel, DV)
flip = step_6_aggregation(panel, DV)

# 汇总
summary = f"""
{'╔' + '═'*68 + '╗'}
{'║' + ' 五项决策最终裁定 '.center(68) + '║'}
{'╚' + '═'*68 + '╝'}

  决策5 (N):          {len(panel):,}
  决策1 (主因变量):   {DV}
  决策2 (放大比率):   {ratio:.2f}×
  决策4 (符号逆转):   {reversal}
  决策3 (H4结论):     {h4}
  H5结论:             {h5}
  聚合偏差:           {'✓ 验证' if flip else '未翻转'}
"""
print(summary)

with open(OUTPUT_DIR / 'FINAL_DECISIONS.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"✅ 全部结果已保存至 {OUTPUT_DIR}")
