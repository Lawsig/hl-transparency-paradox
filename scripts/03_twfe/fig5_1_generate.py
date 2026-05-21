"""
Figure 5.1: Omitted Variable Bias Visualization
(a) No coin FE: Pooled OLS -> beta = -0.041 (sign error)
(b) With coin FE: Within-demeaned OLS -> beta = +0.044 (correct sign)

Usage:
    python fig5_1_generate.py
Input:  panel DataFrame with columns: liq_vol_std, log_spread, ret, log_vol, coin
Output: fig5_1_omitted_variable_bias.pdf / .png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import statsmodels.api as sm
from pathlib import Path

# ── Path configuration (edit as needed) ───────────────────────────────
INPUT_PANEL = Path('E:\\data2\\hyperliquid\\ch5_output\\panel_minute.parquet')
OUTPUT_DIR  = Path('E:\\data2\\hyperliquid\\ch5_output')
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Coin color mapping ─────────────────────────────────────────────────
COIN_COLORS = {
    'BTC':  '#F7931A',
    'ETH':  '#627EEA',
    'SOL':  '#9945FF',
    'XRP':  '#00AAE4',
    'BNB':  '#F3BA2F',
    'DOGE': '#C2A633',
    'AVAX': '#E84142',
    'HYPE': '#00D4AA',
    'LINK': '#2A5ADA',
}

def load_panel(path):
    if str(path).endswith('.parquet'):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)

def compute_within(df, x_col, y_col, group_col):
    df = df.copy()
    for col, new_col in [(x_col, 'x_within'), (y_col, 'y_within')]:
        group_mean = df.groupby(group_col)[col].transform('mean')
        df[new_col] = df[col] - group_mean
    return df

def add_ols_line(ax, x, y, color='black', lw=2.5, label=None, annotate_beta=True):
    X = sm.add_constant(x)
    m = sm.OLS(y, X).fit()
    beta = m.params.iloc[1]
    x_range = np.linspace(x.quantile(0.005), x.quantile(0.995), 200)
    y_pred  = m.params.iloc[0] + beta * x_range
    ax.plot(x_range, y_pred, color=color, lw=lw, zorder=5, label=label)
    if annotate_beta:
        x_ann = x.quantile(0.90)
        y_ann = m.params.iloc[0] + beta * x_ann
        sign  = '+' if beta >= 0 else ''
        ax.annotate(
            f'beta = {sign}{beta:.3f}',
            xy=(x_ann, y_ann),
            xytext=(10, 12),
            textcoords='offset points',
            fontsize=11, fontweight='bold',
            color=color,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color, alpha=0.85)
        )
    return beta

def fwl_residuals(df, y_col, x_col, controls, add_coin_dummies=False):
    """
    Frisch-Waugh-Lovell partial regression.
    Returns (e_x, e_y): residuals after projecting x and y onto controls
    (and optionally coin dummies). Slope of e_y ~ e_x equals the multivariate beta.
    """
    df2 = df[[y_col, x_col] + controls + ['coin']].dropna().copy()
    ctrl_df = df2[controls].copy()
    if add_coin_dummies:
        dummies = pd.get_dummies(df2['coin'], drop_first=True).astype(float)
        ctrl_df = pd.concat([ctrl_df.reset_index(drop=True),
                             dummies.reset_index(drop=True)], axis=1)
        ctrl_df.index = df2.index
    Z = sm.add_constant(ctrl_df.astype(float))
    e_x = sm.OLS(df2[x_col].astype(float), Z).fit().resid
    e_y = sm.OLS(df2[y_col].astype(float), Z).fit().resid
    return e_x, e_y, df2['coin']


def make_fig5_1(panel):
    # ── Full panel for regression (N=116,037, matches Step6 / Table 5.1) ─
    df_reg   = panel.dropna(subset=['liq_vol_std', 'log_spread', 'ret', 'log_vol', 'coin']).copy()
    controls = ['ret', 'log_vol']
    coins    = sorted(df_reg['coin'].unique())

    # ── FWL residuals (slope = multivariate beta by Frisch-Waugh-Lovell) ─
    # Panel (a): partial out ret + log_vol only (no coin dummies)
    ex_a, ey_a, coin_a = fwl_residuals(df_reg, 'log_spread', 'liq_vol_std',
                                        controls, add_coin_dummies=False)
    # Panel (b): partial out ret + log_vol + coin dummies
    ex_b, ey_b, coin_b = fwl_residuals(df_reg, 'log_spread', 'liq_vol_std',
                                        controls, add_coin_dummies=True)

    # ── Scatter subsample (winsorized for visual clarity) ─────────────
    SAMPLE_N = 8000

    def winsor_sample(ex, ey, coin_series, n=SAMPLE_N):
        tmp = pd.DataFrame({'ex': ex.values, 'ey': ey.values,
                            'coin': coin_series.values}, index=ex.index)
        lo_x, hi_x = tmp['ex'].quantile(0.01), tmp['ex'].quantile(0.99)
        lo_y, hi_y = tmp['ey'].quantile(0.005), tmp['ey'].quantile(0.995)
        tmp = tmp[(tmp['ex'] >= lo_x) & (tmp['ex'] <= hi_x) &
                  (tmp['ey'] >= lo_y) & (tmp['ey'] <= hi_y)]
        return tmp.sample(n=min(n, len(tmp)), random_state=42)

    plot_a = winsor_sample(ex_a, ey_a, coin_a)
    plot_b = winsor_sample(ex_b, ey_b, coin_b)

    # ── R2 calculations ───────────────────────────────────────────────
    # Panel (a): overall R² from multivariate OLS without coin FE
    X_a  = sm.add_constant(df_reg[['liq_vol_std', 'ret', 'log_vol']].astype(float))
    r2_a = sm.OLS(df_reg['log_spread'].astype(float), X_a).fit().rsquared

    # Panel (b): within R² — demean ALL variables by coin mean, then run OLS
    # This matches PanelOLS within R² (Table 5.1: 0.187)
    cols_dm = ['log_spread', 'liq_vol_std', 'ret', 'log_vol']
    df_dm   = df_reg[cols_dm + ['coin']].copy()
    for col in cols_dm:
        df_dm[col] = df_dm[col] - df_dm.groupby('coin')[col].transform('mean')
    X_dm = sm.add_constant(df_dm[['liq_vol_std', 'ret', 'log_vol']].astype(float))
    r2_b = sm.OLS(df_dm['log_spread'].astype(float), X_dm).fit().rsquared

    # ── Canvas ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor('white')

    titles = [
        'No Coin Fixed Effects (Pooled OLS)\nSign Error: beta < 0',
        'With Coin Fixed Effects (Within-Demeaned)\nCorrect Sign: beta > 0',
    ]
    x_labels = [
        'liq_vol_std | ret, log_vol  (FWL partial residual)',
        'liq_vol_std | ret, log_vol, coin FE  (FWL partial residual)',
    ]
    y_labels = [
        'log_spread | ret, log_vol  (FWL partial residual)',
        'log_spread | ret, log_vol, coin FE  (FWL partial residual)',
    ]

    for idx, ax in enumerate(axes):
        ax.set_facecolor('#FAFAFA')
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('#CCCCCC')
        ax.grid(True, linestyle='--', linewidth=0.5, color='#E0E0E0', zorder=0)

        plot_df    = plot_a if idx == 0 else plot_b
        reg_ex     = ex_a   if idx == 0 else ex_b
        reg_ey     = ey_a   if idx == 0 else ey_b
        reg_r2     = r2_a   if idx == 0 else r2_b
        line_color = '#CC0000' if idx == 0 else '#0055CC'
        r2_label   = 'R2' if idx == 0 else 'R2 (within)'

        for coin in coins:
            mask = plot_df['coin'] == coin
            ax.scatter(plot_df.loc[mask, 'ex'], plot_df.loc[mask, 'ey'],
                       c=COIN_COLORS.get(coin, '#888888'),
                       s=8, alpha=0.35, linewidths=0, zorder=2,
                       label=coin if idx == 0 else None)

        add_ols_line(ax, reg_ex, reg_ey, color=line_color, lw=2.5)

        ax.text(0.04, 0.96, f'{r2_label} = {reg_r2:.3f}',
                transform=ax.transAxes, fontsize=10, va='top', color=line_color,
                bbox=dict(boxstyle='round', fc='white', ec=line_color, alpha=0.8))

        if idx == 1:
            ax.axhline(0, color='#AAAAAA', lw=0.8, linestyle=':')
            ax.axvline(0, color='#AAAAAA', lw=0.8, linestyle=':')

        ax.set_xlabel(x_labels[idx], fontsize=10)
        ax.set_ylabel(y_labels[idx], fontsize=10)
        ax.set_title(f'{"(a)" if idx==0 else "(b)"}  {titles[idx]}',
                     fontsize=12, fontweight='bold', pad=10)

    # ── Legend ────────────────────────────────────────────────────────
    handles = [mpatches.Patch(color=COIN_COLORS.get(c, '#888'), label=c) for c in coins]
    fig.legend(handles=handles, title='Asset', title_fontsize=10,
               loc='lower center', ncol=len(coins),
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    # ── Title and footnote ────────────────────────────────────────────
    fig.suptitle(
        'Figure 5.1  Omitted Variable Bias: Liquidation Volume vs. Bid-Ask Spread',
        fontsize=13, fontweight='bold', y=1.01
    )
    fig.text(
        0.5, -0.10,
        'Note: Scatter shows FWL partial residuals (8,000 obs. random sample); '
        'regression line slope equals the multivariate beta (N=116,037). '
        '(a) Pooled OLS (controlling for ret, log_vol) yields spurious beta=-0.041 '
        'because BTC (high liq. volume, narrow spread) and LINK (low liq. volume, wide spread) '
        'create cross-sectional confounding. '
        '(b) Adding coin FE removes confounding; beta flips to +0.044*** as predicted by theory. '
        'R2 in (a) is overall R2; R2 (within) in (b) is computed on coin-demeaned data, '
        'matching PanelOLS within R2.',
        ha='center', fontsize=9, style='italic', color='#444444', wrap=True
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    for fmt in ['pdf', 'png']:
        out = OUTPUT_DIR / f'fig5_1_omitted_variable_bias.{fmt}'
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {out}')

    plt.show()
    return fig


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Loading panel data...')
    panel = load_panel(INPUT_PANEL)

    required = ['liq_vol_std', 'log_spread', 'coin']
    missing  = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f'Panel is missing columns: {missing}\n'
                         f'Available columns: {list(panel.columns)}')

    print(f'Panel shape: {panel.shape}, assets: {sorted(panel["coin"].unique())}')
    make_fig5_1(panel)
