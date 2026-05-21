"""
Figure 5.3 (Plan B): V-Shape Spread Pattern Around Liquidation Cascade
9 coins, tau = -30 to +30 minutes, each coin distinct color
Legend: upper left
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────
DATA_DIR   = Path(r'E:\data2\hyperliquid\hyperliquid_s3_data')
OUTPUT_DIR = Path(r'E:\data2\hyperliquid\ch5_output')
OUTPUT_DIR.mkdir(exist_ok=True)

COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'AVAX', 'HYPE', 'LINK']

TAU_LO, TAU_HI = -20, 40

# Hours filter: buffer for baseline calculation
H_FILTER_LO = -2.0          # 120 min before for stable baseline
H_FILTER_HI = TAU_HI / 60 + 0.05

# Baseline: tau in [-60, -1] minutes = hours in [-1.0, -1/60]
BASELINE_LO_H = -1.0
BASELINE_HI_H = -1 / 60

# ── Distinct colors for all 9 coins ────────────────────────────────────
COIN_STYLE = {
    'BTC':  dict(color='#1A6FCC', lw=2.2, zorder=6, ls='-'),
    'ETH':  dict(color='#2E8B57', lw=2.2, zorder=6, ls='-'),
    'SOL':  dict(color='#9B59B6', lw=1.8, zorder=5, ls='-'),
    'XRP':  dict(color='#E67E22', lw=1.8, zorder=5, ls='-'),
    'BNB':  dict(color='#F1C40F', lw=1.8, zorder=5, ls='-'),
    'DOGE': dict(color='#16A085', lw=1.8, zorder=5, ls='-'),
    'AVAX': dict(color='#E74C3C', lw=1.8, zorder=5, ls='-'),
    'HYPE': dict(color='#2980B9', lw=1.8, zorder=5, ls='--'),
    'LINK': dict(color='#CC0000', lw=3.0, zorder=7, ls='-'),  # thickest
}

# ── Load one coin ───────────────────────────────────────────────────────
def load_coin(coin):
    path = DATA_DIR / f'l2_spread_{coin}.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing: {path}')

    df = pd.read_csv(path, usecols=['hours_from_crash', 'spread_pct'])
    df = df.dropna(subset=['spread_pct', 'hours_from_crash'])
    df = df[(df['hours_from_crash'] >= H_FILTER_LO) &
            (df['hours_from_crash'] <= H_FILTER_HI)]

    # tau in minutes
    df['tau'] = (df['hours_from_crash'] * 60).round().astype(int)

    # Baseline median
    baseline_rows = df[(df['hours_from_crash'] >= BASELINE_LO_H) &
                       (df['hours_from_crash'] <= BASELINE_HI_H)]['spread_pct']
    baseline_med = baseline_rows.median()
    if baseline_med <= 0:
        raise ValueError(f'{coin}: non-positive baseline={baseline_med}')

    # Minute-level median, within display window
    minute = (df[df['tau'].between(TAU_LO, TAU_HI)]
              .groupby('tau')['spread_pct']
              .median()
              .reset_index()
              .rename(columns={'spread_pct': 'spread_med'}))

    minute['spread_norm'] = minute['spread_med'] / baseline_med
    minute['coin'] = coin

    tau0_val = minute.loc[minute['tau'] == 0, 'spread_norm'].values
    print(f'  {coin:4s}: baseline={baseline_med:.5f}%  '
          f'tau=0 norm={tau0_val[0]:.1f}x  '
          f'tau pts={len(minute)}')
    return minute


# ── Load all ────────────────────────────────────────────────────────────
def load_all():
    frames = []
    for coin in COINS:
        print(f'Loading {coin}...')
        try:
            frames.append(load_coin(coin))
        except Exception as e:
            print(f'  ERROR {coin}: {e}')
    return pd.concat(frames, ignore_index=True)


# ── Plot ────────────────────────────────────────────────────────────────
def make_fig(df_all):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#CCCCCC')
    ax.grid(True, linestyle='--', linewidth=0.4,
            color='#E8E8E8', zorder=0, which='both')

    # ── Region shadings ────────────────────────────────────────────────
    ax.axvspan(TAU_LO - 0.5, -0.5,
               color='#EEEEEE', alpha=0.55, zorder=1)          # pre-event
    ax.axvspan(-0.5, 2.5,
               color='#D0E8FF', alpha=0.50, zorder=1)          # Stage 1
    ax.axvspan(2.5, TAU_HI + 0.5,
               color='#FFE8E8', alpha=0.22, zorder=1)          # Stage 2

    # ── Draw each coin ─────────────────────────────────────────────────
    for coin in COINS:
        sub = df_all[df_all['coin'] == coin].sort_values('tau')
        if sub.empty:
            continue
        style = COIN_STYLE[coin]
        ax.plot(sub['tau'], sub['spread_norm'],
                color=style['color'], lw=style['lw'],
                ls=style['ls'], zorder=style['zorder'],
                label=coin)

        # End-of-line label at tau=+30
        last = sub[sub['tau'] == sub['tau'].max()]
        if not last.empty:
            ax.text(last['tau'].values[0] + 0.4,
                    last['spread_norm'].values[0],
                    coin,
                    fontsize=8.5,
                    color=style['color'],
                    fontweight='bold' if coin == 'LINK' else 'normal',
                    va='center', zorder=8)

    # ── Reference lines ────────────────────────────────────────────────
    ax.axhline(1.0, color='#555555', lw=1.0, ls='--', zorder=3)
    ax.axvline(0,   color='#E04000', lw=1.8, ls=':',  zorder=5)
    ax.axvline(3,   color='#228B22', lw=1.5, ls='--', zorder=5)

    # ── Vertical line labels: inside axes, top area, staggered vertically ─
    # tau=0: higher position, right of line (avoids legend overlap)
    ax.text(0.4, 0.97, 'tau=0\n(event onset)',
            color='#E04000', fontsize=8.5, ha='left', va='top',
            transform=ax.get_xaxis_transform(), zorder=6)
    # tau=+3: slightly lower, right of line
    ax.text(3.3, 0.88, 'tau=+3\n(sign reversal)',
            color='#228B22', fontsize=8.5, ha='left', va='top',
            transform=ax.get_xaxis_transform(), zorder=6)

    # ── Axes ───────────────────────────────────────────────────────────
    ax.set_xlim(TAU_LO - 0.5, TAU_HI + 3)   # +3 for end labels
    ax.set_yscale('log')
    ax.yaxis.set_minor_locator(plt.NullLocator())   # remove minor tick marks
    ax.set_xticks(range(TAU_LO, TAU_HI + 1, 5))
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlabel('Event time (minutes relative to tau = 0)', fontsize=11)
    ax.set_ylabel('Normalized spread\n(ratio to pre-event baseline median)',
                  fontsize=11)
    ax.set_title(
        'Figure 5.3  Spread V-Shape: Actual Bid-Ask Spreads Around Liquidation Cascade\n'
        'Minute-level median spread / pre-event baseline, tau = \u221220 to +40  '
        '(Oct 10 2025, 9 perpetual contracts)',
        fontsize=12, fontweight='bold', pad=10
    )

    # ── Legend: upper left ─────────────────────────────────────────────
    coin_handles = [
        Line2D([0],[0],
               color=COIN_STYLE[c]['color'],
               lw=COIN_STYLE[c]['lw'],
               ls=COIN_STYLE[c]['ls'],
               label=c)
        for c in COINS
    ]
    ref_handles = [
        Line2D([0],[0], color='#555555', lw=1.0, ls='--',
               label='Pre-event baseline (1\u00d7)'),
        Line2D([0],[0], color='#E04000', lw=1.8, ls=':',
               label='tau=0 (event onset)'),
        Line2D([0],[0], color='#228B22', lw=1.5, ls='--',
               label='tau=+3 (sign reversal)'),
        mpatches.Patch(color='#D0E8FF', alpha=0.6,
                       label='Stage 1: speculative (tau 0\u20132)'),
        mpatches.Patch(color='#FFE8E8', alpha=0.5,
                       label='Stage 2: MM retreat (tau \u22653)'),
    ]
    ax.legend(handles=coin_handles + ref_handles,
              loc='upper left', fontsize=8.5,
              framealpha=0.93, edgecolor='#CCCCCC',
              ncol=2)

    # ── Footnote ───────────────────────────────────────────────────────
    fig.text(
        0.5, -0.03,
        'Note: Each point is the minute-bin median of spread_pct '
        '(bid-ask spread as % of mid price). '
        'Normalized spread = minute median \u00f7 pre-event median '
        '(tau \u2208 [\u221260, \u22121] min). '
        'Y-axis log scale. '
        'LINK (thick red) shows the most extreme spread expansion (189\u00d7 at tau=0). '
        'Source: Hyperliquid S3 l2book snapshots (\u223c250 ms interval).',
        ha='center', fontsize=8, style='italic', color='#444444', wrap=True
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    for fmt in ['pdf', 'png']:
        out = OUTPUT_DIR / f'fig5_3_spread_vshape.{fmt}'
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {out}')

    plt.show()
    return fig


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 55)
    print('Loading l2book spread data for 9 coins...')
    print('=' * 55)
    df_all = load_all()
    print(f'\nTotal coin-minute rows: {len(df_all)}')

    print('\nNormalized spread at tau=0 (sanity check):')
    check = (df_all[df_all['tau'] == 0]
             [['coin', 'spread_norm']]
             .sort_values('spread_norm', ascending=False))
    print(check.to_string(index=False))

    make_fig(df_all)
