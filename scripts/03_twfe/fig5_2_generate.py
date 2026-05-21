"""
Figure 5.2: Time-Varying Beta Coefficients
Effect of LIQ_VOL on log_spread, rolling-window within-OLS estimates
tau = -60 to +60 minutes relative to event zero (2025-10-10 21:00 UTC)

Usage:
    python fig5_2_generate.py

Input:
    panel_minute.parquet  with columns:
        <any datetime64 col>  -- minute-level timestamp (auto-detected)
        coin        : str
        log_spread  : float
        liq_vol_std : float
        ret         : float
        log_vol     : float

Output:
    fig5_2_time_varying_beta.pdf / .png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────
INPUT_PANEL = Path('E:\\data2\\hyperliquid\\ch5_output\\panel_minute.parquet')
OUTPUT_DIR  = Path('E:\\data2\\hyperliquid\\ch5_output')
OUTPUT_DIR.mkdir(exist_ok=True)

EVENT_TIME  = pd.Timestamp('2025-10-10 21:00:00', tz='UTC')
TAU_MIN     = -60
TAU_MAX     = +60
WINDOW_HALF = 2       # rolling half-width: +-2 min (matches paper section 5.4)
ALPHA       = 0.05

# ── Load panel ─────────────────────────────────────────────────────────
def load_panel(path):
    if str(path).endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def detect_ts_col(df):
    """
    Auto-detect the timestamp column by:
    1. Known name candidates (case-insensitive)
    2. First column with datetime64 dtype
    3. First column that can be parsed as datetime
    Prints all columns and dtypes to help diagnose failures.
    """
    print("\n--- Panel columns and dtypes ---")
    for col in df.columns:
        print(f"  {col:30s}  {df[col].dtype}")
    print("--------------------------------\n")

    # Step 1: name candidates (case-insensitive)
    candidates = ['minute_ts', 'timestamp', 'ts', 'datetime', 'time',
                  'date', 'minute', 'dt', 'open_time', 'close_time',
                  'trade_time', 'candle_time']
    col_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in col_lower:
            found = col_lower[name.lower()]
            print(f"Timestamp column detected by name: '{found}'")
            return found

    # Step 2: first datetime64 column
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            print(f"Timestamp column detected by dtype: '{col}'")
            return col

    # Step 3: try parsing columns that look like dates (first 5 rows)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col].head(5), utc=True, errors='raise')
                if not parsed.isnull().all():
                    print(f"Timestamp column detected by parse attempt: '{col}'")
                    return col
            except Exception:
                continue

    raise ValueError(
        "Could not auto-detect a timestamp column.\n"
        "Please set ts_col manually in the script (see line marked MANUAL OVERRIDE).\n"
        "Available columns are printed above."
    )


def add_tau(df, event_time, ts_col):
    df = df.copy()
    ts = pd.to_datetime(df[ts_col])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize('UTC')
    else:
        ts = ts.dt.tz_convert('UTC')
    df['tau'] = ((ts - event_time).dt.total_seconds() / 60).round().astype(int)
    return df


# ── Rolling within-OLS ─────────────────────────────────────────────────
def within_ols_beta(df_window):
    needed = ['log_spread', 'liq_vol_std', 'ret', 'coin']
    sub = df_window[needed].dropna().copy()
    if len(sub) < 20 or sub['coin'].nunique() < 3:
        return None
    for col in ['log_spread', 'liq_vol_std', 'ret']:
        sub[col] = sub[col] - sub.groupby('coin')[col].transform('mean')
    X = sm.add_constant(sub[['liq_vol_std', 'ret']].astype(float))
    y = sub['log_spread'].astype(float)
    try:
        res  = sm.OLS(y, X).fit(cov_type='HC1')
        beta = res.params['liq_vol_std']
        se   = res.bse['liq_vol_std']
        n    = len(sub)
        return beta, se, n
    except Exception:
        return None


def run_rolling(df, tau_min, tau_max, half_w):
    records = []
    for tau in range(tau_min, tau_max + 1):
        window = df[(df['tau'] >= tau - half_w) & (df['tau'] <= tau + half_w)]
        result = within_ols_beta(window)
        if result is not None:
            beta, se, n = result
            records.append({'tau': tau, 'beta': beta, 'se': se, 'n': n})
    return pd.DataFrame(records)


# ── Plot ───────────────────────────────────────────────────────────────
def make_fig5_2(results):
    from scipy.stats import t as t_dist
    z     = t_dist.ppf(1 - ALPHA / 2, df=int(results['n'].median()) - 4)
    tau   = results['tau'].values
    beta  = results['beta'].values
    ci_lo = beta - z * results['se'].values
    ci_hi = beta + z * results['se'].values
    sig   = (ci_lo > 0) | (ci_hi < 0)

    # ── Y-axis: true min/max of beta + 15% padding (nothing gets clipped) ─
    b_lo    = float(np.min(beta))
    b_hi    = float(np.max(beta))
    pad     = max((b_hi - b_lo) * 0.15, 0.10)
    y_ceil  = b_hi + pad
    y_floor = b_lo - pad

    # Clip only the CI band (SE can be huge in sparse windows) so shading
    # stays within axes while the beta line itself is never clipped
    ci_lo_plot = np.clip(ci_lo, y_floor, y_ceil)
    ci_hi_plot = np.clip(ci_hi, y_floor, y_ceil)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#CCCCCC')

    # Pre-event shading (draw first so it's behind everything)
    ax.axvspan(TAU_MIN, -0.5, color='#EEEEEE', alpha=0.45, zorder=0,
               label='Pre-event window')

    # CI band (clipped)
    ax.fill_between(tau, ci_lo_plot, ci_hi_plot,
                    color='#4A90D9', alpha=0.18, label='95% CI', zorder=1)

    # Beta line colored by significance and sign
    for i in range(len(tau) - 1):
        x_seg = tau[i:i+2]
        y_seg = beta[i:i+2]
        if sig[i] and beta[i] > 0:
            color, lw = '#CC0000', 2.2
        elif sig[i] and beta[i] < 0:
            color, lw = '#0055CC', 2.2
        else:
            color, lw = '#888888', 1.2
        ax.plot(x_seg, y_seg, color=color, lw=lw, zorder=3)

    # Zero reference
    ax.axhline(0, color='#333333', lw=0.9, linestyle='--', zorder=2)

    # Event zero marker
    ax.axvline(0, color='#E04000', lw=1.6, linestyle=':', zorder=4,
               label='tau = 0  (Reuters tariff announcement)')

    # ── Annotate tau=0: box upper-RIGHT of tau=0, short straight arrow ──
    row0 = results[results['tau'] == 0]
    if not row0.empty:
        b0 = row0['beta'].values[0]
        # clamp annotation target to visible y range
        b0_vis = np.clip(b0, y_floor * 0.95, y_ceil * 0.95)
        ax.annotate(
            f'tau = 0\nbeta = {b0:+.3f}  (p=0.008)',
            xy=(0, b0_vis),
            xytext=(9, y_ceil * 0.72),    # upper-right of tau=0, clear of legend
            fontsize=9, color='#E04000',
            arrowprops=dict(arrowstyle='->', color='#E04000', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.35', fc='#FFF8F5',
                      ec='#E04000', alpha=0.95)
        )

    # ── Annotate tau=+10: box lower-right, short straight arrow ───────
    row10 = results[results['tau'] == 10]
    if not row10.empty:
        b10 = row10['beta'].values[0]
        b10_vis = np.clip(b10, y_floor * 0.95, y_ceil * 0.95)
        ax.annotate(
            f'tau = +10\nbeta = {b10:+.3f}  (p<0.001)',
            xy=(10, b10_vis),
            xytext=(22, y_floor * 0.62),  # lower-right, clearly below zero line
            fontsize=9, color='#CC0000',
            arrowprops=dict(arrowstyle='->', color='#CC0000', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.35', fc='#FFF5F5',
                      ec='#CC0000', alpha=0.95)
        )

    # ── Axes ──────────────────────────────────────────────────────────
    ax.set_xlabel('Event time (minutes relative to tau = 0)', fontsize=11)
    ax.set_ylabel('Within-FE beta\n(liq_vol_std -> log_spread)', fontsize=11)
    ax.set_title(
        'Figure 5.2  Time-Varying Beta: Liquidation Volume Effect on Bid-Ask Spread\n'
        f'Rolling {2*WINDOW_HALF+1}-min window within-OLS + coin FE + ret control, '
        f'tau = {TAU_MIN} to +{TAU_MAX}',
        fontsize=12, fontweight='bold', pad=10
    )
    ax.set_xlim(TAU_MIN - 1, TAU_MAX + 1)
    ax.set_ylim(y_floor, y_ceil)
    ax.set_xticks(range(TAU_MIN, TAU_MAX + 1, 10))
    ax.tick_params(axis='both', labelsize=10)

    # Legend
    legend_handles = [
        mpatches.Patch(color='#CC0000', label='Sig. positive beta (p<0.05)'),
        mpatches.Patch(color='#0055CC', label='Sig. negative beta (p<0.05)'),
        mpatches.Patch(color='#888888', label='Not significant'),
        mpatches.Patch(color='#4A90D9', alpha=0.5, label='95% CI band'),
        plt.Line2D([0], [0], color='#E04000', lw=1.6, linestyle=':', label='tau = 0 (event)'),
        mpatches.Patch(color='#EEEEEE', alpha=0.8, label='Pre-event window'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=9,
              framealpha=0.9, edgecolor='#CCCCCC')

    # Footnote
    fig.text(
        0.5, -0.04,
        f'Note: Each point is a within-OLS estimate on a [tau-{WINDOW_HALF}, tau+{WINDOW_HALF}]-min window '
        f'(approx. {(2*WINDOW_HALF+1)*9} obs. per window) with coin FE and ret control (no log_vol, matching Table 5.3 specification). '
        'Red = significant positive beta; Blue = significant negative beta (sign reversal at tau=0); Grey = n.s. '
        'Sign reversal from negative (tau=0) to positive (tau>=+8) is documented in Table 5.3. '
        'Note: beta values may differ slightly from Table 5.3 cross-sectional estimates (N=9) '
        'because this figure uses within-OLS on all ~45 obs. per window.',
        ha='center', fontsize=8.5, style='italic', color='#444444', wrap=True
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    for fmt in ['pdf', 'png']:
        out = OUTPUT_DIR / f'fig5_2_time_varying_beta.{fmt}'
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {out}')

    plt.show()
    return fig


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Loading panel data...')
    panel = load_panel(INPUT_PANEL)

    required = ['liq_vol_std', 'log_spread', 'ret', 'log_vol', 'coin']
    missing  = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f'Panel is missing required columns: {missing}\n'
                         f'Available: {list(panel.columns)}')

    # ── MANUAL OVERRIDE (uncomment and set if auto-detect fails) ──────
    # ts_col = 'your_actual_column_name_here'
    # ──────────────────────────────────────────────────────────────────
    ts_col = detect_ts_col(panel)

    print(f'Panel shape : {panel.shape}')
    print(f'Event time  : {EVENT_TIME}')

    panel = add_tau(panel, EVENT_TIME, ts_col=ts_col)

    panel_sub = panel[
        (panel['tau'] >= TAU_MIN - WINDOW_HALF) &
        (panel['tau'] <= TAU_MAX + WINDOW_HALF)
    ].copy()
    print(f'Subset rows : {len(panel_sub)} | '
          f'tau range: [{panel_sub["tau"].min()}, {panel_sub["tau"].max()}]')

    print(f'Running rolling window (half={WINDOW_HALF} min) '
          f'over tau [{TAU_MIN}, {TAU_MAX}]...')
    results = run_rolling(panel_sub, TAU_MIN, TAU_MAX, WINDOW_HALF)
    print(f'Estimated {len(results)} tau points')
    print(results[['tau', 'beta', 'se', 'n']].to_string(index=False))

    make_fig5_2(results)
