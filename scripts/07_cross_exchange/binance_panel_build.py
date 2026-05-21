"""R1.3 Step 2: 从 raw zip 构建 Binance 分钟级 panel

输入：E:\\data2\\binance\\raw\\<type>\\<coin>\\*.zip
输出：E:\\data2\\binance\\binance_panel_minute.parquet

字段（与 HL panel 同结构）：
  minute, coin, vol_usd, n_trades, buy_vol_usd, sell_vol_usd,
  spread_bps (Roll 1984 estimator), top_depth_usd (bookDepth pct=1),
  implied_liq_vol_usd (ΔOI_value 突降), liq_ratio (implied_liq / total vol)
"""
import os, sys, json, zipfile, glob, time
from io import BytesIO
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAW = r"E:\data2\binance\raw"
OUT_PARQ = r"E:\data2\binance\binance_panel_minute.parquet"
COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'HYPEUSDT']
# Map to short names matching HL panel
COIN_MAP = {'BTCUSDT':'BTC','ETHUSDT':'ETH','SOLUSDT':'SOL','XRPUSDT':'XRP','BNBUSDT':'BNB',
            'DOGEUSDT':'DOGE','AVAXUSDT':'AVAX','LINKUSDT':'LINK','HYPEUSDT':'HYPE'}


def read_zip_csv(zip_path, header=None):
    """Read first CSV inside zip"""
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith('.csv'):
                with z.open(name) as f:
                    if header is not None:
                        return pd.read_csv(f, header=header)
                    return pd.read_csv(f)
    return None


def process_aggtrades(coin_full, coin_short, dates):
    """aggTrades → minute vol_usd, n_trades, buy/sell taker vol, Roll's spread"""
    all_trades = []
    for d in dates:
        zp = os.path.join(RAW, 'aggTrades', coin_full, f'{coin_full}-aggTrades-{d}.zip')
        if not os.path.exists(zp): continue
        # Binance aggTrades header: agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker
        df = read_zip_csv(zp, header=0)
        if df is None or len(df) == 0: continue
        # Normalize column names (Binance sometimes ships without header in old format)
        df.columns = [c.strip().lower() for c in df.columns]
        # Build standard columns
        if 'transact_time' not in df.columns:
            df.rename(columns={df.columns[5]:'transact_time'}, inplace=True)
        if 'price' not in df.columns:
            df.rename(columns={df.columns[1]:'price'}, inplace=True)
        if 'quantity' not in df.columns:
            df.rename(columns={df.columns[2]:'quantity'}, inplace=True)
        if 'is_buyer_maker' not in df.columns:
            df.rename(columns={df.columns[6]:'is_buyer_maker'}, inplace=True)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['minute'] = pd.to_datetime(df['transact_time'], unit='ms', utc=True).dt.floor('min')
        df['vol_usd'] = df['price'] * df['quantity']
        # taker side: is_buyer_maker=False → taker BOUGHT (price ↑)
        if df['is_buyer_maker'].dtype == 'object':
            df['is_buyer_maker'] = df['is_buyer_maker'].astype(str).str.lower().isin(['true','1','t'])
        df['taker_buy'] = (~df['is_buyer_maker']).astype(int)
        df['buy_vol_usd'] = df['vol_usd'] * df['taker_buy']
        df['sell_vol_usd'] = df['vol_usd'] * (1 - df['taker_buy'])
        all_trades.append(df[['minute','price','vol_usd','buy_vol_usd','sell_vol_usd','taker_buy']])
    if not all_trades: return pd.DataFrame()
    big = pd.concat(all_trades, ignore_index=True).sort_values('minute')

    # Per-minute aggregation
    grouped = big.groupby('minute')
    panel = grouped.agg(
        vol_usd=('vol_usd','sum'),
        n_trades=('vol_usd','count'),
        buy_vol_usd=('buy_vol_usd','sum'),
        sell_vol_usd=('sell_vol_usd','sum'),
        last_price=('price','last'),
    ).reset_index()

    # Roll's spread estimator: 2·sqrt(max(0, -cov(Δp_t, Δp_{t-1})))
    # Compute per minute on the price series within that minute
    def roll_spread(prices):
        if len(prices) < 3: return np.nan
        dp = np.diff(prices)
        if len(dp) < 2: return np.nan
        c = np.cov(dp[:-1], dp[1:])[0, 1]
        if c >= 0: return 0.0
        return 2 * np.sqrt(-c)
    roll = grouped['price'].apply(lambda s: roll_spread(s.values)).reset_index()
    roll.columns = ['minute', 'roll_spread_abs']
    panel = panel.merge(roll, on='minute', how='left')
    panel['spread_bps'] = (panel['roll_spread_abs'] / panel['last_price']) * 10000  # bps

    panel['coin'] = coin_short
    return panel


def process_bookdepth(coin_full, coin_short, dates):
    """bookDepth → minute top-depth at narrowest band"""
    all_snaps = []
    for d in dates:
        zp = os.path.join(RAW, 'bookDepth', coin_full, f'{coin_full}-bookDepth-{d}.zip')
        if not os.path.exists(zp): continue
        df = read_zip_csv(zp, header=0)
        if df is None or len(df) == 0: continue
        df.columns = [c.strip().lower() for c in df.columns]
        # timestamp, percentage, depth, notional
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['minute'] = df['timestamp'].dt.floor('min')
        df['percentage'] = pd.to_numeric(df['percentage'], errors='coerce')
        df['notional'] = pd.to_numeric(df['notional'], errors='coerce')
        # Keep narrowest bands (|pct| in {1, 2}); top-of-book is closest to 0 but here 1% is finest
        narrow = df[df['percentage'].abs() <= 1].copy()
        all_snaps.append(narrow)
    if not all_snaps: return pd.DataFrame()
    big = pd.concat(all_snaps, ignore_index=True)
    # Sum bid (pct<0) + ask (pct>0) notional at narrowest band per minute
    big['side'] = np.where(big['percentage'] < 0, 'bid', 'ask')
    pivot = big.groupby(['minute','side'])['notional'].mean().unstack(fill_value=0).reset_index()
    if 'bid' not in pivot.columns: pivot['bid'] = 0
    if 'ask' not in pivot.columns: pivot['ask'] = 0
    pivot['top_depth_usd'] = pivot['bid'] + pivot['ask']
    pivot['coin'] = coin_short
    return pivot[['minute','coin','top_depth_usd']]


def process_metrics(coin_full, coin_short, dates):
    """metrics → 5-min OI series → resample 1-min → implied_liq_vol via ΔOI"""
    all_metrics = []
    for d in dates:
        zp = os.path.join(RAW, 'metrics', coin_full, f'{coin_full}-metrics-{d}.zip')
        if not os.path.exists(zp): continue
        df = read_zip_csv(zp, header=0)
        if df is None or len(df) == 0: continue
        df.columns = [c.strip().lower() for c in df.columns]
        df['create_time'] = pd.to_datetime(df['create_time'], utc=True)
        df['sum_open_interest_value'] = pd.to_numeric(df['sum_open_interest_value'], errors='coerce')
        all_metrics.append(df[['create_time','sum_open_interest_value']])
    if not all_metrics: return pd.DataFrame()
    big = pd.concat(all_metrics, ignore_index=True).sort_values('create_time')
    big = big.drop_duplicates('create_time').set_index('create_time')
    # Resample to 1-min (forward fill the 5-min snapshots)
    minute_idx = pd.date_range(big.index.min().floor('min'), big.index.max().ceil('min'), freq='1min', tz='UTC')
    oi = big['sum_open_interest_value'].reindex(minute_idx, method='nearest', tolerance=pd.Timedelta('5min'))
    df_out = pd.DataFrame({'minute': oi.index, 'oi_value_usd': oi.values})
    # Implied liquidation: max(0, -ΔOI / 5)  divided to per-minute basis
    df_out['delta_oi'] = df_out['oi_value_usd'].diff()
    df_out['implied_liq_vol_usd'] = (-df_out['delta_oi']).clip(lower=0) / 5  # spread 5-min interval to per-minute
    df_out['coin'] = coin_short
    return df_out[['minute','coin','oi_value_usd','implied_liq_vol_usd']]


def main():
    t0 = time.time()
    # Get dates from filenames
    dates = sorted({f.split('-')[-3] + '-' + f.split('-')[-2] + '-' + f.split('-')[-1].replace('.zip','')
                    for f in os.listdir(os.path.join(RAW, 'aggTrades', 'BTCUSDT'))
                    if f.endswith('.zip')})
    print(f"Found dates: {dates}")

    all_panels = []
    for coin_full in COINS:
        coin_short = COIN_MAP[coin_full]
        print(f"\n[{coin_short}] processing...")
        t_coin = time.time()

        agg = process_aggtrades(coin_full, coin_short, dates)
        bd = process_bookdepth(coin_full, coin_short, dates)
        met = process_metrics(coin_full, coin_short, dates)
        print(f"  aggTrades: {len(agg)} min rows; bookDepth: {len(bd)} min rows; metrics: {len(met)} min rows")

        # Merge: start from aggTrades (most rows), left join bookDepth + metrics
        p = agg.merge(bd, on=['minute','coin'], how='left').merge(met, on=['minute','coin'], how='left')
        # Compute liq_ratio
        p['liq_ratio'] = p['implied_liq_vol_usd'] / p['vol_usd'].replace(0, np.nan)
        p['log_spread'] = np.log(p['spread_bps'].replace(0, np.nan))
        p['log_depth'] = np.log(p['top_depth_usd'].replace(0, np.nan))
        p['log_vol'] = np.log(p['vol_usd'].replace(0, np.nan))
        p['log_liq_vol'] = np.log(p['implied_liq_vol_usd'].replace(0, np.nan))
        # liq_vol_std normalized by sample std of nonzero implied liq
        nz = p['implied_liq_vol_usd'][p['implied_liq_vol_usd'] > 0]
        sd = nz.std() if len(nz) > 0 else 1.0
        p['liq_vol_std'] = p['implied_liq_vol_usd'] / sd
        all_panels.append(p)
        print(f"  {coin_short} merged: {len(p)} rows; elapsed {time.time()-t_coin:.1f}s")

    panel_all = pd.concat(all_panels, ignore_index=True)
    panel_all['exchange'] = 'BIN'
    # Standardize event period (same as HL)
    event_start = pd.Timestamp('2025-10-10 21:00:00', tz='UTC')
    event_end = pd.Timestamp('2025-10-11 21:00:00', tz='UTC')
    pre_start = pd.Timestamp('2025-10-07 00:00:00', tz='UTC')
    panel_all['D_event'] = ((panel_all['minute'] >= event_start) & (panel_all['minute'] < event_end)).astype(int)
    panel_all['is_pre'] = (panel_all['minute'] < pre_start + pd.Timedelta(days=3))  # pre Oct 7-9
    panel_all['is_event'] = panel_all['D_event'] == 1
    panel_all['is_recovery'] = panel_all['minute'] >= event_end
    panel_all['liq_x_event'] = panel_all['liq_vol_std'] * panel_all['D_event']
    # 1-min return for control
    panel_all = panel_all.sort_values(['coin','minute']).reset_index(drop=True)
    panel_all['ret'] = panel_all.groupby('coin')['last_price'].pct_change()

    # Save
    panel_all.to_parquet(OUT_PARQ, index=False)
    print(f"\nSaved: {OUT_PARQ}")
    print(f"Total rows: {len(panel_all):,}; coins: {panel_all['coin'].nunique()}; minutes: {panel_all['minute'].nunique()}")
    print(f"D_event=1 rows: {(panel_all['D_event']==1).sum():,}")
    print(f"Elapsed: {time.time()-t0:.1f}s")

    # Per-coin summary
    print()
    print("Per-coin summary:")
    summ = panel_all.groupby('coin').agg(
        n_minutes=('minute','nunique'),
        mean_vol=('vol_usd','mean'),
        mean_spread_bps=('spread_bps','mean'),
        mean_implied_liq=('implied_liq_vol_usd','mean'),
    )
    print(summ)


if __name__ == "__main__":
    main()
