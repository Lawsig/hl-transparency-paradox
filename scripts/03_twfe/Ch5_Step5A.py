"""
=================================================================
第五章：面板回归结果 — 五项关键决策 — Step5A 构建分钟级面板 — 可执行脚本 
=================================================================
文件命名：
  fills:  {COIN}_fills.csv        (如 BTC_fills.csv, HYPE_fills.csv)
  l2book: l2book_{COIN}_{YYYYMMDD}.csv  (如 l2book_BTC_20251007.csv)

数据规模：
  fills总计 ~5.5GB (HYPE 2.1GB, BTC 1.3GB) → 分块读取
  l2book 每文件 ~100-130MB × 9品种 × 9天 → 逐文件处理
  
=================================================================
"""

import pandas as pd
import numpy as np
import warnings
import gc
import os
import sys
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# =================================================================
# Step 0: 配置
# =================================================================

# ★★★ 文件路径 ★★★
FILLS_DIR = Path(r'E:\data2\hyperliquid\hyperliquid_s3_data\fills_by_coin')          # 存放 BTC_fills.csv 等
L2BOOK_DIR = Path(r'E:\data2\hyperliquid\hyperliquid_s3_data\l2book\l2_csv')        # 存放 l2book_BTC_20251007.csv 等
OUTPUT_DIR = Path(r'E:\data2\hyperliquid\ch5_output')    # 输出目录

COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'AVAX', 'LINK', 'HYPE']
DATES = pd.date_range('2025-10-07', '2025-10-15', freq='D')
EVENT_ZERO = pd.Timestamp('2025-10-10 21:00:00', tz='UTC')

LEVERAGE_MAP = {
    'BTC': 40, 'ETH': 25, 'SOL': 20, 'XRP': 20,
    'BNB': 10, 'DOGE': 10, 'AVAX': 10, 'LINK': 10, 'HYPE': 10
}

# 分块读取大小（fills文件可达2GB，需要分块）
CHUNK_SIZE = 500_000  # 每次读取50万行

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =================================================================
# Step 5A: 构建分钟级面板 → 确认精确N
# =================================================================
def step_5A_build_panel():
    """
    将双轨原始数据聚合为分钟级面板。
    所有后续分析的基础数据集。
    """
    print("=" * 70)
    print("STEP 5A: 构建分钟级面板数据集")
    print("=" * 70)

    # ----------------------------------------------------------
    # 5A-1: 处理 l2book → 分钟级价差 + 深度
    # ----------------------------------------------------------
    print("\n[1/5] 处理 l2book 数据...")

    all_l2 = []
    l2_raw_count = 0

    for coin in COINS:
        for date in DATES:
            # 文件名格式: l2book_{COIN}_{YYYYMMDD}.csv
            date_str = date.strftime('%Y%m%d')
            filepath = L2BOOK_DIR / f'l2book_{coin}_{date_str}.csv'

            if not filepath.exists():
                print(f"  [SKIP] {filepath.name}")
                continue

            print(f"  读取 {filepath.name} ...", end=' ')

            # 尝试tab分隔，若失败则逗号
            try:
                df = pd.read_csv(filepath, sep='\t')
                if len(df.columns) < 5:
                    df = pd.read_csv(filepath)
            except:
                df = pd.read_csv(filepath)

            l2_raw_count += len(df)
            print(f"{len(df):,} 行")

            # 时间处理：用 exchange_time_ms
            df['timestamp'] = pd.to_datetime(
                df['exchange_time_ms'], unit='ms', utc=True
            )
            df['minute'] = df['timestamp'].dt.floor('min')

            # spread_bps 已预计算，直接用
            # 计算 depth: mid_price ±1% 以内的 USD 深度
            mid = df['mid_price'].values

            bid_depth = np.zeros(len(df))
            for lv in range(1, 21):
                px_col = f'bid{lv}_px'
                sz_col = f'bid{lv}_sz'
                if px_col in df.columns and sz_col in df.columns:
                    px = df[px_col].values
                    sz = df[sz_col].values
                    mask = px >= mid * 0.99
                    bid_depth += np.where(mask, px * sz, 0)

            ask_depth = np.zeros(len(df))
            for lv in range(1, 21):
                px_col = f'ask{lv}_px'
                sz_col = f'ask{lv}_sz'
                if px_col in df.columns and sz_col in df.columns:
                    px = df[px_col].values
                    sz = df[sz_col].values
                    mask = px <= mid * 1.01
                    ask_depth += np.where(mask, px * sz, 0)

            df['bid_depth_usd'] = bid_depth
            df['ask_depth_usd'] = ask_depth
            df['total_depth_usd'] = bid_depth + ask_depth

            # 按分钟聚合
            minute_agg = df.groupby('minute').agg(
                spread_bps=('spread_bps', 'mean'),
                mid_price=('mid_price', 'last'),
                bid_depth_usd=('bid_depth_usd', 'mean'),
                ask_depth_usd=('ask_depth_usd', 'mean'),
                total_depth_usd=('total_depth_usd', 'mean'),
                n_snapshots=('spread_bps', 'count'),
            ).reset_index()

            minute_agg['coin'] = coin
            all_l2.append(minute_agg)

            del df, bid_depth, ask_depth
            gc.collect()

    l2_panel = pd.concat(all_l2, ignore_index=True)
    del all_l2
    gc.collect()
    print(f"\n  l2book 原始快照总数: {l2_raw_count:,}")
    print(f"  l2book 分钟级观测:   {len(l2_panel):,}")

    # ----------------------------------------------------------
    # 5A-2: 处理 node_fills → 分钟级成交量 + 清算量
    #        分块读取，避免内存溢出
    # ----------------------------------------------------------
    print("\n[2/5] 处理 node_fills 数据（分块读取）...")

    all_fills = []
    fills_raw_count = 0

    for coin in COINS:
        # 文件名格式: {COIN}_fills.csv
        filepath = FILLS_DIR / f'{coin}_fills.csv'

        if not filepath.exists():
            print(f"  [SKIP] {filepath.name}")
            continue

        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  读取 {filepath.name} ({file_size_mb:.0f} MB) ...", end=' ')

        # 先读第一行确定分隔符
        with open(filepath, 'r') as f:
            header_line = f.readline()
        sep = '\t' if '\t' in header_line else ','

        # 分块读取 + 逐块聚合
        coin_chunks = []
        chunk_count = 0

        for chunk in pd.read_csv(filepath, sep=sep, chunksize=CHUNK_SIZE):
            fills_raw_count += len(chunk)
            chunk_count += 1

            # 时间处理
            chunk['timestamp'] = pd.to_datetime(
                chunk['time_ms'], unit='ms', utc=True
            )
            chunk['minute'] = chunk['timestamp'].dt.floor('min')

            # 成交金额 (USD)
            chunk['notional_usd'] = chunk['price'] * chunk['size']

            # is_liquidation 已是 0/1
            chunk['liq_notional'] = chunk['notional_usd'] * chunk['is_liquidation']

            # 分钟级聚合（逐块）
            agg = chunk.groupby('minute').agg(
                vol_usd=('notional_usd', 'sum'),
                liq_vol_usd=('liq_notional', 'sum'),
                n_trades=('size', 'count'),
                n_liq_trades=('is_liquidation', 'sum'),
                vol_contracts=('size', 'sum'),
                price_first=('price', 'first'),
                price_last=('price', 'last'),
                price_high=('price', 'max'),
                price_low=('price', 'min'),
            ).reset_index()

            coin_chunks.append(agg)
            del chunk
            gc.collect()

        print(f"{fills_raw_count:,} 行 ({chunk_count} 块)")

        if not coin_chunks:
            continue

        # 合并所有块的分钟聚合 → 再次聚合（同一分钟可能跨块）
        coin_df = pd.concat(coin_chunks, ignore_index=True)
        del coin_chunks
        gc.collect()

        coin_minute = coin_df.groupby('minute').agg(
            vol_usd=('vol_usd', 'sum'),
            liq_vol_usd=('liq_vol_usd', 'sum'),
            n_trades=('n_trades', 'sum'),
            n_liq_trades=('n_liq_trades', 'sum'),
            vol_contracts=('vol_contracts', 'sum'),
            price_first=('price_first', 'first'),   # 第一块的first
            price_last=('price_last', 'last'),       # 最后块的last
            price_high=('price_high', 'max'),
            price_low=('price_low', 'min'),
        ).reset_index()

        # 分钟收益率
        coin_minute['ret'] = np.log(
            coin_minute['price_last'] / coin_minute['price_first']
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        # 价格振幅
        coin_minute['price_range'] = (
            (coin_minute['price_high'] - coin_minute['price_low'])
            / coin_minute['price_first']
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        # 清算占比
        coin_minute['liq_ratio'] = (
            coin_minute['liq_vol_usd'] / coin_minute['vol_usd']
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        coin_minute['coin'] = coin
        all_fills.append(coin_minute)
        del coin_df, coin_minute
        gc.collect()

    fills_panel = pd.concat(all_fills, ignore_index=True)
    del all_fills
    gc.collect()
    print(f"\n  fills 原始成交总数:  {fills_raw_count:,}")
    print(f"  fills 分钟级观测:    {len(fills_panel):,}")

    # ----------------------------------------------------------
    # 5A-3: 双轨合并
    # ----------------------------------------------------------
    print("\n[3/5] 合并双轨数据...")

    panel = pd.merge(
        l2_panel, fills_panel,
        on=['coin', 'minute'],
        how='inner'
    )
    n_after_merge = len(panel)
    del l2_panel, fills_panel
    gc.collect()
    print(f"  Inner join 后: {n_after_merge:,}")

    # ----------------------------------------------------------
    # 5A-4: 数据清洗
    # ----------------------------------------------------------
    print("\n[4/5] 数据清洗...")

    n_before = len(panel)

    m1 = panel['spread_bps'] > 0
    m2 = panel['total_depth_usd'] > 0
    m3 = panel['vol_usd'] > 0

    # 极端异常值阈值
    spread_p999 = panel['spread_bps'].quantile(0.999)
    m4 = panel['spread_bps'] <= spread_p999 * 10

    print(f"  spread_bps <= 0:     剔除 {(~m1).sum():,}")
    print(f"  depth <= 0:          剔除 {(~m2).sum():,}")
    print(f"  vol_usd <= 0:        剔除 {(~m3).sum():,}")
    print(f"  极端spread:          剔除 {(~m4).sum():,} (>{spread_p999*10:.1f} bps)")

    panel = panel[m1 & m2 & m3 & m4].copy()
    print(f"  清洗后: {len(panel):,} (剔除 {n_before - len(panel):,})")

    # ----------------------------------------------------------
    # 5A-5: 构建分析变量
    # ----------------------------------------------------------
    print("\n[5/5] 构建分析变量...")

    # 对数变换
    panel['log_spread'] = np.log(panel['spread_bps'])
    panel['log_depth'] = np.log(panel['total_depth_usd'])
    panel['log_bid_depth'] = np.log(panel['bid_depth_usd'].clip(lower=1))
    panel['log_ask_depth'] = np.log(panel['ask_depth_usd'].clip(lower=1))
    panel['log_vol'] = np.log(panel['vol_usd'] + 1)
    panel['log_liq_vol'] = np.log(panel['liq_vol_usd'] + 1)

    # τ: 距事件零点的分钟数
    panel['tau'] = (panel['minute'] - EVENT_ZERO).dt.total_seconds() / 60

    # 事件窗口标识
    panel['is_pre'] = (panel['tau'] >= -4320) & (panel['tau'] < 0)
    panel['is_event'] = (panel['tau'] >= 0) & (panel['tau'] <= 1440)
    panel['is_recovery'] = panel['tau'] > 1440
    panel['D_event'] = panel['is_event'].astype(int)
    panel['is_estimation'] = (panel['tau'] >= -5580) & (panel['tau'] <= -1501)

    # 杠杆 + 虚拟变量
    panel['leverage'] = panel['coin'].map(LEVERAGE_MAP)
    panel['D_link'] = (panel['coin'] == 'LINK').astype(int)
    panel['post_hour'] = np.where(panel['tau'] > 0, panel['tau'] / 60, 0)

    # 标准化清算量
    liq_positive = panel.loc[panel['liq_vol_usd'] > 0, 'liq_vol_usd']
    liq_std = liq_positive.std() if len(liq_positive) > 0 else 1
    panel['liq_vol_std'] = panel['liq_vol_usd'] / liq_std

    # 交互项
    panel['liq_x_event'] = panel['liq_vol_std'] * panel['D_event']
    panel['liq_x_leverage'] = panel['liq_vol_std'] * panel['leverage']
    panel['liq_x_link'] = panel['liq_vol_std'] * panel['D_link']
    panel['hour_x_link'] = panel['post_hour'] * panel['D_link']

    # ----------------------------------------------------------
    # 样本构建报告
    # ----------------------------------------------------------
    N_final = len(panel)

    report = []
    report.append("=" * 70)
    report.append("第五章面板数据样本构建报告")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    report.append(f"\n{'原始数据':=^60}")
    report.append(f"  l2book  原始快照:  {l2_raw_count:>15,}")
    report.append(f"  fills   原始成交:  {fills_raw_count:>15,}")
    report.append(f"  合计:              {l2_raw_count + fills_raw_count:>15,}")
    report.append(f"\n{'分钟级聚合':=^60}")
    report.append(f"  Inner join 后:     {n_after_merge:>15,}")
    report.append(f"  清洗后最终N:       {N_final:>15,}")
    report.append(f"  理论上限:          {9 * 9 * 1440:>15,} (9×9×1440)")
    report.append(f"  完整率:            {N_final / (9*9*1440)*100:>14.1f}%")

    report.append(f"\n{'按品种分布':=^60}")
    for coin in COINS:
        n_c = len(panel[panel['coin'] == coin])
        report.append(f"  {coin:6s}: {n_c:>8,} ({n_c/N_final*100:5.1f}%)"
                      f"  杠杆={LEVERAGE_MAP[coin]:>2}×")

    report.append(f"\n{'按时间窗口':=^60}")
    report.append(f"  估计窗口(Ch4):     {panel['is_estimation'].sum():>8,}")
    report.append(f"  事前期(Oct 7-9):   {panel['is_pre'].sum():>8,}")
    report.append(f"  事件期(Oct 10-11): {panel['is_event'].sum():>8,}")
    report.append(f"  恢复期(Oct 12-15): {panel['is_recovery'].sum():>8,}")

    report.append(f"\n{'关键变量描述统计':=^60}")
    for var in ['spread_bps', 'total_depth_usd', 'vol_usd',
                'liq_vol_usd', 'ret', 'price_range']:
        if var in panel.columns:
            s = panel[var]
            report.append(f"  {var:20s}: mean={s.mean():>12.4f}  "
                         f"std={s.std():>12.4f}  "
                         f"med={s.median():>12.4f}")

    report.append(f"\n{'缺失分钟':=^60}")
    exp = 9 * 24 * 60
    for coin in COINS:
        n_c = len(panel[panel['coin'] == coin])
        report.append(f"  {coin:6s}: {exp - n_c:>5} 缺失 ({(exp-n_c)/exp*100:.1f}%)")

    report.append(f"\n{'=' * 70}")
    report.append(f"★★★ 第五章统一使用 N = {N_final:,} ★★★")
    report.append(f"{'=' * 70}")

    report_text = '\n'.join(report)
    print(report_text)

    # 保存
    panel.to_parquet(OUTPUT_DIR / 'panel_minute.parquet', index=False)
    with open(OUTPUT_DIR / 'sample_construction_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n✅ 面板保存至 {OUTPUT_DIR / 'panel_minute.parquet'}")
    return panel

if __name__ == '__main__':
    panel = step_5A_build_panel()

