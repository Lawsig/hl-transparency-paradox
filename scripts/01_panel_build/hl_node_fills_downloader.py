"""
=====================================================================
Hyperliquid node_fills tick 20251007-20251015数据下载器
=====================================================================
正确路径：s3://hl-mainnet-node-data/node_fills_by_block/hourly/YYYYMMDD/{hour}.lz4

         实际结构是每行=一个区块，成交在 events[[wallet, fill_data]] 里，
         必须遍历 events 数组才能取到 coin/px/sz 等字段。
         同时加入 tid 去重（同一笔交易买卖双方各出现一次）。

依赖：pip install boto3 lz4 pandas
"""

import boto3
import lz4.frame
import json
import os
import csv
import pandas as pd
from datetime import datetime, timezone

# ============================================================
# 配置
# ============================================================

OUTPUT_DIR  = r"E:\data2\hyperliquid\hyperliquid_s3_data"
NODE_BUCKET = "hl-mainnet-node-data"
S3_REGION   = "ap-northeast-1"

STUDY_DATES = [
    "20251007", "20251008", "20251009",
    "20251010", "20251011",
    "20251012", "20251013", "20251014", "20251015",
]

# 全部9天 × 全部24小时，已存在的文件自动跳过
HOURS_CONFIG = {date: list(range(24)) for date in STUDY_DATES}

COINS = ["BTC", "ETH", "SOL", "DOGE", "BNB", "XRP", "AVAX", "LINK", "HYPE"]

ZERO_HASH      = "0x" + "0" * 64
EVENT_START_MS = 1760054400000   # 2025-10-10 00:00 UTC
EVENT_END_MS   = 1760227200000   # 2025-10-12 00:00 UTC
CRASH_HOUR_MS  = 1760130000000   # 2025-10-10 21:00 UTC

CHUNK_SIZE = 8 * 1024 * 1024  # 8MB 分块读取

# ============================================================
# 工具
# ============================================================

def get_s3():
    return boto3.client("s3", region_name=S3_REGION)

def ms_to_dt(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def label_window(ms):
    if ms < EVENT_START_MS:   return "pre_event"
    elif ms < EVENT_END_MS:   return "event"
    else:                     return "post_event"

# ============================================================
# 步骤1：下载 + 流式解压 → 保存为jsonl
# ============================================================

def download_and_decompress(s3, date: str, hour: int) -> str:
    s3_key     = f"node_fills_by_block/hourly/{date}/{hour}.lz4"
    out_dir    = os.path.join(OUTPUT_DIR, "raw_jsonl", date)
    os.makedirs(out_dir, exist_ok=True)
    local_lz4  = os.path.join(out_dir, f"{hour}.lz4")
    local_json = os.path.join(out_dir, f"{hour}.jsonl")

    if os.path.exists(local_json):
        size_mb = os.path.getsize(local_json) / 1024 / 1024
        print(f"    h{hour:02d} 已存在 ({size_mb:.1f}MB)，跳过")
        return local_json

    try:
        s3.download_file(
            Bucket=NODE_BUCKET, Key=s3_key, Filename=local_lz4,
            ExtraArgs={"RequestPayer": "requester"}
        )
        lz4_mb = os.path.getsize(local_lz4) / 1024 / 1024
        print(f"    h{hour:02d} 下载完成 ({lz4_mb:.1f}MB压缩)，解压中...", end="", flush=True)
    except Exception as e:
        print(f"    h{hour:02d} ✗ 下载失败: {e}")
        return None

    try:
        lines_written = 0
        leftover      = b""

        with lz4.frame.open(local_lz4, mode="rb") as lz4_f, \
             open(local_json, "w", encoding="utf-8") as out_f:
            while True:
                chunk = lz4_f.read(CHUNK_SIZE)
                if not chunk:
                    break
                data  = leftover + chunk
                lines = data.split(b"\n")
                leftover = lines[-1]
                for line in lines[:-1]:
                    if line.strip():
                        out_f.write(line.decode("utf-8", errors="replace") + "\n")
                        lines_written += 1
            if leftover.strip():
                out_f.write(leftover.decode("utf-8", errors="replace") + "\n")
                lines_written += 1

        os.remove(local_lz4)
        out_mb = os.path.getsize(local_json) / 1024 / 1024
        print(f" ✓ {lines_written:,}行  ({out_mb:.1f}MB解压后)")
        return local_json

    except Exception as e:
        print(f" ✗ 解压失败: {e}")
        if os.path.exists(local_lz4):
            os.remove(local_lz4)
        return None

# ============================================================
# 步骤2：解析jsonl → 按币种写CSV  
# ============================================================

def parse_to_csv_by_coin(jsonl_path: str, seen_tids: set):
    """
    实际JSONL结构（每行 = 一个区块，不是一条成交）：
    {
      "block_time": "2025-10-07T00:00:00.113...",
      "block_number": 754645658,
      "events": [
        ["0xWalletAddress", {"coin":"BTC","px":"67000","sz":"0.05",
                             "side":"B","time":1759795200286,
                             "hash":"0x000...000","tid":329302661185082,
                             "fee":"0.02","closedPnl":"0.0", ...}],
        ["0xWalletAddress", {...}],   ← 同一笔交易对手方（tid相同，去重）
        ...
      ]
    }
    coin/px/sz 等字段在 events[i][1] 里，顶层没有这些字段。
    """
    out_dir = os.path.join(OUTPUT_DIR, "fills_by_coin")
    os.makedirs(out_dir, exist_ok=True)

    file_handles = {}
    writers      = {}
    counts       = {coin: 0 for coin in COINS}
    liq_counts   = {coin: 0 for coin in COINS}

    fieldnames = [
        "time_ms", "time_utc", "block_time", "block_number",
        "coin", "price", "size", "side", "dir",
        "hash", "is_liquidation", "tid", "wallet",
        "fee", "closed_pnl", "event_window", "hours_from_crash"
    ]

    try:
        with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    block      = json.loads(line)
                    block_time = block.get("block_time", "")
                    block_num  = block.get("block_number", "")
                    events     = block.get("events", [])

                    for event in events:
                        # event 格式：["0xWallet", {fill_data}]
                        if not isinstance(event, list) or len(event) < 2:
                            continue
                        wallet = event[0]
                        fill   = event[1]
                        if not isinstance(fill, dict):
                            continue

                        coin = fill.get("coin", "")
                        if coin not in COINS:
                            continue

                        # tid 去重：同一笔交易买卖双方各记录一次，只保留第一条
                        tid = fill.get("tid")
                        if tid is not None:
                            if tid in seen_tids:
                                continue
                            seen_tids.add(tid)

                        time_ms = int(fill.get("time", 0))
                        is_liq  = fill.get("hash", "") == ZERO_HASH

                        row = {
                            "time_ms":         time_ms,
                            "time_utc":        ms_to_dt(time_ms),
                            "block_time":      block_time,
                            "block_number":    block_num,
                            "coin":            coin,
                            "price":           fill.get("px",        ""),
                            "size":            fill.get("sz",        ""),
                            "side":            fill.get("side",      ""),
                            "dir":             fill.get("dir",       ""),
                            "hash":            fill.get("hash",      ""),
                            "is_liquidation":  1 if is_liq else 0,
                            "tid":             tid,
                            "wallet":          wallet,
                            "fee":             fill.get("fee",       ""),
                            "closed_pnl":      fill.get("closedPnl",""),
                            "event_window":    label_window(time_ms),
                            "hours_from_crash": round(
                                (time_ms - CRASH_HOUR_MS) / 3_600_000, 4),
                        }

                        if coin not in file_handles:
                            csv_path = os.path.join(out_dir, f"{coin}_fills.csv")
                            is_new   = not os.path.exists(csv_path)
                            fh       = open(csv_path, "a", newline="", encoding="utf-8")
                            w        = csv.DictWriter(fh, fieldnames=fieldnames)
                            if is_new:
                                w.writeheader()
                            file_handles[coin] = fh
                            writers[coin]      = w

                        writers[coin].writerow(row)
                        counts[coin]    += 1
                        if is_liq:
                            liq_counts[coin] += 1

                except Exception:
                    continue

    finally:
        for fh in file_handles.values():
            fh.close()

    total     = sum(counts.values())
    total_liq = sum(liq_counts.values())
    if total > 0:
        print(f"      解析完成: 共{total:,}条  清算单{total_liq:,}条")
        for coin in COINS:
            if counts[coin] > 0:
                pct = liq_counts[coin] / counts[coin] * 100
                print(f"        {coin:<6}: {counts[coin]:>8,}条  "
                      f"清算{liq_counts[coin]:>6,}条 ({pct:.2f}%)")
    else:
        print(f"      ⚠ 未解析到任何成交，请检查JSONL结构")

# ============================================================
# 步骤3：从CSV重建1分钟K线
# ============================================================

def build_1min_candles_for_coin(coin: str) -> pd.DataFrame:
    csv_path = os.path.join(OUTPUT_DIR, "fills_by_coin", f"{coin}_fills.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    chunks = []
    for chunk in pd.read_csv(
        csv_path, chunksize=500_000,
        dtype={"price": float, "size": float, "is_liquidation": int}
    ):
        chunk["bar"] = (chunk["time_ms"] // 60_000) * 60_000
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True).sort_values("time_ms")

    results = []
    for bar, g in df.groupby("bar"):
        liq_mask  = g["is_liquidation"] == 1
        buy_mask  = g["side"] == "B"
        liq_vol   = g.loc[liq_mask, "size"].sum()
        buy_vol   = g.loc[buy_mask,  "size"].sum()
        sell_vol  = g.loc[~buy_mask, "size"].sum()
        total_vol = g["size"].sum()
        results.append({
            "time_ms":           bar,
            "time_utc":          ms_to_dt(bar),
            "coin":              coin,
            "open":              g["price"].iloc[0],
            "high":              g["price"].max(),
            "low":               g["price"].min(),
            "close":             g["price"].iloc[-1],
            "volume":            total_vol,
            "n_trades":          len(g),
            "liq_count":         int(liq_mask.sum()),
            "liq_volume":        liq_vol,
            "buy_volume":        buy_vol,
            "sell_volume":       sell_vol,
            "event_window":      label_window(bar),
            "hours_from_crash":  round((bar - CRASH_HOUR_MS) / 3_600_000, 4),
        })

    candles = pd.DataFrame(results)
    mid = (candles["high"] + candles["low"]) / 2
    candles["price_range_pct"]   = ((candles["high"] - candles["low"]) / mid * 100).round(6)
    candles["liq_pct_of_volume"] = (
        candles["liq_volume"] / candles["volume"].replace(0, float("nan")) * 100
    ).round(3)
    total_side = candles["buy_volume"] + candles["sell_volume"]
    candles["order_imbalance"] = (
        (candles["buy_volume"] - candles["sell_volume"]) /
        total_side.replace(0, float("nan"))
    ).round(4)

    return candles.reset_index(drop=True)

# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 65)
    print("  Hyperliquid node_fills 下载 + 1分钟K线重建（修复版）")
    print("=" * 65)

    s3 = get_s3()

    # ── 阶段1：下载 & 解压 ──────────────────────────────────
    print("\n【阶段1】下载 & 流式解压 node_fills...")
    all_jsonl = []

    for date in STUDY_DATES:
        hours = HOURS_CONFIG[date]
        print(f"\n  [{date}]  共{len(hours)}个小时")
        for hour in hours:
            path = download_and_decompress(s3, date, hour)
            if path:
                all_jsonl.append((date, hour, path))

    # ── 阶段2：解析 → 按币种写CSV ───────────────────────────
    print("\n\n【阶段2】解析tick数据 → 按币种写CSV...")
    print("  (tid跨文件全局去重，同一笔交易只保留一条)\n")

    seen_tids = set()   # 全局去重集合，贯穿所有文件

    for date, hour, path in all_jsonl:
        print(f"  [{date} h{hour:02d}]", flush=True)
        parse_to_csv_by_coin(path, seen_tids)

    # 文件大小汇总
    print("\n  fills_by_coin 文件汇总：")
    fills_dir = os.path.join(OUTPUT_DIR, "fills_by_coin")
    for coin in COINS:
        p = os.path.join(fills_dir, f"{coin}_fills.csv")
        if os.path.exists(p):
            mb   = os.path.getsize(p) / 1024 / 1024
            rows = sum(1 for _ in open(p)) - 1
            print(f"    {coin:<6} {rows:>10,} 条  {mb:>8.1f} MB")
        else:
            print(f"    {coin:<6} ✗ 无文件")

    # ── 阶段3：重建1分钟K线 ─────────────────────────────────
    print("\n\n【阶段3】重建1分钟K线...")
    all_candles = []

    for coin in COINS:
        print(f"  {coin:<6} ", end="", flush=True)
        df = build_1min_candles_for_coin(coin)
        if not df.empty:
            all_candles.append(df)
            out = os.path.join(OUTPUT_DIR, f"1min_candles_{coin}.csv")
            df.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"✓ {len(df):,} 根K线")
        else:
            print("✗ 无数据")

    # ── 阶段4：事件研究摘要 ──────────────────────────────────
    if not all_candles:
        print("\n暂无K线数据。")
        return

    print("\n\n【阶段4】事件研究统计（论文第四章素材）")
    df_all = pd.concat(all_candles, ignore_index=True)
    df_all.to_csv(os.path.join(OUTPUT_DIR, "1min_candles_all.csv"),
                  index=False, encoding="utf-8-sig")

    print(f"\n  {'币种':<6} {'窗口':<14} {'均量':>10} "
          f"{'均价差%':>9} {'均清算笔':>9} {'清算占比%':>10} {'K线数':>7}")
    print("  " + "-" * 65)

    for coin in COINS:
        cdf = df_all[df_all["coin"] == coin]
        if cdf.empty:
            continue
        for window in ["pre_event", "event", "post_event"]:
            w = cdf[cdf["event_window"] == window]
            if w.empty:
                continue
            print(f"  {coin:<6} {window:<14} "
                  f"{w['volume'].mean():>10.4f} "
                  f"{w['price_range_pct'].mean():>9.4f} "
                  f"{w['liq_count'].mean():>9.2f} "
                  f"{w['liq_pct_of_volume'].mean():>10.3f}%"
                  f"{len(w):>8,}")

    print("\n" + "=" * 65)
    print(f"  ✓ 完成！输出目录: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
