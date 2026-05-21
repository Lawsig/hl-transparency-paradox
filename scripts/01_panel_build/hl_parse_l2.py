"""
L2订单簿快照解析器
从 l2book/{date}/{coin}_h{hour}.jsonl 提取价差时间序列

修改说明（相比原版）：
  1. L2_DIR  改为实际路径 E:\data2\hyperliquid\hyperliquid_s3_data\l2book
  2. OUT_DIR 改为同一目录（输出CSV与源数据放在一起）
  3. 逐小时range(24)逻辑不变，自动处理每天24个jsonl文件

输出：
  E:\data2\hyperliquid\hyperliquid_s3_data\l2_spread_{coin}.csv
      每条快照一行，含买卖价差和订单簿深度（tick级）

  E:\data2\hyperliquid\hyperliquid_s3_data\l2_spread_1hour.csv
      按小时聚合（均值/最大/最小价差），直接用于第五章回归

字段说明：
  spread_pct   = (best_ask - best_bid) / mid × 100  （价差百分比，论文核心因变量）
  depth_bid5   = 前5档买方总量（币本位，流动性深度）
  depth_ask5   = 前5档卖方总量
  depth_imbal  = (bid5 - ask5) / (bid5 + ask5)      （深度不平衡度，-1到+1）
  depth_bid20  = 前20档买方总量
  depth_ask20  = 前20档卖方总量
"""

import os, json, csv
from datetime import datetime, timezone

# ============================================================
# 路径配置
# ============================================================
L2_DIR  = r"E:\data2\hyperliquid\hyperliquid_s3_data\l2book"       # JSONL源文件目录
OUT_DIR = r"E:\data2\hyperliquid\hyperliquid_s3_data"               # CSV输出目录

# 论文9个研究品种
COINS = ["BTC", "ETH", "SOL", "DOGE", "BNB", "XRP", "AVAX", "LINK", "HYPE"]

# 研究时间范围（与论文一致：Oct 7-15, 2025）
STUDY_DATES = [
    "20251007", "20251008", "20251009",
    "20251010", "20251011",
    "20251012", "20251013", "20251014", "20251015",
]

# 事件窗口定义
EVENT_START_MS = 1760054400000   # 2025-10-10 00:00 UTC  ← pre/event分界
EVENT_END_MS   = 1760227200000   # 2025-10-12 00:00 UTC  ← event/post分界
CRASH_HOUR_MS  = 1760130000000   # 2025-10-10 21:00 UTC  ← 爆仓核心时刻（hours_from_crash基准）

# ============================================================
# 输出字段定义
# ============================================================
FIELDS = [
    "time_ms", "time_utc", "coin",
    "best_bid", "best_ask", "mid",
    "spread_abs", "spread_pct",
    "depth_bid5", "depth_ask5", "depth_imbal",
    "depth_bid20", "depth_ask20",
    "event_window", "hours_from_crash"
]

FIELDS_1H = [
    "hour_ms", "hour_utc", "coin",
    "spread_pct_mean", "spread_pct_max", "spread_pct_min", "spread_pct_std",
    "depth_bid5_mean", "depth_ask5_mean", "depth_imbal_mean",
    "n_snapshots", "event_window", "hours_from_crash"
]


# ============================================================
# 工具函数
# ============================================================
def ms_to_dt(ms):
    try:
        return datetime.fromtimestamp(int(ms) / 1000,
                                      tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def label_window(ms):
    ms = int(ms)
    if ms < EVENT_START_MS:
        return "pre_event"
    elif ms < EVENT_END_MS:
        return "event"
    else:
        return "post_event"


# ============================================================
# 核心：解析单条L2快照
# ============================================================
def parse_snapshot(rec):
    """
    rec = raw.data 字典，包含 levels 字段
    返回价差和深度指标字典，解析失败返回 None
    """
    try:
        levels = rec["levels"]
        bids   = levels[0]   # 买档，降序，[0]是最优买一
        asks   = levels[1]   # 卖档，升序，[0]是最优卖一

        if not bids or not asks:
            return None

        best_bid = float(bids[0]["px"])
        best_ask = float(asks[0]["px"])
        mid      = (best_bid + best_ask) / 2

        if mid <= 0 or best_ask <= best_bid:
            return None

        spread_abs = best_ask - best_bid
        spread_pct = spread_abs / mid * 100

        bid5  = sum(float(b["sz"]) for b in bids[:5])
        ask5  = sum(float(a["sz"]) for a in asks[:5])
        bid20 = sum(float(b["sz"]) for b in bids[:20])
        ask20 = sum(float(a["sz"]) for a in asks[:20])
        imbal = (bid5 - ask5) / (bid5 + ask5) if (bid5 + ask5) > 0 else 0.0

        return {
            "best_bid":    round(best_bid,  6),
            "best_ask":    round(best_ask,  6),
            "mid":         round(mid,       6),
            "spread_abs":  round(spread_abs, 6),
            "spread_pct":  round(spread_pct, 6),
            "depth_bid5":  round(bid5,  4),
            "depth_ask5":  round(ask5,  4),
            "depth_imbal": round(imbal, 6),
            "depth_bid20": round(bid20, 4),
            "depth_ask20": round(ask20, 4),
        }
    except Exception:
        return None


# ============================================================
# 主解析函数：处理单个品种的全部9天×24小时文件
# ============================================================
def parse_coin(coin):
    """
    遍历 STUDY_DATES × 24小时，读取 {coin}_h{hh}.jsonl，
    写入 tick 级 CSV，同时在内存中聚合小时统计。
    返回 (hour_agg, rows_written)
    """
    out_tick     = os.path.join(OUT_DIR, f"l2_spread_{coin}.csv")
    rows_written = 0
    errors       = 0
    hour_agg     = {}   # hour_ms -> {spreads, bid5s, ask5s, imbals}

    with open(out_tick, "w", newline="", encoding="utf-8-sig") as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDS)
        writer.writeheader()

        for date in STUDY_DATES:
            date_dir = os.path.join(L2_DIR, date)
            if not os.path.exists(date_dir):
                print(f"    ⚠ 目录不存在，跳过: {date_dir}")
                continue

            for hour in range(24):          # h00 ~ h23，全部24小时
                fname = f"{coin}_h{hour:02d}.jsonl"
                fpath = os.path.join(date_dir, fname)
                if not os.path.exists(fpath):
                    # 静默跳过缺失文件（已在转换阶段验证过完整性）
                    continue

                file_rows = 0
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj     = json.loads(line)
                            data    = obj["raw"]["data"]
                            time_ms = int(data["time"])

                            result = parse_snapshot(data)
                            if result is None:
                                errors += 1
                                continue

                            hour_ms = (time_ms // 3_600_000) * 3_600_000

                            row = {
                                "time_ms"         : time_ms,
                                "time_utc"        : ms_to_dt(time_ms),
                                "coin"            : coin,
                                "event_window"    : label_window(time_ms),
                                "hours_from_crash": round(
                                    (time_ms - CRASH_HOUR_MS) / 3_600_000, 4),
                            }
                            row.update(result)
                            writer.writerow(row)
                            file_rows    += 1
                            rows_written += 1

                            # 小时聚合（只保留4个列表，内存极小）
                            if hour_ms not in hour_agg:
                                hour_agg[hour_ms] = {
                                    "spreads": [], "bid5s": [],
                                    "ask5s":   [], "imbals": []
                                }
                            hour_agg[hour_ms]["spreads"].append(result["spread_pct"])
                            hour_agg[hour_ms]["bid5s"].append(result["depth_bid5"])
                            hour_agg[hour_ms]["ask5s"].append(result["depth_ask5"])
                            hour_agg[hour_ms]["imbals"].append(result["depth_imbal"])

                        except Exception:
                            errors += 1
                            continue

                if file_rows > 0:
                    print(f"    [{date} {coin} h{hour:02d}]  {file_rows:,} 条快照",
                          flush=True)

    size_mb = os.path.getsize(out_tick) / 1024 / 1024
    print(f"  {coin:<6} ✓  tick级: {rows_written:,} 条  "
          f"{size_mb:.1f} MB  (解析错误: {errors})")
    return hour_agg, rows_written


# ============================================================
# 写小时聚合表
# ============================================================
def write_hourly(all_hour_agg):
    out  = os.path.join(OUT_DIR, "l2_spread_1hour.csv")
    rows = []

    for coin, hour_agg in all_hour_agg.items():
        for hour_ms, d in sorted(hour_agg.items()):
            sp = d["spreads"]
            b5 = d["bid5s"]
            a5 = d["ask5s"]
            im = d["imbals"]
            n  = len(sp)
            if n == 0:
                continue

            mean_sp = sum(sp) / n
            max_sp  = max(sp)
            min_sp  = min(sp)
            std_sp  = (sum((x - mean_sp) ** 2 for x in sp) / n) ** 0.5 if n > 1 else 0.0

            rows.append({
                "hour_ms"         : hour_ms,
                "hour_utc"        : ms_to_dt(hour_ms),
                "coin"            : coin,
                "spread_pct_mean" : round(mean_sp,       6),
                "spread_pct_max"  : round(max_sp,        6),
                "spread_pct_min"  : round(min_sp,        6),
                "spread_pct_std"  : round(std_sp,        6),
                "depth_bid5_mean" : round(sum(b5) / n,   4),
                "depth_ask5_mean" : round(sum(a5) / n,   4),
                "depth_imbal_mean": round(sum(im) / n,   6),
                "n_snapshots"     : n,
                "event_window"    : label_window(hour_ms),
                "hours_from_crash": round(
                    (hour_ms - CRASH_HOUR_MS) / 3_600_000, 4),
            })

    rows.sort(key=lambda r: (r["coin"], r["hour_ms"]))

    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS_1H)
        writer.writeheader()
        writer.writerows(rows)

    mb = os.path.getsize(out) / 1024 / 1024
    print(f"\n  小时聚合表: {out}")
    print(f"  ({len(rows):,} 行  {mb:.1f} MB)")


# ============================================================
# 打印三阶段汇总（论文描述性统计素材）
# ============================================================
def print_summary(all_hour_agg):
    print("\n" + "=" * 72)
    print("  三阶段价差统计摘要（论文描述性统计素材）")
    print("=" * 72)
    print(f"  {'币种':<6} {'窗口':<14} {'均价差%':>10} {'最大价差%':>11} "
          f"{'均深度(bid5)':>13} {'快照数':>9}")
    print("  " + "-" * 64)

    for coin in COINS:
        if coin not in all_hour_agg:
            continue
        windows = {"pre_event": [], "event": [], "post_event": []}
        depths  = {"pre_event": [], "event": [], "post_event": []}

        for hour_ms, d in all_hour_agg[coin].items():
            w = label_window(hour_ms)
            windows[w].extend(d["spreads"])
            depths[w].extend(d["bid5s"])

        for wname in ["pre_event", "event", "post_event"]:
            sp = windows[wname]
            dp = depths[wname]
            if not sp:
                continue
            n = len(sp)
            print(f"  {coin:<6} {wname:<14} "
                  f"{sum(sp)/n:>10.5f} "
                  f"{max(sp):>11.5f} "
                  f"{sum(dp)/len(dp) if dp else 0:>13.2f} "
                  f"{n:>9,}")
        print()


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 72)
    print("  L2订单簿快照解析器（全量版）")
    print(f"  数据来源: {L2_DIR}")
    print(f"  输出目录: {OUT_DIR}")
    print(f"  处理品种: {', '.join(COINS)}")
    print(f"  研究日期: {STUDY_DATES[0]} ~ {STUDY_DATES[-1]}（9天×24小时）")
    print("=" * 72)
    print()

    # 检查输入目录
    if not os.path.isdir(L2_DIR):
        print(f"❌ L2数据目录不存在: {L2_DIR}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    all_hour_agg = {}
    total_rows   = 0

    for coin in COINS:
        print(f"\n── {coin} ──────────────────────────────────────")
        hour_agg, n   = parse_coin(coin)
        all_hour_agg[coin] = hour_agg
        total_rows    += n

    print(f"\n\n总计解析快照: {total_rows:,} 条")

    write_hourly(all_hour_agg)
    print_summary(all_hour_agg)

    print("\n" + "=" * 72)
    print("  输出文件清单:")
    for coin in COINS:
        p = os.path.join(OUT_DIR, f"l2_spread_{coin}.csv")
        if os.path.exists(p):
            mb = os.path.getsize(p) / 1024 / 1024
            print(f"  l2_spread_{coin}.csv   {mb:.1f} MB  (tick级，每条快照一行)")
    p1h = os.path.join(OUT_DIR, "l2_spread_1hour.csv")
    if os.path.exists(p1h):
        mb = os.path.getsize(p1h) / 1024 / 1024
        print(f"  l2_spread_1hour.csv    {mb:.1f} MB  (小时聚合，用于第五章回归)")
    print("=" * 72)


if __name__ == "__main__":
    main()
