"""
Hyperliquid S3 L2订单簿快照 JSONL → CSV （24小时完整版）
=========================================================
用法: python jsonl_l2book_to_csv_fixed.py <l2book文件夹> <输出文件夹>

示例:
  python jsonl_l2book_to_csv_fixed.py ^
    E:\data\hyperliquid\hyperliquid_s3_data\l2book ^
    E:\data\hyperliquid\hyperliquid_s3_data\l2book\l2_csv

目录结构（下载器每小时一个文件，共24个/品种/天）:
  l2book/
    20251007/
      BTC_h00.jsonl   ← 00:xx UTC
      BTC_h01.jsonl   ← 01:xx UTC
      ...
      BTC_h23.jsonl   ← 23:xx UTC
      ETH_h00.jsonl
      ... （9品种 × 24小时 = 216个文件/天）
    20251008/
      ...

修复说明（相比原版）:
  [修复1] REQUIRED_HOURS 改为 h00~h23 全部24小时（原版只有4个）
  [修复2] 强制按小时顺序处理，不依赖文件名排序
  [修复3] 先写表头清空旧文件，再逐小时追加（"a"模式）
  [修复4] 转换后验证首条=00:xx、末条>=23:xx，不满足标红并汇总报告
"""

import json, os, sys, csv
from collections import defaultdict

# ============ 配置 ============
TARGET_COINS     = ["BTC", "ETH", "LINK", "XRP", "BNB", "SOL", "AVAX", "HYPE", "DOGE"]
TARGET_COINS_SET = set(TARGET_COINS)
NUM_LEVELS       = 20
# 全部24小时：h00, h01, h02, ... h23
REQUIRED_HOURS   = [f"h{h:02d}" for h in range(24)]
# ==============================


def build_header():
    header = ["snapshot_time", "exchange_time_ms", "coin",
               "mid_price", "spread", "spread_bps"]
    for i in range(1, NUM_LEVELS + 1):
        header += [f"bid{i}_px", f"bid{i}_sz", f"bid{i}_n"]
    for i in range(1, NUM_LEVELS + 1):
        header += [f"ask{i}_px", f"ask{i}_sz", f"ask{i}_n"]
    return header

CSV_HEADER = build_header()


def parse_snapshot(line):
    """解析一行JSONL，返回 (coin, row_list) 或 None"""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    raw = obj.get("raw", {})
    if raw.get("channel") != "l2Book":
        return None

    data  = raw.get("data", {})
    coin  = data.get("coin", "")
    if coin not in TARGET_COINS_SET:
        return None

    snapshot_time = obj.get("time", "")
    exchange_time = data.get("time", "")
    levels = data.get("levels", [[], []])
    bids   = levels[0] if len(levels) > 0 else []
    asks   = levels[1] if len(levels) > 1 else []

    best_bid   = float(bids[0]["px"]) if bids else 0
    best_ask   = float(asks[0]["px"]) if asks else 0
    mid_price  = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
    spread     = best_ask - best_bid if best_bid and best_ask else 0
    spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0

    row = [snapshot_time, exchange_time, coin,
           f"{mid_price:.6f}", f"{spread:.6f}", f"{spread_bps:.4f}"]

    for i in range(NUM_LEVELS):
        if i < len(bids):
            row += [bids[i]["px"], bids[i]["sz"], bids[i]["n"]]
        else:
            row += ["", "", ""]
    for i in range(NUM_LEVELS):
        if i < len(asks):
            row += [asks[i]["px"], asks[i]["sz"], asks[i]["n"]]
        else:
            row += ["", "", ""]

    return coin, row


def process_one_day(subdir_path, subdir, output_dir):
    """
    Step A  预检 24x9 个文件是否存在
    Step B  初始化 CSV（写表头，清空旧内容）
    Step C  按 h00->h01->...->h23 顺序逐小时追加写入
    """

    # Step A
    print(f"\n  {'─'*60}")
    print(f"  [{subdir}] Step A  文件预检（{len(REQUIRED_HOURS)}小时 x {len(TARGET_COINS)}品种）")
    print(f"  {'─'*60}")

    file_matrix   = {}
    missing_count = 0
    for coin in TARGET_COINS:
        file_matrix[coin] = {}
        coin_missing = []
        for hour in REQUIRED_HOURS:
            fp = os.path.join(subdir_path, f"{coin}_{hour}.jsonl")
            if os.path.exists(fp):
                file_matrix[coin][hour] = fp
            else:
                file_matrix[coin][hour] = None
                missing_count += 1
                coin_missing.append(hour)

        kb_total = sum(
            os.path.getsize(file_matrix[coin][h]) / 1024
            for h in REQUIRED_HOURS if file_matrix[coin][h]
        )
        present = len(REQUIRED_HOURS) - len(coin_missing)
        if coin_missing:
            print(f"    WARNING {coin:6s}: {present}/24小时  "
                  f"({kb_total:>8,.0f} KB)  缺失: {', '.join(coin_missing)}")
        else:
            print(f"    OK      {coin:6s}: 24/24小时  ({kb_total:>8,.0f} KB)")

    if missing_count == 0:
        print(f"\n    全部 {len(TARGET_COINS)*24} 个文件均存在")
    else:
        print(f"\n    共缺失 {missing_count} 个文件，缺失小时将跳过")

    # Step B
    print(f"\n  [{subdir}] Step B  初始化CSV")
    out_paths = {}
    for coin in TARGET_COINS:
        out_path = os.path.join(output_dir, f"l2book_{coin}_{subdir}.csv")
        out_paths[coin] = out_path
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)
    print(f"    已初始化 {len(TARGET_COINS)} 个CSV文件")

    # Step C
    print(f"\n  [{subdir}] Step C  逐小时转换（h00 -> h23）")
    total_counts = defaultdict(int)
    day_total    = 0

    for hour in REQUIRED_HOURS:
        hour_total = 0
        hour_num   = int(hour[1:])

        for coin in TARGET_COINS:
            fp = file_matrix[coin][hour]
            if fp is None:
                continue

            count = 0
            with open(fp, "r", encoding="utf-8", errors="ignore") as fin, \
                 open(out_paths[coin], "a", newline="", encoding="utf-8") as fout:
                writer = csv.writer(fout)
                for line_raw in fin:
                    line = line_raw.strip()
                    if not line:
                        continue
                    result = parse_snapshot(line)
                    if result is None:
                        continue
                    c, row = result
                    if c != coin:
                        continue
                    writer.writerow(row)
                    count += 1

            total_counts[coin] += count
            hour_total += count

        day_total += hour_total
        print(f"    {hour} ({hour_num:02d}:xx):  {hour_total:>10,} 行  [累计 {day_total:,}]")

    print(f"\n    [{subdir}] 转换完成，共 {day_total:,} 行")
    return total_counts, out_paths


def verify_csv(coin, out_path):
    if not os.path.exists(out_path):
        return {"ok": False, "rows": 0, "reason": "文件不存在",
                "first": "", "last": ""}

    first_time = last_time = ""
    rows = 0
    with open(out_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            if rows == 0:
                first_time = row[0]
            last_time = row[0]
            rows += 1

    if rows == 0:
        return {"ok": False, "rows": 0, "reason": "空文件",
                "first": "", "last": ""}

    try:
        first_hour = int(first_time[11:13])
        last_hour  = int(last_time[11:13])
        covers_all = (first_hour == 0) and (last_hour >= 23)
    except Exception:
        first_hour = last_hour = -1
        covers_all = False

    reason = ""
    if not covers_all:
        if first_hour != 0:
            reason += f"首行从{first_hour:02d}:xx开始（应为00:xx）  "
        if last_hour < 23:
            reason += f"末行到{last_hour:02d}:xx结束（应>=23:xx）"

    return {
        "ok"        : covers_all,
        "rows"      : rows,
        "reason"    : reason.strip(),
        "first"     : first_time[:19],
        "last"      : last_time[:19],
        "first_hour": first_hour,
        "last_hour" : last_hour,
    }


def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    subdirs = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and d.isdigit()
    ])

    if not subdirs:
        print("ERROR: 未找到日期格式子目录（如 20251007）")
        sys.exit(1)

    print(f"  发现 {len(subdirs)} 个日期目录: {', '.join(subdirs)}")
    print(f"  每天处理: {len(REQUIRED_HOURS)}小时 x {len(TARGET_COINS)}品种 "
          f"= {len(REQUIRED_HOURS)*len(TARGET_COINS)} 个文件\n")

    grand_total  = 0
    verify_rows  = []
    problem_list = []

    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)

        print(f"\n{'='*60}")
        print(f"  日期: {subdir}")
        print(f"{'='*60}")

        total_counts, out_paths = process_one_day(subdir_path, subdir, output_dir)

        # Step D 验证
        print(f"\n  [{subdir}] Step D  验证CSV完整性")
        print(f"  {'─'*65}")
        print(f"  {'品种':6s}  {'状态':8s}  {'总行数':>12s}  {'首条时间':20s}  末条时间")
        print(f"  {'─'*65}")

        for coin in TARGET_COINS:
            v = verify_csv(coin, out_paths[coin])
            grand_total += v["rows"]

            if v["ok"]:
                status = "OK 完整"
            else:
                status = "!! 不完整"
                problem_list.append({
                    "日期": subdir, "品种": coin,
                    "问题": v["reason"],
                    "行数": v["rows"],
                    "首条": v["first"],
                    "末条": v["last"],
                })

            suffix = f"  <- {v['reason']}" if not v["ok"] else ""
            print(f"  {coin:6s}  {status:8s}  {v['rows']:>12,}  "
                  f"{v['first']:20s}  {v['last']}{suffix}")

            verify_rows.append({
                "日期"      : subdir,
                "品种"      : coin,
                "总行数"    : v["rows"],
                "首条时间"  : v["first"],
                "末条时间"  : v["last"],
                "覆盖24小时": "是" if v["ok"] else "否",
                "问题说明"  : v.get("reason", ""),
            })

    # 保存报告
    report_path = os.path.join(output_dir, "conversion_report.csv")
    fieldnames  = ["日期","品种","总行数","首条时间","末条时间","覆盖24小时","问题说明"]
    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(verify_rows)

    print(f"\n{'='*60}")
    print(f"  转换完成！总写入: {grand_total:,} 行快照")
    print(f"  完整性报告: {report_path}")
    print(f"{'='*60}")

    if problem_list:
        print(f"\n  发现 {len(problem_list)} 个不完整的CSV：\n")
        for p in problem_list:
            print(f"    !! {p['日期']} {p['品种']:6s}  行数={p['行数']:,}  "
                  f"{p['首条']} -> {p['末条']}")
            print(f"       原因: {p['问题']}")
        print(f"\n  解决: 重新下载上述日期原始JSONL，删除对应CSV后重新运行。")
    else:
        n = len(subdirs) * len(TARGET_COINS)
        print(f"\n  全部 {n} 个CSV均覆盖完整24小时，数据质量验证通过")
        print(f"  下一步: 运行 chapter4_l2book_chunked.py 进行第四章分析")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python jsonl_l2book_to_csv_fixed.py <l2book文件夹> <输出文件夹>")
        print("示例: python jsonl_l2book_to_csv_fixed.py "
              r"E:\data\hyperliquid\hyperliquid_s3_data\l2book "
              r"E:\data\hyperliquid\hyperliquid_s3_data\l2book\l2_csv")
        sys.exit(1)

    input_dir  = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在 -> {input_dir}")
        sys.exit(1)

    print(f"输入目录 : {input_dir}")
    print(f"输出目录 : {output_dir}")
    print(f"目标品种 : {', '.join(sorted(TARGET_COINS))}")
    print(f"处理小时 : h00 ~ h23（共{len(REQUIRED_HOURS)}小时/天）")
    print(f"档位数量 : 每侧 {NUM_LEVELS} 档")
    print(f"{'='*60}\n")

    process_folder(input_dir, output_dir)
