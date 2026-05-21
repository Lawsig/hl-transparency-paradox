"""
=====================================================================
Hyperliquid 官方S3存档 20251007-20251015数据下载脚本（全量版）
=====================================================================
  全部9天 × 全部24小时，确保每天数据完整覆盖00:00~23:59。

  已存在的文件自动跳过，只补充下载缺失部分，不重复下载。

数据来源：
  s3://hyperliquid-archive       → L2订单簿快照（每小时一个文件）
  s3://hl-mainnet-node-data      → tick级成交记录（node_fills）

依赖：pip install boto3 lz4 pandas
注意：需要AWS账号并配置凭证（aws configure），Requester Pays模式
=====================================================================
"""

import boto3
import lz4.frame
import json
import os
import pandas as pd
from datetime import datetime, timezone
from botocore import UNSIGNED
from botocore.config import Config

# ============================================================
# 配置
# ============================================================

OUTPUT_DIR = "./hyperliquid_s3_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 研究目标：Oct 7-15, 2025（全部9天）
STUDY_DATES = [
    "20251007", "20251008", "20251009",
    "20251010", "20251011",
    "20251012", "20251013", "20251014", "20251015",
]

# 研究标的（9个品种）
COINS = ["BTC", "ETH", "SOL", "DOGE", "BNB", "XRP", "AVAX", "LINK", "HYPE"]

# S3桶配置
ARCHIVE_BUCKET   = "hyperliquid-archive"
NODE_DATA_BUCKET = "hl-mainnet-node-data"
S3_REGION        = "ap-northeast-1"

# ============================================================
# S3客户端
# ============================================================

def get_s3_client():
    s3 = boto3.client("s3", region_name=S3_REGION)
    print("  ✓ 使用AWS凭证模式（Requester Pays）")
    return s3, True


# ============================================================
# L2订单簿快照下载
# ============================================================

def download_l2book_snapshots(s3_client, has_auth: bool,
                               date: str, hours: list, coins: list):
    """
    下载指定日期、小时列表、品种列表的L2订单簿快照。
    已存在的文件自动跳过。
    """
    date_dir = os.path.join(OUTPUT_DIR, "l2book", date)
    os.makedirs(date_dir, exist_ok=True)

    downloaded = 0
    skipped    = 0
    failed     = 0

    for hour in hours:
        for coin in coins:
            s3_key     = f"market_data/{date}/{hour}/l2Book/{coin}.lz4"
            local_path = os.path.join(date_dir, f"{coin}_h{hour:02d}.jsonl")

            # 已存在则跳过
            if os.path.exists(local_path):
                size_kb = os.path.getsize(local_path) / 1024
                print(f"    [{date} {hour:02d}:00] {coin:6s} 已存在 ({size_kb:,.0f} KB)，跳过")
                skipped += 1
                continue

            try:
                kwargs = {"Bucket": ARCHIVE_BUCKET, "Key": s3_key}
                if has_auth:
                    kwargs["RequestPayer"] = "requester"

                response    = s3_client.get_object(**kwargs)
                compressed  = response["Body"].read()
                decompressed = lz4.frame.decompress(compressed)
                text        = decompressed.decode("utf-8")

                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(text)

                line_count = text.count("\n")
                size_kb    = len(text.encode()) / 1024
                print(f"    [{date} {hour:02d}:00] {coin:6s} ✓  "
                      f"{line_count:,} 条  {size_kb:,.0f} KB")
                downloaded += 1

            except s3_client.exceptions.NoSuchKey:
                print(f"    [{date} {hour:02d}:00] {coin:6s} ✗  S3文件不存在（跳过）")
                failed += 1
            except Exception as e:
                print(f"    [{date} {hour:02d}:00] {coin:6s} ✗  {e}")
                failed += 1

    return downloaded, skipped, failed


# ============================================================
# node_fills 下载
# ============================================================

def download_node_fills(s3_client, has_auth: bool,
                        date: str, hours: list = None):
    date_dir = os.path.join(OUTPUT_DIR, "node_fills", date)
    os.makedirs(date_dir, exist_ok=True)

    prefixes_to_try = [
        f"node_fills_by_block/{date}/",
        f"node_fills/{date}/",
    ]

    for prefix in prefixes_to_try:
        try:
            kwargs = {"Bucket": NODE_DATA_BUCKET, "Prefix": prefix}
            if has_auth:
                kwargs["RequestPayer"] = "requester"

            response = s3_client.list_objects_v2(**kwargs)
            objects  = response.get("Contents", [])

            if not objects:
                continue

            print(f"    找到 {len(objects)} 个文件（路径: {prefix}）")

            for obj in objects:
                key      = obj["Key"]
                filename = key.split("/")[-1]
                local_path = os.path.join(date_dir, filename)

                if os.path.exists(local_path):
                    print(f"    {filename} 已存在，跳过")
                    continue

                get_kwargs = {"Bucket": NODE_DATA_BUCKET, "Key": key}
                if has_auth:
                    get_kwargs["RequestPayer"] = "requester"

                resp = s3_client.get_object(**get_kwargs)
                data = resp["Body"].read()

                try:
                    decompressed = lz4.frame.decompress(data)
                    text = decompressed.decode("utf-8")
                except Exception:
                    text = data.decode("utf-8")

                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(text)

                size_kb = len(text.encode()) / 1024
                print(f"    {filename} ✓  {size_kb:.1f} KB")

            return True

        except Exception as e:
            print(f"    路径 {prefix} 访问失败: {e}")
            continue

    return False


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 65)
    print("  Hyperliquid S3存档 全量下载器")
    print("  9天 × 24小时 × 9品种 = 1,944 个L2快照文件")
    print("=" * 65)

    # 预估文件数量
    total_files = len(STUDY_DATES) * 24 * len(COINS)
    print(f"\n  计划下载: {len(STUDY_DATES)}天 × 24小时 × {len(COINS)}品种 "
          f"= {total_files:,} 个文件")
    print(f"  日期范围: {STUDY_DATES[0]} ~ {STUDY_DATES[-1]}")
    print(f"  品种列表: {', '.join(COINS)}")
    print(f"  输出目录: {os.path.abspath(OUTPUT_DIR)}/l2book/{{日期}}/")
    print(f"\n  已存在的文件将自动跳过，只下载缺失部分。")
    print()

    s3, has_auth = get_s3_client()

    # ── L2订单簿快照：全部9天 × 全部24小时 ──────────────────
    print("\n【阶段A】下载L2订单簿快照（全量：9天×24小时×9品种）")
    print("  路径: s3://hyperliquid-archive/market_data/{date}/{hour}/l2Book/{coin}.lz4")
    print()

    grand_downloaded = 0
    grand_skipped    = 0
    grand_failed     = 0

    for date in STUDY_DATES:
        hours = list(range(24))    # ← 全部24小时，不再区分日期类型

        print(f"\n  ┌── [{date}]  24小时 × {len(COINS)}品种 = {24*len(COINS)}个文件")
        dl, sk, fa = download_l2book_snapshots(
            s3, has_auth, date, hours, COINS)

        grand_downloaded += dl
        grand_skipped    += sk
        grand_failed     += fa

        print(f"  └── [{date}] 完成: 新下载={dl}  跳过={sk}  失败={fa}")

    print(f"\n  ── L2快照下载汇总 ──")
    print(f"  新下载: {grand_downloaded:,} 个文件")
    print(f"  已跳过: {grand_skipped:,} 个文件（已存在）")
    print(f"  失败  : {grand_failed:,} 个文件")
    print(f"  合计  : {grand_downloaded+grand_skipped+grand_failed:,} 个文件")

    # ── node_fills（保持原版：只下载事件核心4天）────────────
    print("\n【阶段B】下载node_fills tick成交记录（事件核心4天）")
    print("  路径: s3://hl-mainnet-node-data/node_fills_by_block/{date}/")

    for date in ["20251009", "20251010", "20251011", "20251012"]:
        print(f"\n  [{date}]")
        success = download_node_fills(s3, has_auth, date)
        if not success:
            print(f"  ✗ {date} 下载失败")

    print("\n" + "=" * 65)
    print("  ✓ 全部下载完成！")
    print(f"  L2数据目录: {os.path.abspath(OUTPUT_DIR)}/l2book/")
    print()
    print("  下一步：运行转换脚本生成CSV")
    print("  python jsonl_l2book_to_csv_fixed.py "
          f"{os.path.abspath(OUTPUT_DIR)}/l2book "
          f"{os.path.abspath(OUTPUT_DIR)}/l2book/l2_csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
