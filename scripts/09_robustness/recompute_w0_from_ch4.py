"""Recompute W0 (first-touch hourly) for 5 thresholds × 9 coins
using chapter4_l2book_v2.py's actual output as ground truth source.

Reads:
  - ch4_output/table4_5_recovery_detail.csv  (hour × mult × pre_mean per coin)
  - ch4_output/chapter4_report.txt          (pre_event std per coin)

Computes recovery_h for 5 thresholds:
  T1: 1.2 × pre_mean
  T2: 1.5 × pre_mean ★ (matches table 4.5)
  T3: 2.0 × pre_mean
  T4: pre_mean + 1σ
  T5: pre_mean + 2σ

Output: first_touch_grid_5thresholds_ch4.json
        (replaces the previously incorrect first_touch_grid_5thresholds.json
         which used ch5/panel_minute.parquet pipeline)
"""
import os
import sys
import json
import re
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CH4_OUT = os.environ.get("REPL_DATA_DIR", "./data")
DETAIL_CSV = os.path.join(CH4_OUT, "table4_5_recovery_detail.csv")
REPORT_TXT = os.path.join(CH4_OUT, "chapter4_report.txt")
OUT_DIR = r"E:\data2\hyperliquid\H5_recovery_sensitivity"


# ── Load detail csv ────────────────────────────────────────────────────────
df = pd.read_csv(DETAIL_CSV, encoding="utf-8-sig")
print(f"Loaded {DETAIL_CSV}: {len(df)} rows")
print(f"Columns: {list(df.columns)}")
coins = sorted(df["品种"].unique())
print(f"Coins: {coins}\n")

# ── Extract pre_means from detail csv (constant per coin) ─────────────────
pre_means = {}
for coin in coins:
    pre_means[coin] = float(df[df["品种"] == coin]["pre_mean_bps"].iloc[0])


# ── Parse pre_event std from chapter4_report.txt ──────────────────────────
with open(REPORT_TXT, "r", encoding="utf-8") as f:
    report = f.read()

pre_stds = {}
# Pattern: "pre_event  COIN  5580    0.MEAN  0.STD ..."
# 例: " pre_event LINK  5580    0.9498 0.3801   ..."
for coin in coins:
    pat = rf"pre_event\s+{coin}\s+\d+\s+([\d.]+)\s+([\d.]+)"
    m = re.search(pat, report)
    if m:
        std = float(m.group(2))
        pre_stds[coin] = std
    else:
        print(f"WARN: pre_event std for {coin} not found")
        pre_stds[coin] = None

print("Pre-event statistics per coin (from chapter4_l2book_v2.py output):")
print(f"{'Coin':<6} {'pre_mean':<10} {'pre_std':<10}")
for c in coins:
    print(f"{c:<6} {pre_means[c]:<10.4f} {pre_stds[c]:<10.4f}")
print()


# ── Compute thresholds ────────────────────────────────────────────────────
def compute_threshold(coin, mode, k):
    if mode == "mult":
        return pre_means[coin] * k
    if mode == "sigma":
        return pre_means[coin] + k * pre_stds[coin]
    raise ValueError(mode)


THRESHOLDS = [
    ("T1_1.2x_pre",        "mult",  1.2),
    ("T2_1.5x_pre",        "mult",  1.5),  # baseline (matches table 4.5)
    ("T3_2.0x_pre",        "mult",  2.0),
    ("T4_pre_plus_1sigma", "sigma", 1.0),
    ("T5_pre_plus_2sigma", "sigma", 2.0),
]


# ── Find first-touch h for each (threshold, coin) ─────────────────────────
def first_touch(coin_df, threshold_bps):
    """Find first hour where mean_spread ≤ threshold (chapter4_l2book_v2 logic).
    coin_df sorted by h ascending. Returns int h or None."""
    sub = coin_df.sort_values("h").reset_index(drop=True)
    for _, row in sub.iterrows():
        if row["价差均值_bps"] <= threshold_bps:
            return int(row["h"])
    return None


results = {}
for t_label, t_mode, t_k in THRESHOLDS:
    results[t_label] = {}
    for coin in coins:
        thresh = compute_threshold(coin, t_mode, t_k)
        coin_df = df[df["品种"] == coin]
        h = first_touch(coin_df, thresh)
        results[t_label][coin] = {
            "first_touch_hourly_h": h,
            "threshold_bps": round(thresh, 4),
        }


# ── Print results ────────────────────────────────────────────────────────
print("=" * 78)
print("First-touch hourly recovery (chapter4_l2book_v2.py pipeline)")
print("=" * 78)
print(f"{'Coin':<6} | {'T1 1.2x':<8} | {'T2 1.5x ★':<10} | {'T3 2.0x':<8} | "
      f"{'T4 +1σ':<8} | {'T5 +2σ':<8}")
print("-" * 78)
for coin in coins:
    cells = []
    for t_label, _, _ in THRESHOLDS:
        h = results[t_label][coin]["first_touch_hourly_h"]
        cells.append(f"{h}" if h is not None else ">72")
    print(f"{coin:<6} | {cells[0]:<8} | {cells[1]:<10} | {cells[2]:<8} | "
          f"{cells[3]:<8} | {cells[4]:<8}")

# Cross-validate T2 against table4_5_recovery.csv
print()
print("Cross-validation T2 ★ baseline vs table4_5_recovery.csv:")
table45 = pd.read_csv(os.path.join(CH4_OUT, "table4_5_recovery.csv"), encoding="utf-8-sig")
for coin in coins:
    expected = int(table45[table45["品种"] == coin]["恢复小时数"].iloc[0])
    computed = results["T2_1.5x_pre"][coin]["first_touch_hourly_h"]
    match = "✓" if expected == computed else f"✗ ({computed} vs {expected})"
    print(f"  {coin}: {match}")


# ── Ranking check (is LINK slowest under each threshold?) ────────────────
ranking = {}
print()
print("Ranking (LINK slowest under first-touch hourly?):")
for t_label, _, _ in THRESHOLDS:
    coin_hs = {c: results[t_label][c]["first_touch_hourly_h"] for c in coins}
    link_h = coin_hs["LINK"]
    others = [(c, h) for c, h in coin_hs.items() if c != "LINK"]
    censored_others = [c for c, h in others if h is None]
    if link_h is None:
        ranking[t_label] = "LINK_censored(slowest)"
    elif censored_others:
        ranking[t_label] = f"NO (also censored: {','.join(censored_others)})"
    else:
        max_other = max(h for _, h in others if h is not None)
        if link_h > max_other:
            ranking[t_label] = f"YES ({link_h} > {max_other})"
        else:
            ranking[t_label] = f"NO ({link_h} <= {max_other})"
    print(f"  {t_label}: {ranking[t_label]}")


# ── Save JSON ────────────────────────────────────────────────────────────
out = {
    "criterion": "first integer hour where hourly_mean(spread_bps) <= threshold "
                 "(chapter4_l2book_v2.py pipeline, computed from "
                 "ch4_output/table4_5_recovery_detail.csv)",
    "source_pipeline": "chapter4_l2book_v2.py",
    "source_detail_csv": "ch4_output/table4_5_recovery_detail.csv",
    "source_report": "ch4_output/chapter4_report.txt",
    "estimation_window": "tau ∈ [-5580, -1501]  (Oct 7 00:00 to Oct 9 23:59 UTC)",
    "max_search_hour": 72,
    "thresholds": [{"label": t[0], "mode": t[1], "k": t[2]} for t in THRESHOLDS],
    "coins": coins,
    "pre_stats": {c: {"mean": pre_means[c], "std": pre_stds[c]} for c in coins},
    "results": results,
    "ranking_per_threshold": ranking,
    "note": "T2_1.5x_pre row = table 4.5 ground truth (BTC=7, ETH=7, SOL=6, "
            "BNB=7, AVAX=7, DOGE=13, XRP=14, HYPE=15, LINK=53)",
}
out_file = os.path.join(OUT_DIR, "first_touch_grid_5thresholds_ch4.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {out_file}")
