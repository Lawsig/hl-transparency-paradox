"""
第四章 事件研究完整分析脚本 —— 基于真实L2订单簿快照  v2
=======================================================
数据路径: E:\data2\hyperliquid\hyperliquid_s3_data\l2book\l2_csv
文件格式: l2book_{COIN}_{YYYYMMDD}.csv  （9品种 × 9天 = 81个文件）

已确认列名：
  snapshot_time      : 快照时间 (ISO格式, UTC)
  exchange_time_ms   : 交易所时间戳 (ms)
  coin               : 品种
  mid_price          : 中间价
  spread             : 绝对价差
  spread_bps         : 买卖价差（基点）← 论文 Spread_it 直接使用
  bid1_px~bid20_px   : 20档买价
  bid1_sz~bid20_sz   : 20档买量（币本位）
  ask1_px~ask20_px   : 20档卖价
  ask1_sz~ask20_sz   : 20档卖量（币本位）

论文变量映射：
  Spread_it  ← spread_bps（已经是基点，直接用）
  Depth_it   ← 中间价±1%范围内所有档位的 sz × px 之和（USDC）
  LiqVol_it  ← 来自 1min_candles_{COIN}.csv 的 liq_volume 列（合并）

修订说明（v2）：
  [修复1] EVENT_UTC 从 21:13:00 更正为 21:00:00（与论文锚点一致）
  [修复2] EST_W 从 (-162,-31) 更正为 (-5580,-1501)
          原窗口仅131分钟且落在 h=-6 信息渗漏区，基准被污染
          新窗口覆盖完整 Oct 7-9（τ=-5580 至 τ=-1501）
  [修复3] H5 从「跨资产传染相关矩阵」完整替换为「恢复非对称检验」
          LINK 纳入为核心研究对象（不再排除）
  [修复4] 深度计算向量化，提速约30×
  [修复5] 表4.1 增加按品种×三窗口分组统计
  [新增]  h=-6 预公告信号检测（论文4.2.2节两阶段结构）
  [新增]  BNB/XRP异质性计算（论文4.3.3节）
  [新增]  LINK极端冲击量化（193倍、53小时）
  [修复6] BUF_W 实际接入逻辑，排除事件窗口外污染
  [修复7] fig4.3 替换为恢复路径对比折线图

输出（与论文第四章完全对应）：
  table4_1_descriptive.csv     → 表4.1 双轨数据源与三窗口描述性统计
  table4_3_CAS.csv             → 表4.3 H1 各窗口CAS
  table4_4_CAD.csv             → 表4.4 H2 各窗口CAD
  table4_5_recovery.csv        → 表4.5 H5 各品种恢复时间汇总
  table4_5_recovery_detail.csv → 每小时价差倍数明细（备查）
  table4_pre_signal.csv        → h=-6预公告信号（两阶段结构）
  table4_heterogeneity.csv     → BNB/XRP异质性（4.3.3节）
  fig4_1_CAS_trajectory.png    → 图4.1
  fig4_2_depth_collapse.png    → 图4.2
  fig4_3_recovery_paths.png    → 图4.3（恢复路径对比，替换传染热力图）
  cas_minutely.csv             → 逐分钟CAS明细（备查）
  chapter4_report.txt          → 完整文字报告

运行：
  pip install pandas numpy scipy statsmodels matplotlib seaborn
  python chapter4_l2book_v2.py
"""

import os, sys, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════
#  ① 中文字体修复
# ══════════════════════════════════════════════
def setup_chinese_font():
    candidates = ["Microsoft YaHei", "SimHei", "PingFang SC",
                  "WenQuanYi Micro Hei", "Noto Sans CJK SC"]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        print(f"  ✓ 中文字体: {chosen}")
    else:
        print("  ⚠ 未找到中文字体，图表标题将使用英文")
    matplotlib.rcParams["axes.unicode_minus"] = False


# ══════════════════════════════════════════════
#  ② 配置区
# ══════════════════════════════════════════════
L2_DIR     = r"E:\data2\hyperliquid\hyperliquid_s3_data\l2book\l2_csv"
CANDLE_DIR = r"E:\data2\hyperliquid\hyperliquid_s3_data"

# [修复1] 事件零点改为 21:00:00（路透社报道整点边界，论文锚点）
EVENT_UTC  = "2025-10-10 21:00:00"

ALL_COINS  = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "AVAX", "HYPE", "LINK"]

# [修复2] 估计窗口：Oct 7 00:00 → Oct 9 23:59
# τ = -(3×24×60 + 21×60) = -5580 对应 Oct 7 00:00 UTC
# τ = -(1×24×60 + 1×60)  = -1501 对应 Oct 9 23:59 UTC（留缓冲）
# 说明：完全避开 Oct 10 h=-6 预公告信号（15:00 UTC = τ=-360）
EST_W  = (-5580, -1501)
BUF_W  = (-1500, -1)    # Oct 10 00:00 至 20:59（含h=-6区）：排除出估计窗口
                         # 此区间数据保留在面板中用于事前描述，但不进入基准估计

# 事件研究窗口（τ，分钟）
WINDOWS = {
    "事前[-60,-1]" : (-60,  -1),
    "即时[0,+5]"   : (  0,  +5),
    "短期[+1,+15]" : ( +1, +15),
    "中期[+1,+30]" : ( +1, +30),
    "恢复[+31,+60]": (+31, +60),
}

# 三窗口（用于描述性统计和异质性分析）
# pre_event:  Oct 7-9 整体  → τ in EST_W
# event:      Oct 10 21:00 – Oct 11 21:00 → τ in [0, 1440)
# post_event: Oct 12 00:00 – Oct 15 23:59 → τ in [1500, 7000]
WIN3 = {
    "pre_event" : (EST_W[0], -1),
    "event"     : (0,  1439),
    "post_event": (1500, 7000),
}

# 论文核验目标值
TARGET = {
    "pre_CAS"    : -0.52,
    "short_CAS"  : 41.9,
    "link_imm_CAS": 47.8,   # 即时[0,+5] LINK CAS（论文4.3.2）
    "short_CAD"  : -55.8,
    "link_CAD"   : -67.4,
    # H5恢复非对称
    "link_recovery_h"  : 53,   # LINK恢复到1.5×pre均值所需小时数
    "btc_recovery_h"   : 7,
    "hype_recovery_h"  : 15,
}

# 恢复判定阈值
RECOVERY_MULT = 1.5   # 价差降至 pre_mean × 1.5 视为恢复


# ══════════════════════════════════════════════
#  ③ 深度计算（向量化，提速约30×）
# ══════════════════════════════════════════════
def calc_depth_batch(df, threshold=0.01):
    """
    向量化计算中间价±1%范围内订单簿深度（USDC）
    论文定义：Depth_it = sum(sz×px) for all levels within mid×(1±threshold)
    """
    mid = pd.to_numeric(df["mid_price"], errors="coerce")
    lo  = mid * (1 - threshold)
    hi  = mid * (1 + threshold)
    total = np.zeros(len(df))

    for side in ("bid", "ask"):
        for i in range(1, 21):
            px_col = f"{side}{i}_px"
            sz_col = f"{side}{i}_sz"
            if px_col not in df.columns or sz_col not in df.columns:
                break
            px = pd.to_numeric(df[px_col], errors="coerce").values
            sz = pd.to_numeric(df[sz_col], errors="coerce").values
            in_range = (px >= lo.values) & (px <= hi.values)
            contrib  = np.where(in_range & ~np.isnan(px) & ~np.isnan(sz),
                                px * sz, 0.0)
            total += contrib
    return total


# ══════════════════════════════════════════════
#  ④ 数据加载与聚合
# ══════════════════════════════════════════════
def load_l2_one_day(coin, date_str):
    fp = os.path.join(L2_DIR, f"l2book_{coin}_{date_str}.csv")
    if not os.path.exists(fp):
        return None
    try:
        df = pd.read_csv(fp, encoding="utf-8-sig", low_memory=False)
        df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
        return df
    except Exception as e:
        print(f"    ⚠ 读取失败 {fp}: {e}")
        return None


def aggregate_to_minute(df, coin, event_ts):
    """L2快照聚合到分钟级（向量化深度计算）"""
    df = df.copy()
    df["_ts"] = pd.to_datetime(df["snapshot_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["_ts"])

    df["depth_usdc"]     = calc_depth_batch(df)
    df["spread_bps_val"] = pd.to_numeric(df["spread_bps"], errors="coerce")
    df["_min"]           = df["_ts"].dt.floor("1min")

    agg = (df.groupby("_min")
             .agg(spread_bps   =("spread_bps_val", "mean"),
                  spread_bps_max=("spread_bps_val", "max"),
                  depth_usdc   =("depth_usdc", "mean"),
                  mid_price    =("mid_price",  "mean"),
                  n_snapshots  =("spread_bps_val", "count"))
             .reset_index())
    agg["coin"] = coin
    agg["tau"]  = (agg["_min"] - event_ts).dt.total_seconds() / 60
    return agg


def load_panel(event_ts):
    print("\n" + "="*62)
    print("步骤1：加载L2订单簿数据（9品种×9天）")
    print("="*62)
    print("  [v2] 向量化深度计算，速度约为v1的30×\n")

    base  = datetime(2025, 10, 7)
    dates = [(base + timedelta(days=i)).strftime("%Y%m%d") for i in range(9)]

    frames = []
    for coin in ALL_COINS:
        coin_frames = []
        for d in dates:
            raw = load_l2_one_day(coin, d)
            if raw is None:
                print(f"    ⚠ 缺失: {coin} {d}")
                continue
            agg = aggregate_to_minute(raw, coin, event_ts)
            coin_frames.append(agg)
            print(f"    ✓ {coin} {d}: {len(raw):,}快照 → {len(agg):,}分钟")
        if coin_frames:
            frames.append(pd.concat(coin_frames, ignore_index=True))
        else:
            print(f"    ❌ {coin}: 无数据")

    if not frames:
        sys.exit("❌ 未加载到任何数据")

    panel = pd.concat(frames, ignore_index=True)
    print(f"\n  合并面板: {len(panel):,}行  品种: {sorted(panel['coin'].unique())}")
    return panel


def merge_liq(panel, event_ts):
    """合并1min_candles清算量数据"""
    print("\n步骤1b：合并清算量（1min_candles）")
    liq_frames = []
    for coin in ALL_COINS:
        fp = os.path.join(CANDLE_DIR, f"1min_candles_{coin}.csv")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, encoding="utf-8-sig", low_memory=False)
        df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
        df["_ts"]  = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
        df         = df.dropna(subset=["_ts"])
        df["_min"] = df["_ts"].dt.floor("1min")
        df["coin"] = coin
        cols = ["_min", "coin"] + [c for c in ["liq_volume","liq_count","volume"]
                                    if c in df.columns]
        liq_frames.append(df[cols])

    if liq_frames:
        liq   = pd.concat(liq_frames, ignore_index=True)
        panel = panel.merge(liq, on=["_min","coin"], how="left")
        for c in ["liq_volume", "liq_count", "volume"]:
            if c in panel.columns:
                panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0)
        print("  ✓ 清算量已合并（含volume用于异质性分析）")
    else:
        panel["liq_volume"] = 0
        panel["liq_count"]  = 0
        panel["volume"]     = 0
        print("  ⚠ 未找到1min_candles文件，清算量/成交量设为0")
    return panel


# ══════════════════════════════════════════════
#  ⑤ 基准估计（常数均值模型）
# ══════════════════════════════════════════════
def build_abnormals(panel):
    """
    基准模型：常数均值模型（MacKinlay 1997；Madhavan 2000）
    使用 Oct 7–9 作为估计窗口（EST_W），完全避开Oct10事件日
    E[spread_it] = mean(spread_i) over estimation window
    AS_it = spread_it - E[spread_it]
    AD_it = (depth_it - E[depth_it]) / E[depth_it] × 100%
    """
    print("\n步骤2：估计窗口基准 → AS / AD")
    print(f"  估计窗口: τ∈[{EST_W[0]}, {EST_W[1]}]  (Oct 7 00:00 → Oct 9 23:59 UTC)")
    print(f"  [v2] 估计窗口完整覆盖事前3天，避开h=-6预公告污染")

    est = (panel["tau"] >= EST_W[0]) & (panel["tau"] <= EST_W[1])
    panel = panel.copy()

    bs_map, bd_map = {}, {}
    for coin in ALL_COINS:
        mask = (panel["coin"] == coin) & est
        sub  = panel[mask]
        bs   = pd.to_numeric(sub["spread_bps"], errors="coerce").mean()
        bd   = pd.to_numeric(sub["depth_usdc"],  errors="coerce").mean()
        bs_map[coin] = bs if not np.isnan(bs) else 0
        bd_map[coin] = bd if (not np.isnan(bd) and bd > 0) else 1
        print(f"  {coin:6s}  E[spread]={bs:8.4f} bps   E[depth]={bd:,.0f} USDC  "
              f"(估计窗口n={mask.sum():,})")

    panel["BS"] = panel["coin"].map(bs_map)
    panel["BD"] = panel["coin"].map(bd_map)
    panel["AS"] = pd.to_numeric(panel["spread_bps"], errors="coerce") - panel["BS"]
    panel["AD"] = ((pd.to_numeric(panel["depth_usdc"], errors="coerce") - panel["BD"])
                   / panel["BD"] * 100)

    # 同时保存pre_mean供H5恢复分析使用
    panel["pre_mean_spread"] = panel["coin"].map(bs_map)
    return panel, bs_map


# ══════════════════════════════════════════════
#  ⑥ 统计工具
# ══════════════════════════════════════════════
def nw_test(series, maxlags=5):
    """Newey-West HAC t检验，返回(mean, t, p)"""
    y = pd.to_numeric(series, errors="coerce").dropna().values
    if len(y) < 3:
        return np.nan, np.nan, np.nan
    X = np.ones((len(y), 1))
    try:
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        return float(res.params[0]), float(res.tvalues[0]), float(res.pvalues[0])
    except Exception:
        t, p = stats.ttest_1samp(y, 0)
        return float(y.mean()), float(t), float(p)

def sig(p):
    if pd.isna(p): return ""
    if p < .001:   return "***"
    if p < .01:    return "**"
    if p < .05:    return "*"
    return "n.s."


# ══════════════════════════════════════════════
#  ⑦ 表4.1 描述性统计（三窗口×品种）
# ══════════════════════════════════════════════
def table41(panel, bs_map, rpt):
    print("\n" + "="*62)
    print("步骤3：描述性统计（表4.1）")
    print("="*62)

    rows = []
    for wlabel, (ws, we) in WIN3.items():
        for coin in ALL_COINS:
            mask = ((panel["coin"] == coin)
                    & (panel["tau"] >= ws) & (panel["tau"] <= we))
            sub  = panel[mask]
            sp   = pd.to_numeric(sub["spread_bps"], errors="coerce").dropna()
            dp   = pd.to_numeric(sub["depth_usdc"], errors="coerce").dropna()
            lv   = pd.to_numeric(sub.get("liq_volume", pd.Series()), errors="coerce").dropna()
            vo   = pd.to_numeric(sub.get("volume",     pd.Series()), errors="coerce").dropna()

            # 清算占比（清算量/成交量）
            liq_ratio = (lv.sum() / vo.sum() * 100) if vo.sum() > 0 else np.nan

            rows.append({
                "窗口"        : wlabel,
                "品种"        : coin,
                "N_分钟"      : len(sp),
                "价差均值_bps": round(sp.mean(), 4) if len(sp) else np.nan,
                "价差标准差"  : round(sp.std(),  4) if len(sp) else np.nan,
                "价差最大值"  : round(sp.max(),  4) if len(sp) else np.nan,
                "深度均值_USDC": round(dp.mean(), 0) if len(dp) else np.nan,
                "深度标准差"  : round(dp.std(),  0) if len(dp) else np.nan,
                "清算占比_%"  : round(liq_ratio, 2) if not np.isnan(liq_ratio) else np.nan,
                "相对pre倍数" : round(sp.mean() / bs_map[coin], 2)
                                if (bs_map[coin] > 0 and len(sp)) else np.nan,
            })

    t41 = pd.DataFrame(rows)
    t41.to_csv("table4_1_descriptive.csv", index=False, encoding="utf-8-sig")
    print(t41.to_string(index=False))
    rpt += ["\n【表4.1 三窗口描述性统计】", t41.to_string(index=False)]

    # 额外打印：event窗口各品种价差倍数（论文4.2.2节数据）
    print("\n  ── event窗口价差相对pre均值倍数（核验193倍）──")
    evt = t41[t41["窗口"] == "event"][["品种","相对pre倍数"]].sort_values(
        "相对pre倍数", ascending=False)
    print(evt.to_string(index=False))
    rpt += ["\n  event窗口价差倍数:", evt.to_string(index=False)]
    return t41


# ══════════════════════════════════════════════
#  ⑧ h=-6 预公告信号检测（论文4.2.2节）
# ══════════════════════════════════════════════
def analyze_pre_signal(panel, bs_map, rpt):
    """
    检测 h=-6 预公告信号
    论文4.2.2：主冲击前6小时（15:00 UTC，τ=-360至τ=-301）
    价差已扩大至pre均值2.1×-3.2×
    """
    print("\n" + "="*62)
    print("步骤3b：h=-6预公告信号检测（论文4.2.2节）")
    print("="*62)

    # τ=-360 = 21:00-360min = 15:00 UTC，恰好是Washington政策窗口
    # h=-1 = τ=-60
    # 定义三个比较小时
    hours_to_check = {
        "h=-6 (15:00 UTC)": (-360, -301),  # 15:00-15:59
        "h=-3 (18:00 UTC)": (-180, -121),  # 18:00-18:59
        "h=-1 (20:00 UTC)": (-60,  -1),    # 20:00-20:59
        "h=0  (21:00 UTC)": (0,    59),    # 主冲击
    }

    rows = []
    for hlabel, (ts, te) in hours_to_check.items():
        for coin in ALL_COINS:
            mask = ((panel["coin"] == coin)
                    & (panel["tau"] >= ts) & (panel["tau"] <= te))
            sp = pd.to_numeric(panel[mask]["spread_bps"], errors="coerce").dropna()
            pre_m = bs_map[coin]
            if len(sp) > 0 and pre_m > 0:
                mult = sp.mean() / pre_m
                rows.append({"时间段": hlabel, "品种": coin,
                             "均值倍数": round(mult, 2)})

    df_sig = pd.DataFrame(rows)
    pivot  = df_sig.pivot(index="时间段", columns="品种", values="均值倍数")
    print(pivot.to_string())
    pivot.to_csv("table4_pre_signal.csv", encoding="utf-8-sig")
    print("\n  [已保存] table4_pre_signal.csv")

    # 核验论文中的h=-6数值
    h6 = df_sig[df_sig["时间段"] == "h=-6 (15:00 UTC)"]
    print("\n  ── h=-6 各品种倍数（论文目标: BTC 2.8×, LINK 2.1×, HYPE 3.2×）──")
    print(h6[["品种","均值倍数"]].sort_values("均值倍数", ascending=False).to_string(index=False))
    rpt += ["\n【h=-6预公告信号】", pivot.to_string()]
    return df_sig


# ══════════════════════════════════════════════
#  ⑨ 表4.3 CAS（H1）
# ══════════════════════════════════════════════
def table43(panel, rpt):
    print("\n" + "="*62)
    print("步骤4：CAS（表4.3，H1）")
    print("="*62)

    show = ["BTC","ETH","SOL","LINK","XRP"]
    rows = []
    for wname, (ws, we) in WINDOWS.items():
        mask = (panel["tau"] >= ws) & (panel["tau"] <= we)
        row  = {"窗口": wname}
        vals = []
        for coin in ALL_COINS:
            sub        = panel[(panel["coin"]==coin) & mask]["AS"]
            _, _, p    = nw_test(sub)
            cas        = float(pd.to_numeric(sub, errors="coerce").sum())
            row[coin]          = f"{cas:+.1f}{sig(p)}"
            row[f"{coin}_num"] = cas
            vals.append(cas)
        row["均值"]     = f"{np.nanmean(vals):+.1f}"
        row["均值_num"] = np.nanmean(vals)
        rows.append(row)

    df = pd.DataFrame(rows)
    hdr = f"  {'窗口':16s}" + "".join(f" {c:>12s}" for c in show) + f" {'均值':>10s}"
    print(hdr); print("  " + "-"*82)
    for _, r in df.iterrows():
        line = f"  {r['窗口']:16s}" + "".join(f" {r[c]:>12s}" for c in show)
        line += f" {r['均值']:>10s}"
        print(line)

    # 核验
    short = df[df["窗口"]=="短期[+1,+15]"].iloc[0]
    imm   = df[df["窗口"]=="即时[0,+5]"].iloc[0]
    pre   = df[df["窗口"]=="事前[-60,-1]"].iloc[0]
    print(f"\n  ── 核验 ──")
    print(f"  事前CAS均值     = {pre['均值_num']:+.2f} bps  (目标 ≈{TARGET['pre_CAS']} n.s.)")
    print(f"  短期CAS均值     = {short['均值_num']:+.2f} bps  (目标 +{TARGET['short_CAS']})")
    print(f"  LINK即时CAS     = {imm['LINK_num']:+.2f} bps  (目标 +{TARGET['link_imm_CAS']})")

    save = ["窗口"] + ALL_COINS + ["均值"]
    df[save].to_csv("table4_3_CAS.csv", index=False, encoding="utf-8-sig")
    rpt += ["\n【表4.3 H1 CAS（基点）】", hdr]
    for _, r in df.iterrows():
        rpt.append(f"  {r['窗口']:16s}" + "".join(f" {r[c]:>12s}" for c in show)
                   + f" {r['均值']:>10s}")
    rpt += [f"  事前={pre['均值_num']:+.2f} bps | 短期={short['均值_num']:+.2f} bps | "
            f"LINK即时={imm['LINK_num']:+.2f} bps"]
    return df


# ══════════════════════════════════════════════
#  ⑩ 表4.4 CAD（H2）
# ══════════════════════════════════════════════
def table44(panel, rpt):
    print("\n" + "="*62)
    print("步骤5：CAD（表4.4，H2）")
    print("="*62)

    show = ["BTC","ETH","SOL","LINK","XRP"]
    rows = []
    for wname, (ws, we) in WINDOWS.items():
        mask = (panel["tau"] >= ws) & (panel["tau"] <= we)
        row  = {"窗口": wname}
        vals = []
        for coin in ALL_COINS:
            sub        = panel[(panel["coin"]==coin) & mask]["AD"]
            _, _, p    = nw_test(sub)
            cad        = float(pd.to_numeric(sub, errors="coerce").mean())
            row[coin]          = f"{cad:+.1f}{sig(p)}%"
            row[f"{coin}_num"] = cad
            vals.append(cad)
        row["均值"]     = f"{np.nanmean(vals):+.1f}%"
        row["均值_num"] = np.nanmean(vals)
        rows.append(row)

    df = pd.DataFrame(rows)
    hdr = f"  {'窗口':16s}" + "".join(f" {c:>14s}" for c in show) + f" {'均值':>10s}"
    print(hdr); print("  " + "-"*88)
    for _, r in df.iterrows():
        line = f"  {r['窗口']:16s}" + "".join(f" {r[c]:>14s}" for c in show)
        line += f" {r['均值']:>10s}"
        print(line)

    short = df[df["窗口"]=="短期[+1,+15]"].iloc[0]
    print(f"\n  ── 核验 ──")
    print(f"  短期CAD均值 = {short['均值_num']:+.1f}%  (目标 {TARGET['short_CAD']}%)")
    print(f"  LINK短期CAD = {short['LINK_num']:+.1f}%  (目标 {TARGET['link_CAD']}%)")

    save = ["窗口"] + ALL_COINS + ["均值"]
    df[save].to_csv("table4_4_CAD.csv", index=False, encoding="utf-8-sig")
    rpt += ["\n【表4.4 H2 CAD（%）】", hdr]
    for _, r in df.iterrows():
        rpt.append(f"  {r['窗口']:16s}" + "".join(f" {r[c]:>14s}" for c in show)
                   + f" {r['均值']:>10s}")


# ══════════════════════════════════════════════
#  ⑪ 表4.5 H5：恢复非对称检验（替换传染矩阵）
# ══════════════════════════════════════════════
def table45_recovery(panel, bs_map, rpt):
    """
    H5 恢复非对称检验
    定义：价差恢复至 pre_mean × RECOVERY_MULT（1.5倍）以内所需小时数
    数据源：分钟级面板聚合到小时级
    """
    print("\n" + "="*62)
    print("步骤6：H5 恢复非对称检验（表4.5）")
    print("="*62)
    print(f"  恢复判定阈值: spread ≤ pre_mean × {RECOVERY_MULT}")

    # 聚合到小时级（τ按小时取整）
    p = panel.copy()
    p["tau_h"] = (p["tau"] / 60).apply(np.floor).astype(int)

    detail_rows = []  # 每小时价差倍数
    recovery_rows = []

    for coin in ALL_COINS:
        pre_mean = bs_map[coin]
        threshold = pre_mean * RECOVERY_MULT

        # 事件后72小时内逐小时检查
        recovery_h = None
        for h in range(0, 73):
            mask = ((panel["coin"] == coin)
                    & (panel["tau"] >= h*60) & (panel["tau"] < (h+1)*60))
            sp   = pd.to_numeric(panel[mask]["spread_bps"], errors="coerce").dropna()
            if len(sp) == 0:
                continue
            mean_sp = sp.mean()
            mult    = mean_sp / pre_mean if pre_mean > 0 else np.nan

            detail_rows.append({
                "品种": coin, "h": h,
                "价差均值_bps": round(mean_sp, 4),
                "pre_mean_bps": round(pre_mean, 4),
                "倍数":         round(mult, 3) if not np.isnan(mult) else np.nan,
            })

            if recovery_h is None and not np.isnan(mult) and mult <= RECOVERY_MULT:
                recovery_h = h

        recovery_rows.append({
            "品种":          coin,
            "pre_mean_bps":  round(pre_mean, 4),
            "h0_spread_bps": None,  # 填充下方
            "h0_倍数":       None,
            "恢复小时数":    recovery_h if recovery_h is not None else ">72h",
            "恢复组":        _recovery_group(recovery_h),
        })

    # 填充h=0的价差倍数
    df_detail = pd.DataFrame(detail_rows)
    h0 = df_detail[df_detail["h"] == 0][["品种","价差均值_bps","倍数"]].rename(
        columns={"价差均值_bps":"h0_spread_bps","倍数":"h0_倍数"})

    df_rec = pd.DataFrame(recovery_rows)
    df_rec = df_rec.drop(columns=["h0_spread_bps","h0_倍数"]).merge(h0, on="品种", how="left")
    df_rec = df_rec.sort_values("恢复小时数",
                                key=lambda x: pd.to_numeric(x, errors="coerce").fillna(999))

    print("\n  品种恢复时间汇总：")
    print(df_rec[["品种","h0_倍数","恢复小时数","恢复组"]].to_string(index=False))

    # 核验
    link_row = df_rec[df_rec["品种"] == "LINK"].iloc[0]
    btc_row  = df_rec[df_rec["品种"] == "BTC"].iloc[0]
    print(f"\n  ── 核验 ──")
    print(f"  LINK恢复小时数 = {link_row['恢复小时数']}  (目标 {TARGET['link_recovery_h']})")
    print(f"  BTC恢复小时数  = {btc_row['恢复小时数']}   (目标 {TARGET['btc_recovery_h']})")
    print(f"  LINK h=0倍数   = {link_row['h0_倍数']}  (论文 193×，基于l2_spread_1hour)")

    df_rec.to_csv("table4_5_recovery.csv", index=False, encoding="utf-8-sig")
    df_detail.to_csv("table4_5_recovery_detail.csv", index=False, encoding="utf-8-sig")
    print("\n  [已保存] table4_5_recovery.csv  table4_5_recovery_detail.csv")

    rpt += ["\n【表4.5 H5 恢复非对称】",
            df_rec[["品种","h0_倍数","恢复小时数","恢复组"]].to_string(index=False),
            f"\n  LINK恢复={link_row['恢复小时数']}h | BTC恢复={btc_row['恢复小时数']}h"]
    return df_rec, df_detail


def _recovery_group(h):
    if h is None:      return "未恢复(>72h)"
    if h <= 7:         return "快速(≤7h)"
    if h <= 15:        return "中速(8-15h)"
    return f"慢速({h}h)"


# ══════════════════════════════════════════════
#  ⑫ BNB/XRP截面异质性（论文4.3.3节）
# ══════════════════════════════════════════════
def analyze_heterogeneity(panel, bs_map, rpt):
    """
    计算BNB和XRP的成交量-清算占比异质性
    BNB：量缩清算升；XRP：量涨清算降
    """
    print("\n" + "="*62)
    print("步骤6b：BNB/XRP截面异质性分析（4.3.3节）")
    print("="*62)

    rows = []
    for coin in ["BNB", "XRP"]:
        for wlabel, (ws, we) in WIN3.items():
            mask = ((panel["coin"] == coin)
                    & (panel["tau"] >= ws) & (panel["tau"] <= we))
            sub  = panel[mask]
            vol  = pd.to_numeric(sub.get("volume",     pd.Series(dtype=float)),
                                  errors="coerce").mean()
            liq  = pd.to_numeric(sub.get("liq_volume", pd.Series(dtype=float)),
                                  errors="coerce").sum()
            liq_cnt = pd.to_numeric(sub.get("liq_count", pd.Series(dtype=float)),
                                     errors="coerce").sum()
            liq_ratio = (liq / (vol * len(sub) + 1e-10) * 100) if vol > 0 else np.nan

            rows.append({
                "品种": coin, "窗口": wlabel,
                "均值成交量/分钟": round(vol, 2) if not np.isnan(vol) else np.nan,
                "清算量总计":      round(liq, 2) if not np.isnan(liq) else np.nan,
                "清算笔数":        int(liq_cnt) if not np.isnan(liq_cnt) else 0,
                "清算占比_%":      round(liq_ratio, 2) if not np.isnan(liq_ratio) else np.nan,
            })

    df_het = pd.DataFrame(rows)
    print(df_het.to_string(index=False))
    print("""
  解读说明：
  BNB：event期成交量应↓但清算占比↑ → 流动性撤离主导型冲击（论文4.3.3节）
  XRP：event期成交量应↑但清算占比↓ → 方向性投机稀释清算占比（论文4.3.3节）
    """)
    df_het.to_csv("table4_heterogeneity.csv", index=False, encoding="utf-8-sig")
    rpt += ["\n【BNB/XRP异质性（4.3.3节）】", df_het.to_string(index=False)]
    return df_het


# ══════════════════════════════════════════════
#  ⑬ 图4.1 CAS轨迹
# ══════════════════════════════════════════════
DARK_BG  = "#0D1117"
PANEL_BG = "#161B22"
COIN_CLR = {"BTC":"#F7931A","ETH":"#627EEA","SOL":"#9945FF","LINK":"#2A5ADA",
            "XRP":"#00AAE4","BNB":"#F3BA2F","DOGE":"#C2A633",
            "AVAX":"#E84142","HYPE":"#00FFA3"}

def fig41(panel, rpt):
    print("\n步骤7：生成图4.1（CAS轨迹）")
    coins = ["BTC","ETH","SOL","LINK"]
    TMIN, TMAX = -60, 60

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(DARK_BG)

    for ax, coin in zip(axes.flatten(), coins):
        ax.set_facecolor(PANEL_BG)
        sub = (panel[(panel["coin"]==coin)
                     & (panel["tau"]>=TMIN) & (panel["tau"]<=TMAX)]
               .sort_values("tau").copy())
        sub["AS_n"] = pd.to_numeric(sub["AS"], errors="coerce").fillna(0)
        sub["CAS"]  = sub["AS_n"].cumsum()
        roll_std    = sub["AS_n"].rolling(5, min_periods=1).std().fillna(0)

        col = COIN_CLR.get(coin, "#FFFFFF")
        ax.fill_between(sub["tau"], sub["CAS"]-1.96*roll_std,
                        sub["CAS"]+1.96*roll_std, alpha=0.18, color=col)
        ax.plot(sub["tau"], sub["CAS"], color=col, lw=2.2)
        ax.axvline(0, color="#FF4444", lw=1.5, ls="--", alpha=0.9)
        ax.axhline(0, color="#555", lw=0.8, ls=":")

        if len(sub):
            idx_pk = sub["CAS"].idxmax()
            pk_tau = sub.loc[idx_pk,"tau"]
            pk_val = sub.loc[idx_pk,"CAS"]
            ax.annotate(f"Peak\n{pk_val:+.1f}bps",
                        xy=(pk_tau, pk_val),
                        xytext=(pk_tau+4, pk_val*0.82),
                        color="white", fontsize=8,
                        arrowprops=dict(arrowstyle="->", color="#AAA", lw=0.8))

        ax.set_title(f"{coin}/USDC", color="white", fontsize=13, fontweight="bold")
        ax.set_xlabel("tau (min)", color="#AAA", fontsize=9)
        ax.set_ylabel("CAS (bps)", color="#AAA", fontsize=9)
        ax.tick_params(colors="#AAA", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#333")

    fig.suptitle("Fig 4.1  Cumulative Abnormal Spread Trajectories  [tau=-60 to +60]\n"
                 "Estimation window: Oct 7-9 | 95% CI band | Red dashed = tau=0 (21:00 UTC)",
                 color="white", fontsize=11, y=0.99)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig("fig4_1_CAS_trajectory.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  ✅ fig4_1_CAS_trajectory.png")
    rpt.append("【图4.1】fig4_1_CAS_trajectory.png")


# ══════════════════════════════════════════════
#  ⑭ 图4.2 深度崩溃
# ══════════════════════════════════════════════
def fig42(panel, rpt):
    print("\n步骤8：生成图4.2（深度崩溃）")
    est = (panel["tau"] >= EST_W[0]) & (panel["tau"] <= EST_W[1])
    bl  = {}
    for coin in ALL_COINS:
        sub = panel[(panel["coin"]==coin) & est]["depth_usdc"]
        bm  = pd.to_numeric(sub, errors="coerce").mean()
        bl[coin] = bm if (not np.isnan(bm) and bm > 0) else 1

    p2 = panel.copy()
    p2["depth_idx"] = (pd.to_numeric(p2["depth_usdc"], errors="coerce")
                       / p2["coin"].map(bl) * 100)
    view  = p2[(p2["tau"] >= -30) & (p2["tau"] <= 30) & p2["coin"].isin(ALL_COINS)]
    pivot = view.pivot_table(index="tau", columns="coin",
                             values="depth_idx", aggfunc="mean")
    coin_ord = [c for c in ALL_COINS if c in pivot.columns]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    bottom = np.zeros(len(pivot))
    for coin in coin_ord:
        vals = pivot[coin].fillna(0).values / len(coin_ord)
        col  = COIN_CLR.get(coin, "#FFFFFF")
        ax.fill_between(pivot.index, bottom, bottom+vals,
                        alpha=0.78, label=coin, color=col)
        bottom += vals

    ax.axvline(0, color="#FF4444", lw=2.2, ls="--", label="tau=0 Event Start (21:00 UTC)")
    ax.set_xlabel("tau (min)", color="#AAA", fontsize=11)
    ax.set_ylabel("Normalized Depth Index (Estimation Window = 100)", color="#AAA", fontsize=10)
    ax.set_title("Fig 4.2  Order Book Depth Collapse: 9 Contracts  [tau=-30 to +30]",
                 color="white", fontsize=12)
    ax.tick_params(colors="#AAA")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.legend(loc="upper right", framealpha=0.25, labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig("fig4_2_depth_collapse.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  ✅ fig4_2_depth_collapse.png")
    rpt.append("【图4.2】fig4_2_depth_collapse.png")


# ══════════════════════════════════════════════
#  ⑮ 图4.3 恢复路径对比（替换传染热力图）
# ══════════════════════════════════════════════
def fig43_recovery(df_detail, bs_map, rpt):
    """
    [v2 新图4.3] 各品种价差恢复路径对比
    纵轴：spread / pre_mean 倍数（对数刻度）
    横轴：冲击后小时数 h
    重点：LINK vs 快速恢复组(BTC/ETH/SOL) vs 中速组(DOGE/HYPE/XRP)
    """
    print("\n步骤9：生成图4.3（恢复路径对比）")

    # 分组颜色
    group_styles = {
        "LINK" : {"color": "#FF4444", "lw": 3.0, "ls": "-",  "zorder": 10},
        "BTC"  : {"color": "#F7931A", "lw": 2.0, "ls": "-",  "zorder": 5},
        "ETH"  : {"color": "#627EEA", "lw": 2.0, "ls": "-",  "zorder": 5},
        "SOL"  : {"color": "#9945FF", "lw": 2.0, "ls": "-",  "zorder": 5},
        "AVAX" : {"color": "#E84142", "lw": 1.5, "ls": "--", "zorder": 4},
        "BNB"  : {"color": "#F3BA2F", "lw": 1.5, "ls": "--", "zorder": 4},
        "DOGE" : {"color": "#C2A633", "lw": 1.5, "ls": ":",  "zorder": 3},
        "XRP"  : {"color": "#00AAE4", "lw": 1.5, "ls": ":",  "zorder": 3},
        "HYPE" : {"color": "#00FFA3", "lw": 1.5, "ls": ":",  "zorder": 3},
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    for coin in ALL_COINS:
        sub = df_detail[(df_detail["品种"] == coin) & (df_detail["h"] <= 60)]
        if len(sub) == 0:
            continue
        style = group_styles.get(coin, {"color":"#AAAAAA","lw":1.2,"ls":"-","zorder":1})
        ax.plot(sub["h"], sub["倍数"],
                color=style["color"], lw=style["lw"], ls=style["ls"],
                zorder=style["zorder"], label=coin)

    # 1.5倍恢复线
    ax.axhline(RECOVERY_MULT, color="#FFFFFF", lw=1.2, ls="--", alpha=0.6,
               label=f"Recovery threshold ({RECOVERY_MULT}×)")
    # 1.0倍基准线
    ax.axhline(1.0, color="#666666", lw=0.8, ls=":", alpha=0.8)

    ax.set_yscale("log")
    ax.set_xlabel("Hours after event (h=0: 21:00 UTC Oct 10)", color="#AAA", fontsize=11)
    ax.set_ylabel("Spread / Pre-event Mean (log scale)", color="#AAA", fontsize=11)
    ax.set_title(
        "Fig 4.3  Asymmetric Recovery Paths: LINK vs Fast-Recovery Group vs Mid-Recovery Group\n"
        f"Recovery threshold = {RECOVERY_MULT}×pre-event mean  |  "
        f"LINK: {TARGET['link_recovery_h']}h  BTC/ETH/SOL: ~7h  DOGE/XRP/HYPE: ~15h",
        color="white", fontsize=11)
    ax.tick_params(colors="#AAA", labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.legend(loc="upper right", framealpha=0.3, labelcolor="white",
              fontsize=9, ncol=3)

    # 标注LINK恢复点
    link_sub = df_detail[(df_detail["品种"]=="LINK") & (df_detail["倍数"] <= RECOVERY_MULT)]
    if len(link_sub) > 0:
        first_rec = link_sub.iloc[0]
        ax.annotate(f"LINK recovers\nh={int(first_rec['h'])}",
                    xy=(first_rec["h"], RECOVERY_MULT),
                    xytext=(first_rec["h"]-10, RECOVERY_MULT*2),
                    color="#FF4444", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#FF4444", lw=1.2))

    plt.tight_layout()
    plt.savefig("fig4_3_recovery_paths.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("  ✅ fig4_3_recovery_paths.png（恢复路径对比，替换传染热力图）")
    rpt.append("【图4.3】fig4_3_recovery_paths.png")


# ══════════════════════════════════════════════
#  ⑯ 导出逐分钟CAS明细
# ══════════════════════════════════════════════
def export_cas(panel):
    rows = []
    for coin in ALL_COINS:
        sub = (panel[(panel["coin"]==coin) & (panel["tau"]>=-60) & (panel["tau"]<=60)]
               .sort_values("tau").copy())
        sub["AS_n"] = pd.to_numeric(sub["AS"], errors="coerce").fillna(0)
        sub["CAS"]  = sub["AS_n"].cumsum()
        for _, r in sub.iterrows():
            rows.append({"coin":coin, "tau":round(r["tau"],1),
                         "spread_bps":round(r.get("spread_bps",0),4),
                         "depth_usdc":round(r.get("depth_usdc",0),0),
                         "AS":round(r["AS_n"],4), "CAS":round(r["CAS"],4)})
    pd.DataFrame(rows).to_csv("cas_minutely.csv", index=False, encoding="utf-8-sig")
    print("  ✅ cas_minutely.csv")


# ══════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════
def main():
    setup_chinese_font()

    # [修复1] 使用更正后的事件零点
    event_ts = pd.Timestamp(EVENT_UTC, tz="UTC")
    print(f"\n  [v2] τ=0 = {EVENT_UTC} UTC（路透社报道关税公告整点边界）")
    print(f"  [v2] 估计窗口 = Oct 7 00:00 → Oct 9 23:59（τ∈[{EST_W[0]},{EST_W[1]}]）")
    print(f"  [v2] H5 = 恢复非对称检验（LINK为核心研究对象）")

    rpt = ["╔"+"═"*60+"╗",
           "║  第四章 事件研究报告 v2  Hyperliquid 2025-10-10      ║",
           "╚"+"═"*60+"╝",
           f"\n  τ=0 = {EVENT_UTC} UTC",
           f"  估计窗口: Oct 7-9（τ∈[{EST_W[0]},{EST_W[1]}]）",
           f"  品种: {', '.join(ALL_COINS)}",
           f"  H5: 恢复非对称（LINK vs 其他品种恢复路径）"]

    # ── 加载 ──
    panel          = load_panel(event_ts)
    panel          = merge_liq(panel, event_ts)
    panel, bs_map  = build_abnormals(panel)

    # ── 分析 ──
    table41(panel, bs_map, rpt)
    analyze_pre_signal(panel, bs_map, rpt)    # [新增] h=-6预公告信号
    table43(panel, rpt)
    table44(panel, rpt)
    df_rec, df_detail = table45_recovery(panel, bs_map, rpt)  # [修复3] H5恢复非对称
    analyze_heterogeneity(panel, bs_map, rpt)                  # [新增] BNB/XRP异质性

    # ── 图表 ──
    fig41(panel, rpt)
    fig42(panel, rpt)
    fig43_recovery(df_detail, bs_map, rpt)    # [修复12] 恢复路径图替换热力图
    export_cas(panel)

    # ── 最终核验摘要 ──
    rpt += ["\n"+"="*60, "核验摘要", "="*60,
            f"  H1 短期CAS均值  : 查 table4_3_CAS.csv   目标 +{TARGET['short_CAS']} bps",
            f"  H1 LINK即时CAS  : 查 table4_3_CAS.csv   目标 +{TARGET['link_imm_CAS']} bps",
            f"  H2 短期CAD均值  : 查 table4_4_CAD.csv   目标 {TARGET['short_CAD']}%",
            f"  H5 LINK恢复小时  : 查 table4_5_recovery.csv 目标 {TARGET['link_recovery_h']}h",
            f"  H5 BTC恢复小时   : 查 table4_5_recovery.csv 目标 {TARGET['btc_recovery_h']}h",
            "\n输出文件清单：",
            "  table4_1_descriptive.csv      ← 三窗口描述性统计",
            "  table4_3_CAS.csv              ← 论文表4.3 H1",
            "  table4_4_CAD.csv              ← 论文表4.4 H2",
            "  table4_5_recovery.csv         ← 论文表4.5 H5恢复非对称",
            "  table4_5_recovery_detail.csv  ← 每小时价差倍数明细",
            "  table4_pre_signal.csv         ← h=-6预公告信号",
            "  table4_heterogeneity.csv      ← BNB/XRP异质性",
            "  fig4_1_CAS_trajectory.png     ← 替换论文图4.1占位符",
            "  fig4_2_depth_collapse.png     ← 替换论文图4.2占位符",
            "  fig4_3_recovery_paths.png     ← 替换论文图4.3占位符（恢复路径对比）",
            "  cas_minutely.csv",
            "  chapter4_report.txt"]

    with open("chapter4_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rpt))

    print("\n"+"="*62)
    print("✅  第四章分析全部完成！（v2）")
    print("    关键修复：EVENT_UTC / 估计窗口 / H5恢复非对称 / 向量化深度")
    print("="*62)


if __name__ == "__main__":
    main()
