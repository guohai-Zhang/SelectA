#!/usr/bin/env python3
"""
涨停板策略历史回测
- 从K线历史找涨停日（涨幅>=9.8%）
- 分析次日收益率
- 按多维度拆解胜率
"""

import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import t1_trader as t1

def run_ztb_backtest(sample_size=300, days=250):
    print("=" * 65)
    print("  涨停板策略历史回测")
    print(f"  样本: {sample_size} 只股票 × {days} 天K线")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print()

    # 获取股票列表
    print("[1/3] 获取股票列表...")
    stock_list = t1.fetch_stock_list_sina()
    if stock_list.empty:
        print("获取股票列表失败")
        return

    # 筛选活跃股
    active = stock_list[
        (stock_list["最新价"] >= 5) & (stock_list["最新价"] <= 80) &
        (stock_list["换手率"] >= 1) &
        (~stock_list["名称"].str.contains("ST|退市|N |C ", na=False))
    ]
    if len(active) > sample_size:
        active = active.sample(sample_size, random_state=42)
    codes = active["代码"].tolist()
    print(f"  筛选 {len(codes)} 只活跃股")

    # 批量获取K线
    print(f"[2/3] 批量获取K线（{days}天）...")
    all_klines = {}
    done = 0

    def fetch_one(code):
        kl = t1.fetch_kline_long(str(code), days=days)
        if kl.empty or len(kl) < 30:
            return code, None
        kl = t1.calc_all_indicators(kl)
        return code, kl

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_one, c): c for c in codes}
        for f in as_completed(futures):
            done += 1
            if done % 50 == 0:
                print(f"  进度 {done}/{len(codes)}")
            try:
                code, kl = f.result()
                if kl is not None:
                    all_klines[code] = kl
            except Exception:
                pass
            time.sleep(0.05)

    print(f"  有效股票 {len(all_klines)} 只")

    # 扫描涨停日，分析次日表现
    print("[3/3] 扫描涨停日 & 分析次日收益...")

    trades = []  # 所有涨停->次日交易记录

    for code, kl in all_klines.items():
        for j in range(30, len(kl) - 1):
            row = kl.iloc[j]
            chg = row.get("涨跌幅", 0) or 0

            # 涨停判定: 涨幅 >= 9.8%（覆盖四舍五入误差）
            if chg < 9.8:
                continue

            # 次日数据
            next_row = kl.iloc[j + 1]
            buy_price = row["收盘"]  # 涨停价买入（假设排板买到）
            sell_open = next_row["开盘"]  # 次日开盘卖
            sell_close = next_row["收盘"]  # 次日收盘卖

            if buy_price <= 0:
                continue

            ret_open = (sell_open - buy_price) / buy_price * 100  # 开盘卖收益
            ret_close = (sell_close - buy_price) / buy_price * 100  # 收盘卖收益
            next_chg = next_row.get("涨跌幅", 0) or 0

            # 分析涨停日特征
            # 量能: 今日成交量 vs 5日均量
            vol_today = row.get("成交量", 0) or 0
            vol_ma5 = kl["成交量"].iloc[max(0, j-5):j].mean() if j >= 5 else vol_today
            vol_ratio = vol_today / vol_ma5 if vol_ma5 > 0 else 1.0

            # 换手率
            turnover = row.get("换手率", 0) or 0

            # 振幅（低振幅=一字板/秒板，高振幅=冲板封）
            amplitude = row.get("振幅", 0) or 0

            # 连板: 往回看前几天是否也是涨停
            consecutive = 1
            for k in range(j - 1, max(j - 10, -1), -1):
                if kl.iloc[k].get("涨跌幅", 0) >= 9.8:
                    consecutive += 1
                else:
                    break

            # 技术位置
            close = row["收盘"]
            ma5 = kl["收盘"].iloc[max(0, j-4):j+1].mean()
            ma10 = kl["收盘"].iloc[max(0, j-9):j+1].mean()
            ma20 = kl["收盘"].iloc[max(0, j-19):j+1].mean()
            above_ma20 = close > ma20
            ma_bull = ma5 > ma10 > ma20

            # MACD
            dif = row.get("DIF", 0) or 0
            dea = row.get("DEA", 0) or 0
            macd_bull = dif > dea

            # 前5日涨幅（涨停前的趋势）
            if j >= 5:
                pre5_ret = (kl.iloc[j-1]["收盘"] - kl.iloc[j-5]["收盘"]) / kl.iloc[j-5]["收盘"] * 100
            else:
                pre5_ret = 0

            # 价格区间
            price = row["收盘"]
            if price < 10:
                price_zone = "低价(<10)"
            elif price < 30:
                price_zone = "中价(10-30)"
            elif price < 50:
                price_zone = "中高(30-50)"
            else:
                price_zone = "高价(50+)"

            trades.append({
                "code": code,
                "date": str(row.get("日期", "")),
                "buy_price": buy_price,
                "sell_open": sell_open,
                "sell_close": sell_close,
                "ret_open": ret_open,
                "ret_close": ret_close,
                "next_chg": next_chg,
                "chg": chg,
                "vol_ratio": vol_ratio,
                "turnover": turnover,
                "amplitude": amplitude,
                "consecutive": consecutive,
                "above_ma20": above_ma20,
                "ma_bull": ma_bull,
                "macd_bull": macd_bull,
                "pre5_ret": pre5_ret,
                "price_zone": price_zone,
            })

    print(f"  共找到 {len(trades)} 笔涨停->次日交易记录")
    print()

    if not trades:
        print("无涨停记录，无法回测")
        return

    df = pd.DataFrame(trades)

    # ============================================================
    # 总体统计
    # ============================================================
    print("=" * 65)
    print("  总体统计")
    print("=" * 65)

    def stats(series, label=""):
        n = len(series)
        if n == 0:
            return
        wins = (series > 0).sum()
        wr = wins / n * 100
        avg = series.mean()
        avg_win = series[series > 0].mean() if wins > 0 else 0
        avg_loss = series[series <= 0].mean() if (n - wins) > 0 else 0
        median = series.median()
        print(f"  {label}")
        print(f"    样本: {n} | 胜率: {wr:.1f}% | 平均: {avg:+.2f}%")
        print(f"    均盈: {avg_win:+.2f}% | 均亏: {avg_loss:+.2f}% | 中位数: {median:+.2f}%")
        print(f"    最大盈: {series.max():+.2f}% | 最大亏: {series.min():+.2f}%")
        return {"n": n, "wr": round(wr, 1), "avg": round(avg, 2)}

    print()
    stats(df["ret_open"], "次日开盘卖（竞价出）")
    print()
    stats(df["ret_close"], "次日收盘卖（持有一整天）")
    print()

    # 次日也涨停的比例
    next_zt = (df["next_chg"] >= 9.8).sum()
    print(f"  次日继续涨停: {next_zt} 笔 ({next_zt/len(df)*100:.1f}%)")
    # 次日跌停
    next_dt = (df["next_chg"] <= -9.8).sum()
    print(f"  次日跌停(天地板): {next_dt} 笔 ({next_dt/len(df)*100:.1f}%)")

    # ============================================================
    # 按连板数分析
    # ============================================================
    print()
    print("=" * 65)
    print("  按连板数分析")
    print("=" * 65)
    print()
    for lb in sorted(df["consecutive"].unique()):
        sub = df[df["consecutive"] == lb]
        label = f"{lb}连板" if lb > 1 else "首板"
        s = stats(sub["ret_close"], f"{label} (次日收盘卖)")
        print()

    # ============================================================
    # 按量比分析
    # ============================================================
    print("=" * 65)
    print("  按量比(涨停日)分析 — 次日收盘卖")
    print("=" * 65)
    print()
    vol_bins = [(0, 0.8, "缩量(<0.8)"), (0.8, 1.5, "正常(0.8-1.5)"),
                (1.5, 3.0, "放量(1.5-3)"), (3.0, 5.0, "大放量(3-5)"),
                (5.0, 100, "巨量(5+)")]
    for lo, hi, label in vol_bins:
        sub = df[(df["vol_ratio"] >= lo) & (df["vol_ratio"] < hi)]
        if len(sub) > 0:
            stats(sub["ret_close"], f"量比 {label}")
            print()

    # ============================================================
    # 按振幅分析（区分一字板 vs 换手板）
    # ============================================================
    print("=" * 65)
    print("  按振幅分析（区分封板类型）— 次日收盘卖")
    print("=" * 65)
    print()
    amp_bins = [(0, 2, "一字板(振幅<2%)"), (2, 5, "T字板(2-5%)"),
                (5, 10, "换手板(5-10%)"), (10, 100, "天地板(10%+)")]
    for lo, hi, label in amp_bins:
        sub = df[(df["amplitude"] >= lo) & (df["amplitude"] < hi)]
        if len(sub) > 0:
            stats(sub["ret_close"], label)
            print()

    # ============================================================
    # 按换手率分析
    # ============================================================
    print("=" * 65)
    print("  按换手率分析 — 次日收盘卖")
    print("=" * 65)
    print()
    tr_bins = [(0, 3, "低换手(<3%)"), (3, 8, "中换手(3-8%)"),
               (8, 15, "高换手(8-15%)"), (15, 100, "超高换手(15%+)")]
    for lo, hi, label in tr_bins:
        sub = df[(df["turnover"] >= lo) & (df["turnover"] < hi)]
        if len(sub) > 0:
            stats(sub["ret_close"], label)
            print()

    # ============================================================
    # 按技术位置分析
    # ============================================================
    print("=" * 65)
    print("  按技术位置分析 — 次日收盘卖")
    print("=" * 65)
    print()
    stats(df[df["ma_bull"]]["ret_close"], "多头排列(MA5>MA10>MA20)")
    print()
    stats(df[~df["ma_bull"]]["ret_close"], "非多头排列")
    print()
    stats(df[df["above_ma20"]]["ret_close"], "站上MA20")
    print()
    stats(df[~df["above_ma20"]]["ret_close"], "MA20下方")
    print()
    stats(df[df["macd_bull"]]["ret_close"], "MACD多头(DIF>DEA)")
    print()
    stats(df[~df["macd_bull"]]["ret_close"], "MACD空头")
    print()

    # ============================================================
    # 按涨停前5日涨幅分析
    # ============================================================
    print("=" * 65)
    print("  按涨停前5日涨幅分析 — 次日收盘卖")
    print("=" * 65)
    print()
    pre_bins = [(-100, -5, "前5日跌>5%"), (-5, 0, "前5日微跌"),
                (0, 5, "前5日微涨(0-5%)"), (5, 15, "前5日涨5-15%"),
                (15, 100, "前5日涨>15%")]
    for lo, hi, label in pre_bins:
        sub = df[(df["pre5_ret"] >= lo) & (df["pre5_ret"] < hi)]
        if len(sub) > 0:
            stats(sub["ret_close"], label)
            print()

    # ============================================================
    # 按价格区间
    # ============================================================
    print("=" * 65)
    print("  按价格区间分析 — 次日收盘卖")
    print("=" * 65)
    print()
    for zone in ["低价(<10)", "中价(10-30)", "中高(30-50)", "高价(50+)"]:
        sub = df[df["price_zone"] == zone]
        if len(sub) > 0:
            stats(sub["ret_close"], zone)
            print()

    # ============================================================
    # 组合条件测试：找最优策略
    # ============================================================
    print("=" * 65)
    print("  策略优化扫描（组合条件 → 提升胜率）")
    print("=" * 65)
    print()

    strategies = []

    def test_strategy(name, mask):
        sub = df[mask]
        n = len(sub)
        if n < 20:
            return
        wr = (sub["ret_close"] > 0).sum() / n * 100
        avg = sub["ret_close"].mean()
        strategies.append({"name": name, "n": n, "wr": round(wr, 1), "avg": round(avg, 2)})

    # 基准
    test_strategy("基准(所有涨停)", pd.Series([True] * len(df)))

    # 连板
    test_strategy("首板only", df["consecutive"] == 1)
    test_strategy("2板+", df["consecutive"] >= 2)
    test_strategy("3板+", df["consecutive"] >= 3)

    # 量能
    test_strategy("放量涨停(1.5-3)", (df["vol_ratio"] >= 1.5) & (df["vol_ratio"] < 3))
    test_strategy("缩量涨停(<0.8)", df["vol_ratio"] < 0.8)

    # 振幅
    test_strategy("一字板(振幅<2%)", df["amplitude"] < 2)
    test_strategy("换手板(振幅5-10%)", (df["amplitude"] >= 5) & (df["amplitude"] < 10))

    # 换手率
    test_strategy("低换手涨停(<5%)", df["turnover"] < 5)
    test_strategy("适中换手(5-12%)", (df["turnover"] >= 5) & (df["turnover"] < 12))

    # 技术
    test_strategy("多头+涨停", df["ma_bull"])
    test_strategy("MACD多头+涨停", df["macd_bull"])
    test_strategy("站上MA20+涨停", df["above_ma20"])

    # 趋势
    test_strategy("前5日微跌后涨停", (df["pre5_ret"] >= -5) & (df["pre5_ret"] < 0))
    test_strategy("前5日平稳后涨停(<5%)", (df["pre5_ret"] >= -2) & (df["pre5_ret"] < 5))

    # 组合
    test_strategy("首板+放量+多头",
                  (df["consecutive"] == 1) & (df["vol_ratio"] >= 1.5) & (df["vol_ratio"] < 3) & df["ma_bull"])
    test_strategy("首板+放量+MACD多头",
                  (df["consecutive"] == 1) & (df["vol_ratio"] >= 1.5) & (df["vol_ratio"] < 3) & df["macd_bull"])
    test_strategy("首板+换手板+站上MA20",
                  (df["consecutive"] == 1) & (df["amplitude"] >= 5) & (df["amplitude"] < 10) & df["above_ma20"])
    test_strategy("首板+前5日微跌+放量",
                  (df["consecutive"] == 1) & (df["pre5_ret"] >= -5) & (df["pre5_ret"] < 0) & (df["vol_ratio"] >= 1.5))
    test_strategy("2板+多头排列",
                  (df["consecutive"] >= 2) & df["ma_bull"])
    test_strategy("一字板(缩量秒板)",
                  (df["amplitude"] < 2) & (df["vol_ratio"] < 1))
    test_strategy("换手板+适中换手+多头",
                  (df["amplitude"] >= 5) & (df["turnover"] >= 5) & (df["turnover"] < 12) & df["ma_bull"])
    test_strategy("低价首板+放量",
                  (df["consecutive"] == 1) & (df["price_zone"] == "低价(<10)") & (df["vol_ratio"] >= 1.5))
    test_strategy("中价首板+多头",
                  (df["consecutive"] == 1) & (df["price_zone"] == "中价(10-30)") & df["ma_bull"])

    # 开盘卖 vs 收盘卖
    test_strategy("[开盘卖]所有涨停", pd.Series([True] * len(df)))

    strategies.sort(key=lambda x: x["wr"], reverse=True)
    base_wr = next((s["wr"] for s in strategies if s["name"] == "基准(所有涨停)"), 50)

    print(f"  {'策略':<28} {'样本':>5} {'胜率':>6} {'均盈':>7} {'vs基准':>7}")
    print("  " + "-" * 56)
    for s in strategies:
        delta = s["wr"] - base_wr
        delta_str = f"{delta:+.1f}%" if s["name"] != "基准(所有涨停)" else "  ---"
        if s["name"].startswith("[开盘卖]"):
            # 重新计算开盘卖
            wr_open = (df["ret_open"] > 0).sum() / len(df) * 100
            avg_open = df["ret_open"].mean()
            print(f"  {s['name']:<28} {len(df):>5} {wr_open:>5.1f}% {avg_open:>+6.2f}%    ---")
        else:
            mark = " ★" if s["wr"] >= base_wr + 3 and s["n"] >= 30 else ""
            print(f"  {s['name']:<28} {s['n']:>5} {s['wr']:>5.1f}% {s['avg']:>+6.2f}% {delta_str}{mark}")

    # ============================================================
    # 收益分布
    # ============================================================
    print()
    print("=" * 65)
    print("  次日收盘卖收益分布")
    print("=" * 65)
    print()
    bins = [(-100, -7), (-7, -5), (-5, -3), (-3, -1), (-1, 0),
            (0, 1), (1, 3), (3, 5), (5, 7), (7, 100)]
    for lo, hi in bins:
        sub = df[(df["ret_close"] >= lo) & (df["ret_close"] < hi)]
        pct = len(sub) / len(df) * 100
        bar = "█" * int(pct)
        label = f"{lo:+d}% ~ {hi:+d}%"
        if hi == 100:
            label = f"{lo:+d}% 以上"
        if lo == -100:
            label = f"{hi:+d}% 以下"
        print(f"  {label:>12}: {len(sub):>4} ({pct:>5.1f}%) {bar}")

    print()
    print("=" * 65)
    print("  回测结论")
    print("=" * 65)
    wr_close = (df["ret_close"] > 0).sum() / len(df) * 100
    wr_open = (df["ret_open"] > 0).sum() / len(df) * 100
    print(f"  涨停次日收盘卖胜率: {wr_close:.1f}%")
    print(f"  涨停次日开盘卖胜率: {wr_open:.1f}%")
    if strategies:
        best = max(strategies, key=lambda x: x["wr"] if x["n"] >= 30 and not x["name"].startswith("[") else 0)
        print(f"  最优策略: {best['name']} → 胜率 {best['wr']}% (样本{best['n']})")
    print()

    return df, strategies


if __name__ == "__main__":
    sample = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 300
    run_ztb_backtest(sample_size=sample, days=250)
