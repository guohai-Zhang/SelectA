#!/usr/bin/env python3
"""
涨停预测策略回测
- 模拟早盘10:00-10:30买入涨幅0.5-5%的股票
- 验证：哪些因子能预测当天涨停？次日卖出收益如何？
- 用历史数据校准评分权重

限制：只有日线OHLC数据，无法完美模拟10:30盘中状态
代理指标：开盘涨幅 0.5-5% ≈ "10:30时涨幅0.5-5%"
买入价 = 当日开盘价（保守估计）
"""

import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import t1_trader as t1


def run_predict_zt_backtest(sample_size=300, days=250):
    print("=" * 65)
    print("  涨停预测策略回测")
    print(f"  样本: {sample_size} 只活跃股 × {days} 天K线")
    print(f"  策略: 开盘涨0.5-5%时买入 → 预期当天涨停 → 次日卖出")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print()

    # ── 获取股票列表 ──
    print("[1/4] 获取股票列表...")
    stock_list = t1.fetch_stock_list_sina()
    if stock_list.empty:
        print("获取股票列表失败")
        return

    active = stock_list[
        (stock_list["最新价"] >= 5) & (stock_list["最新价"] <= 80) &
        (stock_list["换手率"] >= 1) &
        (~stock_list["名称"].str.contains("ST|退市|N |C ", na=False))
    ]
    if len(active) > sample_size:
        active = active.sample(sample_size, random_state=42)
    codes = active["代码"].tolist()
    print(f"  筛选 {len(codes)} 只活跃股")

    # ── 批量获取K线 ──
    print(f"[2/4] 批量获取K线（{days}天）...")
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

    # ── 扫描候选日 & 分析 ──
    print("[3/4] 扫描候选日（开盘涨0.5-5%）& 分析因子...")

    trades = []

    for code, kl in all_klines.items():
        for j in range(30, len(kl) - 1):
            row = kl.iloc[j]
            prev_row = kl.iloc[j - 1]

            prev_close = prev_row["收盘"]
            if prev_close <= 0:
                continue

            open_price = row["开盘"]
            close_price = row["收盘"]
            high_price = row["最高"]
            low_price = row["最低"]

            # 开盘涨幅（代理10:30涨幅）
            open_gap = (open_price - prev_close) / prev_close * 100

            # 筛选：开盘涨 0.5% ~ 5%
            if open_gap < 0.5 or open_gap > 5:
                continue

            # 当日涨跌幅
            day_chg = row.get("涨跌幅", 0) or 0
            # 是否涨停
            hit_zt = day_chg >= 9.8
            # 振幅
            amplitude = (high_price - low_price) / prev_close * 100 if prev_close > 0 else 0

            # 次日数据
            next_row = kl.iloc[j + 1]
            # 买入价 = 今日开盘（模拟10:30左右买入）
            buy_price = open_price
            sell_open = next_row["开盘"]
            sell_close = next_row["收盘"]

            if buy_price <= 0:
                continue

            # 收益（基于买入价=今日开盘）
            ret_open = (sell_open - buy_price) / buy_price * 100
            ret_close = (sell_close - buy_price) / buy_price * 100

            # ── 因子特征 ──

            # 量比：当日成交量 vs 5日均量
            vol_today = row.get("成交量", 0) or 0
            vol_ma5 = kl["成交量"].iloc[max(0, j - 5):j].mean() if j >= 5 else vol_today
            vol_ratio = vol_today / vol_ma5 if vol_ma5 > 0 else 1.0

            # 换手率
            turnover = row.get("换手率", 0) or 0

            # 前3日缩量程度（蓄势）
            if j >= 3:
                prev3_vol = kl["成交量"].iloc[j - 3:j].mean()
                vol_expand = vol_today / prev3_vol if prev3_vol > 0 else 1.0
            else:
                vol_expand = 1.0

            # 前3日振幅（横盘程度）
            if j >= 3:
                prev3_chg = [abs(kl.iloc[j - k].get("涨跌幅", 0) or 0) for k in range(1, 4)]
                prev3_max_chg = max(prev3_chg)
                prev3_avg_chg = sum(prev3_chg) / 3
            else:
                prev3_max_chg = 10
                prev3_avg_chg = 5

            # 均线位置
            price = close_price
            ma5 = row.get("MA5", 0) or 0
            ma10 = row.get("MA10", 0) or 0
            ma20 = row.get("MA20", 0) or 0
            ma_bull = (ma5 > ma10 > ma20) if (ma5 > 0 and ma10 > 0 and ma20 > 0) else False
            above_ma20 = price > ma20 if ma20 > 0 else False

            # MACD
            dif = row.get("DIF", 0) or 0
            dea = row.get("DEA", 0) or 0
            macd_bull = dif > dea
            macd_reversal = dif < dea and dif < 0  # 空头反转

            # 前5日涨幅
            if j >= 5:
                pre5_ret = (prev_row["收盘"] - kl.iloc[j - 5]["收盘"]) / kl.iloc[j - 5]["收盘"] * 100
            else:
                pre5_ret = 0

            # 价格区间
            if price < 10:
                price_zone = "低价(<10)"
            elif price < 30:
                price_zone = "中价(10-30)"
            elif price < 50:
                price_zone = "中高(30-50)"
            else:
                price_zone = "高价(50+)"

            # 开盘到收盘方向（是否盘中走强）
            intraday_up = close_price > open_price

            # 回撤幅度：(振幅 - 涨幅) 越小=走势越坚决
            pullback = amplitude - day_chg if amplitude > day_chg else 0

            trades.append({
                "code": code,
                "date": str(row.get("日期", "")),
                "buy_price": buy_price,
                "sell_open": sell_open,
                "sell_close": sell_close,
                "ret_open": ret_open,
                "ret_close": ret_close,
                "open_gap": open_gap,
                "day_chg": day_chg,
                "hit_zt": hit_zt,
                "amplitude": amplitude,
                "pullback": pullback,
                "vol_ratio": vol_ratio,
                "vol_expand": vol_expand,
                "turnover": turnover,
                "prev3_max_chg": prev3_max_chg,
                "prev3_avg_chg": prev3_avg_chg,
                "ma_bull": ma_bull,
                "above_ma20": above_ma20,
                "macd_bull": macd_bull,
                "macd_reversal": macd_reversal,
                "pre5_ret": pre5_ret,
                "price_zone": price_zone,
                "intraday_up": intraday_up,
            })

    print(f"  共找到 {len(trades)} 笔候选记录（开盘涨0.5-5%的日子）")
    if not trades:
        print("无候选记录")
        return

    df = pd.DataFrame(trades)
    zt_df = df[df["hit_zt"]]
    nozt_df = df[~df["hit_zt"]]

    print(f"  其中涨停 {len(zt_df)} 笔 ({len(zt_df)/len(df)*100:.1f}%)")
    print(f"  未涨停 {len(nozt_df)} 笔 ({len(nozt_df)/len(df)*100:.1f}%)")
    print()

    # ============================================================
    # 统计函数
    # ============================================================
    def stats(series, label="", show=True):
        n = len(series)
        if n == 0:
            return None
        wins = (series > 0).sum()
        wr = wins / n * 100
        avg = series.mean()
        avg_win = series[series > 0].mean() if wins > 0 else 0
        avg_loss = series[series <= 0].mean() if (n - wins) > 0 else 0
        if show:
            print(f"  {label}")
            print(f"    样本: {n} | 胜率: {wr:.1f}% | 平均: {avg:+.2f}%")
            print(f"    均盈: {avg_win:+.2f}% | 均亏: {avg_loss:+.2f}%")
        return {"n": n, "wr": round(wr, 1), "avg": round(avg, 2)}

    # ============================================================
    print("[4/4] 分析结果...")
    print()

    # ============================================================
    # 基准：所有候选的次日收益
    # ============================================================
    print("=" * 65)
    print("  基准统计（所有开盘涨0.5-5%的股票）")
    print("=" * 65)
    print()
    stats(df["ret_open"], "次日开盘卖（今日开盘买→次日开盘卖）")
    print()
    stats(df["ret_close"], "次日收盘卖（今日开盘买→次日收盘卖）")
    print()
    print(f"  涨停命中率: {len(zt_df)/len(df)*100:.1f}% ({len(zt_df)}/{len(df)})")
    print()

    # 涨停 vs 未涨停 次日收益对比
    print("=" * 65)
    print("  涨停 vs 未涨停 次日收益对比")
    print("=" * 65)
    print()
    stats(zt_df["ret_open"], "涨停股 → 次日开盘卖")
    print()
    stats(zt_df["ret_close"], "涨停股 → 次日收盘卖")
    print()
    stats(nozt_df["ret_open"], "未涨停 → 次日开盘卖")
    print()
    stats(nozt_df["ret_close"], "未涨停 → 次日收盘卖")
    print()

    # ============================================================
    # 因子分析：哪些因子能提高涨停命中率？
    # ============================================================
    print("=" * 65)
    print("  因子分析：提升涨停命中率")
    print("=" * 65)
    print()

    base_zt_rate = len(zt_df) / len(df) * 100 if len(df) > 0 else 0

    def factor_analysis(mask, label):
        sub = df[mask]
        n = len(sub)
        if n < 20:
            return None
        zt_n = sub["hit_zt"].sum()
        zt_rate = zt_n / n * 100
        ret_avg = sub["ret_close"].mean()
        wr = (sub["ret_close"] > 0).sum() / n * 100
        delta = zt_rate - base_zt_rate
        mark = " ★" if zt_rate >= base_zt_rate * 1.3 and n >= 30 else ""
        print(f"  {label:<30} 样本{n:>4} 涨停{zt_n:>3}({zt_rate:>5.1f}%,{delta:>+5.1f}) "
              f"次日胜率{wr:>5.1f}% 次日均盈{ret_avg:>+5.2f}%{mark}")
        return {"label": label, "n": n, "zt_n": zt_n, "zt_rate": zt_rate,
                "wr": wr, "ret_avg": ret_avg}

    results = []

    print(f"  {'基准':<30} 样本{len(df):>4} 涨停{len(zt_df):>3}({base_zt_rate:>5.1f}%)")
    print()

    # 按开盘涨幅
    print("  ── 按开盘涨幅 ──")
    for lo, hi, label in [(0.5, 1, "微涨(0.5-1%)"), (1, 2, "小涨(1-2%)"),
                           (2, 3, "中涨(2-3%)"), (3, 4, "较强(3-4%)"),
                           (4, 5, "强势(4-5%)")]:
        r = factor_analysis((df["open_gap"] >= lo) & (df["open_gap"] < hi), label)
        if r:
            results.append(r)
    print()

    # 按量比
    print("  ── 按量比 ──")
    for lo, hi, label in [(1.0, 2, "温和放量(1-2)"), (2, 3, "放量(2-3)"),
                           (3, 5, "大放量(3-5)"), (5, 8, "巨量(5-8)"),
                           (8, 100, "超巨量(8+)")]:
        r = factor_analysis((df["vol_ratio"] >= lo) & (df["vol_ratio"] < hi), label)
        if r:
            results.append(r)
    print()

    # 按量能爆发（vs前3日）
    print("  ── 按量能爆发(vs前3日) ──")
    for lo, hi, label in [(1, 1.5, "微放(1-1.5x)"), (1.5, 2, "放量(1.5-2x)"),
                           (2, 3, "倍量(2-3x)"), (3, 5, "巨量爆发(3-5x)"),
                           (5, 100, "超级爆发(5x+)")]:
        r = factor_analysis((df["vol_expand"] >= lo) & (df["vol_expand"] < hi), label)
        if r:
            results.append(r)
    print()

    # 前3日横盘程度
    print("  ── 前3日形态 ──")
    for hi, label in [(2, "3日窄幅横盘(<2%)"), (3, "3日小幅波动(<3%)"),
                       (5, "3日正常波动(<5%)")]:
        r = factor_analysis(df["prev3_max_chg"] < hi, f"{label}")
        if r:
            results.append(r)
    r = factor_analysis(df["prev3_max_chg"] >= 5, "3日有大波动(>=5%)")
    if r:
        results.append(r)
    print()

    # 技术形态
    print("  ── 技术形态 ──")
    factor_analysis(df["ma_bull"], "多头排列(MA5>10>20)")
    factor_analysis(~df["ma_bull"], "非多头排列")
    factor_analysis(df["above_ma20"], "站上MA20")
    factor_analysis(~df["above_ma20"], "MA20下方")
    factor_analysis(df["macd_bull"], "MACD多头")
    factor_analysis(df["macd_reversal"], "MACD空头反转(DIF<DEA且<0)")
    print()

    # 前5日趋势
    print("  ── 前5日趋势 ──")
    for lo, hi, label in [(-100, -5, "前5日跌>5%"), (-5, 0, "前5日微跌"),
                           (0, 5, "前5日微涨(0-5%)"), (5, 15, "前5日涨5-15%"),
                           (15, 100, "前5日涨>15%")]:
        r = factor_analysis((df["pre5_ret"] >= lo) & (df["pre5_ret"] < hi), label)
        if r:
            results.append(r)
    print()

    # 价格区间
    print("  ── 价格区间 ──")
    for zone in ["低价(<10)", "中价(10-30)", "中高(30-50)", "高价(50+)"]:
        r = factor_analysis(df["price_zone"] == zone, zone)
        if r:
            results.append(r)
    print()

    # 换手率
    print("  ── 换手率 ──")
    for lo, hi, label in [(0, 3, "低换手(<3%)"), (3, 8, "适中(3-8%)"),
                           (8, 15, "高换手(8-15%)"), (15, 100, "超高(15%+)")]:
        r = factor_analysis((df["turnover"] >= lo) & (df["turnover"] < hi), label)
        if r:
            results.append(r)
    print()

    # ============================================================
    # 组合策略扫描：找最优
    # ============================================================
    print("=" * 65)
    print("  组合策略扫描（找高胜率组合）")
    print("=" * 65)
    print()

    combos = []

    def test_combo(name, mask):
        sub = df[mask]
        n = len(sub)
        if n < 15:
            return
        zt_n = sub["hit_zt"].sum()
        zt_rate = zt_n / n * 100
        wr_close = (sub["ret_close"] > 0).sum() / n * 100
        avg_close = sub["ret_close"].mean()
        wr_open = (sub["ret_open"] > 0).sum() / n * 100
        avg_open = sub["ret_open"].mean()
        combos.append({
            "name": name, "n": n, "zt_n": zt_n, "zt_rate": zt_rate,
            "wr_close": wr_close, "avg_close": avg_close,
            "wr_open": wr_open, "avg_open": avg_open,
        })

    # 基准
    test_combo("基准(全部)", pd.Series([True] * len(df)))

    # 单因子
    test_combo("量比>=3", df["vol_ratio"] >= 3)
    test_combo("量比>=5", df["vol_ratio"] >= 5)
    test_combo("量能爆发>=2x", df["vol_expand"] >= 2)
    test_combo("量能爆发>=3x", df["vol_expand"] >= 3)
    test_combo("前3日横盘(<3%)", df["prev3_max_chg"] < 3)
    test_combo("前3日横盘(<2%)", df["prev3_max_chg"] < 2)
    test_combo("多头排列", df["ma_bull"])
    test_combo("MACD空头反转", df["macd_reversal"])
    test_combo("站上MA20", df["above_ma20"])
    test_combo("开盘涨2-4%", (df["open_gap"] >= 2) & (df["open_gap"] < 4))
    test_combo("开盘涨3-5%", (df["open_gap"] >= 3) & (df["open_gap"] < 5))

    # 双因子组合
    test_combo("量比>=3 + 多头排列",
               (df["vol_ratio"] >= 3) & df["ma_bull"])
    test_combo("量比>=3 + 站上MA20",
               (df["vol_ratio"] >= 3) & df["above_ma20"])
    test_combo("量比>=3 + MACD空头反转",
               (df["vol_ratio"] >= 3) & df["macd_reversal"])
    test_combo("量能爆发2x + 多头排列",
               (df["vol_expand"] >= 2) & df["ma_bull"])
    test_combo("量能爆发2x + 前3日横盘",
               (df["vol_expand"] >= 2) & (df["prev3_max_chg"] < 3))
    test_combo("前3日横盘 + 多头排列",
               (df["prev3_max_chg"] < 3) & df["ma_bull"])
    test_combo("前3日横盘 + 站上MA20",
               (df["prev3_max_chg"] < 3) & df["above_ma20"])
    test_combo("开盘2-4% + 量比>=3",
               (df["open_gap"] >= 2) & (df["open_gap"] < 4) & (df["vol_ratio"] >= 3))
    test_combo("开盘2-4% + 多头排列",
               (df["open_gap"] >= 2) & (df["open_gap"] < 4) & df["ma_bull"])
    test_combo("开盘1-3% + 量能爆发2x",
               (df["open_gap"] >= 1) & (df["open_gap"] < 3) & (df["vol_expand"] >= 2))

    # 三因子组合
    test_combo("量比>=3 + 多头 + 前3日横盘",
               (df["vol_ratio"] >= 3) & df["ma_bull"] & (df["prev3_max_chg"] < 3))
    test_combo("量能爆发2x + 多头 + 前3日横盘",
               (df["vol_expand"] >= 2) & df["ma_bull"] & (df["prev3_max_chg"] < 3))
    test_combo("量比>=3 + 站上MA20 + 前3日横盘",
               (df["vol_ratio"] >= 3) & df["above_ma20"] & (df["prev3_max_chg"] < 3))
    test_combo("量比>=3 + MACD反转 + 前3日横盘",
               (df["vol_ratio"] >= 3) & df["macd_reversal"] & (df["prev3_max_chg"] < 3))
    test_combo("开盘2-4% + 量比>=3 + 多头",
               (df["open_gap"] >= 2) & (df["open_gap"] < 4) & (df["vol_ratio"] >= 3) & df["ma_bull"])
    test_combo("开盘2-4% + 量能爆发2x + 站上MA20",
               (df["open_gap"] >= 2) & (df["open_gap"] < 4) & (df["vol_expand"] >= 2) & df["above_ma20"])
    test_combo("开盘1-3% + 量比>=3 + 前3日横盘",
               (df["open_gap"] >= 1) & (df["open_gap"] < 3) & (df["vol_ratio"] >= 3) & (df["prev3_max_chg"] < 3))
    test_combo("开盘1-3% + 量能爆发3x + 多头",
               (df["open_gap"] >= 1) & (df["open_gap"] < 3) & (df["vol_expand"] >= 3) & df["ma_bull"])

    # 四因子组合
    test_combo("量比>=3 + 多头 + 前3日横盘 + 开盘2-4%",
               (df["vol_ratio"] >= 3) & df["ma_bull"] & (df["prev3_max_chg"] < 3) &
               (df["open_gap"] >= 2) & (df["open_gap"] < 4))
    test_combo("量能爆发2x + 站上MA20 + 前3日横盘 + 开盘1-3%",
               (df["vol_expand"] >= 2) & df["above_ma20"] & (df["prev3_max_chg"] < 3) &
               (df["open_gap"] >= 1) & (df["open_gap"] < 3))
    test_combo("量比>=3 + 站上MA20 + 换手3-8% + 前3日横盘",
               (df["vol_ratio"] >= 3) & df["above_ma20"] &
               (df["turnover"] >= 3) & (df["turnover"] < 8) & (df["prev3_max_chg"] < 3))
    test_combo("量能爆发2x + 多头 + 换手3-8% + 中价股",
               (df["vol_expand"] >= 2) & df["ma_bull"] &
               (df["turnover"] >= 3) & (df["turnover"] < 8) & (df["price_zone"] == "中价(10-30)"))

    # 按次日收盘胜率排序
    combos.sort(key=lambda x: x["wr_close"], reverse=True)

    print(f"  {'策略':<38} {'样本':>4} {'涨停':>4} {'命中率':>6} {'次日胜率':>7} {'次日均盈':>7}")
    print("  " + "-" * 70)
    for c in combos:
        mark = " ★" if c["wr_close"] >= 60 and c["n"] >= 20 else ""
        print(f"  {c['name']:<38} {c['n']:>4} {c['zt_n']:>4} {c['zt_rate']:>5.1f}% "
              f"{c['wr_close']:>6.1f}% {c['avg_close']:>+6.2f}%{mark}")

    # ============================================================
    # 分析：涨停命中时 vs 未命中时的盈亏
    # ============================================================
    print()
    print("=" * 65)
    print("  盈亏分析：涨停命中 vs 未命中")
    print("=" * 65)
    print()

    # 对最优策略分析
    best_combos = [c for c in combos if c["n"] >= 20 and c["name"] != "基准(全部)"]
    if best_combos:
        best = best_combos[0]
        print(f"  最优策略: {best['name']}")
        print(f"  样本 {best['n']} 笔, 涨停命中 {best['zt_n']} 笔 ({best['zt_rate']:.1f}%)")
        print(f"  次日收盘卖胜率: {best['wr_close']:.1f}%, 均盈: {best['avg_close']:+.2f}%")
        print(f"  次日开盘卖胜率: {best['wr_open']:.1f}%, 均盈: {best['avg_open']:+.2f}%")
        print()

    # 全量数据分析
    if len(zt_df) > 0:
        print(f"  命中涨停 ({len(zt_df)}笔):")
        print(f"    次日开盘卖: 胜率{(zt_df['ret_open']>0).sum()/len(zt_df)*100:.1f}% "
              f"均盈{zt_df['ret_open'].mean():+.2f}%")
        print(f"    次日收盘卖: 胜率{(zt_df['ret_close']>0).sum()/len(zt_df)*100:.1f}% "
              f"均盈{zt_df['ret_close'].mean():+.2f}%")
    if len(nozt_df) > 0:
        print(f"  未命中 ({len(nozt_df)}笔):")
        print(f"    次日开盘卖: 胜率{(nozt_df['ret_open']>0).sum()/len(nozt_df)*100:.1f}% "
              f"均盈{nozt_df['ret_open'].mean():+.2f}%")
        print(f"    次日收盘卖: 胜率{(nozt_df['ret_close']>0).sum()/len(nozt_df)*100:.1f}% "
              f"均盈{nozt_df['ret_close'].mean():+.2f}%")
    print()

    # ============================================================
    # 收益分布
    # ============================================================
    print("=" * 65)
    print("  次日收盘卖收益分布（全部候选）")
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

    # ============================================================
    # 结论
    # ============================================================
    print()
    print("=" * 65)
    print("  回测结论")
    print("=" * 65)
    base_c = next((c for c in combos if c["name"] == "基准(全部)"), None)
    if base_c:
        print(f"  基准(开盘涨0.5-5%全买):")
        print(f"    涨停命中率: {base_c['zt_rate']:.1f}%")
        print(f"    次日胜率: {base_c['wr_close']:.1f}% | 次日均盈: {base_c['avg_close']:+.2f}%")
    top3 = [c for c in combos if c["n"] >= 20 and c["name"] != "基准(全部)"][:3]
    if top3:
        print()
        print("  TOP3 高胜率策略:")
        for i, c in enumerate(top3):
            print(f"    {i+1}. {c['name']}")
            print(f"       样本{c['n']} | 涨停{c['zt_rate']:.1f}% | "
                  f"次日胜率{c['wr_close']:.1f}% | 均盈{c['avg_close']:+.2f}%")
    print()

    return df, combos


if __name__ == "__main__":
    sample = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 300
    days = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 250
    run_predict_zt_backtest(sample_size=sample, days=days)
