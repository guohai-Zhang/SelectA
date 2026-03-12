#!/usr/bin/env python3
"""
涨停预测策略回测 v2
核心指标：次日卖出是否盈利（胜率+均盈）
涨停是加分项，不是必要条件

模拟：
- 早盘10:00-10:30，股票涨幅0.5-5%时买入
- 买入价 ≈ 今日开盘价（日线代理）
- 次日卖出（开盘 or 收盘）
- 胜 = 次日卖出价 > 买入价
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
    print("  涨停预测策略回测 v2（以次日盈利为核心指标）")
    print(f"  样本: {sample_size} 只活跃股 × {days} 天K线")
    print(f"  策略: 开盘涨0.5-5%买入 → 次日卖出 → 胜=盈利")
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

    # ── 扫描候选日 ──
    print("[3/4] 扫描候选日 & 提取因子...")

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

            # 开盘涨幅
            open_gap = (open_price - prev_close) / prev_close * 100

            # 筛选：开盘涨 0.5% ~ 5%
            if open_gap < 0.5 or open_gap > 5:
                continue

            # 当日涨跌幅
            day_chg = row.get("涨跌幅", 0) or 0
            hit_zt = day_chg >= 9.8
            amplitude = (high_price - low_price) / prev_close * 100 if prev_close > 0 else 0

            # 次日数据
            next_row = kl.iloc[j + 1]
            buy_price = open_price
            sell_open = next_row["开盘"]
            sell_close = next_row["收盘"]

            if buy_price <= 0:
                continue

            ret_open = (sell_open - buy_price) / buy_price * 100
            ret_close = (sell_close - buy_price) / buy_price * 100

            # ── 因子 ──

            # 今日收盘强度（关键！收盘>买入价=今天涨了，明天更可能继续）
            today_gain = (close_price - buy_price) / buy_price * 100  # 买入后今日收益
            close_strong = close_price >= high_price * 0.97  # 收在最高价附近
            close_near_zt = day_chg >= 8  # 收盘接近涨停

            # 量比
            vol_today = row.get("成交量", 0) or 0
            vol_ma5 = kl["成交量"].iloc[max(0, j - 5):j].mean() if j >= 5 else vol_today
            vol_ratio = vol_today / vol_ma5 if vol_ma5 > 0 else 1.0

            # 量能爆发
            if j >= 3:
                prev3_vol = kl["成交量"].iloc[j - 3:j].mean()
                vol_expand = vol_today / prev3_vol if prev3_vol > 0 else 1.0
            else:
                vol_expand = 1.0

            # 换手率
            turnover = row.get("换手率", 0) or 0

            # 前3日横盘
            if j >= 3:
                prev3_chg = [abs(kl.iloc[j - k].get("涨跌幅", 0) or 0) for k in range(1, 4)]
                prev3_max_chg = max(prev3_chg)
            else:
                prev3_max_chg = 10

            # 均线
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

            # 回撤
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
                "today_gain": today_gain,
                "hit_zt": hit_zt,
                "close_strong": close_strong,
                "close_near_zt": close_near_zt,
                "amplitude": amplitude,
                "pullback": pullback,
                "vol_ratio": vol_ratio,
                "vol_expand": vol_expand,
                "turnover": turnover,
                "prev3_max_chg": prev3_max_chg,
                "ma_bull": ma_bull,
                "above_ma20": above_ma20,
                "macd_bull": macd_bull,
                "pre5_ret": pre5_ret,
                "price_zone": price_zone,
            })

    print(f"  共 {len(trades)} 笔候选（开盘涨0.5-5%的日子）")
    if not trades:
        print("无候选记录")
        return

    df = pd.DataFrame(trades)
    print(f"  涨停: {df['hit_zt'].sum()} ({df['hit_zt'].mean()*100:.1f}%)")
    print(f"  今日收阳(买入后涨): {(df['today_gain']>0).sum()} ({(df['today_gain']>0).mean()*100:.1f}%)")
    print(f"  次日盈利: {(df['ret_close']>0).sum()} ({(df['ret_close']>0).mean()*100:.1f}%)")
    print()

    # ── 统计函数 ──
    def stats(series, label=""):
        n = len(series)
        if n == 0:
            return None
        wins = (series > 0).sum()
        wr = wins / n * 100
        avg = series.mean()
        avg_win = series[series > 0].mean() if wins > 0 else 0
        avg_loss = series[series <= 0].mean() if (n - wins) > 0 else 0
        print(f"  {label}")
        print(f"    样本: {n} | 胜率: {wr:.1f}% | 均盈: {avg:+.2f}% | 均赢: {avg_win:+.2f}% | 均亏: {avg_loss:+.2f}%")
        return {"n": n, "wr": round(wr, 1), "avg": round(avg, 2)}

    # ============================================================
    print("[4/4] 分析结果...")
    print()
    print("=" * 65)
    print("  一、基准 & 今日收盘强度 vs 次日收益")
    print("=" * 65)
    print()
    stats(df["ret_close"], "基准（全部候选 → 次日收盘卖）")
    print()
    stats(df["ret_open"], "基准（全部候选 → 次日开盘卖）")
    print()

    # ── 核心分析：今天涨幅 vs 明天收益 ──
    print("=" * 65)
    print("  二、今日收盘涨幅 vs 次日收益（核心！）")
    print("  逻辑：今天买入后继续涨，说明趋势强，明天更可能盈利")
    print("=" * 65)
    print()

    for lo, hi, label in [(-100, 0, "今日回落(收盘<开盘)"),
                           (0, 2, "今日微涨(0-2%)"),
                           (2, 5, "今日中涨(2-5%)"),
                           (5, 8, "今日大涨(5-8%)"),
                           (8, 9.8, "今日接近涨停(8-9.8%)"),
                           (9.8, 100, "今日涨停(>=9.8%)")]:
        sub = df[(df["today_gain"] >= lo) & (df["today_gain"] < hi)]
        if len(sub) >= 20:
            stats(sub["ret_close"], f"{label} → 次日收盘卖")
            print()

    # 收盘强度
    print("=" * 65)
    print("  三、收盘位置 vs 次日收益")
    print("  收在最高价附近 = 尾盘强势 = 明天惯性上涨")
    print("=" * 65)
    print()
    stats(df[df["close_strong"]]["ret_close"], "收在日内高点(close>=97%high) → 次日")
    print()
    stats(df[~df["close_strong"]]["ret_close"], "未收在高点 → 次日")
    print()

    # ============================================================
    print("=" * 65)
    print("  四、单因子分析（按次日胜率排序）")
    print("=" * 65)
    print()

    base_wr = (df["ret_close"] > 0).sum() / len(df) * 100

    def fa(mask, label):
        sub = df[mask]
        n = len(sub)
        if n < 20:
            return None
        wr = (sub["ret_close"] > 0).sum() / n * 100
        avg = sub["ret_close"].mean()
        zt_rate = sub["hit_zt"].sum() / n * 100
        delta = wr - base_wr
        mark = " ★" if wr >= base_wr + 5 and n >= 30 else ""
        print(f"  {label:<35} n={n:>5} 次日胜率{wr:>5.1f}%({delta:>+5.1f}) "
              f"均盈{avg:>+5.2f}% 涨停{zt_rate:>4.1f}%{mark}")
        return {"label": label, "n": n, "wr": wr, "avg": avg, "zt_rate": zt_rate}

    # 量比
    print("  ── 量比 ──")
    fa(df["vol_ratio"] < 1.5, "量比<1.5")
    fa((df["vol_ratio"] >= 1.5) & (df["vol_ratio"] < 2), "量比1.5-2")
    fa((df["vol_ratio"] >= 2) & (df["vol_ratio"] < 3), "量比2-3")
    fa((df["vol_ratio"] >= 3) & (df["vol_ratio"] < 5), "量比3-5")
    fa(df["vol_ratio"] >= 5, "量比>=5")
    print()

    # 量能爆发
    print("  ── 量能爆发(vs前3日) ──")
    fa(df["vol_expand"] < 1.5, "爆发<1.5x")
    fa((df["vol_expand"] >= 1.5) & (df["vol_expand"] < 2), "爆发1.5-2x")
    fa((df["vol_expand"] >= 2) & (df["vol_expand"] < 3), "爆发2-3x")
    fa(df["vol_expand"] >= 3, "爆发>=3x")
    print()

    # 前3日横盘
    print("  ── 前3日波动 ──")
    fa(df["prev3_max_chg"] < 2, "前3日窄幅横盘(<2%)")
    fa((df["prev3_max_chg"] >= 2) & (df["prev3_max_chg"] < 3), "前3日小幅(2-3%)")
    fa((df["prev3_max_chg"] >= 3) & (df["prev3_max_chg"] < 5), "前3日正常(3-5%)")
    fa(df["prev3_max_chg"] >= 5, "前3日大波动(>=5%)")
    print()

    # 趋势
    print("  ── 趋势 ──")
    fa(df["ma_bull"], "多头排列")
    fa(~df["ma_bull"], "非多头排列")
    fa(df["above_ma20"], "站上MA20")
    fa(~df["above_ma20"], "MA20下方")
    fa(df["macd_bull"], "MACD多头")
    fa(~df["macd_bull"], "MACD空头")
    print()

    # 开盘涨幅
    print("  ── 开盘涨幅 ──")
    fa((df["open_gap"] >= 0.5) & (df["open_gap"] < 1), "开盘0.5-1%")
    fa((df["open_gap"] >= 1) & (df["open_gap"] < 2), "开盘1-2%")
    fa((df["open_gap"] >= 2) & (df["open_gap"] < 3), "开盘2-3%")
    fa((df["open_gap"] >= 3) & (df["open_gap"] < 5), "开盘3-5%")
    print()

    # 今日收盘强度
    print("  ── 今日走势（买入后） ──")
    fa(df["today_gain"] < 0, "今日回落（买入后跌）")
    fa((df["today_gain"] >= 0) & (df["today_gain"] < 3), "今日微涨(0-3%)")
    fa((df["today_gain"] >= 3) & (df["today_gain"] < 6), "今日中涨(3-6%)")
    fa(df["today_gain"] >= 6, "今日大涨(6%+)")
    fa(df["close_strong"], "收盘在高点")
    fa(df["pullback"] < 1, "低回撤(<1%)")
    fa(df["pullback"] >= 3, "高回撤(>=3%)")
    print()

    # 前5日趋势
    print("  ── 前5日趋势 ──")
    fa(df["pre5_ret"] < -5, "前5日跌>5%")
    fa((df["pre5_ret"] >= -5) & (df["pre5_ret"] < 0), "前5日微跌")
    fa((df["pre5_ret"] >= 0) & (df["pre5_ret"] < 5), "前5日微涨(0-5%)")
    fa(df["pre5_ret"] >= 5, "前5日涨>5%")
    print()

    # 价格
    print("  ── 价格区间 ──")
    for zone in ["低价(<10)", "中价(10-30)", "中高(30-50)", "高价(50+)"]:
        fa(df["price_zone"] == zone, zone)
    print()

    # ============================================================
    print("=" * 65)
    print("  五、组合策略扫描（以次日胜率排序）")
    print("=" * 65)
    print()

    combos = []

    def tc(name, mask):
        sub = df[mask]
        n = len(sub)
        if n < 15:
            return
        wr_close = (sub["ret_close"] > 0).sum() / n * 100
        avg_close = sub["ret_close"].mean()
        wr_open = (sub["ret_open"] > 0).sum() / n * 100
        avg_open = sub["ret_open"].mean()
        zt_rate = sub["hit_zt"].sum() / n * 100
        today_up = (sub["today_gain"] > 0).sum() / n * 100
        combos.append({
            "name": name, "n": n,
            "wr_close": wr_close, "avg_close": avg_close,
            "wr_open": wr_open, "avg_open": avg_open,
            "zt_rate": zt_rate, "today_up": today_up,
        })

    # === 基准 ===
    tc("基准(全部)", pd.Series([True] * len(df)))

    # === 单因子 ===
    tc("量比>=3", df["vol_ratio"] >= 3)
    tc("量比>=2", df["vol_ratio"] >= 2)
    tc("量能爆发>=2x", df["vol_expand"] >= 2)
    tc("量能爆发>=3x", df["vol_expand"] >= 3)
    tc("前3日横盘(<3%)", df["prev3_max_chg"] < 3)
    tc("多头排列", df["ma_bull"])
    tc("站上MA20", df["above_ma20"])
    tc("收盘在高点", df["close_strong"])
    tc("今日涨>=3%(买后)", df["today_gain"] >= 3)
    tc("今日涨>=5%(买后)", df["today_gain"] >= 5)
    tc("低回撤(<1%)", df["pullback"] < 1)

    # === 核心双因子 ===
    tc("量比>=3 + 多头",
       (df["vol_ratio"] >= 3) & df["ma_bull"])
    tc("量比>=3 + 站上MA20",
       (df["vol_ratio"] >= 3) & df["above_ma20"])
    tc("量比>=2 + 前3日横盘",
       (df["vol_ratio"] >= 2) & (df["prev3_max_chg"] < 3))
    tc("量比>=3 + 前3日横盘",
       (df["vol_ratio"] >= 3) & (df["prev3_max_chg"] < 3))
    tc("量能爆发2x + 前3日横盘",
       (df["vol_expand"] >= 2) & (df["prev3_max_chg"] < 3))
    tc("量能爆发2x + 多头",
       (df["vol_expand"] >= 2) & df["ma_bull"])
    tc("收盘高点 + 量比>=2",
       df["close_strong"] & (df["vol_ratio"] >= 2))
    tc("收盘高点 + 多头",
       df["close_strong"] & df["ma_bull"])
    tc("今日涨3%+ + 量比>=2",
       (df["today_gain"] >= 3) & (df["vol_ratio"] >= 2))
    tc("今日涨3%+ + 多头",
       (df["today_gain"] >= 3) & df["ma_bull"])
    tc("低回撤 + 量比>=2",
       (df["pullback"] < 1) & (df["vol_ratio"] >= 2))
    tc("低回撤 + 多头",
       (df["pullback"] < 1) & df["ma_bull"])

    # === 三因子 ===
    tc("量比>=3 + 多头 + 前3日横盘",
       (df["vol_ratio"] >= 3) & df["ma_bull"] & (df["prev3_max_chg"] < 3))
    tc("量比>=3 + 站上MA20 + 前3日横盘",
       (df["vol_ratio"] >= 3) & df["above_ma20"] & (df["prev3_max_chg"] < 3))
    tc("量能爆发2x + 多头 + 前3日横盘",
       (df["vol_expand"] >= 2) & df["ma_bull"] & (df["prev3_max_chg"] < 3))
    tc("量能爆发2x + 站上MA20 + 前3日横盘",
       (df["vol_expand"] >= 2) & df["above_ma20"] & (df["prev3_max_chg"] < 3))
    tc("收盘高点 + 量比>=2 + 多头",
       df["close_strong"] & (df["vol_ratio"] >= 2) & df["ma_bull"])
    tc("收盘高点 + 量比>=3 + 站上MA20",
       df["close_strong"] & (df["vol_ratio"] >= 3) & df["above_ma20"])
    tc("今日涨3%+ + 量比>=2 + 多头",
       (df["today_gain"] >= 3) & (df["vol_ratio"] >= 2) & df["ma_bull"])
    tc("今日涨3%+ + 量比>=2 + 前3日横盘",
       (df["today_gain"] >= 3) & (df["vol_ratio"] >= 2) & (df["prev3_max_chg"] < 3))
    tc("低回撤 + 量比>=2 + 多头",
       (df["pullback"] < 1) & (df["vol_ratio"] >= 2) & df["ma_bull"])
    tc("低回撤 + 量比>=3 + 前3日横盘",
       (df["pullback"] < 1) & (df["vol_ratio"] >= 3) & (df["prev3_max_chg"] < 3))
    tc("低回撤 + 量能爆发2x + 多头",
       (df["pullback"] < 1) & (df["vol_expand"] >= 2) & df["ma_bull"])

    # === 四因子 ===
    tc("量比>=3 + 多头 + 前3日横盘 + 收盘高点",
       (df["vol_ratio"] >= 3) & df["ma_bull"] & (df["prev3_max_chg"] < 3) & df["close_strong"])
    tc("量比>=3 + 站上MA20 + 前3日横盘 + 低回撤",
       (df["vol_ratio"] >= 3) & df["above_ma20"] & (df["prev3_max_chg"] < 3) & (df["pullback"] < 1))
    tc("量能爆发2x + 多头 + 前3日横盘 + 收盘高点",
       (df["vol_expand"] >= 2) & df["ma_bull"] & (df["prev3_max_chg"] < 3) & df["close_strong"])
    tc("量能爆发2x + 多头 + 前3日横盘 + 低回撤",
       (df["vol_expand"] >= 2) & df["ma_bull"] & (df["prev3_max_chg"] < 3) & (df["pullback"] < 1))
    tc("量比>=2 + 多头 + 前3日横盘 + 今日涨3%+",
       (df["vol_ratio"] >= 2) & df["ma_bull"] & (df["prev3_max_chg"] < 3) & (df["today_gain"] >= 3))
    tc("量比>=2 + 站上MA20 + 前3日横盘 + 收盘高点",
       (df["vol_ratio"] >= 2) & df["above_ma20"] & (df["prev3_max_chg"] < 3) & df["close_strong"])
    tc("量比>=2 + 站上MA20 + 低回撤 + 今日涨3%+",
       (df["vol_ratio"] >= 2) & df["above_ma20"] & (df["pullback"] < 1) & (df["today_gain"] >= 3))

    # === 五因子 ===
    tc("量比>=3 + 多头 + 前3日横盘 + 收盘高点 + 低回撤",
       (df["vol_ratio"] >= 3) & df["ma_bull"] & (df["prev3_max_chg"] < 3)
       & df["close_strong"] & (df["pullback"] < 1))
    tc("量比>=2 + 多头 + 前3日横盘 + 收盘高点 + 今日涨3%+",
       (df["vol_ratio"] >= 2) & df["ma_bull"] & (df["prev3_max_chg"] < 3)
       & df["close_strong"] & (df["today_gain"] >= 3))

    combos.sort(key=lambda x: x["wr_close"], reverse=True)

    print(f"  {'策略':<42} {'n':>4} {'次日胜率':>7} {'均盈':>6} {'今日涨':>6} {'涨停':>5}")
    print("  " + "-" * 74)
    for c in combos:
        mark = " ★" if c["wr_close"] >= 65 and c["n"] >= 20 else ""
        print(f"  {c['name']:<42} {c['n']:>4} {c['wr_close']:>6.1f}% {c['avg_close']:>+5.2f}% "
              f"{c['today_up']:>5.0f}% {c['zt_rate']:>4.1f}%{mark}")

    # ============================================================
    print()
    print("=" * 65)
    print("  六、收益分布（次日收盘卖）")
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
    # 七、v3评分模拟回测
    print()
    print("=" * 65)
    print("  七、v3评分模拟回测（验证新评分体系）")
    print("=" * 65)
    print()

    def sim_v3_score(row):
        """模拟v3评分（用回测可见因子）"""
        chg = row["day_chg"]
        pullback = row["pullback"]
        vr = row["vol_ratio"]
        ve = row["vol_expand"]
        prev3_max = row["prev3_max_chg"]
        ma_b = row["ma_bull"]
        above20 = row["above_ma20"]
        macd_b = row["macd_bull"]
        p5r = row["pre5_ret"]

        # 1. 走势 (0-40)
        m = 0
        if chg <= 0:
            m = -20
        elif chg >= 5:
            m = 22
        elif chg >= 3:
            m = 16
        elif chg >= 2:
            m = 10
        elif chg >= 1:
            m = 5
        else:
            m = 2
        if chg > 0:
            if pullback < 1:
                m += 15
            elif pullback < 2:
                m += 8
            elif pullback < 3:
                m += 2
            else:
                m -= 5
        m = max(-20, min(40, m))

        # 2. 量能 (0-25)
        v = 0
        if vr >= 5: v += 13
        elif vr >= 3: v += 11
        elif vr >= 2: v += 8
        elif vr >= 1.5: v += 4
        else: v += 1
        if ve >= 3: v += 10
        elif ve >= 2: v += 7
        elif ve >= 1.5: v += 3
        to = row.get("turnover", 0)
        if 3 <= to <= 8: v += 2
        v = min(25, v)

        # 3. 蓄势 (0-15)
        k = 0
        if prev3_max < 3:
            k += 8
            if prev3_max < 1.5: k += 2
        elif prev3_max < 5:
            k += 3
        if ve >= 2 and prev3_max < 3: k += 2
        if p5r < -5: k += 4
        elif p5r > 5: k -= 3
        k = max(-3, min(15, k))

        # 4. 趋势 (0-10)
        t = 0
        if above20: t += 3
        else: t -= 5
        if ma_b: t += 2
        elif vr >= 1.5: t += 1  # ma5>ma10 approximation
        if macd_b: t += 2
        t = max(-5, min(10, t))

        return max(0, min(100, m + v + k + t))

    df["v3_score"] = df.apply(sim_v3_score, axis=1)

    print("  v3评分分布 vs 次日胜率:")
    print()
    for lo, hi, label in [(0, 30, " 0-29(弱)"),
                           (30, 40, "30-39"),
                           (40, 50, "40-49"),
                           (50, 60, "50-59(门槛)"),
                           (60, 70, "60-69"),
                           (70, 80, "70-79"),
                           (80, 100, "80+ (强)")]:
        sub = df[(df["v3_score"] >= lo) & (df["v3_score"] < hi)]
        n = len(sub)
        if n < 10:
            print(f"  {label}: n={n:>5} (样本不足)")
            continue
        wr = (sub["ret_close"] > 0).sum() / n * 100
        avg = sub["ret_close"].mean()
        zt = sub["hit_zt"].sum() / n * 100
        print(f"  {label}: n={n:>5} 次日胜率{wr:>5.1f}% 均盈{avg:>+5.2f}% 涨停{zt:>4.1f}%")

    # v3>=50 vs <50
    print()
    above50 = df[df["v3_score"] >= 50]
    below50 = df[df["v3_score"] < 50]
    if len(above50) > 0:
        wr_a = (above50["ret_close"] > 0).sum() / len(above50) * 100
        avg_a = above50["ret_close"].mean()
        print(f"  v3>=50(推荐): n={len(above50):>5} 次日胜率{wr_a:.1f}% 均盈{avg_a:+.2f}%")
    if len(below50) > 0:
        wr_b = (below50["ret_close"] > 0).sum() / len(below50) * 100
        avg_b = below50["ret_close"].mean()
        print(f"  v3< 50(过滤): n={len(below50):>5} 次日胜率{wr_b:.1f}% 均盈{avg_b:+.2f}%")

    # v3>=60
    above60 = df[df["v3_score"] >= 60]
    if len(above60) > 0:
        wr_60 = (above60["ret_close"] > 0).sum() / len(above60) * 100
        avg_60 = above60["ret_close"].mean()
        print(f"  v3>=60(高质量): n={len(above60):>5} 次日胜率{wr_60:.1f}% 均盈{avg_60:+.2f}%")

    print()

    # ============================================================
    print()
    print("=" * 65)
    print("  回测结论")
    print("=" * 65)
    base_c = next((c for c in combos if c["name"] == "基准(全部)"), None)
    if base_c:
        print(f"  基准(开盘涨0.5-5%全买):")
        print(f"    次日胜率: {base_c['wr_close']:.1f}% | 均盈: {base_c['avg_close']:+.2f}%")
    top5 = [c for c in combos if c["n"] >= 20 and c["name"] != "基准(全部)"][:5]
    if top5:
        print()
        print("  TOP5 高胜率策略:")
        for i, c in enumerate(top5):
            print(f"    {i+1}. {c['name']}")
            print(f"       样本{c['n']} | 次日胜率{c['wr_close']:.1f}% | "
                  f"均盈{c['avg_close']:+.2f}% | 今日涨{c['today_up']:.0f}% | 涨停{c['zt_rate']:.1f}%")
    print()

    return df, combos


if __name__ == "__main__":
    sample = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 300
    days = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 250
    run_predict_zt_backtest(sample_size=sample, days=days)
