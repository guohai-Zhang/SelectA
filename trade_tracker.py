#!/usr/bin/env python3
"""
交易追踪模块 - 记录推荐、跟踪结果、统计胜率

功能：
- 记录每次推荐（T+1 / T+5）的完整信息
- 自动拉取 D+1 / D+5 收盘价，计算实际盈亏
- 按类型、推荐等级汇总胜率和平均收益
- CLI 格式化报告
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

TRADES_FILE = Path(__file__).parent / "trades.json"


# ── 持久化 ────────────────────────────────────────────────

def _load_trades():
    """从 trades.json 加载全部交易记录"""
    if not TRADES_FILE.exists():
        return []
    try:
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_trades(trades):
    """写回 trades.json"""
    with open(TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(trades, f, ensure_ascii=False, indent=2)


# ── 记录推荐 ──────────────────────────────────────────────

def record_recommendation(rec_type, code, name, price, score,
                          rec_level, signals=None, risk_params=None):
    """
    记录一条推荐

    Parameters
    ----------
    rec_type : str
        "T+1" 或 "T+5"
    code : str
        股票代码，如 "000001"
    name : str
        股票名称
    price : float
        推荐时价格（当日收盘/现价）
    score : float
        综合评分
    rec_level : str
        推荐等级：强推荐 / 推荐 / 弱推荐 / 仅参考
    signals : dict, optional
        信号明细（技术分、基本面分等）
    risk_params : dict, optional
        风控参数（止损价、止盈价、仓位等）
    """
    trades = _load_trades()

    record = {
        "id": len(trades) + 1,
        "rec_type": rec_type,
        "code": str(code),
        "name": name,
        "price": float(price),
        "score": float(score),
        "rec_level": rec_level,
        "signals": signals or {},
        "risk_params": _serialize_risk(risk_params),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rec_date": datetime.now().strftime("%Y-%m-%d"),
        # 结果字段，后续填充
        "d1_close": None,
        "d1_pnl_pct": None,
        "d5_close": None,
        "d5_pnl_pct": None,
        "outcome_updated": False,
    }

    trades.append(record)
    _save_trades(trades)
    return record


def _serialize_risk(risk_params):
    """确保 risk_params 里的值都可以 JSON 序列化"""
    if not risk_params:
        return {}
    clean = {}
    for k, v in risk_params.items():
        try:
            json.dumps(v)
            clean[k] = v
        except (TypeError, ValueError):
            clean[k] = str(v)
    return clean


# ── 更新结果 ──────────────────────────────────────────────

def _get_fetch_kline():
    """延迟导入 fetch_kline 避免循环依赖"""
    from t1_trader import fetch_kline
    return fetch_kline


def _trading_day_offset(base_date_str, offset):
    """
    从 base_date 开始往后找第 offset 个交易日的日期。
    简单处理：跳过周末，不处理节假日（节假日在拉K线时自然取到最近交易日）。
    """
    d = datetime.strptime(base_date_str, "%Y-%m-%d")
    count = 0
    while count < offset:
        d += timedelta(days=1)
        if d.weekday() < 5:  # 周一~周五
            count += 1
    return d.strftime("%Y-%m-%d")


def update_outcomes():
    """
    遍历所有未完成的交易，尝试拉取 D+1 / D+5 收盘价并计算盈亏。

    Returns
    -------
    int
        本次更新了多少条记录
    """
    fetch_kline = _get_fetch_kline()
    trades = _load_trades()
    today = datetime.now().strftime("%Y-%m-%d")
    updated_count = 0

    for t in trades:
        if t["outcome_updated"]:
            continue

        rec_date = t["rec_date"]
        price = t["price"]
        if not price or price <= 0:
            continue

        # 需要的目标日期
        d1_target = _trading_day_offset(rec_date, 1)
        d5_target = _trading_day_offset(rec_date, 5)

        # 如果目标日期还没到，跳过
        need_d1 = t["d1_close"] is None and d1_target <= today
        need_d5 = t["d5_close"] is None and d5_target <= today

        if not need_d1 and not need_d5:
            continue

        # 拉取K线
        kline = fetch_kline(t["code"], days=30)
        if kline is None or kline.empty:
            continue

        kline_dates = kline["日期"].tolist()
        changed = False

        # 填充 D+1
        if need_d1:
            close = _find_close_on_or_before(kline, kline_dates, d1_target)
            if close is not None:
                t["d1_close"] = round(close, 3)
                t["d1_pnl_pct"] = round((close - price) / price * 100, 2)
                changed = True

        # 填充 D+5
        if need_d5:
            close = _find_close_on_or_before(kline, kline_dates, d5_target)
            if close is not None:
                t["d5_close"] = round(close, 3)
                t["d5_pnl_pct"] = round((close - price) / price * 100, 2)
                changed = True

        # 两个都有了就标记完成
        if t["d1_close"] is not None and t["d5_close"] is not None:
            t["outcome_updated"] = True

        if changed:
            updated_count += 1

    _save_trades(trades)
    return updated_count


def _find_close_on_or_before(kline, kline_dates, target_date):
    """
    在K线中找 target_date 当天的收盘价。
    如果恰好该日无数据（节假日），取最近的前一个交易日。
    """
    if target_date in kline_dates:
        row = kline[kline["日期"] == target_date].iloc[-1]
        return float(row["收盘"])
    # 往前找最近的交易日（最多回退5天）
    for i in range(1, 6):
        d = datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=i)
        ds = d.strftime("%Y-%m-%d")
        if ds in kline_dates:
            row = kline[kline["日期"] == ds].iloc[-1]
            return float(row["收盘"])
    return None


# ── 查询 ──────────────────────────────────────────────────

def get_trade_history(rec_type=None, days=30):
    """
    返回最近 N 天的交易记录

    Parameters
    ----------
    rec_type : str, optional
        过滤类型："T+1" 或 "T+5"，None 表示全部
    days : int
        往前看多少天

    Returns
    -------
    list[dict]
    """
    trades = _load_trades()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    result = []
    for t in trades:
        if t["rec_date"] < cutoff:
            continue
        if rec_type and t["rec_type"] != rec_type:
            continue
        result.append(t)

    return result


def get_performance_summary():
    """
    汇总统计：总体、按类型、按推荐等级

    Returns
    -------
    dict  包含 overall / by_type / by_rec_level 三个维度
    """
    trades = _load_trades()

    def _calc_stats(subset, pnl_key):
        """计算一组交易的统计指标"""
        filled = [t for t in subset if t.get(pnl_key) is not None]
        if not filled:
            return {"count": len(subset), "filled": 0,
                    "win_rate": None, "avg_pnl": None,
                    "max_win": None, "max_loss": None}
        pnls = [t[pnl_key] for t in filled]
        wins = [p for p in pnls if p > 0]
        return {
            "count": len(subset),
            "filled": len(filled),
            "win_rate": round(len(wins) / len(filled) * 100, 1),
            "avg_pnl": round(sum(pnls) / len(pnls), 2),
            "max_win": round(max(pnls), 2),
            "max_loss": round(min(pnls), 2),
        }

    # 整体 D+1
    overall_d1 = _calc_stats(trades, "d1_pnl_pct")
    # 整体 D+5
    overall_d5 = _calc_stats(trades, "d5_pnl_pct")

    # 按 rec_type 拆分
    by_type = {}
    for rt in ("T+1", "T+5"):
        sub = [t for t in trades if t["rec_type"] == rt]
        if sub:
            pnl_key = "d1_pnl_pct" if rt == "T+1" else "d5_pnl_pct"
            by_type[rt] = _calc_stats(sub, pnl_key)

    # 按 rec_level 拆分
    by_rec_level = {}
    levels = sorted(set(t["rec_level"] for t in trades))
    for lv in levels:
        sub = [t for t in trades if t["rec_level"] == lv]
        if sub:
            by_rec_level[lv] = {
                "d1": _calc_stats(sub, "d1_pnl_pct"),
                "d5": _calc_stats(sub, "d5_pnl_pct"),
            }

    return {
        "total_trades": len(trades),
        "overall_d1": overall_d1,
        "overall_d5": overall_d5,
        "by_type": by_type,
        "by_rec_level": by_rec_level,
    }


# ── CLI 报告 ─────────────────────────────────────────────

def show_trade_report(days=30):
    """打印格式化的交易追踪报告"""
    from tabulate import tabulate

    trades = get_trade_history(days=days)
    summary = get_performance_summary()

    print()
    print("=" * 70)
    print("  交易追踪报告")
    print(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  统计范围: 最近 {days} 天 ({len(trades)} 条推荐)")
    print("=" * 70)

    # ── 最近推荐列表 ──
    if trades:
        print()
        print("  [ 最近推荐 ]")
        print()

        rows = []
        for t in trades[-20:]:  # 最多显示最近20条
            d1_str = f"{t['d1_pnl_pct']:+.2f}%" if t["d1_pnl_pct"] is not None else "-"
            d5_str = f"{t['d5_pnl_pct']:+.2f}%" if t["d5_pnl_pct"] is not None else "-"
            rows.append([
                t["rec_date"],
                t["rec_type"],
                f"{t['name']}({t['code']})",
                f"{t['price']:.2f}",
                f"{t['score']:.0f}",
                t["rec_level"],
                d1_str,
                d5_str,
            ])

        headers = ["日期", "类型", "股票", "价格", "评分", "等级", "D+1盈亏", "D+5盈亏"]
        print(tabulate(rows, headers=headers, tablefmt="simple",
                        stralign="right", numalign="right"))
    else:
        print()
        print("  暂无推荐记录")

    # ── 整体统计 ──
    print()
    print("  [ 整体表现 ]")
    print()

    stat_rows = []
    for label, stats in [("D+1 整体", summary["overall_d1"]),
                         ("D+5 整体", summary["overall_d5"])]:
        if stats["filled"]:
            stat_rows.append([
                label,
                f"{stats['filled']}/{stats['count']}",
                f"{stats['win_rate']:.1f}%",
                f"{stats['avg_pnl']:+.2f}%",
                f"{stats['max_win']:+.2f}%",
                f"{stats['max_loss']:+.2f}%",
            ])
        else:
            stat_rows.append([label, f"0/{stats['count']}", "-", "-", "-", "-"])

    print(tabulate(stat_rows,
                   headers=["维度", "已出结果", "胜率", "平均盈亏", "最大盈利", "最大亏损"],
                   tablefmt="simple", stralign="right", numalign="right"))

    # ── 按类型统计 ──
    if summary["by_type"]:
        print()
        print("  [ 按类型 ]")
        print()
        type_rows = []
        for rt, stats in summary["by_type"].items():
            if stats["filled"]:
                type_rows.append([
                    rt,
                    f"{stats['filled']}/{stats['count']}",
                    f"{stats['win_rate']:.1f}%",
                    f"{stats['avg_pnl']:+.2f}%",
                    f"{stats['max_win']:+.2f}%",
                    f"{stats['max_loss']:+.2f}%",
                ])
            else:
                type_rows.append([rt, f"0/{stats['count']}", "-", "-", "-", "-"])
        print(tabulate(type_rows,
                       headers=["类型", "已出结果", "胜率", "平均盈亏", "最大盈利", "最大亏损"],
                       tablefmt="simple", stralign="right", numalign="right"))

    # ── 按推荐等级统计 ──
    if summary["by_rec_level"]:
        print()
        print("  [ 按推荐等级 ]")
        print()
        level_rows = []
        for lv, data in summary["by_rec_level"].items():
            d1 = data["d1"]
            d5 = data["d5"]
            d1_wr = f"{d1['win_rate']:.1f}%" if d1["win_rate"] is not None else "-"
            d1_avg = f"{d1['avg_pnl']:+.2f}%" if d1["avg_pnl"] is not None else "-"
            d5_wr = f"{d5['win_rate']:.1f}%" if d5["win_rate"] is not None else "-"
            d5_avg = f"{d5['avg_pnl']:+.2f}%" if d5["avg_pnl"] is not None else "-"
            level_rows.append([
                lv,
                f"{d1['filled']}/{d1['count']}",
                d1_wr, d1_avg,
                d5_wr, d5_avg,
            ])
        print(tabulate(level_rows,
                       headers=["等级", "数量", "D+1胜率", "D+1均盈亏",
                                "D+5胜率", "D+5均盈亏"],
                       tablefmt="simple", stralign="right", numalign="right"))

    print()
    print("-" * 70)
    print("  * D+1盈亏 = (推荐后第1个交易日收盘 - 推荐价) / 推荐价")
    print("  * D+5盈亏 = (推荐后第5个交易日收盘 - 推荐价) / 推荐价")
    print("  * 胜率 = 盈利次数 / 已出结果数")
    print("-" * 70)
    print()


# ── 入口 ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "update":
        n = update_outcomes()
        print(f"已更新 {n} 条记录")
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        show_trade_report(days=days)
    else:
        print("用法:")
        print("  python trade_tracker.py update    # 更新所有待定结果")
        print("  python trade_tracker.py report    # 打印报告（默认30天）")
        print("  python trade_tracker.py report 7  # 打印最近7天报告")
