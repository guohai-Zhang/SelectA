#!/usr/bin/env python3
"""
A股 T+1 短线分析工具 v3.0
- 今天买，明天卖
- 十二大维度：技术指标 + 主力资金 + 板块热度 + 市场情绪 + 量价配合 + K线形态
                + 基本面 + 龙虎榜 + 北向资金 + 融资融券 + 连板分析 + 股东增持
- 智能风控：动态仓位 + ATR自适应止损 + 三级止盈 + 行业景气度加成
- 内置回测引擎：用历史数据验证策略真实胜率
- 数据来源：腾讯财经(K线) + 新浪财经(行情) + 东方财富(资金流向/龙虎榜/融资/增持)
- 交易平台：同花顺（手动下单）

用法:
    python3 t1_trader.py                # 全市场扫描（推荐下午2点后运行）
    python3 t1_trader.py 002185         # 分析单只股票（华天科技）
    python3 t1_trader.py --top 20       # 显示前20只候选
    python3 t1_trader.py --backtest     # 回测策略历史胜率
    python3 t1_trader.py --backtest 002185  # 回测单只股票
    python3 t1_trader.py --calibrate    # 信号校准：用历史数据计算每个信号真实胜率→自动优化权重
    python3 t1_trader.py --market       # 查看大盘情绪
    python3 t1_trader.py --sector       # 查看板块资金流向
    python3 t1_trader.py --go           # 一键决策：直接推荐3只股票+明日卖出策略
    python3 t1_trader.py --etf          # ETF波段扫描（持有1-5天）
    python3 t1_trader.py --etf-go       # ETF波段一键决策：推荐3只ETF+操作计划
"""

import requests
import pandas as pd
import numpy as np
import json
import sys
import time
import re
import warnings
from datetime import datetime, timedelta
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ============================================================
# 清除代理（国内网站直连）
# ============================================================
import os as _os
for _k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    _os.environ.pop(_k, None)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
UT = "b2884a393a59ad64002292a3e90d46a5"
FS_ALL_A = "m:0+t:6+f:!2,m:0+t:13+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2"


def _mkt_prefix(code):
    """股票代码 -> 市场前缀 sz/sh"""
    code = str(code)
    return "sh" if code.startswith("6") else "sz"


def _get(url, params=None, timeout=15, retries=2, **kwargs):
    """统一 GET 请求，带重试"""
    hdrs = dict(HEADERS)
    hdrs.update(kwargs.get("headers", {}))
    for i in range(retries):
        try:
            return requests.get(url, params=params, headers=hdrs,
                                timeout=timeout, allow_redirects=True)
        except Exception:
            if i < retries - 1:
                time.sleep(1)
            else:
                raise


# ============================================================
# 数据获取 - 腾讯财经（K线，稳定免费）
# ============================================================

def fetch_kline(code, days=120):
    """从腾讯财经获取日K线（前复权）"""
    code = str(code)
    prefix = _mkt_prefix(code)
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{prefix}{code},day,,,{days},qfq"}
    try:
        resp = _get(url, params=params, timeout=10)
        data = resp.json()
        stock_data = data.get("data", {}).get(f"{prefix}{code}", {})
        klines = stock_data.get("qfqday", stock_data.get("day", []))
        if not klines:
            return pd.DataFrame()
        records = []
        for k in klines:
            # [日期, 开盘, 收盘, 最高, 最低, 成交量, ...]
            if len(k) < 6:
                continue
            records.append({
                "日期": k[0],
                "开盘": float(k[1]),
                "收盘": float(k[2]),
                "最高": float(k[3]),
                "最低": float(k[4]),
                "成交量": float(k[5]),
            })
        df = pd.DataFrame(records)
        if df.empty:
            return df
        # 计算衍生字段
        df["涨跌幅"] = df["收盘"].pct_change() * 100
        df["振幅"] = (df["最高"] - df["最低"]) / df["收盘"].shift(1) * 100
        df["换手率"] = 0.0  # 腾讯K线不含换手率，后面从实时行情补充
        return df
    except Exception:
        return pd.DataFrame()


def fetch_kline_long(code, days=500):
    """获取更长的K线（用于回测），分段拉取"""
    code = str(code)
    prefix = _mkt_prefix(code)
    # 腾讯一次最多约640条
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{prefix}{code},day,,,{min(days, 640)},qfq"}
    try:
        resp = _get(url, params=params, timeout=15)
        data = resp.json()
        stock_data = data.get("data", {}).get(f"{prefix}{code}", {})
        klines = stock_data.get("qfqday", stock_data.get("day", []))
        if not klines:
            return pd.DataFrame()
        records = []
        for k in klines:
            if len(k) < 6:
                continue
            records.append({
                "日期": k[0], "开盘": float(k[1]), "收盘": float(k[2]),
                "最高": float(k[3]), "最低": float(k[4]), "成交量": float(k[5]),
            })
        df = pd.DataFrame(records)
        if df.empty:
            return df
        df["涨跌幅"] = df["收盘"].pct_change() * 100
        df["振幅"] = (df["最高"] - df["最低"]) / df["收盘"].shift(1) * 100
        df["换手率"] = 0.0
        return df
    except Exception:
        return pd.DataFrame()


# ============================================================
# 数据获取 - 新浪财经（实时行情 + 股票列表）
# ============================================================

def fetch_realtime_sina(codes):
    """
    新浪实时行情，批量获取
    返回 dict: code -> {名称, 今开, 昨收, 最新价, 最高, 最低, 成交量, 成交额, 涨跌幅, 换手率, ...}
    """
    if not codes:
        return {}
    code_list = [f"{_mkt_prefix(c)}{c}" for c in codes]
    result = {}
    # 新浪一次最多请求约100只
    batch_size = 80
    for i in range(0, len(code_list), batch_size):
        batch = code_list[i:i+batch_size]
        url = f"http://hq.sinajs.cn/list={','.join(batch)}"
        try:
            resp = _get(url, headers={"Referer": "http://finance.sina.com.cn"}, timeout=10)
            resp.encoding = "gbk"
            for line in resp.text.strip().split("\n"):
                line = line.strip()
                if not line or '="' not in line:
                    continue
                # var hq_str_sz002185="华天科技,今开,昨收,最新价,最高,最低,买一,卖一,成交量,成交额,...";
                m = re.match(r'var hq_str_(\w+)="(.+)"', line)
                if not m:
                    continue
                symbol = m.group(1)  # sz002185
                code = symbol[2:]    # 002185
                parts = m.group(2).split(",")
                if len(parts) < 32:
                    continue
                try:
                    close = float(parts[3]) if float(parts[3]) > 0 else float(parts[2])
                    prev_close = float(parts[2])
                    chg = (close - prev_close) / prev_close * 100 if prev_close > 0 else 0
                    result[code] = {
                        "名称": parts[0],
                        "今开": float(parts[1]),
                        "昨收": prev_close,
                        "最新价": close,
                        "最高": float(parts[4]),
                        "最低": float(parts[5]),
                        "成交量": float(parts[8]),   # 股
                        "成交额": float(parts[9]),   # 元
                        "涨跌幅": chg,
                    }
                except (ValueError, IndexError):
                    continue
        except Exception:
            continue
        if i + batch_size < len(code_list):
            time.sleep(0.1)
    return result


def fetch_stock_list_sina():
    """通过东方财富接口获取A股列表（含实时行情），分页获取全量数据"""
    url = "http://push2delay.eastmoney.com/api/qt/clist/get"
    fields = "f12,f14,f2,f3,f5,f6,f7,f8,f9,f10,f15,f16,f17,f18,f20,f21"
    col_map = {
        "f12": "代码", "f14": "名称", "f2": "最新价", "f3": "涨跌幅",
        "f5": "成交量", "f6": "成交额", "f7": "振幅", "f8": "换手率",
        "f9": "市盈率", "f10": "量比", "f15": "最高", "f16": "最低",
        "f17": "今开", "f18": "昨收", "f20": "总市值", "f21": "流通市值"
    }
    all_items = []
    page_size = 100
    try:
        for pn in range(1, 80):  # 最多80页 = 8000只
            params = {
                "pn": pn, "pz": page_size, "po": 0, "np": 1, "fltt": 2, "invt": 2,
                "fid": "f12", "fs": FS_ALL_A, "ut": UT,
                "fields": fields,
            }
            resp = _get(url, params=params, headers={"Referer": "http://data.eastmoney.com/"})
            data = resp.json()
            diff = data.get("data", {}).get("diff", [])
            if not diff:
                break
            all_items.extend(diff)
            if len(diff) < page_size:
                break
        if not all_items:
            return pd.DataFrame()
        df = pd.DataFrame(all_items)
        df = df.rename(columns=col_map)
        df["代码"] = df["代码"].astype(str)
        df["名称"] = df["名称"].astype(str)
        df = df[df["最新价"] != "-"]
        df = df[~df["名称"].str.contains("ST|退市|N |C ", na=False)]
        df = df[~df["代码"].str.startswith("3")]  # 剔除创业板(300xxx)
        num_cols = ["最新价","涨跌幅","成交量","成交额","振幅","换手率","量比","最高","最低","今开","昨收","总市值","流通市值"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["最新价"])
        df = df[df["最新价"] > 0]
        return df
    except Exception as e:
        print(f"[错误] 获取股票列表失败: {e}")
        return pd.DataFrame()


# ============================================================
# 数据获取 - 东方财富（资金流向 + 板块 + 情绪）
# ============================================================

def fetch_capital_flow_rank(top_n=100):
    """个股主力资金流向排名"""
    url = "http://push2delay.eastmoney.com/api/qt/clist/get"
    params = {
        "fid": "f62", "po": 1, "pz": top_n, "pn": 1, "np": 1, "fltt": 2,
        "fields": "f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f124",
        "fs": FS_ALL_A, "ut": UT,
    }
    try:
        resp = _get(url, params=params, headers={"Referer": "http://data.eastmoney.com/"})
        data = resp.json()
        if not (data.get("data") and data["data"].get("diff")):
            return {}
        result = {}
        for item in data["data"]["diff"]:
            code = str(item.get("f12", ""))
            result[code] = {
                "主力净流入": item.get("f62", 0),
                "主力净流入占比": item.get("f184", 0),
                "超大单净流入": item.get("f66", 0),
                "超大单占比": item.get("f69", 0),
                "大单净流入": item.get("f72", 0),
                "大单占比": item.get("f75", 0),
            }
        return result
    except Exception:
        return {}


def fetch_stock_capital_flow(code):
    """单只股票历史资金流向"""
    code = str(code)
    market = 1 if code.startswith("6") else 0
    url = "http://push2delay.eastmoney.com/api/qt/stock/fflow/daykline/get"
    params = {
        "lmt": 0, "klt": 101,
        "secid": f"{market}.{code}",
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "ut": UT,
    }
    try:
        resp = _get(url, params=params, headers={"Referer": "http://data.eastmoney.com/"}, timeout=10)
        data = resp.json()
        if not (data.get("data") and data["data"].get("klines")):
            return pd.DataFrame()
        records = []
        for line in data["data"]["klines"]:
            p = line.split(",")
            records.append({
                "日期": p[0],
                "主力净流入": float(p[1]),
                "小单净流入": float(p[2]),
                "中单净流入": float(p[3]),
                "大单净流入": float(p[4]),
                "超大单净流入": float(p[5]),
            })
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()


def fetch_sector_flow(sector_type="concept", top_n=30):
    """板块资金流向排名"""
    fs_map = {"industry": "m:90+t:2", "concept": "m:90+t:3"}
    url = "http://push2delay.eastmoney.com/api/qt/clist/get"
    params = {
        "fid": "f62", "po": 1, "pz": top_n, "pn": 1, "np": 1, "fltt": 2,
        "fields": "f12,f14,f2,f3,f62,f184,f124",
        "fs": fs_map.get(sector_type, "m:90+t:3"), "ut": UT,
    }
    try:
        resp = _get(url, params=params, headers={"Referer": "http://data.eastmoney.com/"})
        data = resp.json()
        if not (data.get("data") and data["data"].get("diff")):
            return []
        results = []
        for item in data["data"]["diff"]:
            results.append({
                "板块代码": str(item.get("f12", "")),
                "板块名称": str(item.get("f14", "")),
                "涨跌幅": item.get("f3", 0),
                "主力净流入": item.get("f62", 0),
                "净流入占比": item.get("f184", 0),
            })
        return results
    except Exception:
        return []


def fetch_market_sentiment():
    """大盘指数 + 涨跌家数"""
    url = "http://push2delay.eastmoney.com/api/qt/ulist.np/get"
    params = {
        "fltt": 2,
        "fields": "f1,f2,f3,f4,f6,f12,f13,f14,f104,f105,f106",
        "secids": "1.000001,0.399001,0.399006,1.000300",
        "ut": UT,
    }
    try:
        resp = _get(url, params=params, headers={"Referer": "http://data.eastmoney.com/"}, timeout=10)
        data = resp.json()
        if not (data.get("data") and data["data"].get("diff")):
            return []
        results = []
        for item in data["data"]["diff"]:
            results.append({
                "名称": str(item.get("f14", "")),
                "点位": item.get("f2", 0),
                "涨跌幅": item.get("f3", 0),
                "成交额": item.get("f6", 0),
                "上涨家数": item.get("f104", 0),
                "下跌家数": item.get("f105", 0),
                "平盘家数": item.get("f106", 0),
            })
        return results
    except Exception:
        return []


# ============================================================
# 数据获取 - 公司基本面（东方财富）
# ============================================================

def fetch_fundamentals_batch():
    """批量获取全市场基本面数据：PE/PB/营收增长/利润增长"""
    url = "http://push2delay.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": 1, "pz": 6000, "po": 1, "np": 1, "fltt": 2,
        "fid": "f3", "fs": FS_ALL_A, "ut": UT,
        "fields": "f12,f9,f23,f24,f25,f115",
    }
    try:
        resp = _get(url, params=params, headers={"Referer": "http://data.eastmoney.com/"})
        data = resp.json()
        if not (data.get("data") and data["data"].get("diff")):
            return {}
        result = {}
        for item in data["data"]["diff"]:
            code = str(item.get("f12", ""))
            pe = item.get("f9", None)       # PE(静)
            pb = item.get("f23", None)      # PB
            rev_growth = item.get("f24", None)   # 营收同比增长%
            profit_growth = item.get("f25", None) # 净利润同比增长%
            pe_ttm = item.get("f115", None) # PE(TTM)
            result[code] = {
                "PE": pe if pe and pe != "-" else None,
                "PB": pb if pb and pb != "-" else None,
                "营收增长": rev_growth if rev_growth and rev_growth != "-" else None,
                "利润增长": profit_growth if profit_growth and profit_growth != "-" else None,
                "PE_TTM": pe_ttm if pe_ttm and pe_ttm != "-" else None,
            }
        return result
    except Exception:
        return {}


def fetch_fundamental_detail(code):
    """获取单只股票详细基本面：ROE/毛利率/净利率/负债率/营收增长/总市值"""
    code = str(code)
    market = 1 if code.startswith("6") else 0
    url = "http://push2delay.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": f"{market}.{code}",
        "fields": "f57,f58,f116,f117,f162,f173,f183,f184,f185,f187,f188",
        "ut": UT,
    }
    try:
        resp = _get(url, params=params, headers={"Referer": "http://data.eastmoney.com/"}, timeout=10)
        data = resp.json().get("data", {})
        if not data:
            return {}

        def safe_float(v, div=1):
            try:
                return float(v) / div if v and v != "-" else None
            except (ValueError, TypeError):
                return None

        return {
            "代码": data.get("f57", code),
            "名称": data.get("f58", ""),
            "总市值": safe_float(data.get("f116"), 1e8),       # 亿
            "流通市值": safe_float(data.get("f117"), 1e8),     # 亿
            "PE动态": safe_float(data.get("f162"), 100),       # 动态PE(x100存储)
            "ROE": safe_float(data.get("f173")),               # %
            "每股收益": safe_float(data.get("f183")),
            "毛利率": safe_float(data.get("f184")),            # %
            "净利率": safe_float(data.get("f185")),            # %
            "负债率": safe_float(data.get("f187")),            # %
            "营收增长": safe_float(data.get("f188")),          # %
        }
    except Exception:
        return {}


def evaluate_fundamentals(fund_info):
    """
    基本面评分 (满分50)：
    1. 成长性 (0-20)：营收增长 + 利润增长
    2. 盈利能力 (0-15)：ROE + 毛利率
    3. 估值合理性 (0-15)：PE + PB
    """
    if not fund_info:
        return 8, {}, ""  # 无数据给低分（未知 ≠ 中性，应保守）

    score = 0
    details = {}
    reasons = []

    # 1. 成长性 (0-20)
    rev = fund_info.get("营收增长")
    profit = fund_info.get("利润增长")
    growth_s = 0
    if rev is not None and profit is not None:
        if profit > 50 and rev > 30:
            growth_s = 20; details["成长"] = f"高增长(营收+{rev:.0f}%,利润+{profit:.0f}%)"
            reasons.append(f"高增长(利润+{profit:.0f}%)")
        elif profit > 20 and rev > 15:
            growth_s = 15; details["成长"] = f"稳增长(营收+{rev:.0f}%,利润+{profit:.0f}%)"
            reasons.append("业绩稳增长")
        elif profit > 0 and rev > 0:
            growth_s = 10; details["成长"] = f"正增长(营收+{rev:.0f}%,利润+{profit:.0f}%)"
        elif profit > -10:
            growth_s = 5; details["成长"] = f"微降(利润{profit:+.0f}%)"
        else:
            growth_s = 0; details["成长"] = f"下滑(利润{profit:+.0f}%)"
    elif rev is not None:
        if rev > 30:
            growth_s = 12; details["成长"] = f"营收+{rev:.0f}%"
        elif rev > 0:
            growth_s = 7; details["成长"] = f"营收+{rev:.0f}%"
        else:
            growth_s = 2; details["成长"] = f"营收{rev:+.0f}%"
    else:
        growth_s = 3; details["成长"] = "无数据(保守)"
    score += growth_s

    # 2. 盈利能力 (0-15)
    roe = fund_info.get("ROE")
    margin = fund_info.get("毛利率")
    profit_s = 0
    if roe is not None:
        if roe > 15:
            profit_s += 10; details["ROE"] = f"{roe:.1f}%(优秀)"
            reasons.append(f"ROE {roe:.0f}%")
        elif roe > 8:
            profit_s += 7; details["ROE"] = f"{roe:.1f}%(良好)"
        elif roe > 3:
            profit_s += 4; details["ROE"] = f"{roe:.1f}%(一般)"
        elif roe > 0:
            profit_s += 2; details["ROE"] = f"{roe:.1f}%(偏低)"
        else:
            profit_s += 0; details["ROE"] = f"{roe:.1f}%(亏损)"
    else:
        profit_s += 2; details["ROE"] = "无数据"

    if margin is not None:
        if margin > 40:
            profit_s += 5; details["毛利率"] = f"{margin:.1f}%(高)"
        elif margin > 20:
            profit_s += 3; details["毛利率"] = f"{margin:.1f}%(中)"
        else:
            profit_s += 1; details["毛利率"] = f"{margin:.1f}%(低)"
    else:
        profit_s += 2; details["毛利率"] = "无数据"
    score += profit_s

    # 3. 估值合理性 (0-15)
    pe = fund_info.get("PE") or fund_info.get("PE动态") or fund_info.get("PE_TTM")
    pb = fund_info.get("PB")
    val_s = 0
    if pe is not None and pe > 0:
        if pe < 20:
            val_s += 10; details["PE"] = f"{pe:.1f}(低估)"
            reasons.append(f"PE仅{pe:.0f}")
        elif pe < 40:
            val_s += 7; details["PE"] = f"{pe:.1f}(合理)"
        elif pe < 80:
            val_s += 4; details["PE"] = f"{pe:.1f}(偏高)"
        else:
            val_s += 1; details["PE"] = f"{pe:.1f}(高估)"
    elif pe is not None and pe < 0:
        val_s += 0; details["PE"] = "亏损"
    else:
        val_s += 2; details["PE"] = "无数据"

    if pb is not None and pb > 0:
        if pb < 2:
            val_s += 5; details["PB"] = f"{pb:.1f}(低)"
        elif pb < 5:
            val_s += 3; details["PB"] = f"{pb:.1f}(中)"
        else:
            val_s += 1; details["PB"] = f"{pb:.1f}(高)"
    else:
        val_s += 2; details["PB"] = "无数据"
    score += val_s

    reason_str = "；".join(reasons) if reasons else ""
    return score, details, reason_str


# ============================================================
# 数据获取 - 龙虎榜 / 北向资金 / 融资融券 / 涨停板
# ============================================================

def fetch_billboard():
    """获取龙虎榜净买入股票（机构/游资动向）"""
    url = "http://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "reportName": "RPT_DAILYBILLBOARD_DETAILSNEW",
        "columns": "SECURITY_CODE,SECURITY_NAME_ABBR,TRADE_DATE,CHANGE_RATE,BILLBOARD_NET_AMT,BILLBOARD_BUY_AMT,BILLBOARD_SELL_AMT",
        "pageSize": 100, "pageNumber": 1,
        "sortColumns": "BILLBOARD_NET_AMT", "sortTypes": "-1",
        "source": "WEB", "client": "WEB",
    }
    try:
        resp = _get(url, params=params, timeout=10)
        data = resp.json()
        if not (data.get("result") and data["result"].get("data")):
            return {}
        result = {}
        for item in data["result"]["data"]:
            code = str(item.get("SECURITY_CODE", ""))
            net = item.get("BILLBOARD_NET_AMT", 0) or 0
            buy = item.get("BILLBOARD_BUY_AMT", 0) or 0
            sell = item.get("BILLBOARD_SELL_AMT", 0) or 0
            result[code] = {
                "龙虎榜净买入": net,
                "龙虎榜买入": buy,
                "龙虎榜卖出": sell,
                "龙虎榜涨幅": item.get("CHANGE_RATE", 0) or 0,
            }
        return result
    except Exception:
        return {}


def fetch_northbound_flow():
    """获取北向资金（沪股通+深股通）今日净流入"""
    url = "http://push2delay.eastmoney.com/api/qt/kamt.rtmin/get"
    params = {
        "fields1": "f1,f2,f3,f4",
        "fields2": "f51,f52,f53,f54,f55,f56",
        "ut": UT,
    }
    try:
        resp = _get(url, params=params, headers={"Referer": "http://data.eastmoney.com/"}, timeout=10)
        data = resp.json().get("data", {})
        if not data:
            return 0, ""
        s2n = data.get("s2n", [])
        if not s2n:
            return 0, ""
        # 最后一条数据
        last = s2n[-1].split(",")
        # last: 时间, 沪股通净买入, 深股通净买入, 合计, ...
        try:
            sh_net = float(last[1]) if last[1] != "-" else 0
            sz_net = float(last[2]) if last[2] != "-" else 0
            total = sh_net + sz_net
            info = f"沪股通:{sh_net/1e4:+.1f}亿 深股通:{sz_net/1e4:+.1f}亿 合计:{total/1e4:+.1f}亿"
            return total, info
        except (ValueError, IndexError):
            return 0, ""
    except Exception:
        return 0, ""


def fetch_margin_data_top():
    """获取融资余额TOP股票（融资买入=杠杆资金看多）"""
    url = "http://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "reportName": "RPTA_WEB_RZRQ_GGMX",
        "columns": "SCODE,SECNAME,RZYE,RZMRE,RZJME,RQYL",
        "pageSize": 200, "pageNumber": 1,
        "sortColumns": "RZJME", "sortTypes": "-1",
        "source": "WEB", "client": "WEB",
    }
    try:
        resp = _get(url, params=params, timeout=10)
        data = resp.json()
        if not (data.get("result") and data["result"].get("data")):
            return {}
        result = {}
        for item in data["result"]["data"]:
            code = str(item.get("SCODE", ""))
            result[code] = {
                "融资余额": item.get("RZYE", 0) or 0,
                "融资净买入": item.get("RZJME", 0) or 0,
                "融资买入额": item.get("RZMRE", 0) or 0,
            }
        return result
    except Exception:
        return {}


def count_limit_up():
    """统计今日涨停/跌停数量（市场情绪）"""
    url = "http://push2delay.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": 1, "pz": 6000, "po": 1, "np": 1, "fltt": 2,
        "fid": "f3", "fs": FS_ALL_A, "ut": UT,
        "fields": "f12,f14,f3",
    }
    try:
        resp = _get(url, params=params, headers={"Referer": "http://data.eastmoney.com/"})
        data = resp.json()
        if not (data.get("data") and data["data"].get("diff")):
            return 0, 0
        limit_up = 0
        limit_down = 0
        for item in data["data"]["diff"]:
            chg = item.get("f3", 0)
            if chg is None:
                continue
            if chg >= 9.8:
                limit_up += 1
            elif chg <= -9.8:
                limit_down += 1
        return limit_up, limit_down
    except Exception:
        return 0, 0


def fetch_limit_up_pool():
    """获取涨停板池：连板数、首板/二板/三板+、封单额"""
    url = "http://push2ex.eastmoney.com/getTopicZTPool"
    params = {
        "ut": UT, "dpt": "wz.ztzt",
        "Ession": "0",  # 0=全部
        "sort": "fbt:asc",
        "date": datetime.now().strftime("%Y%m%d"),
    }
    try:
        resp = _get(url, params=params, timeout=10)
        data = resp.json()
        if not (data.get("data") and data["data"].get("pool")):
            return {}
        result = {}
        for item in data["data"]["pool"]:
            code = str(item.get("c", ""))
            result[code] = {
                "连板数": item.get("zbc", 1),           # 连板数
                "涨停原因": item.get("hybk", ""),        # 行业板块
                "封单额": item.get("fund", 0),           # 封单金额
                "首次封板时间": item.get("fbt", ""),      # 首次封板时间
                "最后封板时间": item.get("lbt", ""),      # 最后封板时间
                "炸板次数": item.get("zbc2", 0),         # 开板次数
                "涨停统计": item.get("zttj", {}),
            }
        return result
    except Exception:
        return {}


def fetch_billboard_detail():
    """获取龙虎榜买卖席位明细（区分机构/游资）"""
    url = "http://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "reportName": "RPT_BILLBOARD_DAILYDETAILSBUY",
        "columns": "SECURITY_CODE,OPERATEDEPT_NAME,BUY_AMT,SELL_AMT,NET_AMT,RANK",
        "pageSize": 500, "pageNumber": 1,
        "sortColumns": "NET_AMT", "sortTypes": "-1",
        "source": "WEB", "client": "WEB",
    }
    try:
        resp = _get(url, params=params, timeout=10)
        data = resp.json()
        if not (data.get("result") and data["result"].get("data")):
            return {}
        # 按股票代码汇总：统计机构席位 vs 游资(营业部)
        result = {}
        for item in data["result"]["data"]:
            code = str(item.get("SECURITY_CODE", ""))
            dept = str(item.get("OPERATEDEPT_NAME", ""))
            net = item.get("NET_AMT", 0) or 0
            if code not in result:
                result[code] = {"机构买入": 0, "游资买入": 0, "机构席位数": 0, "游资席位数": 0}
            if "机构" in dept:
                result[code]["机构买入"] += net
                result[code]["机构席位数"] += 1
            else:
                result[code]["游资买入"] += net
                result[code]["游资席位数"] += 1
        return result
    except Exception:
        return {}


def fetch_shareholder_increase():
    """获取近期股东/机构增持数据"""
    url = "http://datacenter-web.eastmoney.com/api/data/v1/get"
    # 近30天内的增持公告
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    params = {
        "reportName": "RPT_CUSTOM_HOLDER_INCREASE",
        "columns": "SECURITY_CODE,SECURITY_NAME_ABBR,HOLDER_NAME,CHANGE_SHARES,CHANGE_RATIO,AFTER_RATIO,END_DATE,HOLDER_TYPE",
        "pageSize": 200, "pageNumber": 1,
        "sortColumns": "END_DATE", "sortTypes": "-1",
        "filter": f"(END_DATE>='{start_date}')(END_DATE<='{end_date}')",
        "source": "WEB", "client": "WEB",
    }
    try:
        resp = _get(url, params=params, timeout=10)
        data = resp.json()
        if not (data.get("result") and data["result"].get("data")):
            return {}
        result = {}
        for item in data["result"]["data"]:
            code = str(item.get("SECURITY_CODE", ""))
            ratio = item.get("CHANGE_RATIO", 0) or 0
            holder_type = str(item.get("HOLDER_TYPE", ""))
            if code not in result:
                result[code] = {"增持比例": 0, "增持次数": 0, "增持类型": set()}
            result[code]["增持比例"] += ratio
            result[code]["增持次数"] += 1
            if holder_type:
                result[code]["增持类型"].add(holder_type)
        # 转set为str
        for code in result:
            result[code]["增持类型"] = "/".join(result[code]["增持类型"]) if result[code]["增持类型"] else "其他"
        return result
    except Exception:
        return {}


def fetch_industry_prosperity():
    """获取行业景气度：通过行业板块涨幅排名+资金流入+个股数量综合评估"""
    url = "http://push2delay.eastmoney.com/api/qt/clist/get"
    params = {
        "fid": "f3", "po": 1, "pz": 80, "pn": 1, "np": 1, "fltt": 2,
        "fields": "f12,f14,f2,f3,f62,f184,f104,f105",
        "fs": "m:90+t:2", "ut": UT,  # 行业板块
    }
    try:
        resp = _get(url, params=params, headers={"Referer": "http://data.eastmoney.com/"})
        data = resp.json()
        if not (data.get("data") and data["data"].get("diff")):
            return {}
        result = {}
        rank = 0
        for item in data["data"]["diff"]:
            rank += 1
            name = str(item.get("f14", ""))
            result[name] = {
                "行业排名": rank,
                "涨跌幅": item.get("f3", 0),
                "主力净流入": item.get("f62", 0),
                "净流入占比": item.get("f184", 0),
                "上涨家数": item.get("f104", 0),
                "下跌家数": item.get("f105", 0),
            }
        return result
    except Exception:
        return {}


def get_stock_industry(code):
    """获取个股所属行业"""
    market = 1 if str(code).startswith("6") else 0
    url = "http://push2delay.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": f"{market}.{code}",
        "fields": "f57,f127",
        "ut": UT,
    }
    try:
        resp = _get(url, params=params, headers={"Referer": "http://data.eastmoney.com/"}, timeout=5)
        data = resp.json().get("data", {})
        return str(data.get("f127", "")) if data else ""
    except Exception:
        return ""


def evaluate_extra_dimensions(code, billboard_data, margin_data, northbound_total, limit_up, limit_down,
                              limit_up_pool=None, billboard_detail=None, shareholder_data=None, industry_data=None, stock_industry=""):
    """
    额外维度评分 (满分80)：
    1. 龙虎榜 (0-20)：机构/游资净买入 + 席位类型加分
    2. 北向资金 (0-10)：外资净流入方向
    3. 融资融券 (0-10)：融资净买入=杠杆看多
    4. 涨跌停/连板 (0-15)：涨停多=情绪好 + 连板溢价
    5. 股东增持 (0-10)：大股东/机构增持=内部人看好
    6. 行业景气度 (0-10)：行业处于上升周期
    7. 风控扣分 (-25~0)：高风险特征扣分
    """
    if limit_up_pool is None:
        limit_up_pool = {}
    if billboard_detail is None:
        billboard_detail = {}
    if shareholder_data is None:
        shareholder_data = {}
    if industry_data is None:
        industry_data = {}
    score = 0
    details = {}
    reasons = []

    # 1. 龙虎榜 (0-20)：净买入 + 机构/游资席位加分
    bb = billboard_data.get(code)
    bb_detail = billboard_detail.get(code)
    if bb:
        net = bb["龙虎榜净买入"]
        if net > 5e7:
            score += 15; details["龙虎榜"] = f"净买入{net/1e4:.0f}万"
            reasons.append(f"龙虎榜净买入{net/1e8:.1f}亿")
        elif net > 1e7:
            score += 10; details["龙虎榜"] = f"净买入{net/1e4:.0f}万"
            reasons.append("龙虎榜净买入")
        elif net > 0:
            score += 5; details["龙虎榜"] = f"净买入{net/1e4:.0f}万"
        elif net < -1e7:
            score -= 5; details["龙虎榜"] = f"净卖出{net/1e4:.0f}万"
        else:
            details["龙虎榜"] = "小额"
        # 席位类型加分
        if bb_detail:
            inst_buy = bb_detail.get("机构买入", 0)
            inst_seats = bb_detail.get("机构席位数", 0)
            if inst_buy > 1e7 and inst_seats >= 2:
                score += 5; details["龙虎榜"] += f"+机构{inst_seats}席"
                reasons.append(f"机构{inst_seats}席位买入")
            elif inst_buy > 0 and inst_seats >= 1:
                score += 3; details["龙虎榜"] += f"+机构{inst_seats}席"
    else:
        details["龙虎榜"] = "未上榜"
        score += 2  # 未上榜是中性

    # 2. 北向资金 (0-10)
    if northbound_total > 50e4:  # >50亿净流入
        score += 10; details["北向资金"] = f"大幅流入"
    elif northbound_total > 10e4:
        score += 7; details["北向资金"] = f"净流入"
    elif northbound_total > 0:
        score += 4; details["北向资金"] = f"小幅流入"
    elif northbound_total > -10e4:
        score += 2; details["北向资金"] = f"小幅流出"
    else:
        score += 0; details["北向资金"] = f"大幅流出"

    # 3. 融资融券 (0-10)
    mg = margin_data.get(code)
    if mg:
        net_buy = mg.get("融资净买入", 0)
        if net_buy > 5e7:
            score += 10; details["融资"] = f"净买入{net_buy/1e4:.0f}万"
            reasons.append("融资净买入")
        elif net_buy > 1e7:
            score += 7; details["融资"] = f"净买入{net_buy/1e4:.0f}万"
        elif net_buy > 0:
            score += 4; details["融资"] = f"小幅净买入"
        elif net_buy > -1e7:
            score += 2; details["融资"] = f"小幅净卖出"
        else:
            score += 0; details["融资"] = f"净卖出"
    else:
        details["融资"] = "无数据"
        score += 3  # 中性

    # 4. 涨跌停/连板 (0-15)
    zt_base = 0
    if limit_up > 80:
        zt_base = 8; details["涨停"] = f"涨{limit_up}跌{limit_down}(强)"
    elif limit_up > 40:
        zt_base = 5; details["涨停"] = f"涨{limit_up}跌{limit_down}(较强)"
    elif limit_up > 20:
        zt_base = 3; details["涨停"] = f"涨{limit_up}跌{limit_down}(一般)"
    else:
        zt_base = 1; details["涨停"] = f"涨{limit_up}跌{limit_down}(弱)"
    score += zt_base

    # 连板分析：首板次日溢价率统计有优势
    zt_info = limit_up_pool.get(code)
    if zt_info:
        boards = zt_info.get("连板数", 1)
        seal_amt = zt_info.get("封单额", 0)
        if boards == 1 and seal_amt > 1e8:
            score += 7; details["连板"] = f"首板(封单{seal_amt/1e8:.1f}亿)"
            reasons.append("首板强封单")
        elif boards == 1:
            score += 5; details["连板"] = "首板"
            reasons.append("首板溢价")
        elif boards == 2:
            score += 4; details["连板"] = "二连板"
            reasons.append("二连板")
        elif boards >= 3:
            # 高位连板风险增大，降低加分
            score += 2; details["连板"] = f"{boards}连板(高位注意)"
    else:
        details["连板"] = "-"

    # 5. 股东增持 (0-10)
    sh_data = shareholder_data.get(code)
    if sh_data:
        inc_ratio = sh_data.get("增持比例", 0)
        inc_count = sh_data.get("增持次数", 0)
        inc_type = sh_data.get("增持类型", "")
        if inc_ratio > 1 and inc_count >= 2:
            score += 10; details["增持"] = f"大幅增持{inc_ratio:.2f}%×{inc_count}次"
            reasons.append(f"股东大幅增持")
        elif inc_ratio > 0.5:
            score += 7; details["增持"] = f"增持{inc_ratio:.2f}%({inc_type})"
            reasons.append("股东增持")
        elif inc_ratio > 0:
            score += 4; details["增持"] = f"小幅增持{inc_ratio:.2f}%"
        else:
            details["增持"] = "无增持"
    else:
        details["增持"] = "-"
        score += 1  # 中性

    # 6. 行业景气度 (0-10)
    if stock_industry and industry_data:
        ind = industry_data.get(stock_industry)
        if ind:
            ind_rank = ind.get("行业排名", 50)
            ind_chg = ind.get("涨跌幅", 0)
            ind_flow = ind.get("主力净流入", 0)
            ind_up = ind.get("上涨家数", 0)
            ind_down = ind.get("下跌家数", 0)
            ind_ratio = ind_up / max(ind_up + ind_down, 1)
            if ind_rank <= 10 and ind_flow > 0:
                score += 10; details["行业"] = f"{stock_industry}(排名{ind_rank},涨{ind_chg:+.1f}%)"
                reasons.append(f"行业景气({stock_industry})")
            elif ind_rank <= 20 and ind_chg > 0:
                score += 7; details["行业"] = f"{stock_industry}(排名{ind_rank})"
            elif ind_rank <= 40:
                score += 4; details["行业"] = f"{stock_industry}(排名{ind_rank})"
            else:
                score += 1; details["行业"] = f"{stock_industry}(排名{ind_rank},偏弱)"
        else:
            details["行业"] = stock_industry or "-"
            score += 3
    else:
        details["行业"] = "-"
        score += 3  # 中性

    # 7. 风控扣分 (-25~0)
    risk_warnings = []
    # 如果龙虎榜大幅净卖出
    if bb and bb["龙虎榜净买入"] < -1e8:
        score -= 10; risk_warnings.append("龙虎榜大额净卖出")
    # 如果跌停多于涨停
    if limit_down > limit_up * 1.5 and limit_down > 30:
        score -= 5; risk_warnings.append("跌停多于涨停")
    # 北向大幅流出
    if northbound_total < -50e4:
        score -= 5; risk_warnings.append("北向大幅流出")
    # 高位连板风险
    if zt_info and zt_info.get("连板数", 0) >= 4:
        score -= 5; risk_warnings.append(f"高位{zt_info['连板数']}连板")
    # 融资净卖出 + 龙虎榜净卖出 = 双杀
    if mg and mg.get("融资净买入", 0) < -5e7 and bb and bb["龙虎榜净买入"] < -5e7:
        score -= 5; risk_warnings.append("融资+龙虎榜双杀")

    if risk_warnings:
        details["风控"] = "；".join(risk_warnings)
    else:
        details["风控"] = "正常"

    reason_str = "；".join(reasons) if reasons else ""
    return max(score, 0), details, reason_str


# ============================================================
# 智能仓位管理 & 风控系统
# ============================================================

def calc_position_and_risk(stock_score, sentiment_score, northbound_total, limit_up, limit_down, price, kline, largecap_adj=None):
    """
    根据综合评分 + 市场环境，计算：
    1. 建议仓位比例
    2. 止损价 / 止盈价
    3. 风险等级
    """
    latest = kline.iloc[-1]
    recent_20 = kline.tail(20)
    atr = recent_20["振幅"].mean()  # 平均振幅
    recent_5 = kline.tail(5)

    # ★ 动态止损：基于ATR倍数（标准风控是1.5-2倍ATR）
    # 高波动股止损不能太紧（否则被震出），低波动股止损不能太松（否则亏损扩大）
    atr_stop = atr * 0.8  # 约0.8倍ATR作为止损幅度
    if atr > 6:
        stop_pct = -max(2.5, min(atr_stop, 4.0))   # 高波动：-2.5%~-4%
    elif atr > 4:
        stop_pct = -max(2.0, min(atr_stop, 3.0))   # 中波动：-2%~-3%
    else:
        stop_pct = -max(1.5, min(atr_stop, 2.5))   # 低波动：-1.5%~-2.5%

    # ★ 结合支撑位优化止损：如果支撑位在止损范围内，以支撑位下方为止损
    support_20 = kline.tail(20)["最低"].min()
    support_pct = (support_20 - price) / price * 100
    if -6 < support_pct < stop_pct:
        # 支撑位比计算止损更近，用支撑位下方0.5%
        stop_pct = support_pct - 0.5

    stop_loss = price * (1 + stop_pct / 100)

    # ★ 动态止盈：基于ATR倍数，确保盈亏比 >= 2:1
    min_tp1 = abs(stop_pct) * 1.5  # 第一目标至少是止损的1.5倍
    tp1_pct = max(min_tp1, atr * 0.6)
    tp2_pct = max(min_tp1 * 1.8, atr * 1.0)
    tp3_pct = max(min_tp1 * 2.5, atr * 1.5)

    tp1 = price * (1 + tp1_pct / 100)
    tp2 = price * (1 + tp2_pct / 100)
    tp3 = price * (1 + tp3_pct / 100)

    # 支撑/压力（用20日范围，比5日更可靠）
    support = kline.tail(20)["最低"].min()
    resistance = kline.tail(20)["最高"].max()

    # ★ 仓位计算 - 更保守的基础仓位，避免过度集中
    base_position = 30  # 基础仓位30%（比之前的40%更保守）
    if stock_score >= 180:
        base_position = 40
    elif stock_score >= 150:
        base_position = 35
    elif stock_score >= 120:
        base_position = 30
    else:
        base_position = 20

    # 市场环境调整（使用乘数而非加减法，避免极端偏移）
    env_mult = 1.0
    if sentiment_score >= 10:
        env_mult += 0.15
    elif sentiment_score <= 3:
        env_mult -= 0.25

    if northbound_total > 30e4:
        env_mult += 0.1
    elif northbound_total < -30e4:
        env_mult -= 0.2

    if limit_down > limit_up and limit_down > 30:
        env_mult -= 0.2

    # ★ 波动率调整：高波动降仓
    if atr > 6:
        env_mult -= 0.15
    elif atr > 5:
        env_mult -= 0.08

    base_position = int(base_position * max(env_mult, 0.4))
    position = max(10, min(base_position, 40))  # ★ 上限从50%降至40%

    # 大盘股风控调整
    largecap_label = ""
    if largecap_adj:
        position = min(position, largecap_adj.get("仓位上限", position))
        extra_stop = largecap_adj.get("止损收紧", 0)
        if extra_stop:
            stop_pct += extra_stop  # extra_stop is negative, makes stop tighter
            stop_loss = price * (1 + stop_pct / 100)
        largecap_label = largecap_adj.get("标签", "")

    # 风险等级
    risk_factors = 0
    if atr > 5:
        risk_factors += 1
    if sentiment_score <= 4:
        risk_factors += 1
    if northbound_total < -20e4:
        risk_factors += 1
    if limit_down > limit_up:
        risk_factors += 1

    if risk_factors >= 3:
        risk_level = "高风险"
    elif risk_factors >= 2:
        risk_level = "中等风险"
    elif risk_factors >= 1:
        risk_level = "低风险"
    else:
        risk_level = "极低风险"

    if largecap_label:
        risk_level = f"{risk_level}({largecap_label})"

    return {
        "仓位": position,
        "止损价": stop_loss,
        "止损幅度": stop_pct,
        "止盈一": tp1,
        "止盈一幅度": tp1_pct,
        "止盈二": tp2,
        "止盈二幅度": tp2_pct,
        "止盈三": tp3,
        "止盈三幅度": tp3_pct,
        "支撑位": support,
        "压力位": resistance,
        "ATR": atr,
        "风险等级": risk_level,
    }


def calc_position_and_risk_t5(tech_score, position_mult, sentiment_score, northbound_total,
                               limit_up, limit_down, price, kline, largecap_adj=None):
    """
    T+5 波段风控：比T+1更宽的止损止盈，带移动止损
    """
    latest = kline.iloc[-1]
    recent_20 = kline.tail(20)
    atr = recent_20["振幅"].mean()
    recent_10 = kline.tail(10)

    # T+5止损：比T+1宽（-3%~-5%）
    if atr > 6:
        stop_pct = -3.0
    elif atr > 4:
        stop_pct = -4.0
    else:
        stop_pct = -5.0

    stop_loss = price * (1 + stop_pct / 100)

    # T+5止盈：三级，目标更高
    tp1_pct = max(3.0, atr * 1.0)
    tp2_pct = max(5.0, atr * 2.0)
    tp3_pct = max(8.0, atr * 3.0)

    tp1 = price * (1 + tp1_pct / 100)
    tp2 = price * (1 + tp2_pct / 100)
    tp3 = price * (1 + tp3_pct / 100)

    # 支撑/压力（看10天）
    support = recent_10["最低"].min()
    resistance = recent_10["最高"].max()

    # 仓位：基于技术分+倍率
    base_position = 30
    if tech_score >= 60:
        base_position = 40
    elif tech_score >= 45:
        base_position = 35
    # 基本面/聪明钱倍率调整
    base_position = int(base_position * position_mult)

    # 市场环境
    if sentiment_score >= 10:
        base_position += 5
    elif sentiment_score <= 3:
        base_position -= 10
    if northbound_total < -30e4:
        base_position -= 5

    position = max(10, min(base_position, 40))  # T+5最高40%

    if largecap_adj:
        position = min(position, largecap_adj.get("仓位上限", position))

    risk_factors = 0
    if atr > 5: risk_factors += 1
    if sentiment_score <= 4: risk_factors += 1
    if northbound_total < -20e4: risk_factors += 1
    if limit_down > limit_up: risk_factors += 1

    risk_level = "高风险" if risk_factors >= 3 else "中等风险" if risk_factors >= 2 else \
                 "低风险" if risk_factors >= 1 else "极低风险"

    return {
        "仓位": position,
        "止损价": stop_loss, "止损幅度": stop_pct,
        "止盈一": tp1, "止盈一幅度": tp1_pct,
        "止盈二": tp2, "止盈二幅度": tp2_pct,
        "止盈三": tp3, "止盈三幅度": tp3_pct,
        "移动止损": "TP1达成→止损移至成本价；TP2达成→止损移至TP1",
        "时间止损": "第5天收盘无条件清仓",
        "支撑位": support, "压力位": resistance,
        "ATR": atr, "风险等级": risk_level,
    }


# ============================================================
# 技术指标计算
# ============================================================

def calc_ma(df, periods=[5, 10, 20, 60]):
    for p in periods:
        df[f"MA{p}"] = df["收盘"].rolling(p).mean()
    return df

def calc_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["收盘"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["收盘"].ewm(span=slow, adjust=False).mean()
    df["DIF"] = ema_fast - ema_slow
    df["DEA"] = df["DIF"].ewm(span=signal, adjust=False).mean()
    df["MACD柱"] = 2 * (df["DIF"] - df["DEA"])
    return df

def calc_kdj(df, n=9, m1=3, m2=3):
    low_n = df["最低"].rolling(n).min()
    high_n = df["最高"].rolling(n).max()
    denom = high_n - low_n
    # ★ 修复：当最高=最低(涨停/跌停全天)时分母为0，RSV设为50
    rsv = np.where(denom > 0, (df["收盘"] - low_n) / denom * 100, 50)
    rsv = pd.Series(rsv, index=df.index)
    rsv = rsv.fillna(50)
    df["K"] = rsv.ewm(com=m1-1, adjust=False).mean()
    df["D"] = df["K"].ewm(com=m2-1, adjust=False).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]
    return df

def calc_rsi(df, period=6, period2=14):
    delta = df["收盘"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"RSI{period}"] = 100 - (100 / (1 + rs))
    # ★ 增加RSI14作为趋势确认
    gain14 = delta.where(delta > 0, 0).rolling(period2).mean()
    loss14 = (-delta.where(delta < 0, 0)).rolling(period2).mean()
    rs14 = gain14 / loss14.replace(0, np.nan)
    df[f"RSI{period2}"] = 100 - (100 / (1 + rs14))
    return df

def calc_volume_ratio(df):
    df["量比计算"] = df["成交量"] / df["成交量"].rolling(5).mean()
    return df

def calc_boll(df, n=20, k=2):
    df["BOLL中"] = df["收盘"].rolling(n).mean()
    std = df["收盘"].rolling(n).std()
    df["BOLL上"] = df["BOLL中"] + k * std
    df["BOLL下"] = df["BOLL中"] - k * std
    return df

def calc_all_indicators(df):
    df = calc_ma(df)
    df = calc_macd(df)
    df = calc_kdj(df)
    df = calc_rsi(df)
    df = calc_volume_ratio(df)
    df = calc_boll(df)
    return df


# ============================================================
# 新增数据源：筹码/尾盘资金/机构调研/赚钱效应
# ============================================================

def fetch_chip_data(code):
    """获取筹码集中度和获利盘比例"""
    code = str(code)
    market = 1 if code.startswith("6") else 0
    url = "http://push2delay.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": f"{market}.{code}",
        "fields": "f57,f164,f165,f166,f167,f168,f169,f170",
        "ut": UT,
    }
    try:
        resp = _get(url, params=params, headers={"Referer": "http://data.eastmoney.com/"}, timeout=8)
        data = resp.json().get("data", {})
        if not data:
            return None

        def sf(v):
            try:
                return float(v) if v and v != "-" else None
            except (ValueError, TypeError):
                return None

        profit_ratio = sf(data.get("f164"))  # 获利盘比例 %
        avg_cost = sf(data.get("f165"))      # 平均成本
        chip_conc_90 = sf(data.get("f166"))  # 90%筹码集中度
        chip_conc_70 = sf(data.get("f167"))  # 70%筹码集中度

        if profit_ratio is None and avg_cost is None:
            return None
        return {
            "获利盘": profit_ratio,
            "平均成本": avg_cost,
            "筹码集中90": chip_conc_90,
            "筹码集中70": chip_conc_70,
        }
    except Exception:
        return None


def fetch_tail_capital_flow(code):
    """获取尾盘(最后30分钟)资金流向"""
    code = str(code)
    market = 1 if code.startswith("6") else 0
    url = "http://push2delay.eastmoney.com/api/qt/stock/fflow/kline/get"
    params = {
        "lmt": 0, "klt": 1,  # 1分钟级别
        "secid": f"{market}.{code}",
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63",
        "ut": UT,
    }
    try:
        resp = _get(url, params=params, headers={"Referer": "http://data.eastmoney.com/"}, timeout=8)
        data = resp.json()
        if not (data.get("data") and data["data"].get("klines")):
            return None
        klines = data["data"]["klines"]
        # 取最后30条（最后30分钟）
        tail = klines[-30:] if len(klines) >= 30 else klines
        total_main = 0
        total_super = 0
        day_main = 0
        for line in klines:
            p = line.split(",")
            if len(p) >= 6:
                day_main += float(p[1])  # 主力净流入
        for line in tail:
            p = line.split(",")
            if len(p) >= 6:
                total_main += float(p[1])
                total_super += float(p[2]) if len(p) > 2 else 0
        return {
            "尾盘主力净流入": total_main,
            "尾盘超大单": total_super,
            "全天主力": day_main,
            "尾盘占比": total_main / day_main * 100 if day_main != 0 else 0,
        }
    except Exception:
        return None


def fetch_institutional_research():
    """获取近30天机构调研数据"""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    url = "http://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "reportName": "RPT_ORG_SURVEYNEW",
        "columns": "SECURITY_CODE,SECURITY_NAME_ABBR,SURVEY_DATE,ORG_TYPE,RECEIVE_WAY,RESEARCHERS",
        "pageSize": 500, "pageNumber": 1,
        "sortColumns": "SURVEY_DATE", "sortTypes": "-1",
        "filter": f"(SURVEY_DATE>='{start_date}')(SURVEY_DATE<='{end_date}')",
        "source": "WEB", "client": "WEB",
    }
    try:
        resp = _get(url, params=params, timeout=10)
        data = resp.json()
        if not (data.get("result") and data["result"].get("data")):
            return {}
        result = {}
        for item in data["result"]["data"]:
            code = str(item.get("SECURITY_CODE", ""))
            org_type = str(item.get("ORG_TYPE", ""))
            if code not in result:
                result[code] = {"调研次数": 0, "基金": 0, "券商": 0, "保险QFII": 0}
            result[code]["调研次数"] += 1
            if "基金" in org_type:
                result[code]["基金"] += 1
            elif "券商" in org_type or "证券" in org_type:
                result[code]["券商"] += 1
            elif "保险" in org_type or "QFII" in org_type or "外资" in org_type:
                result[code]["保险QFII"] += 1
        return result
    except Exception:
        return {}


def calc_money_effect():
    """
    计算赚钱效应：
    - 今日上涨比例
    - 用上证指数近N日K线推算连续赚钱/亏钱天数
    """
    sentiment = fetch_market_sentiment()
    sh = next((s for s in sentiment if "上证" in s["名称"]), None) if sentiment else None

    today_up_ratio = 0.5
    today_chg = 0
    if sh:
        up = sh.get("上涨家数", 0)
        down = sh.get("下跌家数", 0)
        total = up + down if (up + down) > 0 else 1
        today_up_ratio = up / total
        today_chg = sh.get("涨跌幅", 0)

    # 用上证指数K线算连续天数
    sh_kline = fetch_kline("000001", days=15)
    consecutive_bad = 0
    consecutive_good = 0
    if not sh_kline.empty and len(sh_kline) >= 3:
        for i in range(len(sh_kline) - 1, 0, -1):
            chg = sh_kline.iloc[i]["涨跌幅"]
            if chg is not None and chg < -0.3:
                consecutive_bad += 1
            else:
                break
        for i in range(len(sh_kline) - 1, 0, -1):
            chg = sh_kline.iloc[i]["涨跌幅"]
            if chg is not None and chg > 0.3:
                consecutive_good += 1
            else:
                break

    return {
        "今日上涨比例": today_up_ratio,
        "今日大盘涨幅": today_chg,
        "连续下跌天数": consecutive_bad,
        "连续上涨天数": consecutive_good,
    }


def calc_sector_leader_score(code, stock_chg, stock_mktcap, industry_data, stock_industry, stock_list_df):
    """
    板块龙头识别：
    - 个股涨幅 vs 行业平均涨幅
    - 个股市值在行业中的排名
    返回 0-8 分
    """
    if not stock_industry or not industry_data:
        return 0, ""
    ind = industry_data.get(stock_industry)
    if not ind:
        return 0, ""

    ind_chg = ind.get("涨跌幅", 0)
    # 个股涨幅 > 行业平均2倍 → 龙头
    excess_return = stock_chg - ind_chg if ind_chg else stock_chg

    score = 0
    label = ""
    if excess_return > 5:
        score = 8; label = f"板块龙头(超额+{excess_return:.1f}%)"
    elif excess_return > 3:
        score = 6; label = f"板块强势(超额+{excess_return:.1f}%)"
    elif excess_return > 1:
        score = 4; label = f"板块跟涨(超额+{excess_return:.1f}%)"
    elif excess_return > -1:
        score = 2; label = "板块同步"
    else:
        score = 0; label = "弱于板块"

    return score, label


# ============================================================
# 大宗商品关联风险
# ============================================================

# 行业 → 关联大宗商品
INDUSTRY_COMMODITY_MAP = {
    "石油石化": ["原油"], "石油": ["原油"], "石油开采": ["原油"],
    "化工": ["原油"], "基础化工": ["原油"],
    "有色金属": ["铜", "黄金"], "贵金属": ["黄金"], "铜": ["铜"],
    "农业": ["大豆"], "农林牧渔": ["大豆"], "养殖": ["大豆"],
    "煤炭": ["原油"],  # 煤炭受油价影响（替代品逻辑）
    "钢铁": ["铜", "原油"],  # 钢铁受工业金属和能源成本影响
    "电力": ["原油"],  # 火电受燃料成本影响
    "航空": ["原油"],  # 航油成本
    "航运": ["原油"],  # 燃油成本
}

# 大宗商品名 → 关联的行业关键词（用于模糊匹配股票名称）
COMMODITY_NAME_KEYWORDS = {
    "原油": ["石油", "油气", "油服", "中海油", "中石油", "中石化", "海油"],
    "黄金": ["黄金", "金矿", "贵金属"],
    "铜": ["铜", "有色"],
    "大豆": ["大豆", "粮油", "农业"],
}


def fetch_commodity_prices():
    """获取国际大宗商品期货价格变动（东方财富外盘期货接口）"""
    # 外盘期货代码: 原油CL, 黄金GC, 铜HG, 大豆S
    secids = {
        "原油": "113.CL00Y",
        "黄金": "113.GC00Y",
        "铜": "113.HG00Y",
        "大豆": "113.S00Y",
    }
    result = {}
    try:
        ids_str = ",".join(secids.values())
        resp = _get("http://push2delay.eastmoney.com/api/qt/ulist.np/get",
                     params={"fltt": 2, "invt": 2, "ut": UT,
                             "secids": ids_str,
                             "fields": "f12,f14,f2,f3"},
                     headers={"Referer": "http://quote.eastmoney.com/"},
                     timeout=10)
        data = resp.json()
        items = data.get("data", {}).get("diff", []) if data.get("data") else []
        # 按secid顺序映射
        secid_to_name = {v: k for k, v in secids.items()}
        for item in items:
            code = f"{item.get('f12', '')}"
            name = item.get("f14", "")
            price = item.get("f2", 0)
            chg = item.get("f3", 0)
            # 匹配商品名
            for cn, sid in secids.items():
                if cn in name or code in sid:
                    result[cn] = {"价格": price, "涨跌幅": chg}
                    break
    except Exception:
        pass

    # 备用方案：如果上面没获取到，尝试逐个请求
    if not result:
        for cn, sid in secids.items():
            try:
                resp = _get("http://push2delay.eastmoney.com/api/qt/stock/get",
                            params={"secid": sid, "ut": UT,
                                    "fields": "f43,f170,f14"},
                            headers={"Referer": "http://quote.eastmoney.com/"},
                            timeout=8)
                d = resp.json().get("data", {})
                if d:
                    price = d.get("f43", 0)
                    chg = d.get("f170", 0)
                    if isinstance(price, (int, float)) and price > 0:
                        price = price / 100 if price > 10000 else price
                    if isinstance(chg, (int, float)):
                        chg = chg / 100 if abs(chg) > 100 else chg
                    result[cn] = {"价格": price, "涨跌幅": chg}
            except Exception:
                pass

    return result


def check_commodity_risk(stock_industry, stock_name, commodity_data):
    """
    检查个股是否受大宗商品暴跌影响
    返回 (penalty: int, warning: str)
    """
    if not commodity_data or not (stock_industry or stock_name):
        return 0, ""

    # 通过行业匹配
    linked = set()
    if stock_industry:
        for ind_key, commodities in INDUSTRY_COMMODITY_MAP.items():
            if ind_key in stock_industry or stock_industry in ind_key:
                linked.update(commodities)

    # 通过股票名称匹配
    if stock_name:
        for commodity, keywords in COMMODITY_NAME_KEYWORDS.items():
            for kw in keywords:
                if kw in stock_name:
                    linked.add(commodity)

    if not linked:
        return 0, ""

    worst_penalty = 0
    warnings = []
    for commodity in linked:
        info = commodity_data.get(commodity)
        if not info:
            continue
        chg = info.get("涨跌幅", 0)
        if chg >= -2:
            continue
        if chg <= -7:
            penalty = -35
        elif chg <= -5:
            penalty = -25
        elif chg <= -3:
            penalty = -15
        else:
            penalty = -8
        if penalty < worst_penalty:
            worst_penalty = penalty
        warnings.append(f"{commodity}{chg:+.1f}%")

    if warnings:
        return worst_penalty, f"大宗商品风险({','.join(warnings)})"
    return 0, ""


def check_consecutive_rally(kline):
    """
    检测连续上涨/累计涨幅过大的追高风险
    返回 (penalty: int, warning: str)
    """
    if kline.empty or len(kline) < 10:
        return 0, ""

    # 连涨天数
    consecutive_up = 0
    for i in range(len(kline) - 1, 0, -1):
        if kline.iloc[i]["涨跌幅"] > 0:
            consecutive_up += 1
        else:
            break

    # 10日累计涨幅
    recent_10 = kline.tail(10)
    first_close = recent_10.iloc[0]["收盘"]
    last_close = recent_10.iloc[-1]["收盘"]
    cum_gain = (last_close / first_close - 1) * 100 if first_close > 0 else 0

    penalty = 0
    warnings = []

    if consecutive_up >= 7:
        penalty -= 20; warnings.append(f"连涨{consecutive_up}天")
    elif consecutive_up >= 6:
        penalty -= 15; warnings.append(f"连涨{consecutive_up}天")
    elif consecutive_up >= 5:
        penalty -= 8; warnings.append(f"连涨{consecutive_up}天")

    if cum_gain >= 30:
        penalty -= 25; warnings.append(f"10日涨{cum_gain:.0f}%")
    elif cum_gain >= 20:
        penalty -= 18; warnings.append(f"10日涨{cum_gain:.0f}%")
    elif cum_gain >= 15:
        penalty -= 10; warnings.append(f"10日涨{cum_gain:.0f}%")

    if warnings:
        return penalty, f"追高风险({','.join(warnings)})"
    return 0, ""


def apply_largecap_adjustments(market_cap_yi):
    """
    大盘股(>500亿)专属风控调整
    返回 (threshold_bonus: int, risk_adj: dict or None)
    threshold_bonus: 该股需要额外多少分才能入选
    risk_adj: 风控参数覆盖
    """
    if not market_cap_yi or market_cap_yi <= 0:
        return 0, None

    if market_cap_yi >= 2000:
        return 25, {
            "仓位上限": 20,
            "止损收紧": -1.0,  # 额外收紧1%
            "标签": "超大盘股风控",
        }
    elif market_cap_yi >= 500:
        return 15, {
            "仓位上限": 30,
            "止损收紧": -0.5,
            "标签": "大盘股风控",
        }
    return 0, None


# ============================================================
# 国际市场 & AH溢价 & 换手率深度 & 宏观趋势
# ============================================================

def fetch_global_markets():
    """
    获取全球主要指数隔夜表现（东方财富全球指数接口）
    返回 dict: {指数名: {价格, 涨跌幅}}
    """
    # 全球主要指数secid
    indices = {
        "道琼斯": "100.DJIA",
        "标普500": "100.SPX",
        "纳斯达克": "100.NDX",
        "恒生指数": "100.HSI",
        "日经225": "100.N225",
        "德国DAX": "100.GDAXI",
        "英国富时": "100.FTSE",
        "韩国KOSPI": "100.KS11",
    }
    result = {}
    try:
        ids_str = ",".join(indices.values())
        resp = _get("http://push2delay.eastmoney.com/api/qt/ulist.np/get",
                     params={"fltt": 2, "invt": 2, "ut": UT,
                             "secids": ids_str,
                             "fields": "f12,f14,f2,f3,f4"},
                     headers={"Referer": "http://quote.eastmoney.com/"},
                     timeout=10)
        data = resp.json()
        items = data.get("data", {}).get("diff", []) if data.get("data") else []
        for item in items:
            name = item.get("f14", "")
            chg = item.get("f3", 0)
            price = item.get("f2", 0)
            # 匹配已知指数
            for cn, sid in indices.items():
                if cn[:2] in name or name in cn:
                    result[cn] = {"价格": price, "涨跌幅": chg}
                    break
            else:
                # fallback: 用名称作key
                if name and chg is not None:
                    result[name] = {"价格": price, "涨跌幅": chg}
    except Exception:
        pass

    # 备用：逐个请求
    if len(result) < 3:
        for cn, sid in list(indices.items())[:4]:
            if cn in result:
                continue
            try:
                resp = _get("http://push2delay.eastmoney.com/api/qt/stock/get",
                            params={"secid": sid, "ut": UT,
                                    "fields": "f43,f170,f14"},
                            headers={"Referer": "http://quote.eastmoney.com/"},
                            timeout=8)
                d = resp.json().get("data", {})
                if d:
                    price = d.get("f43", 0)
                    chg = d.get("f170", 0)
                    if isinstance(price, (int, float)) and price > 0:
                        price = price / 100 if price > 10000 else price
                    if isinstance(chg, (int, float)):
                        chg = chg / 100 if abs(chg) > 100 else chg
                    result[cn] = {"价格": price, "涨跌幅": chg}
            except Exception:
                pass

    return result


def calc_global_risk(global_markets):
    """
    根据全球市场表现计算隔夜风险
    返回 (penalty: int, warning: str, detail: dict)
    penalty: 0~-30
    """
    if not global_markets:
        return 0, "", {}

    # 美股权重最大
    us_indices = ["道琼斯", "标普500", "纳斯达克"]
    asia_indices = ["恒生指数", "日经225", "韩国KOSPI"]
    eu_indices = ["德国DAX", "英国富时"]

    def avg_chg(names):
        vals = [global_markets[n]["涨跌幅"] for n in names if n in global_markets]
        return sum(vals) / len(vals) if vals else 0

    us_chg = avg_chg(us_indices)
    asia_chg = avg_chg(asia_indices)
    eu_chg = avg_chg(eu_indices)
    all_chg = avg_chg(list(global_markets.keys()))

    penalty = 0
    warnings = []

    # 美股大跌对A股影响最大
    if us_chg <= -3:
        penalty -= 20; warnings.append(f"美股暴跌{us_chg:.1f}%")
    elif us_chg <= -2:
        penalty -= 12; warnings.append(f"美股大跌{us_chg:.1f}%")
    elif us_chg <= -1:
        penalty -= 5; warnings.append(f"美股下跌{us_chg:.1f}%")

    # 亚太联动
    if asia_chg <= -2:
        penalty -= 8; warnings.append(f"亚太下跌{asia_chg:.1f}%")
    elif asia_chg <= -1:
        penalty -= 3

    # 全球普跌
    if all_chg <= -2:
        penalty -= 5; warnings.append("全球普跌")

    # 美股大涨 → 小幅利好
    if us_chg >= 2:
        penalty += 5

    detail = {"美股": us_chg, "亚太": asia_chg, "欧洲": eu_chg}
    warn_str = f"全球市场风险({', '.join(warnings)})" if warnings else ""
    return max(penalty, -30), warn_str, detail


def fetch_ah_premium():
    """
    获取AH股溢价数据（东方财富AH比价接口）
    返回 dict: {A股代码: {溢价率, H股价格, A股价格, 名称}}
    """
    result = {}
    try:
        resp = _get("http://push2delay.eastmoney.com/api/qt/clist/get",
                     params={
                         "pn": 1, "pz": 200, "po": 1, "np": 1, "fltt": 2,
                         "invt": 2, "fid": "f164", "fs": "b:DLMK0101",
                         "ut": UT,
                         "fields": "f12,f14,f2,f164,f166",
                     },
                     headers={"Referer": "http://data.eastmoney.com/"},
                     timeout=10)
        data = resp.json()
        items = data.get("data", {}).get("diff", []) if data.get("data") else []
        for item in items:
            code = str(item.get("f12", ""))
            name = item.get("f14", "")
            a_price = item.get("f2", 0)
            premium = item.get("f164", 0)  # 溢价率%
            if code and premium is not None:
                result[code] = {
                    "溢价率": premium,
                    "A股价格": a_price,
                    "名称": name,
                }
    except Exception:
        pass
    return result


def check_ah_premium_risk(code, stock_name, ah_data):
    """
    AH股溢价风险：溢价率过高 → H股下跌会带动A股补跌
    返回 (penalty: int, warning: str)
    """
    if not ah_data:
        return 0, ""

    info = ah_data.get(code)
    if not info:
        return 0, ""

    premium = info.get("溢价率", 0)
    if premium >= 150:
        return -20, f"AH溢价{premium:.0f}%(极高风险)"
    elif premium >= 100:
        return -12, f"AH溢价{premium:.0f}%(高)"
    elif premium >= 60:
        return -5, f"AH溢价{premium:.0f}%(偏高)"
    return 0, ""


def analyze_turnover_depth(kline, current_turnover=0):
    """
    换手率深度分析：不只是过滤，结合价格判断主力行为
    - 高换手+下跌 = 出货（利空）
    - 高换手+上涨 = 抢筹（利好）
    - 低换手+上涨 = 缩量上涨（利好）
    - 低换手+下跌 = 阴跌（中性偏空）
    - 换手率突然放大 = 异动
    返回 (score_adj: int, label: str)
    """
    if kline.empty or len(kline) < 5:
        return 0, ""

    latest = kline.iloc[-1]
    chg = latest.get("涨跌幅", 0)

    # 近5日平均换手率（从实时数据获取）
    # 由于腾讯K线不含换手率，我们用成交量变化来推算
    recent_5 = kline.tail(5)
    avg_vol = recent_5["成交量"].mean()
    latest_vol = latest["成交量"]
    vol_ratio = latest_vol / avg_vol if avg_vol > 0 else 1

    # 用传入的实时换手率
    turnover = current_turnover if current_turnover > 0 else 0

    score_adj = 0
    labels = []

    if turnover > 0:
        # 高换手率分析
        if turnover >= 15:
            if chg > 2:
                score_adj += 3; labels.append(f"天量抢筹(换手{turnover:.1f}%)")
            elif chg < -2:
                score_adj -= 8; labels.append(f"天量出货(换手{turnover:.1f}%)")
            else:
                score_adj -= 3; labels.append(f"高换手博弈(换手{turnover:.1f}%)")
        elif turnover >= 8:
            if chg > 1:
                score_adj += 2; labels.append(f"放量上攻(换手{turnover:.1f}%)")
            elif chg < -1:
                score_adj -= 5; labels.append(f"放量下跌(换手{turnover:.1f}%)")

    # 成交量异动
    if vol_ratio >= 3:
        if chg < -1:
            score_adj -= 5; labels.append(f"量比异动{vol_ratio:.1f}+下跌")
        elif chg > 1:
            score_adj += 2; labels.append(f"量比异动{vol_ratio:.1f}+上涨")
    elif vol_ratio <= 0.3 and chg > 0.5:
        score_adj += 3; labels.append("地量上涨(惜售)")

    label = "；".join(labels) if labels else ""
    return score_adj, label


def fetch_global_news():
    """
    获取国际财经新闻（东方财富国际频道）
    返回新闻列表，格式同 fetch_news
    """
    all_news = []

    # 东方财富全球频道
    try:
        resp = _get("https://np-listapi.eastmoney.com/comm/web/getFastNewsList",
                     params={"client": "web", "biz": "web_news_col",
                             "fastColumn": "230", "sortEnd": "",  # 230=全球
                             "pageSize": 30, "req_trace": "t1"},
                     timeout=10)
        data = resp.json()
        for n in data.get("result", {}).get("fastNewsList", []):
            title = n.get("title", "")
            summary = n.get("digest", n.get("summary", ""))
            all_news.append({
                "来源": "东方财富(国际)",
                "时间": n.get("showTime", ""),
                "标题": title,
                "摘要": summary,
                "文本": title + " " + summary,
            })
    except Exception:
        pass

    # 新浪全球
    try:
        resp = _get("https://feed.mix.sina.com.cn/api/roll/get",
                     params={"pageid": "155", "lid": "2526",  # 国际财经
                             "num": 20, "page": 1},
                     timeout=10)
        data = resp.json()
        for item in data.get("result", {}).get("data", []):
            title = item.get("title", "")
            intro = item.get("intro", "")
            all_news.append({
                "来源": "新浪(国际)",
                "时间": item.get("ctime", ""),
                "标题": title,
                "摘要": intro,
                "文本": title + " " + intro,
            })
    except Exception:
        pass

    return all_news


# 国际宏观风险关键词
GLOBAL_RISK_KEYWORDS = {
    # 地缘政治
    "战争": -8, "军事冲突": -8, "导弹": -6, "封锁": -5,
    "制裁": -5, "贸易战": -6, "关税": -4, "脱钩": -4,
    "台海": -7, "台湾海峡": -7, "南海冲突": -6, "南海对峙": -5,
    "领土争端": -5, "主权争议": -4, "军事演习": -4, "实弹演练": -5,
    "核武器": -8, "核试验": -7, "朝鲜": -3, "中东战争": -6,
    "入侵": -7, "空袭": -7, "武装冲突": -7, "军事部署": -4,
    "外交降级": -4, "外交召回": -5, "断交": -6,
    "技术封锁": -5, "实体清单": -5, "投资禁令": -4,
    "联合军演": -3, "航母编队": -3, "无人机袭击": -5,
    # 美联储/货币政策
    "加息": -5, "缩表": -4, "鹰派": -4, "紧缩": -3,
    "美债收益率飙升": -5, "美元走强": -3,
    # 金融风险
    "银行危机": -8, "债务违约": -7, "金融危机": -10, "衰退": -5,
    "崩盘": -6, "暴跌": -4, "恐慌": -5, "黑天鹅": -8,
    # 供应链
    "芯片禁令": -5, "出口管制": -5, "断供": -6,
}

GLOBAL_POSITIVE_KEYWORDS = {
    # 宏观利好
    "降息": 6, "宽松": 4, "鸽派": 4, "刺激": 4,
    "合作": 3, "协议": 3, "缓和": 3,
    # 科技突破
    "突破": 2, "创新": 2, "里程碑": 3,
    # 地缘缓和
    "和平协议": 5, "停火": 5, "和谈": 4, "建交": 4,
    "外交突破": 4, "关系回暖": 3, "对话重启": 3,
    "取消制裁": 5, "关税减免": 4, "贸易协定": 4,
    # 军事利好（国防订单等）
    "国防预算增长": 3, "军费增长": 3, "装备列装": 3,
    # 文化软实力
    "文化出海": 3, "文化输出": 3, "票房纪录": 2, "海外爆红": 2,
}

# 宏观产业趋势 → 受益/受损行业映射
MACRO_TREND_MAP = {
    # 利好
    "降息": {"受益": ["金融", "地产", "消费", "科技"], "权重": 5},
    "芯片": {"受益": ["半导体", "信创", "科技"], "权重": 4},
    "人工智能": {"受益": ["AI", "算力", "软件"], "权重": 4},
    "新能源": {"受益": ["新能源", "光伏", "锂电"], "权重": 3},
    "消费复苏": {"受益": ["消费", "白酒", "零售"], "权重": 3},
    # 利空
    "加息": {"受损": ["地产", "金融", "消费"], "权重": -4},
    "贸易战": {"受损": ["出口", "电子", "纺织"], "权重": -5},
    "芯片禁令": {"受损": ["半导体", "芯片"], "权重": -6},
    # ── 地缘政治 ──
    "台海": {"受益": ["军工", "国防", "航天", "船舶"], "受损": ["航空", "旅游", "半导体"], "权重": 6},
    "南海": {"受益": ["军工", "国防", "船舶", "海洋"], "受损": ["航运", "旅游"], "权重": 5},
    "制裁": {"受益": ["信创", "国产替代", "军工", "安全"], "受损": ["出口", "半导体"], "权重": 5},
    "关税": {"受益": ["内需", "国产替代", "农业"], "受损": ["出口", "电子", "纺织", "家电"], "权重": 4},
    "缓和": {"受益": ["航空", "旅游", "出口", "消费"], "受损": ["军工"], "权重": 3},
    "脱钩": {"受益": ["信创", "国产替代", "军工", "安全"], "受损": ["出口", "外贸"], "权重": 5},
    "技术封锁": {"受益": ["信创", "国产替代", "安全"], "受损": ["半导体", "科技"], "权重": 5},
    "一带一路": {"受益": ["基建", "建筑", "交运", "港口"], "权重": 4},
    "中东": {"受益": ["石油", "军工", "黄金"], "受损": ["航空", "化工"], "权重": 4},
    # ── 军事 ──
    "军费增长": {"受益": ["军工", "国防", "航天", "电子"], "权重": 5},
    "国防预算": {"受益": ["军工", "国防", "航天", "船舶"], "权重": 5},
    "军事演习": {"受益": ["军工", "国防"], "受损": ["航空", "旅游"], "权重": 4},
    "装备列装": {"受益": ["军工", "航天", "电子", "新材料"], "权重": 4},
    "导弹": {"受益": ["军工", "航天", "国防"], "受损": ["航空", "旅游"], "权重": 5},
    "航母": {"受益": ["船舶", "军工", "国防", "钢铁"], "权重": 4},
    "军民融合": {"受益": ["军工", "通信", "电子", "新材料"], "权重": 3},
    "太空": {"受益": ["航天", "卫星", "军工"], "权重": 3},
    "无人机": {"受益": ["军工", "航天", "电子"], "权重": 3},
    "网络安全": {"受益": ["安全", "信创", "软件"], "权重": 4},
    # ── 文化 ──
    "文化产业": {"受益": ["传媒", "影视", "游戏", "文化"], "权重": 3},
    "文化出海": {"受益": ["传媒", "游戏", "影视"], "权重": 3},
    "票房": {"受益": ["影视", "传媒", "院线"], "权重": 3},
    "版权": {"受益": ["传媒", "影视", "出版"], "权重": 3},
    "电竞": {"受益": ["游戏", "传媒", "电竞"], "权重": 3},
    "文旅": {"受益": ["旅游", "酒店", "消费", "文化"], "权重": 3},
    "国潮": {"受益": ["消费", "服装", "食品", "白酒"], "权重": 3},
    "非遗": {"受益": ["文化", "消费", "旅游"], "权重": 2},
    "动漫": {"受益": ["传媒", "影视", "游戏"], "权重": 2},
    "短视频": {"受益": ["传媒", "互联网", "广告"], "权重": 3},
}


def analyze_global_news(global_news_list):
    """
    分析国际新闻对A股的影响
    返回 (risk_score: int, trend_industries: dict, key_headlines: list)
    risk_score: -30 到 +10
    trend_industries: {行业: 加减分}
    """
    risk_score = 0
    trend_industries = {}  # 行业 -> 累计分
    key_headlines = []

    for news in global_news_list:
        text = news["文本"]

        # 风险关键词
        for kw, pts in GLOBAL_RISK_KEYWORDS.items():
            if kw in text:
                risk_score += pts
                key_headlines.append(f"[国际] {news['标题'][:40]}")
                break

        # 正面关键词
        for kw, pts in GLOBAL_POSITIVE_KEYWORDS.items():
            if kw in text:
                risk_score += pts
                break

        # 产业趋势匹配
        for kw, info in MACRO_TREND_MAP.items():
            if kw in text:
                weight = info["权重"]
                for ind in info.get("受益", []):
                    trend_industries[ind] = trend_industries.get(ind, 0) + weight
                for ind in info.get("受损", []):
                    trend_industries[ind] = trend_industries.get(ind, 0) + weight  # weight已经是负数

    risk_score = max(-30, min(risk_score, 10))
    return risk_score, trend_industries, list(dict.fromkeys(key_headlines))[:8]


def check_macro_trend_fit(stock_industry, stock_name, trend_industries):
    """
    检查个股是否契合当前宏观产业趋势
    返回 (score_adj: int, label: str)
    """
    if not trend_industries or not (stock_industry or stock_name):
        return 0, ""

    best_score = 0
    best_label = ""

    for ind, pts in trend_industries.items():
        matched = False
        if stock_industry and (ind in stock_industry or stock_industry in ind):
            matched = True
        if stock_name and ind in stock_name:
            matched = True
        if matched:
            if abs(pts) > abs(best_score):
                best_score = pts
                if pts > 0:
                    best_label = f"契合趋势({ind}+{pts})"
                else:
                    best_label = f"逆势行业({ind}{pts})"

    return max(-10, min(best_score, 10)), best_label


# ============================================================
# 市场级熔断：不该买的时候不买
# ============================================================

def market_go_nogo(money_effect=None, sentiment_data=None, limit_up=0, limit_down=0):
    """
    市场级过滤，返回 (should_trade: bool, reason: str, severity: int)
    severity: 0=正常, 1=谨慎, 2=不建议, 3=禁止
    """
    if money_effect is None:
        money_effect = calc_money_effect()

    reasons = []
    severity = 0

    # Rule 1: 赚钱效应太差
    up_ratio = money_effect.get("今日上涨比例", 0.5)
    bad_days = money_effect.get("连续下跌天数", 0)
    if up_ratio < 0.25:
        severity = max(severity, 3); reasons.append(f"赚钱效应极差({up_ratio:.0%})")
    elif up_ratio < 0.35 and bad_days >= 2:
        severity = max(severity, 2); reasons.append(f"连续{bad_days}天弱势+赚钱效应{up_ratio:.0%}")
    elif up_ratio < 0.35:
        severity = max(severity, 1); reasons.append(f"赚钱效应偏差({up_ratio:.0%})")

    # Rule 2: 大盘大跌
    today_chg = money_effect.get("今日大盘涨幅", 0)
    if today_chg < -2.0:
        severity = max(severity, 3); reasons.append(f"大盘暴跌{today_chg:.1f}%")
    elif today_chg < -1.5:
        severity = max(severity, 2); reasons.append(f"大盘大跌{today_chg:.1f}%")
    elif today_chg < -1.0:
        severity = max(severity, 1); reasons.append(f"大盘下跌{today_chg:.1f}%")

    # Rule 3: 跌停 > 涨停
    if limit_down > limit_up and limit_down > 30:
        severity = max(severity, 2); reasons.append(f"跌停{limit_down}>涨停{limit_up}")
    elif limit_down > limit_up:
        severity = max(severity, 1); reasons.append(f"跌停{limit_down}>涨停{limit_up}")

    # Rule 4: 连续下跌
    if bad_days >= 3:
        severity = max(severity, 2); reasons.append(f"大盘连跌{bad_days}天")

    if not reasons:
        return True, "市场环境正常", 0

    should_trade = severity < 2
    return should_trade, "；".join(reasons), severity


# ============================================================
# 信号检测 → 分离信号与评分
# ============================================================

def detect_signals(df, capital_info=None):
    """
    纯信号检测，返回每个信号是否触发 + 数值特征。
    不含评分逻辑，评分由 apply_weights() 完成。
    """
    if len(df) < 30:
        return {}

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) >= 3 else prev

    signals = {}

    # === MACD ===
    signals["MACD_金叉"] = (latest["DIF"] > latest["DEA"] and prev["DIF"] <= prev["DEA"])
    signals["MACD_红柱放大"] = (latest["DIF"] > latest["DEA"] and latest["MACD柱"] > prev["MACD柱"] > 0)
    signals["MACD_即将金叉"] = (latest["DIF"] < latest["DEA"] and
                              (latest["DIF"] - latest["DEA"]) > (prev["DIF"] - prev["DEA"]))
    signals["MACD_红柱"] = (latest["MACD柱"] > 0)

    # === KDJ ===
    signals["KDJ_超卖金叉"] = (latest["K"] > latest["D"] and prev["K"] <= prev["D"] and latest["K"] < 50)
    signals["KDJ_J超卖"] = (latest["J"] < 20)
    signals["KDJ_多头"] = (latest["K"] > latest["D"] and latest["K"] < 80)

    # === RSI ===
    rsi = latest.get("RSI6", np.nan)
    rsi14 = latest.get("RSI14", np.nan)
    prev_rsi = prev.get("RSI6", np.nan) if not pd.isna(prev.get("RSI6", np.nan)) else rsi
    # ★ RSI回升需要确认：RSI6在回升区间 + RSI比前日上升（动量确认）
    signals["RSI_回升"] = (not pd.isna(rsi) and 30 <= rsi <= 50
                          and not pd.isna(prev_rsi) and rsi > prev_rsi)
    signals["RSI_超卖"] = (not pd.isna(rsi) and 20 <= rsi < 30)
    signals["RSI_强势"] = (not pd.isna(rsi) and 50 < rsi <= 70)
    signals["RSI_极度超卖"] = (not pd.isna(rsi) and rsi < 20)
    signals["RSI_偏高"] = (not pd.isna(rsi) and rsi > 70)
    # ★ RSI14趋势确认标记（用于其他信号增强）
    signals["RSI14_超卖确认"] = (not pd.isna(rsi14) and rsi14 < 35)
    signals["RSI值"] = rsi if not pd.isna(rsi) else 50

    # === 均线 ===
    ma20 = latest.get("MA20", np.nan)
    has_ma = not pd.isna(ma20)
    signals["MA_多头排列"] = (has_ma and latest["MA5"] > latest["MA10"] > ma20)
    signals["MA_短期多头"] = (has_ma and latest["MA5"] > latest["MA10"] and not signals["MA_多头排列"])
    signals["MA_突破MA5"] = (has_ma and latest["收盘"] > latest["MA5"] and prev["收盘"] <= prev["MA5"])
    signals["MA_站上MA5"] = (has_ma and latest["收盘"] > latest["MA5"])
    signals["MA_弱势"] = (has_ma and latest["收盘"] <= latest["MA5"])

    # === 布林带 ===
    boll_lower = latest.get("BOLL下", np.nan)
    has_boll = not pd.isna(boll_lower)
    signals["BOLL_触及下轨"] = (has_boll and latest["收盘"] <= boll_lower * 1.01)
    signals["BOLL_中轨下方"] = (has_boll and latest["收盘"] < latest["BOLL中"] and not signals["BOLL_触及下轨"])

    # === 量价 ===
    vr = latest.get("量比计算", np.nan)
    chg = latest.get("涨跌幅", 0)
    has_vr = not pd.isna(vr)
    signals["量价_放量上涨"] = (has_vr and 1.5 <= vr <= 3.0 and chg > 0)
    signals["量价_量升价涨"] = (has_vr and 1.2 <= vr <= 5.0 and chg > 0 and not signals["量价_放量上涨"])
    signals["量价_放量横盘"] = (has_vr and vr >= 1.5 and -1 <= chg <= 0.5)
    signals["量价_缩量上涨"] = (has_vr and 0.8 <= vr < 1.5 and chg > 0)
    signals["量价_严重缩量"] = (has_vr and vr < 0.5)
    signals["量比值"] = vr if has_vr else 1.0
    signals["涨跌幅"] = chg

    # === K线形态 ===
    body = abs(latest["收盘"] - latest["开盘"])
    lower_shadow = min(latest["开盘"], latest["收盘"]) - latest["最低"]
    upper_shadow = latest["最高"] - max(latest["开盘"], latest["收盘"])
    total_range = latest["最高"] - latest["最低"]

    signals["形态_锤子线"] = (total_range > 0 and lower_shadow > body * 2 and lower_shadow > upper_shadow * 2)
    signals["形态_阳包阴"] = (total_range > 0 and latest["收盘"] > latest["开盘"]
                            and prev["收盘"] < prev["开盘"]
                            and latest["收盘"] > prev["开盘"] and latest["开盘"] < prev["收盘"])
    signals["形态_早晨之星"] = (total_range > 0 and len(df) >= 3
                             and prev2["收盘"] < prev2["开盘"]
                             and abs(prev["收盘"] - prev["开盘"]) < body * 0.3
                             and latest["收盘"] > latest["开盘"])
    signals["形态_阳线"] = (total_range > 0 and latest["收盘"] > latest["开盘"])

    # === 资金面 ===
    if capital_info:
        net_pct = capital_info.get("主力净流入占比", 0)
        signals["资金_大幅流入"] = (net_pct > 10)
        signals["资金_明显流入"] = (5 < net_pct <= 10)
        signals["资金_温和流入"] = (2 < net_pct <= 5)
        signals["资金_小幅流入"] = (0 < net_pct <= 2)
        signals["资金_流出"] = (net_pct < -2)
        signals["主力净流入占比"] = net_pct
    else:
        signals["主力净流入占比"] = 0

    # ★ 底部背离检测：价格创新低但RSI/MACD不创新低 → 强烈反转信号
    signals["底部背离"] = False
    if len(df) >= 20:
        recent_20 = df.tail(20)
        price_low_idx = recent_20["收盘"].idxmin()
        # 检查最近5天价格是否接近20日新低
        recent_5_low = df.tail(5)["收盘"].min()
        low_20 = recent_20["收盘"].min()
        near_low = (recent_5_low <= low_20 * 1.02)  # 价格在20日低点2%以内
        if near_low and not pd.isna(rsi):
            # 价格接近新低但RSI比上次低点时更高 → 底部背离
            rsi_at_price_low = recent_20.loc[price_low_idx].get("RSI6", rsi)
            if not pd.isna(rsi_at_price_low) and rsi > rsi_at_price_low + 3:
                signals["底部背离"] = True

    # ★ 量价背离检测：价格下跌但成交量萎缩 → 抛压减轻
    signals["量价背离_抛压衰竭"] = False
    if len(df) >= 10:
        recent_10 = df.tail(10)
        price_falling = recent_10["收盘"].iloc[-1] < recent_10["收盘"].iloc[0]
        vol_shrinking = recent_10["成交量"].iloc[-3:].mean() < recent_10["成交量"].iloc[:3].mean() * 0.7
        if price_falling and vol_shrinking:
            signals["量价背离_抛压衰竭"] = True

    return signals


# 默认权重（后续会被校准回测覆盖）
# ★ 基于回测数据修正：低胜率信号(<49%)降权/归零，高胜率信号(>53%)提权
DEFAULT_WEIGHTS = {
    # MACD — 互斥，取最高
    "MACD_金叉": 12, "MACD_红柱放大": 3, "MACD_即将金叉": 8, "MACD_红柱": 1,
    # KDJ — 互斥
    "KDJ_超卖金叉": 12, "KDJ_J超卖": 14, "KDJ_多头": 2,
    # RSI — 互斥
    "RSI_回升": 12, "RSI_超卖": 10, "RSI_强势": 3, "RSI_极度超卖": 14, "RSI_偏高": 0,
    # 均线 — 互斥
    "MA_多头排列": 3, "MA_短期多头": 3, "MA_突破MA5": 10, "MA_站上MA5": 1,
    # 布林 — 互斥
    "BOLL_触及下轨": 15, "BOLL_中轨下方": 6,
    # 量价 — 互斥（★放量上涨回测胜率仅46%，大幅降权）
    "量价_放量上涨": 4, "量价_量升价涨": 3, "量价_放量横盘": 3, "量价_缩量上涨": 5, "量价_严重缩量": 5,
    # 形态 — 互斥（★阳包阴回测胜率仅47%，降权）
    "形态_阳包阴": 2, "形态_锤子线": 3, "形态_早晨之星": 3, "形态_阳线": 1,
    # 资金 — 互斥
    "资金_大幅流入": 30, "资金_明显流入": 24, "资金_温和流入": 18, "资金_小幅流入": 10,
}

# 互斥组：同组内只取第一个触发的信号
SIGNAL_GROUPS = {
    "MACD": ["MACD_金叉", "MACD_红柱放大", "MACD_即将金叉", "MACD_红柱"],
    "KDJ": ["KDJ_超卖金叉", "KDJ_J超卖", "KDJ_多头"],
    "RSI": ["RSI_极度超卖", "RSI_超卖", "RSI_回升", "RSI_强势", "RSI_偏高"],
    "MA": ["MA_多头排列", "MA_短期多头", "MA_突破MA5", "MA_站上MA5"],
    "BOLL": ["BOLL_触及下轨", "BOLL_中轨下方"],
    "量价": ["量价_放量上涨", "量价_量升价涨", "量价_放量横盘", "量价_缩量上涨", "量价_严重缩量"],
    "形态": ["形态_阳包阴", "形态_锤子线", "形态_早晨之星", "形态_阳线"],
    "资金": ["资金_大幅流入", "资金_明显流入", "资金_温和流入", "资金_小幅流入"],
}


def apply_weights(signals, weights=None):
    """用权重表给信号打分，互斥组内只取最高分"""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    score = 0
    fired_signals = []
    for group_name, group_signals in SIGNAL_GROUPS.items():
        for sig in group_signals:
            if signals.get(sig):
                w = weights.get(sig, 0)
                score += w
                if w > 0:
                    fired_signals.append(sig)
                break  # 互斥：只取第一个
    return score, fired_signals


# ============================================================
# 多维度增强分析：大盘趋势/周线/历史形态/日历效应
# ============================================================

def calc_market_regime():
    """
    大盘趋势研判：上证指数在MA20/MA60之上还是之下
    返回 (regime: str, score_adj: int, label: str)
    regime: 'bull' / 'neutral' / 'bear'
    """
    try:
        kl = fetch_kline("000001", days=80)
        if kl.empty or len(kl) < 60:
            return "neutral", 0, ""
        kl = calc_all_indicators(kl)
        latest = kl.iloc[-1]
        close = latest["收盘"]
        ma20 = latest.get("MA20", close)
        ma5 = latest.get("MA5", close)
        ma10 = latest.get("MA10", close)
        # 计算60日均线
        if len(kl) >= 60:
            ma60 = kl["收盘"].iloc[-60:].mean()
        else:
            ma60 = close
        above_ma20 = close > ma20
        above_ma60 = close > ma60
        short_bull = ma5 > ma10

        # 计算大盘RSI
        rsi = latest.get("RSI6", 50)

        if above_ma20 and above_ma60 and short_bull:
            return "bull", 10, "大盘多头(MA20+MA60上方)"
        elif above_ma20 and short_bull:
            return "bull", 5, "大盘偏多(MA20上方)"
        elif not above_ma20 and not above_ma60 and not short_bull:
            return "bear", -15, "大盘空头(MA20+MA60下方)"
        elif not above_ma20 and not short_bull:
            return "bear", -10, "大盘偏空(MA20下方)"
        else:
            return "neutral", 0, "大盘震荡"
    except Exception:
        return "neutral", 0, ""


def calc_weekly_trend(df):
    """
    周线趋势确认：将日线重采样为周线，判断周级趋势
    返回 (score_adj: int, label: str)
    """
    if len(df) < 30:
        return 0, ""
    try:
        # 重采样为周线
        wdf = df.copy()
        wdf.index = pd.to_datetime(wdf["日期"])
        weekly = wdf.resample("W-FRI").agg({
            "开盘": "first", "最高": "max", "最低": "min",
            "收盘": "last", "成交量": "sum"
        }).dropna()
        if len(weekly) < 5:
            return 0, ""
        weekly["MA5"] = weekly["收盘"].rolling(5).mean()
        weekly["MA10"] = weekly["收盘"].rolling(10).mean()
        latest_w = weekly.iloc[-1]
        prev_w = weekly.iloc[-2]
        ma5w = latest_w.get("MA5", np.nan)
        ma10w = latest_w.get("MA10", np.nan)
        if pd.isna(ma5w) or pd.isna(ma10w):
            return 0, ""
        # 周线多头：周MA5>MA10 且 本周收涨
        weekly_bull = ma5w > ma10w and latest_w["收盘"] > prev_w["收盘"]
        # 周线空头：周MA5<MA10 且 本周收跌
        weekly_bear = ma5w < ma10w and latest_w["收盘"] < prev_w["收盘"]
        if weekly_bull:
            return 8, "周线多头"
        elif weekly_bear:
            return -10, "周线空头"
        else:
            return 0, ""
    except Exception:
        return 0, ""


def classify_trend_context(df):
    """
    历史趋势形态识别：上升趋势回调(最佳) / 下降趋势反弹(最差) / 横盘
    返回 (score_adj: int, label: str)
    """
    if len(df) < 60:
        return 0, ""
    close = df["收盘"].values
    latest_close = close[-1]

    # 20日和60日均线斜率
    ma20_now = np.mean(close[-20:])
    ma20_10ago = np.mean(close[-30:-10]) if len(close) >= 30 else ma20_now
    ma20_rising = ma20_now > ma20_10ago

    # 距60日高点的回撤
    high_60 = np.max(close[-60:])
    low_60 = np.min(close[-60:])
    drawdown = (high_60 - latest_close) / high_60 if high_60 > 0 else 0
    recovery = (latest_close - low_60) / low_60 if low_60 > 0 else 0

    if ma20_rising and drawdown < 0.15:
        # 上升趋势回调：MA20上行，距高点<15%
        return 10, "上升回调(最佳形态)"
    elif ma20_rising and drawdown < 0.25:
        return 5, "上升趋势中"
    elif not ma20_rising and drawdown > 0.25:
        # 下降趋势反弹：MA20下行，距高点>25%
        return -10, "下降反弹(警惕)"
    elif not ma20_rising and drawdown > 0.15:
        return -5, "弱势整理"
    else:
        return 0, "横盘整理"


def calc_calendar_adjustment():
    """
    日历/季节效应调整（A股特有）
    返回 (score_adj: int, label: str)
    """
    from datetime import datetime
    now = datetime.now()
    weekday = now.weekday()  # 0=周一
    month = now.month
    day = now.day

    adj = 0
    labels = []

    # 周几效应（T+1意味着今天买明天卖）
    if weekday == 4:  # 周五买入 → 周一卖（持过周末风险大）
        adj -= 8
        labels.append("周五(持周末风险)")
    elif weekday == 0:  # 周一买入 → 周二卖
        adj -= 3
        labels.append("周一(开盘不稳)")
    elif weekday in (1, 2):  # 周二/三最佳
        adj += 3
        labels.append("周中(最优T+1窗口)")

    # 月份效应
    if month == 1 or (month == 2 and day <= 15):
        adj += 3; labels.append("年初效应(偏多)")
    elif month in (3, 4):
        adj += 2; labels.append("春季行情")
    elif month in (7, 8):
        adj -= 3; labels.append("暑期淡季")

    # 两会效应（3月初）
    if month == 3 and day <= 15:
        adj += 3; labels.append("两会行情")

    # 国庆/春节前（减仓期）
    if (month == 9 and day >= 25) or (month == 1 and day >= 20):
        adj -= 5; labels.append("长假前(资金撤离)")

    # 年末效应（机构调仓）
    if month == 12 and day >= 20:
        adj -= 3; labels.append("年末调仓")

    # 季报窗口（业绩预告期偏多）
    if (month == 1 and 10 <= day <= 31) or (month == 4 and 1 <= day <= 20):
        adj += 2; labels.append("业绩窗口")

    return adj, "；".join(labels) if labels else ""


# ── 高胜率信号白名单（回测胜率>53%的信号 + 逻辑强信号）──
HIGH_WINRATE_SIGNALS = {
    "BOLL_触及下轨",     # 59.9%
    "KDJ_J超卖",         # 54.5%
    "RSI_回升",           # 54.0%（现已加动量确认）
    "RSI_极度超卖",       # 53.4%
    "RSI_超卖",           # 53.0%
    "MA_突破MA5",         # 52.9%
    "BOLL_中轨下方",      # 52.9%
    "MACD_即将金叉",      # 52.1%
    "KDJ_超卖金叉",       # 51.6%
    "量价_严重缩量",      # 51.4%
    "底部背离",           # 新增：底部背离是经典强反转信号
    "量价背离_抛压衰竭",  # 新增：抛压衰竭是底部确认信号
}

# ── 低胜率信号（回测<49%，不应给正分）──
LOW_WINRATE_SIGNALS = {
    "KDJ_多头",           # 49.4%
    "形态_阳线",          # 48.4%
    "MACD_红柱",          # 48.5%
    "MACD_红柱放大",      # 48.1%
    "MA_多头排列",        # 48.1%
    "MA_站上MA5",         # 48.1%
    "RSI_偏高",           # 47.4%（实际应为看空信号）
    "RSI_强势",           # 48.1%
    "形态_阳包阴",        # 47.5%
    "形态_锤子线",        # 47.6%
    "量价_放量上涨",      # 45.8%（★最容易误导的信号之一）
    "量价_放量横盘",      # 42.9%（最差信号）
    "量价_量升价涨",      # 46.8%
    "MACD_金叉",          # 47.7%
    "MA_短期多头",        # 49.3%
}


def calc_signal_quality(fired_signals):
    """
    计算信号质量等级
    返回 (quality: str, high_count: int, low_count: int)
    quality: 'A' (>=2个高胜率) / 'B' (1个高胜率) / 'C' (无高胜率)
    """
    high = sum(1 for s in fired_signals if s in HIGH_WINRATE_SIGNALS)
    low = sum(1 for s in fired_signals if s in LOW_WINRATE_SIGNALS)
    if high >= 2:
        return 'A', high, low
    elif high >= 1:
        return 'B', high, low
    else:
        return 'C', high, low


# ============================================================
# 综合评分引擎 v3.2 — 信号驱动 + 胜率优先
# ============================================================

def evaluate_signals_v2(df, capital_info=None, sector_score=0, sentiment_score=0,
                        fundamental_score=0, fundamental_details=None,
                        extra_score=0, extra_details=None,
                        chip_data=None, tail_flow=None, leader_score=0, leader_label="",
                        research_data=None, calibrated_weights=None, golden_combos=None,
                        commodity_penalty=0, rally_penalty=0,
                        global_risk=0, ah_penalty=0, turnover_adj=0, macro_adj=0,
                        market_regime_adj=0, weekly_trend_adj=0,
                        trend_context_adj=0, calendar_adj=0):
    """
    综合评分，十五大维度：
    技术面(信号驱动) + 资金面 + 量价 + 板块 + 情绪 + 形态
    + 基本面 + 聪明钱 + 筹码 + 尾盘资金 + 板块龙头 + 机构调研
    """
    if len(df) < 30:
        return 0, {}, ""

    latest = df.iloc[-1]
    details = {}
    reasons = []

    # === 信号检测 + 加权评分 ===
    signals = detect_signals(df, capital_info)
    tech_score, fired = apply_weights(signals, calibrated_weights)
    score = tech_score

    # ★ 底部背离额外加分（强烈反转信号）
    if signals.get("底部背离"):
        score += 15
        fired.append("底部背离")

    # ★ RSI14超卖确认：当RSI6超卖 + RSI14也超卖 → 额外加分
    if signals.get("RSI14_超卖确认") and any(s in fired for s in ["RSI_极度超卖", "RSI_超卖"]):
        score += 5

    # ★ 量价背离加分
    if signals.get("量价背离_抛压衰竭"):
        score += 8
        fired.append("量价背离_抛压衰竭")

    # === 信号质量评估 ===
    sig_quality, high_count, low_count = calc_signal_quality(fired)

    # ★ 低胜率信号惩罚加强
    if low_count >= 3 and high_count == 0:
        score -= 15  # 全是低胜率信号，重扣（从-10加到-15）
    elif low_count >= 2 and high_count == 0:
        score -= 8   # ★ 新增：2个低胜率也要扣分

    # 高胜率信号奖励
    if high_count >= 3:
        score += 12  # 3个以上高胜率信号共振，强烈看多
    elif high_count >= 2:
        score += 6   # 2个高胜率信号

    # === 黄金组合加分（加大力度）===
    combo_bonus = 0
    matched_combo = ""
    if golden_combos and fired:
        fired_set = set(fired)
        for combo in golden_combos:
            combo_sigs = set(combo.get("signals", []))
            if combo_sigs.issubset(fired_set):
                wr = combo.get("win_rate", 0)
                # 加大黄金组合奖励：70%→15, 75%→20, 80%→25
                bonus = int((wr - 0.55) * 100)
                if bonus > combo_bonus:
                    combo_bonus = min(bonus, 30)
                    matched_combo = f"黄金组合({wr:.0%})"
    score += combo_bonus

    # 生成详情和理由
    sig_label_map = {
        "MACD_金叉": ("MACD", "当日金叉", True), "MACD_红柱放大": ("MACD", "红柱放大", False),
        "MACD_即将金叉": ("MACD", "即将金叉", False), "MACD_红柱": ("MACD", "红柱", False),
        "KDJ_超卖金叉": ("KDJ", "超卖金叉", True), "KDJ_J超卖": ("KDJ", f"J超卖({latest['J']:.0f})", True),
        "KDJ_多头": ("KDJ", "多头", False),
        "RSI_回升": ("RSI", f"回升({signals.get('RSI值', 0):.0f})", False),
        "RSI_超卖": ("RSI", f"超卖({signals.get('RSI值', 0):.0f})", True),
        "RSI_强势": ("RSI", f"强势({signals.get('RSI值', 0):.0f})", False),
        "RSI_极度超卖": ("RSI", f"极度超卖({signals.get('RSI值', 0):.0f})", True),
        "RSI_偏高": ("RSI", f"偏高({signals.get('RSI值', 0):.0f})", False),
        "MA_多头排列": ("均线", "多头排列", True), "MA_短期多头": ("均线", "短期多头", False),
        "MA_突破MA5": ("均线", "突破MA5", True), "MA_站上MA5": ("均线", "站上MA5", False),
        "BOLL_触及下轨": ("布林", "触及下轨", True), "BOLL_中轨下方": ("布林", "中轨下方", False),
        "量价_放量上涨": ("量价", f"放量上涨(量比{signals.get('量比值', 0):.1f})", True),
        "量价_量升价涨": ("量价", f"量升价涨(量比{signals.get('量比值', 0):.1f})", False),
        "量价_放量横盘": ("量价", f"放量横盘(量比{signals.get('量比值', 0):.1f})", False),
        "量价_缩量上涨": ("量价", f"缩量上涨(量比{signals.get('量比值', 0):.1f})", False),
        "量价_严重缩量": ("量价", f"严重缩量(量比{signals.get('量比值', 0):.1f})", False),
        "形态_阳包阴": ("形态", "阳包阴", True), "形态_锤子线": ("形态", "锤子线", True),
        "形态_早晨之星": ("形态", "早晨之星", True), "形态_阳线": ("形态", "阳线", False),
        "资金_大幅流入": ("资金", f"大幅流入({signals.get('主力净流入占比', 0):.1f}%)", True),
        "资金_明显流入": ("资金", f"明显流入({signals.get('主力净流入占比', 0):.1f}%)", True),
        "资金_温和流入": ("资金", f"温和流入({signals.get('主力净流入占比', 0):.1f}%)", True),
        "资金_小幅流入": ("资金", f"小幅流入({signals.get('主力净流入占比', 0):.1f}%)", False),
        "底部背离": ("背离", "底部背离(强反转)", True),
        "量价背离_抛压衰竭": ("背离", "抛压衰竭", True),
    }
    # 未触发的维度设为 "-"
    seen_dims = set()
    for sig in fired:
        if sig in sig_label_map:
            dim, label, is_reason = sig_label_map[sig]
            details[dim] = label
            seen_dims.add(dim)
            if is_reason:
                reasons.append(label)
    for dim in ["MACD", "KDJ", "RSI", "均线", "布林", "量价", "形态", "资金"]:
        if dim not in seen_dims:
            details[dim] = "-" if dim != "资金" else ("无数据" if not capital_info else f"流出({signals.get('主力净流入占比', 0):.1f}%)")

    # === 板块热度 (0-15) ===
    sector_s = min(sector_score, 15)
    if sector_s >= 12:
        details["板块"] = "热门板块"; reasons.append("热门板块")
    elif sector_s >= 6:
        details["板块"] = "板块偏强"
    else:
        details["板块"] = "板块一般"
    score += sector_s

    # === 市场情绪 (0-15) ===
    sent_s = min(sentiment_score, 15)
    if sent_s >= 10:
        details["情绪"] = "大盘偏强"
    elif sent_s >= 5:
        details["情绪"] = "大盘中性"
    else:
        details["情绪"] = "大盘偏弱"
    score += sent_s

    # === 基本面 (0-50) ===
    fund_s = min(fundamental_score, 50)
    score += fund_s
    if fundamental_details:
        details.update(fundamental_details)
    if fund_s >= 35:
        if "基本面" not in details: details["基本面"] = "优良"
    elif fund_s >= 20:
        if "基本面" not in details: details["基本面"] = "一般"
    else:
        if "基本面" not in details: details["基本面"] = "偏弱"

    # === 聪明钱 (0-80) ===
    ext_s = min(max(extra_score, 0), 80)
    score += ext_s
    if extra_details:
        details.update(extra_details)

    # === 筹码分析 (0-8) ===
    chip_s = 0
    if chip_data:
        profit = chip_data.get("获利盘")
        if profit is not None:
            if profit < 15:
                chip_s = 8; details["筹码"] = f"获利盘{profit:.0f}%(低位,惜售)"
                reasons.append(f"获利盘仅{profit:.0f}%")
            elif profit < 40:
                chip_s = 6; details["筹码"] = f"获利盘{profit:.0f}%(安全)"
            elif profit < 70:
                chip_s = 3; details["筹码"] = f"获利盘{profit:.0f}%(中位)"
            else:
                chip_s = 0; details["筹码"] = f"获利盘{profit:.0f}%(高位抛压)"
        else:
            details["筹码"] = "无数据"
    else:
        details["筹码"] = "-"
    score += chip_s

    # === 尾盘资金 (0-8) ===
    tail_s = 0
    if tail_flow:
        tail_main = tail_flow.get("尾盘主力净流入", 0)
        tail_pct = tail_flow.get("尾盘占比", 0)
        if tail_main > 0 and tail_pct > 50:
            tail_s = 8; details["尾盘"] = f"尾盘抢筹({tail_pct:.0f}%集中)"
            reasons.append("尾盘主力抢筹")
        elif tail_main > 0 and tail_pct > 30:
            tail_s = 6; details["尾盘"] = f"尾盘流入({tail_pct:.0f}%)"
        elif tail_main > 0:
            tail_s = 3; details["尾盘"] = "尾盘小幅流入"
        elif tail_main < 0 and abs(tail_pct) > 50:
            tail_s = -3; details["尾盘"] = f"尾盘出逃({tail_pct:.0f}%)"
        else:
            tail_s = 0; details["尾盘"] = "尾盘中性"
    else:
        details["尾盘"] = "-"
    score += tail_s

    # === 板块龙头 (0-8) ===
    if leader_score > 0:
        score += leader_score
        details["龙头"] = leader_label
        if leader_score >= 6:
            reasons.append(leader_label)
    else:
        details["龙头"] = "-"

    # === 机构调研 (0-6) ===
    research_s = 0
    if research_data:
        visits = research_data.get("调研次数", 0)
        top_inst = research_data.get("保险QFII", 0) + research_data.get("基金", 0)
        if visits >= 5 and top_inst >= 2:
            research_s = 6; details["调研"] = f"{visits}次(含{top_inst}家机构)"
            reasons.append(f"机构{visits}次调研")
        elif visits >= 3:
            research_s = 4; details["调研"] = f"{visits}次"
        elif visits >= 1:
            research_s = 2; details["调研"] = f"{visits}次"
        else:
            details["调研"] = "-"
    else:
        details["调研"] = "-"
    score += research_s

    if combo_bonus > 0:
        details["组合"] = matched_combo
        details["黄金组合匹配"] = True
        reasons.append(matched_combo)
    else:
        details["黄金组合匹配"] = False

    # === 信号质量标签 ===
    if sig_quality == 'A':
        details["信号质量"] = f"A级({high_count}个高胜率)"
    elif sig_quality == 'B':
        details["信号质量"] = f"B级({high_count}个高胜率)"
    else:
        details["信号质量"] = f"C级(无高胜率信号)"

    # === 大宗商品风险扣分 ===
    if commodity_penalty and commodity_penalty < 0:
        score += commodity_penalty
        details["商品风险"] = f"扣{abs(commodity_penalty)}分"

    # === 追高风险扣分 ===
    if rally_penalty and rally_penalty < 0:
        score += rally_penalty
        details["追高风险"] = f"扣{abs(rally_penalty)}分"

    # === 国际市场隔夜风险 ===
    if global_risk and global_risk < 0:
        score += global_risk
        details["全球市场"] = f"扣{abs(global_risk)}分"
    elif global_risk and global_risk > 0:
        score += global_risk
        details["全球市场"] = f"加{global_risk}分"

    # === AH溢价风险 ===
    if ah_penalty and ah_penalty < 0:
        score += ah_penalty
        details["AH溢价"] = f"扣{abs(ah_penalty)}分"

    # === 换手率深度分析 ===
    if turnover_adj:
        score += turnover_adj
        if turnover_adj > 0:
            details["换手分析"] = f"加{turnover_adj}分"
        else:
            details["换手分析"] = f"扣{abs(turnover_adj)}分"

    # === 宏观趋势匹配 ===
    if macro_adj:
        score += macro_adj
        if macro_adj > 0:
            details["宏观趋势"] = f"契合+{macro_adj}"
        else:
            details["宏观趋势"] = f"逆势{macro_adj}"

    # ★ 大盘趋势研判（熊市重扣，牛市小加 — 不对称但比之前好）
    if market_regime_adj < 0:
        adj = max(market_regime_adj, -12)  # ★ 熊市扣分上限从-8提到-12
        score += adj
        details["大盘趋势"] = f"偏空({adj})"
    elif market_regime_adj > 0:
        adj = min(market_regime_adj, 5)  # ★ 牛市也给加分，但上限5
        score += adj
        details["大盘趋势"] = f"偏多(+{adj})"

    # ★ 周线趋势确认（空头重扣，多头小加）
    if weekly_trend_adj < 0:
        adj = max(weekly_trend_adj, -8)  # ★ 从-5加到-8
        score += adj
        details["周线"] = f"空头({adj})"
    elif weekly_trend_adj > 0:
        adj = min(weekly_trend_adj, 3)
        score += adj
        details["周线"] = f"多头(+{adj})"

    # ★ 历史趋势形态（下降反弹重扣）
    if trend_context_adj < 0:
        adj = max(trend_context_adj, -8)  # ★ 从-5加到-8
        score += adj
        details["趋势形态"] = f"不利({adj})"
    elif trend_context_adj > 0:
        adj = min(trend_context_adj, 5)
        score += adj
        details["趋势形态"] = f"上升回调(+{adj})"

    # === 日历效应（周五扣分，其他不动）===
    if calendar_adj < -5:
        score += -5  # 最多扣5分（周五）
        details["日历"] = "周五(-5)"
    elif calendar_adj > 0:
        details["日历"] = "有利"

    # ★ 利空信号惩罚（买入前检查明确看空信号）
    today_chg = signals.get("涨跌幅", 0)
    bearish_count = 0
    if signals.get("MA_弱势"):
        bearish_count += 1
    if signals.get("资金_流出"):
        bearish_count += 1
    if signals.get("RSI_偏高"):
        bearish_count += 1
    if today_chg < -3:  # 当日大跌
        bearish_count += 1

    if bearish_count >= 3:
        score -= 15
        details["利空叠加"] = f"{bearish_count}项看空"
    elif bearish_count >= 2:
        score -= 8
        details["利空叠加"] = f"{bearish_count}项看空"

    reason_str = "；".join(reasons) if reasons else "信号不足"
    return score, details, reason_str


# ============================================================
# T+5 波段评分引擎 — 技术面独立门槛 + 基本面调仓
# ============================================================

TECH_GATE_THRESHOLD = 42  # ★ 技术面独立门槛从35提到42（T+5持仓期更长，需要更强信号）

def evaluate_signals_gated(df, capital_info=None, sector_score=0, sentiment_score=0,
                           fundamental_score=0, fundamental_details=None,
                           extra_score=0, extra_details=None,
                           calibrated_weights=None, golden_combos=None,
                           market_regime_adj=0, weekly_trend_adj=0,
                           trend_context_adj=0, calendar_adj=0):
    """
    T+5 门控评分：技术面必须独立达标，基本面/聪明钱只影响仓位倍率
    返回 (tech_score, position_mult, details, reason_str, passed_gate)
    """
    if len(df) < 30:
        return 0, 1.0, {}, "数据不足", False

    # === 技术面评分（门槛）===
    signals = detect_signals(df, capital_info)
    tech_score, fired = apply_weights(signals, calibrated_weights)
    sig_quality, high_count, low_count = calc_signal_quality(fired)

    # 低胜率惩罚
    if low_count >= 3 and high_count == 0:
        tech_score -= 10

    # 高胜率奖励
    if high_count >= 3:
        tech_score += 12
    elif high_count >= 2:
        tech_score += 6

    # 黄金组合
    combo_bonus = 0
    matched_combo = ""
    has_golden = False
    if golden_combos and fired:
        fired_set = set(fired)
        for combo in golden_combos:
            combo_sigs = set(combo.get("signals", []))
            if combo_sigs.issubset(fired_set):
                wr = combo.get("win_rate", 0)
                bonus = int((wr - 0.55) * 100)
                if bonus > combo_bonus:
                    combo_bonus = min(bonus, 30)
                    matched_combo = f"黄金组合({wr:.0%})"
                    has_golden = True
    tech_score += combo_bonus

    # 趋势过滤（只扣不加）
    if market_regime_adj < 0:
        tech_score += max(market_regime_adj, -8)
    if weekly_trend_adj < 0:
        tech_score += max(weekly_trend_adj, -5)
    if trend_context_adj < 0:
        tech_score += max(trend_context_adj, -5)
    if calendar_adj < -5:
        tech_score -= 5

    # === 技术面门槛判断 ===
    passed_gate = tech_score >= TECH_GATE_THRESHOLD and (high_count >= 1 or has_golden)

    # === 基本面/聪明钱 → 仓位倍率（不影响买入决策）===
    fund_s = min(fundamental_score, 50)
    ext_s = min(max(extra_score, 0), 80)
    # 基本面倍率: 0-50分 → 0.7x-1.3x
    fund_mult = 0.7 + (fund_s / 50) * 0.6
    # 聪明钱倍率: 0-80分 → 0.7x-1.3x
    smart_mult = 0.7 + (ext_s / 80) * 0.6
    position_mult = round((fund_mult + smart_mult) / 2, 2)
    position_mult = max(0.5, min(position_mult, 1.5))

    # === 构建详情 ===
    details = {}
    latest = df.iloc[-1]
    sig_label_map = {
        "MACD_金叉": ("MACD", "当日金叉"), "MACD_红柱放大": ("MACD", "红柱放大"),
        "MACD_即将金叉": ("MACD", "即将金叉"), "MACD_红柱": ("MACD", "红柱"),
        "KDJ_超卖金叉": ("KDJ", "超卖金叉"), "KDJ_J超卖": ("KDJ", f"J超卖({latest['J']:.0f})"),
        "KDJ_多头": ("KDJ", "多头"),
        "RSI_回升": ("RSI", f"回升({signals.get('RSI值', 0):.0f})"),
        "RSI_超卖": ("RSI", f"超卖({signals.get('RSI值', 0):.0f})"),
        "RSI_极度超卖": ("RSI", f"极度超卖({signals.get('RSI值', 0):.0f})"),
        "RSI_强势": ("RSI", f"强势({signals.get('RSI值', 0):.0f})"),
        "MA_多头排列": ("均线", "多头排列"), "MA_短期多头": ("均线", "短期多头"),
        "MA_突破MA5": ("均线", "突破MA5"), "MA_站上MA5": ("均线", "站上MA5"),
        "BOLL_触及下轨": ("布林", "触及下轨"), "BOLL_中轨下方": ("布林", "中轨下方"),
    }
    reasons = []
    seen_dims = set()
    for sig in fired:
        if sig in sig_label_map:
            dim, label = sig_label_map[sig]
            details[dim] = label
            seen_dims.add(dim)
            reasons.append(label)

    if has_golden:
        details["黄金组合匹配"] = True
        details["组合"] = matched_combo
        reasons.append(matched_combo)
    else:
        details["黄金组合匹配"] = False

    details["技术门槛"] = f"{'PASS' if passed_gate else 'FAIL'}({tech_score:.0f}/{TECH_GATE_THRESHOLD})"
    details["信号质量"] = f"{'A' if high_count >= 2 else 'B' if high_count >= 1 else 'C'}级({high_count}个高胜率)"
    details["仓位倍率"] = f"{position_mult:.1f}x(基本面{fund_s}/50 聪明钱{ext_s}/80)"
    if weekly_trend_adj > 0:
        details["周线"] = "多头"
    elif weekly_trend_adj < 0:
        details["周线"] = "空头"
    if trend_context_adj > 0:
        details["趋势形态"] = "上升回调"
        reasons.append("上升趋势回调")
    elif trend_context_adj < 0:
        details["趋势形态"] = "下降反弹"

    reason_str = "；".join(reasons) if reasons else "信号不足"
    return tech_score, position_mult, details, reason_str, passed_gate


# ============================================================
# 市场情绪
# ============================================================

def get_sentiment_score():
    sentiments = fetch_market_sentiment()
    if not sentiments:
        return 7, "数据获取失败，默认中性"
    sh = next((s for s in sentiments if "上证" in s["名称"]), None)
    if not sh:
        return 7, "无数据"
    up = sh.get("上涨家数", 0)
    down = sh.get("下跌家数", 0)
    chg = sh.get("涨跌幅", 0)
    total = up + down if (up + down) > 0 else 1
    up_ratio = up / total
    score = 0
    info_parts = []
    if chg > 1.0:
        score += 8; info_parts.append(f"大盘涨{chg:.2f}%")
    elif chg > 0:
        score += 6; info_parts.append(f"大盘涨{chg:.2f}%")
    elif chg > -1.0:
        score += 4; info_parts.append(f"大盘跌{chg:.2f}%")
    else:
        score += 1; info_parts.append(f"大盘跌{chg:.2f}%")
    if up_ratio > 0.6:
        score += 7; info_parts.append(f"涨跌比{up}:{down}")
    elif up_ratio > 0.45:
        score += 4; info_parts.append(f"涨跌比{up}:{down}")
    else:
        score += 1; info_parts.append(f"涨跌比{up}:{down}")
    return min(score, 15), " | ".join(info_parts)


def get_hot_sectors():
    hot = set()
    for stype in ["concept", "industry"]:
        sectors = fetch_sector_flow(stype, top_n=15)
        for s in sectors:
            if s.get("主力净流入", 0) > 0 and s.get("涨跌幅", 0) > 0:
                hot.add(s["板块名称"])
    return hot


# ============================================================
# 全市场扫描
# ============================================================

def scan_market_v2(top_n=15):
    print("=" * 65)
    print("  A股 T+1 短线扫描工具 v3.0")
    print(f"  扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print()

    # 加载校准权重
    cal_weights, cal_threshold, cal_combos = load_calibrated_weights()
    if cal_weights:
        print(f"  [校准模式] 使用数据驱动权重 (阈值={cal_threshold})")

    # 市场级熔断检测
    print("[0/9] 市场环境检测...")
    money_effect = calc_money_effect()
    limit_up_pre, limit_down_pre = count_limit_up()
    can_trade, market_reason, market_severity = market_go_nogo(
        money_effect, limit_up=limit_up_pre, limit_down=limit_down_pre)
    if market_severity >= 3:
        print(f"  !! 市场熔断: {market_reason}")
        print("  !! 今日禁止操作，空仓观望！")
        return
    elif market_severity >= 2:
        print(f"  [警告] {market_reason} — 结果仅供参考，建议观望")
    elif market_severity >= 1:
        print(f"  [注意] {market_reason}")
    else:
        print(f"  市场环境正常 (赚钱效应:{money_effect['今日上涨比例']:.0%})")
    print()

    print("[1/9] 获取市场情绪...")
    sentiment_score, sentiment_info = get_sentiment_score()
    print(f"  {sentiment_info} → 情绪分: {sentiment_score}/15")

    print("[2/9] 获取板块资金流向...")
    hot_sectors = get_hot_sectors()
    if hot_sectors:
        print(f"  热门板块: {', '.join(list(hot_sectors)[:8])}")

    print("[3/9] 获取主力资金流向 TOP200...")
    capital_rank = fetch_capital_flow_rank(top_n=200)
    print(f"  获取到 {len(capital_rank)} 只股票的资金数据")

    print("[4/9] 获取龙虎榜+席位...")
    billboard_data = fetch_billboard()
    billboard_detail = fetch_billboard_detail()
    bb_buy = sum(1 for v in billboard_data.values() if v["龙虎榜净买入"] > 0)
    print(f"  龙虎榜 {len(billboard_data)} 只(净买入{bb_buy}只), 席位明细 {len(billboard_detail)} 只")

    print("[5/9] 获取北向+融资+涨停...")
    northbound_total, northbound_info = fetch_northbound_flow()
    margin_data = fetch_margin_data_top()
    limit_up, limit_down = count_limit_up()
    limit_up_pool = fetch_limit_up_pool()
    if northbound_info:
        print(f"  北向: {northbound_info}")
    print(f"  融资 {len(margin_data)} 只 | 涨停 {limit_up} 跌停 {limit_down} | 涨停池 {len(limit_up_pool)} 只")

    print("[6/9] 获取股东增持...")
    shareholder_data = fetch_shareholder_increase()
    print(f"  近30日增持 {len(shareholder_data)} 只")

    print("[7/9] 获取行业景气度...")
    industry_data = fetch_industry_prosperity()
    if industry_data:
        top3 = list(industry_data.items())[:3]
        print("  热门行业: " + ", ".join([f"{n}({d['涨跌幅']:+.1f}%)" for n, d in top3]))

    print("[8/9] 获取股票列表...")
    stock_list = fetch_stock_list_sina()
    if stock_list.empty:
        print("[错误] 无法获取股票列表")
        return

    print("[9/9] 获取基本面数据...")
    fund_batch = fetch_fundamentals_batch()
    print(f"  获取 {len(fund_batch)} 只股票的基本面数据")

    # 预筛选
    stock_list = stock_list[
        (stock_list["最新价"] >= 3) & (stock_list["最新价"] <= 100) &
        (stock_list["涨跌幅"] > -5) & (stock_list["涨跌幅"] < 9.8) &
        (stock_list["换手率"] >= 2) & (stock_list["量比"] >= 0.8)
    ]

    capital_codes = set(capital_rank.keys())
    priority_codes = stock_list[stock_list["代码"].isin(capital_codes)]["代码"].tolist()
    other_codes = stock_list[~stock_list["代码"].isin(capital_codes)]["代码"].tolist()
    ordered = priority_codes + other_codes
    code_to_row = stock_list.set_index("代码").to_dict("index")

    total = len(ordered)
    print(f"[6/6] 分析 {total} 只股票...")

    results = []

    # 预取所有候选股票的行业（批量获取太慢，用缓存+按需获取）
    _industry_cache = {}

    def analyze_one(code):
        kline = fetch_kline(code, days=120)
        if kline.empty or len(kline) < 30:
            return None
        kline = calc_all_indicators(kline)
        cap_info = capital_rank.get(code)
        sec_score = 8 if code in capital_codes else 3
        # 基本面评分
        fund_info = fund_batch.get(code, {})
        fund_s, fund_d, fund_r = evaluate_fundamentals(fund_info)
        # 额外维度评分（龙虎榜/北向/融资/连板/增持/行业）
        stock_ind = ""
        if code in billboard_data or code in limit_up_pool or code in shareholder_data:
            stock_ind = get_stock_industry(code)
            _industry_cache[code] = stock_ind
        extra_s, extra_d, extra_r = evaluate_extra_dimensions(
            code, billboard_data, margin_data, northbound_total, limit_up, limit_down,
            limit_up_pool, billboard_detail, shareholder_data, industry_data, stock_ind)
        s, d, r = evaluate_signals_v2(kline, cap_info, sec_score, sentiment_score,
                                       fund_s, fund_d, extra_s, extra_d,
                                       calibrated_weights=cal_weights, golden_combos=cal_combos)
        combined_reasons = r
        if fund_r:
            combined_reasons = r + "；" + fund_r if r != "信号不足" else fund_r
        if extra_r:
            combined_reasons += "；" + extra_r if combined_reasons != "信号不足" else extra_r
        scan_threshold = cal_threshold if cal_weights else 90
        if s >= scan_threshold:
            row = code_to_row.get(code, {})
            return {
                "代码": code, "名称": row.get("名称", ""),
                "最新价": row.get("最新价", 0), "涨跌幅": row.get("涨跌幅", 0),
                "换手率": row.get("换手率", 0), "评分": s,
                "信号": d, "理由": combined_reasons,
                "主力净流入占比": cap_info.get("主力净流入占比", 0) if cap_info else 0,
            }
        return None

    done = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(analyze_one, code): code for code in ordered}
        for future in as_completed(futures):
            done += 1
            if done % 200 == 0:
                print(f"  进度: {done}/{total} ({done/total*100:.0f}%) | 候选: {len(results)}")
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception:
                pass

    print(f"\n  分析完成！{len(results)} 只股票评分>=90")
    print()

    if not results:
        print("  今日未发现高评分候选，建议观望。")
        return

    results.sort(key=lambda x: x["评分"], reverse=True)
    results = results[:top_n]

    print("=" * 65)
    print(f"  T+1 短线候选 TOP {len(results)}（满分280）")
    print("=" * 65)
    print()

    table_data = []
    for r in results:
        table_data.append([
            r["代码"], r["名称"], f"{r['最新价']:.2f}",
            f"{r['涨跌幅']:+.2f}%", f"{r['换手率']:.1f}%",
            f"{r['主力净流入占比']:+.1f}%",
            r["评分"], r["理由"][:30],
        ])
    print(tabulate(
        table_data,
        headers=["代码", "名称", "现价", "涨跌幅", "换手率", "主力净流入%", "评分", "买入理由"],
        tablefmt="simple_grid", stralign="center",
    ))

    print()
    print("── 详细信号 ──")
    for r in results[:5]:
        sigs = " | ".join([f"{k}:{v}" for k, v in r["信号"].items()])
        print(f"  {r['代码']} {r['名称']}: {sigs}")

    print()
    print("提示：")
    print("  - 评分>=170 强烈关注，>=130 可以关注，>=90 谨慎参与")
    print("  - 下午2:30后扫描最准确（数据更完整）")
    print("  - 严格止损-2%，止盈+3%~5%")
    print("  - 分散持仓，单只不超过总资金20%")


# ============================================================
# 单只股票深度分析
# ============================================================

def analyze_single_v2(code):
    code = str(code)
    print(f"\n获取 {code} 数据...")

    kline = fetch_kline(code, days=120)
    if kline.empty or len(kline) < 30:
        print(f"[错误] 数据不足，无法分析 {code}")
        return

    cap_flow = fetch_stock_capital_flow(code)
    cap_rank = fetch_capital_flow_rank(200)
    sentiment_score, sentiment_info = get_sentiment_score()
    fund_detail = fetch_fundamental_detail(code)
    fund_batch = fetch_fundamentals_batch()
    fund_info = fund_batch.get(code, {})
    # 合并批量和详细数据
    if fund_detail:
        fund_info.update({k: v for k, v in fund_detail.items() if v is not None})
    fund_s, fund_d, fund_r = evaluate_fundamentals(fund_info)

    # 新增维度数据
    billboard_data = fetch_billboard()
    billboard_detail = fetch_billboard_detail()
    northbound_total, northbound_info = fetch_northbound_flow()
    margin_data = fetch_margin_data_top()
    limit_up, limit_down = count_limit_up()
    limit_up_pool = fetch_limit_up_pool()
    shareholder_data = fetch_shareholder_increase()
    industry_data = fetch_industry_prosperity()
    stock_industry = get_stock_industry(code)

    extra_s, extra_d, extra_r = evaluate_extra_dimensions(
        code, billboard_data, margin_data, northbound_total, limit_up, limit_down,
        limit_up_pool, billboard_detail, shareholder_data, industry_data, stock_industry)

    kline = calc_all_indicators(kline)
    latest = kline.iloc[-1]

    cap_info = cap_rank.get(code)
    score, details, reasons = evaluate_signals_v2(kline, cap_info, 8, sentiment_score, fund_s, fund_d, extra_s, extra_d)

    # 获取股票名称
    rt = fetch_realtime_sina([code])
    name = rt.get(code, {}).get("名称", code)

    print()
    print("=" * 55)
    print(f"  {name}({code}) 深度分析报告")
    print(f"  日期: {latest['日期']}    综合评分: {score}/280")
    print("=" * 55)

    print()
    print("  ── 行情 ──")
    print(f"  收盘: {latest['收盘']:.2f}  涨跌幅: {latest['涨跌幅']:+.2f}%  振幅: {latest['振幅']:.2f}%")
    print(f"  最高: {latest['最高']:.2f}  最低: {latest['最低']:.2f}")

    print()
    print("  ── 技术指标 ──")
    print(f"  MA5: {latest['MA5']:.2f}  MA10: {latest['MA10']:.2f}  MA20: {latest['MA20']:.2f}")
    print(f"  MACD: DIF={latest['DIF']:.3f} DEA={latest['DEA']:.3f} 柱={latest['MACD柱']:.3f}")
    print(f"  KDJ: K={latest['K']:.1f} D={latest['D']:.1f} J={latest['J']:.1f}")
    rsi_val = latest.get('RSI6', np.nan)
    if not pd.isna(rsi_val):
        print(f"  RSI6: {rsi_val:.1f}")
    boll_upper = latest.get("BOLL上", np.nan)
    if not pd.isna(boll_upper):
        print(f"  BOLL: 上={boll_upper:.2f} 中={latest['BOLL中']:.2f} 下={latest['BOLL下']:.2f}")

    # 基本面
    print()
    print("  ── 公司基本面 ──")
    if fund_info:
        parts = []
        for k in ["总市值", "PE动态", "PE", "PB", "ROE", "毛利率", "净利率", "负债率", "营收增长", "利润增长"]:
            v = fund_info.get(k)
            if v is not None:
                if k == "总市值":
                    parts.append(f"{k}: {v:.0f}亿")
                elif k in ("PE动态", "PE", "PB"):
                    parts.append(f"{k}: {v:.1f}")
                else:
                    parts.append(f"{k}: {v:.1f}%")
        for i in range(0, len(parts), 3):
            print(f"  {'  '.join(parts[i:i+3])}")
        print(f"  基本面评分: {fund_s}/50" + (f" ({fund_r})" if fund_r else ""))
    else:
        print("  无基本面数据")

    print()
    print("  ── 主力资金 ──")
    if cap_info:
        print(f"  今日主力净流入占比: {cap_info['主力净流入占比']:+.2f}%")
        print(f"  超大单: {cap_info['超大单占比']:+.2f}%  大单: {cap_info['大单占比']:+.2f}%")
    else:
        print("  今日无主力资金排名数据")

    if not cap_flow.empty and len(cap_flow) >= 5:
        recent_flow = cap_flow.tail(5)
        print("  近5日资金流向:")
        flow_table = []
        for _, r in recent_flow.iterrows():
            mn = r["主力净流入"]
            flow_table.append([
                r["日期"],
                f"{mn/1e4:+.0f}万" if abs(mn) < 1e8 else f"{mn/1e8:+.2f}亿",
            ])
        print(tabulate(flow_table, headers=["日期", "主力净流入"], tablefmt="simple_grid"))
        consecutive = 0
        for _, r in cap_flow.iloc[::-1].iterrows():
            if r["主力净流入"] > 0:
                consecutive += 1
            else:
                break
        if consecutive > 0:
            print(f"  主力连续流入: {consecutive} 天")

    print()
    print(f"  ── 市场情绪 ──")
    print(f"  {sentiment_info}")

    print()
    print("  ── 信号详情 ──")
    for key, val in details.items():
        print(f"  {key}: {val}")

    print()
    print(f"  ★ 综合评分: {score}/280")
    print(f"  买入理由: {reasons}")
    if score >= 170:
        print("  结论: 强烈关注！技术面+基本面+资金面+聪明钱共振")
    elif score >= 130:
        print("  结论: 可以关注，信号较好")
    elif score >= 90:
        print("  结论: 谨慎参与，信号一般")
    else:
        print("  结论: 建议观望，信号不足")

    print()
    print("  ── 近10日走势 ──")
    recent = kline.tail(10)
    table_data = []
    for _, r in recent.iterrows():
        table_data.append([
            r["日期"], f"{r['开盘']:.2f}", f"{r['收盘']:.2f}",
            f"{r['最高']:.2f}", f"{r['最低']:.2f}",
            f"{r['涨跌幅']:+.2f}%",
        ])
    print(tabulate(
        table_data,
        headers=["日期", "开盘", "收盘", "最高", "最低", "涨跌幅"],
        tablefmt="simple_grid",
    ))
    print()


# ============================================================
# 回测引擎
# ============================================================

def backtest(code=None, days=250):
    print("=" * 55)
    print("  T+1 策略回测")
    print("=" * 55)
    print()

    # 加载校准权重
    cal_weights, cal_threshold, cal_combos = load_calibrated_weights()
    if cal_weights:
        bt_threshold = cal_threshold
        print(f"  [校准模式] 阈值={bt_threshold}")
    else:
        bt_threshold = 110
        print(f"  [默认模式] 阈值={bt_threshold}")

    if code:
        codes = [str(code)]
        print(f"  回测标的: {code}")
    else:
        print("  选取活跃股进行回测...")
        stock_list = fetch_stock_list_sina()
        if stock_list.empty:
            print("[错误] 无法获取股票列表")
            return
        active = stock_list[(stock_list["最新价"] >= 5) & (stock_list["最新价"] <= 80)]
        if active.empty:
            active = stock_list.head(50)
        codes = active.sample(min(50, len(active)), random_state=42)["代码"].tolist()
        print(f"  选取 {len(codes)} 只活跃股")

    print(f"  回测天数: ~{days} 个交易日")
    print(f"  策略: 综合评分>={bt_threshold}买入，次日收盘卖出")
    print()

    all_trades = []
    for i, c in enumerate(codes):
        if (i + 1) % 10 == 0:
            print(f"  回测进度: {i+1}/{len(codes)}")

        kline = fetch_kline_long(c, days=days)
        if kline.empty or len(kline) < 60:
            continue

        kline = calc_all_indicators(kline)

        # 回测中用模拟的资金面/板块/情绪
        cap_sim = {"主力净流入占比": 3, "主力净流入": 1e7,
                   "超大单净流入": 0, "超大单占比": 0, "大单净流入": 0, "大单占比": 0}
        for j in range(30, len(kline) - 1):
            window = kline.iloc[:j+1].copy()
            sc, details, _ = evaluate_signals_v2(
                window, cap_sim, 8, 8, 25, {},
                calibrated_weights=cal_weights, golden_combos=cal_combos)
            if sc >= bt_threshold:
                buy_price = kline.iloc[j]["收盘"]
                sell_price = kline.iloc[j+1]["收盘"]
                pnl = (sell_price - buy_price) / buy_price * 100
                all_trades.append({
                    "代码": c, "买入日": kline.iloc[j]["日期"],
                    "买入价": buy_price, "卖出价": sell_price,
                    "收益率": pnl,
                    "黄金组合": details.get("黄金组合匹配", False),
                })

        time.sleep(0.15)

    if not all_trades:
        print("  回测期间无交易信号")
        return

    df_trades = pd.DataFrame(all_trades)
    total_trades = len(df_trades)
    win_trades = len(df_trades[df_trades["收益率"] > 0])
    win_rate = win_trades / total_trades * 100
    avg_return = df_trades["收益率"].mean()
    avg_win = df_trades[df_trades["收益率"] > 0]["收益率"].mean() if win_trades > 0 else 0
    avg_loss = df_trades[df_trades["收益率"] <= 0]["收益率"].mean() if (total_trades - win_trades) > 0 else 0
    max_win = df_trades["收益率"].max()
    max_loss = df_trades["收益率"].min()

    df_trades["止损收益"] = df_trades["收益率"].clip(lower=-2)
    stop_avg = df_trades["止损收益"].mean()

    print("  ── 回测结果 ──")
    print()
    print(f"  总交易次数:  {total_trades}")
    print(f"  胜率:        {win_rate:.1f}%")
    print(f"  平均收益:    {avg_return:+.2f}%")
    print(f"  平均盈利:    {avg_win:+.2f}%")
    print(f"  平均亏损:    {avg_loss:+.2f}%")
    if avg_loss != 0:
        print(f"  盈亏比:      {abs(avg_win/avg_loss):.2f}")
    print(f"  最大单笔盈利: {max_win:+.2f}%")
    print(f"  最大单笔亏损: {max_loss:+.2f}%")
    print()
    print(f"  ── 加入止损(-2%)后 ──")
    print(f"  平均收益:    {stop_avg:+.2f}%")
    print()

    # 黄金组合子集胜率
    if "黄金组合" in df_trades.columns:
        combo_trades = df_trades[df_trades["黄金组合"] == True]
        if len(combo_trades) > 0:
            combo_win = len(combo_trades[combo_trades["收益率"] > 0])
            combo_wr = combo_win / len(combo_trades) * 100
            combo_avg = combo_trades["收益率"].mean()
            print(f"  ── 黄金组合子集 ──")
            print(f"  交易次数:    {len(combo_trades)}")
            print(f"  胜率:        {combo_wr:.1f}%")
            print(f"  平均收益:    {combo_avg:+.2f}%")
            print()

    bins = [-999, -5, -3, -2, -1, 0, 1, 2, 3, 5, 999]
    labels = ["<-5%", "-5~-3%", "-3~-2%", "-2~-1%", "-1~0%",
              "0~1%", "1~2%", "2~3%", "3~5%", ">5%"]
    df_trades["区间"] = pd.cut(df_trades["收益率"], bins=bins, labels=labels)
    dist = df_trades["区间"].value_counts().sort_index()

    print("  ── 收益分布 ──")
    dist_table = []
    for label in labels:
        count = dist.get(label, 0)
        pct = count / total_trades * 100
        bar = "#" * int(pct / 2)
        dist_table.append([label, count, f"{pct:.1f}%", bar])
    print(tabulate(dist_table, headers=["区间", "次数", "占比", "分布"], tablefmt="simple_grid"))

    print()
    print("  ── 最近20笔交易 ──")
    recent = df_trades.tail(20)
    trade_table = []
    for _, t in recent.iterrows():
        emoji = "+" if t["收益率"] > 0 else "-" if t["收益率"] < 0 else "="
        trade_table.append([
            t["代码"], t["买入日"], f"{t['买入价']:.2f}",
            f"{t['卖出价']:.2f}", f"{t['收益率']:+.2f}%", emoji
        ])
    print(tabulate(trade_table, headers=["代码", "买入日", "买入价", "卖出价", "收益率", ""], tablefmt="simple_grid"))
    print()


# ============================================================
# 校准回测：计算每个信号的真实次日胜率 → 数据驱动权重
# ============================================================

CALIBRATION_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "signal_weights.json")


def load_calibrated_weights():
    """加载校准权重文件，不存在则返回None"""
    try:
        with open(CALIBRATION_FILE, "r") as f:
            data = json.load(f)
        return data.get("weights"), data.get("threshold", 120), data.get("best_combos", [])
    except Exception:
        return None, 120, []


def calibrate(days=250, sample_size=200):
    """
    校准回测：遍历历史数据，统计每个信号的真实次日胜率和平均收益。
    用胜率×平均收益×样本量 计算数据驱动的权重。
    """
    print("=" * 65)
    print("  信号校准回测 — 用数据替代拍脑袋")
    print("=" * 65)
    print()

    # 选取样本股
    stock_list = fetch_stock_list_sina()
    if stock_list.empty:
        print("[错误] 无法获取股票列表")
        return
    active = stock_list[(stock_list["最新价"] >= 5) & (stock_list["最新价"] <= 80) &
                        (stock_list["换手率"] >= 1)]
    if len(active) < sample_size:
        active = stock_list.head(sample_size)
    codes = active.sample(min(sample_size, len(active)), random_state=42)["代码"].tolist()
    print(f"  样本: {len(codes)} 只股票 × ~{days} 交易日")
    print(f"  目标: 统计每个信号触发后次日的真实胜率")
    print()

    # 统计容器
    signal_stats = {}  # signal_name -> {"wins": int, "total": int, "returns": [float]}

    for i, code in enumerate(codes):
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(codes)}")
        kline = fetch_kline_long(code, days=days)
        if kline.empty or len(kline) < 60:
            continue
        kline = calc_all_indicators(kline)

        # 模拟资金面（回测中无法获取实时数据）
        cap_sim = {"主力净流入占比": 3, "主力净流入": 1e7,
                   "超大单净流入": 0, "超大单占比": 0, "大单净流入": 0, "大单占比": 0}

        for j in range(30, len(kline) - 1):
            window = kline.iloc[:j + 1].copy()
            signals = detect_signals(window, cap_sim)

            buy_price = kline.iloc[j]["收盘"]
            sell_price = kline.iloc[j + 1]["收盘"]
            ret = (sell_price - buy_price) / buy_price * 100

            # 记录每个触发的信号
            for group_name, group_signals in SIGNAL_GROUPS.items():
                for sig in group_signals:
                    if signals.get(sig):
                        if sig not in signal_stats:
                            signal_stats[sig] = {"wins": 0, "total": 0, "returns": []}
                        signal_stats[sig]["total"] += 1
                        if ret > 0:
                            signal_stats[sig]["wins"] += 1
                        signal_stats[sig]["returns"].append(ret)
                        break  # 互斥

        time.sleep(0.15)

    if not signal_stats:
        print("  校准失败：无信号数据")
        return

    # === 计算校准权重 ===
    print()
    print("  ── 信号真实胜率排名 ──")
    print()

    calibrated = {}
    table_data = []
    for sig_name, stats in sorted(signal_stats.items(),
                                    key=lambda x: x[1]["wins"] / max(x[1]["total"], 1), reverse=True):
        n = stats["total"]
        if n < 20:  # 样本太少的信号不可靠
            continue
        win_rate = stats["wins"] / n
        avg_ret = np.mean(stats["returns"])
        avg_win = np.mean([r for r in stats["returns"] if r > 0]) if stats["wins"] > 0 else 0
        avg_loss = np.mean([r for r in stats["returns"] if r <= 0]) if (n - stats["wins"]) > 0 else 0

        # 权重公式：胜率偏移 × 平均收益 × 样本量对数 × 缩放
        # 胜率50%=中性，偏离越远权重越大
        wr_factor = (win_rate - 0.45) * 20  # 45%以下为负权重
        ret_factor = max(avg_ret, -1)  # 限制负面影响
        sample_factor = min(np.log(n + 1) / np.log(100), 1.5)  # 样本量加成，但有上限
        weight = wr_factor * sample_factor * 3  # 缩放到合理范围

        # 限制在原权重的 0.3x ~ 2.5x
        default_w = DEFAULT_WEIGHTS.get(sig_name, 5)
        weight = max(default_w * 0.3, min(weight, default_w * 2.5))
        weight = round(weight, 1)

        calibrated[sig_name] = weight

        emoji = "+" if win_rate > 0.52 else "-" if win_rate < 0.48 else "="
        table_data.append([
            sig_name, n, f"{win_rate:.1%}", f"{avg_ret:+.2f}%",
            f"{avg_win:+.2f}%", f"{avg_loss:.2f}%",
            f"{DEFAULT_WEIGHTS.get(sig_name, '?')}", f"{weight:.1f}", emoji
        ])

    print(tabulate(table_data,
                   headers=["信号", "样本", "胜率", "平均收益", "平均盈利", "平均亏损", "旧权重", "新权重", ""],
                   tablefmt="simple_grid"))

    # === 组合信号分析：找到最强共振模式 ===
    print()
    print("  ── 信号组合分析（找最强共振模式）──")
    print()

    # 重新遍历，收集每个交易点的信号组合 + 收益
    combo_data = []  # [(fired_set, return)]
    for code in codes[:100]:  # 用更大子集提高组合分析可靠性
        kline = fetch_kline_long(code, days=days)
        if kline.empty or len(kline) < 60:
            continue
        kline = calc_all_indicators(kline)
        cap_sim = {"主力净流入占比": 3, "主力净流入": 1e7,
                   "超大单净流入": 0, "超大单占比": 0, "大单净流入": 0, "大单占比": 0}
        for j in range(30, len(kline) - 1):
            window = kline.iloc[:j + 1].copy()
            signals = detect_signals(window, cap_sim)
            _, fired = apply_weights(signals, calibrated)
            ret = (kline.iloc[j + 1]["收盘"] - kline.iloc[j]["收盘"]) / kline.iloc[j]["收盘"] * 100
            combo_data.append((frozenset(fired), fired, ret))
        time.sleep(0.1)

    # 分析常见的2-3信号组合
    from itertools import combinations
    # 收集所有触发过的信号
    all_fired_signals = set()
    for _, fired, _ in combo_data:
        all_fired_signals.update(fired)

    # 测试所有2信号组合
    combo_stats = {}
    top_signals = [s for s in all_fired_signals if signal_stats.get(s, {}).get("total", 0) >= 50]

    for pair in combinations(top_signals, 2):
        pair_set = set(pair)
        wins = 0
        total = 0
        rets = []
        for fired_set, _, ret in combo_data:
            if pair_set.issubset(fired_set):
                total += 1
                if ret > 0:
                    wins += 1
                rets.append(ret)
        if total >= 15:
            combo_stats[pair] = {"wins": wins, "total": total, "returns": rets}

    # 测试3信号组合（只测top信号）
    top10 = sorted(top_signals, key=lambda s: signal_stats.get(s, {}).get("wins", 0) /
                   max(signal_stats.get(s, {}).get("total", 1), 1), reverse=True)[:12]
    for triple in combinations(top10, 3):
        triple_set = set(triple)
        wins = 0
        total = 0
        rets = []
        for fired_set, _, ret in combo_data:
            if triple_set.issubset(fired_set):
                total += 1
                if ret > 0:
                    wins += 1
                rets.append(ret)
        if total >= 10:
            combo_stats[triple] = {"wins": wins, "total": total, "returns": rets}

    # 排序输出
    combo_ranked = sorted(combo_stats.items(),
                          key=lambda x: x[1]["wins"] / max(x[1]["total"], 1), reverse=True)

    combo_table = []
    best_combos = []
    for combo, stats in combo_ranked[:20]:
        n = stats["total"]
        wr = stats["wins"] / n
        avg = np.mean(stats["returns"])
        stop_avg = np.mean([max(r, -2) for r in stats["returns"]])
        ev = stop_avg * wr
        combo_name = " + ".join(sorted(combo))
        combo_table.append([combo_name, n, f"{wr:.1%}", f"{avg:+.2f}%",
                           f"{stop_avg:+.2f}%", f"{ev:.3f}"])
        if wr >= 0.58 and n >= 10:
            best_combos.append({"signals": list(combo), "win_rate": wr,
                                "avg_return": avg, "n": n})

    if combo_table:
        print(tabulate(combo_table,
                       headers=["信号组合", "样本", "胜率", "平均收益", "止损后收益", "期望值"],
                       tablefmt="simple_grid"))

    if best_combos:
        print(f"\n  ★ 发现 {len(best_combos)} 个高胜率组合（>=58%）")
        print("  这些组合将作为\"黄金模式\"加入实盘筛选")

    # === 阈值优化 ===
    print()
    print("  ── 阈值优化（找最优买入门槛）──")
    print()

    # 重新跑一遍，用校准权重，测试不同阈值
    all_scores_returns = []
    for code in codes[:80]:  # 用更大子集提高阈值优化可靠性
        kline = fetch_kline_long(code, days=days)
        if kline.empty or len(kline) < 60:
            continue
        kline = calc_all_indicators(kline)
        cap_sim = {"主力净流入占比": 3, "主力净流入": 1e7,
                   "超大单净流入": 0, "超大单占比": 0, "大单净流入": 0, "大单占比": 0}

        for j in range(30, len(kline) - 1):
            window = kline.iloc[:j + 1].copy()
            # 用校准权重计算分数
            sc, _, _ = evaluate_signals_v2(window, cap_sim, 8, 8, 25, {},
                                           calibrated_weights=calibrated)
            ret = (kline.iloc[j + 1]["收盘"] - kline.iloc[j]["收盘"]) / kline.iloc[j]["收盘"] * 100
            all_scores_returns.append((sc, ret))
        time.sleep(0.1)

    if all_scores_returns:
        threshold_table = []
        best_threshold = 120
        best_metric = -999
        for thresh in range(80, 200, 10):
            trades = [(s, r) for s, r in all_scores_returns if s >= thresh]
            if len(trades) < 10:
                continue
            wins = sum(1 for _, r in trades if r > 0)
            wr = wins / len(trades)
            avg = np.mean([r for _, r in trades])
            stop_avg = np.mean([max(r, -2) for _, r in trades])  # 加止损
            metric = stop_avg * wr  # 期望值=收益×胜率
            threshold_table.append([thresh, len(trades), f"{wr:.1%}", f"{avg:+.2f}%",
                                    f"{stop_avg:+.2f}%", f"{metric:.3f}"])
            if metric > best_metric:
                best_metric = metric
                best_threshold = thresh

        print(tabulate(threshold_table,
                       headers=["阈值", "交易数", "胜率", "平均收益", "止损后收益", "期望值"],
                       tablefmt="simple_grid"))
        print(f"\n  ★ 最优阈值: {best_threshold}  (期望值最大化)")

    # === 保存 ===
    save_data = {
        "weights": calibrated,
        "threshold": best_threshold,
        "calibrated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_size": len(codes),
        "signal_stats": {k: {"win_rate": v["wins"] / max(v["total"], 1),
                              "avg_return": float(np.mean(v["returns"])),
                              "n": v["total"]}
                         for k, v in signal_stats.items() if v["total"] >= 20},
        "best_combos": best_combos[:10] if best_combos else [],
    }
    try:
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"\n  校准结果已保存到: {CALIBRATION_FILE}")
        print("  后续 --go 和扫描将自动使用校准权重")
    except Exception as e:
        print(f"\n  [警告] 保存失败: {e}")

    print()


# ============================================================
# 板块资金流向
# ============================================================

def show_sector_flow():
    print("=" * 55)
    print("  板块资金流向")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    for stype, title in [("concept", "概念板块"), ("industry", "行业板块")]:
        sectors = fetch_sector_flow(stype, top_n=15)
        print(f"\n  ── {title} TOP15 资金流入 ──")
        table = []
        for s in sectors:
            net = s["主力净流入"]
            net_str = f"{net/1e8:+.2f}亿" if abs(net) >= 1e8 else f"{net/1e4:+.0f}万"
            table.append([
                s["板块名称"], f"{s['涨跌幅']:+.2f}%", net_str, f"{s['净流入占比']:+.2f}%"
            ])
        print(tabulate(table, headers=["板块", "涨跌幅", "主力净流入", "净流入占比"], tablefmt="simple_grid"))
    print()


def show_market_sentiment():
    sentiments = fetch_market_sentiment()
    if not sentiments:
        print("[错误] 无法获取市场数据")
        return

    print("=" * 55)
    print("  大盘情绪面板")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    print()

    table = []
    for s in sentiments:
        vol_str = f"{s['成交额']/1e8:.0f}亿" if s['成交额'] > 0 else "-"
        table.append([
            s["名称"], f"{s['点位']:.2f}", f"{s['涨跌幅']:+.2f}%",
            vol_str, s["上涨家数"], s["下跌家数"],
        ])
    print(tabulate(table, headers=["指数", "点位", "涨跌幅", "成交额", "上涨", "下跌"], tablefmt="simple_grid"))

    sh = sentiments[0] if sentiments else {}
    up = sh.get("上涨家数", 0)
    down = sh.get("下跌家数", 0)
    total = up + down if (up + down) > 0 else 1
    ratio = up / total
    print()
    if ratio > 0.65:
        print("  情绪: 市场强势，适合做多")
    elif ratio > 0.5:
        print("  情绪: 市场偏强，可适度参与")
    elif ratio > 0.35:
        print("  情绪: 市场偏弱，谨慎操作")
    else:
        print("  情绪: 市场弱势，建议观望")
    print()


# ============================================================
# 新闻/政策/热点分析
# ============================================================

# 利好关键词 → 对应板块/概念
NEWS_POSITIVE_KEYWORDS = {
    # 政策利好
    "降准": ("金融", 15), "降息": ("金融", 15), "减税": ("消费", 12),
    "刺激": ("消费", 12), "扶持": ("科技", 12), "补贴": ("新能源", 12),
    "国产替代": ("半导体", 15), "自主可控": ("信创", 15),
    "数字经济": ("数字经济", 12), "新基建": ("基建", 10),
    "碳中和": ("新能源", 10), "新能源": ("新能源", 10),
    # AI/科技
    "人工智能": ("AI", 15), "AI": ("AI", 15), "大模型": ("AI", 15),
    "芯片": ("半导体", 12), "半导体": ("半导体", 12),
    "算力": ("算力", 12), "数据中心": ("算力", 10),
    "机器人": ("机器人", 12), "无人驾驶": ("智能驾驶", 10),
    "DeepSeek": ("AI", 12), "ChatGPT": ("AI", 10),
    # 消费/医药
    "消费": ("消费", 8), "医药": ("医药", 8), "中药": ("中药", 10),
    # 利好信号
    "大涨": ("", 8), "涨停": ("", 10), "暴涨": ("", 8),
    "突破": ("", 6), "新高": ("", 8), "放量": ("", 6),
    # ── 地缘政治利好 ──
    "国产替代加速": ("信创", 15), "自主可控突破": ("信创", 15),
    "一带一路": ("基建", 12), "中欧合作": ("出口", 10),
    "贸易协定": ("出口", 10), "关税减免": ("出口", 10),
    "外交突破": ("", 8), "关系回暖": ("", 6),
    # ── 军事/国防 ──
    "军工订单": ("军工", 15), "国防订单": ("军工", 15),
    "军费增长": ("军工", 15), "国防预算": ("军工", 15),
    "装备列装": ("军工", 12), "武器出口": ("军工", 12),
    "军民融合": ("军工", 12), "航天发射": ("航天", 12),
    "卫星导航": ("卫星", 10), "北斗": ("卫星", 12),
    "网络安全": ("安全", 12), "信息安全": ("安全", 12),
    "无人机": ("军工", 10), "航母": ("军工", 10),
    # ── 文化产业 ──
    "票房": ("影视", 10), "爆款": ("传媒", 8),
    "文化出海": ("传媒", 12), "海外发行": ("游戏", 10),
    "国潮": ("消费", 10), "文旅": ("旅游", 10),
    "电竞": ("游戏", 10), "版号": ("游戏", 12),
    "短视频": ("传媒", 8), "直播": ("传媒", 8),
    "文化产业": ("传媒", 10), "影视": ("影视", 8),
    "动漫": ("传媒", 8), "元宇宙": ("传媒", 10),
}

# 利空关键词
NEWS_NEGATIVE_KEYWORDS = [
    "暴跌", "崩盘", "大跌", "跌停", "利空", "制裁", "打压",
    "退市", "爆雷", "违规", "处罚", "下调", "减持", "清仓",
    "战争", "冲突", "加息",  # 美联储加息
    # 大宗商品暴跌
    "油价暴跌", "油价大跌", "原油暴跌", "原油大跌", "原油崩盘",
    "金价暴跌", "铜价暴跌", "大宗商品暴跌", "商品期货大跌",
    "OPEC", "供应过剩", "库存大增",
    # 地缘政治利空
    "军事冲突", "武装冲突", "导弹袭击", "空袭", "入侵",
    "断交", "外交降级", "技术封锁", "实体清单", "投资禁令",
    "台海危机", "南海冲突", "核试验",
    # 文化监管利空
    "版号收紧", "游戏限制", "未成年限制", "整顿", "封杀",
]

# 新闻一票否决：匹配到这些关键词 → 相关行业的股票直接排除
NEWS_VETO_KEYWORDS = {
    "油价暴跌": ["石油", "石油石化", "化工", "油气", "油服", "海油", "中石油", "中石化"],
    "原油暴跌": ["石油", "石油石化", "化工", "油气", "油服", "海油", "中石油", "中石化"],
    "原油崩盘": ["石油", "石油石化", "化工", "油气", "油服", "海油", "中石油", "中石化"],
    "油价大跌": ["石油", "石油石化", "化工", "油气", "海油"],
    "原油大跌": ["石油", "石油石化", "化工", "油气", "海油"],
    "金价暴跌": ["贵金属", "有色金属", "黄金"],
    "铜价暴跌": ["有色金属", "铜"],
    "大宗商品暴跌": ["石油", "有色金属", "化工", "农业", "煤炭", "钢铁"],
    # 地缘政治否决
    "台海危机": ["航空", "旅游", "酒店", "免税"],
    "台海冲突": ["航空", "旅游", "酒店", "免税"],
    "南海冲突": ["航运", "港口", "旅游"],
    # 文化监管否决
    "版号收紧": ["游戏"],
    "游戏严管": ["游戏"],
    "影视整顿": ["影视", "传媒", "院线"],
    "娱乐整顿": ["影视", "传媒", "娱乐"],
}


def fetch_news():
    """获取最新财经新闻（东方财富 + 新浪）"""
    all_news = []

    # 东方财富快讯
    try:
        resp = _get("https://np-listapi.eastmoney.com/comm/web/getFastNewsList",
                     params={"client": "web", "biz": "web_news_col",
                             "fastColumn": "102", "sortEnd": "",
                             "pageSize": 30, "req_trace": "t1"},
                     timeout=10)
        data = resp.json()
        for n in data.get("result", {}).get("fastNewsList", []):
            title = n.get("title", "")
            summary = n.get("digest", n.get("summary", ""))
            all_news.append({
                "来源": "东方财富",
                "时间": n.get("showTime", ""),
                "标题": title,
                "摘要": summary,
                "文本": title + " " + summary,
            })
    except Exception:
        pass

    # 新浪财经滚动新闻
    try:
        resp = _get("https://feed.mix.sina.com.cn/api/roll/get",
                     params={"pageid": "153", "lid": "2516",
                             "num": 20, "page": 1},
                     timeout=10)
        data = resp.json()
        for item in data.get("result", {}).get("data", []):
            title = item.get("title", "")
            intro = item.get("intro", "")
            all_news.append({
                "来源": "新浪",
                "时间": item.get("ctime", ""),
                "标题": title,
                "摘要": intro,
                "文本": title + " " + intro,
            })
    except Exception:
        pass

    return all_news


def analyze_news_sentiment(news_list):
    """
    分析新闻情绪，返回：
    - overall_score: 整体市场情绪 (-10 到 +10)
    - hot_concepts: 热门概念及其新闻热度分
    - key_headlines: 关键新闻标题
    - veto_industries: 被一票否决的行业/关键词集合
    """
    concept_scores = {}  # 概念 -> 累计分数
    positive_count = 0
    negative_count = 0
    key_headlines = []
    veto_industries = set()

    for news in news_list:
        text = news["文本"]

        # 利好扫描
        for keyword, (concept, pts) in NEWS_POSITIVE_KEYWORDS.items():
            if keyword in text:
                positive_count += 1
                if concept:
                    concept_scores[concept] = concept_scores.get(concept, 0) + pts
                key_headlines.append(f"[+] {news['标题'][:40]}")
                break  # 每条新闻只计一次

        # 利空扫描
        for keyword in NEWS_NEGATIVE_KEYWORDS:
            if keyword in text:
                negative_count += 1
                key_headlines.append(f"[-] {news['标题'][:40]}")
                break

        # 一票否决扫描
        for keyword, industries in NEWS_VETO_KEYWORDS.items():
            if keyword in text:
                veto_industries.update(industries)

    # 整体情绪分 (-10 到 +10)
    total = positive_count + negative_count
    if total == 0:
        overall = 0
    else:
        overall = (positive_count - negative_count) / total * 10

    # 排序热门概念
    hot_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)

    return overall, hot_concepts, list(dict.fromkeys(key_headlines))[:10], veto_industries


# ============================================================
# --go 一键决策
# ============================================================

def go_decision():
    """一键决策：综合所有维度，推荐3只股票 + 明日卖出策略"""

    print()
    print("=" * 65)
    print("  T+1 一键决策系统 v3.0")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # 加载校准权重
    cal_weights, cal_threshold, cal_combos = load_calibrated_weights()
    if cal_weights:
        print(f"  [校准模式] 使用数据驱动权重 (阈值={cal_threshold})")
    else:
        print("  [默认模式] 未校准，使用默认权重。运行 --calibrate 可优化")

    # ── Step 0: 市场级熔断 ──
    print()
    print("[0/13] 市场环境检测...")
    money_effect = calc_money_effect()
    limit_up_pre, limit_down_pre = count_limit_up()
    can_trade, market_reason, market_severity = market_go_nogo(
        money_effect, limit_up=limit_up_pre, limit_down=limit_down_pre)

    up_ratio = money_effect["今日上涨比例"]
    bad_days = money_effect["连续下跌天数"]
    good_days = money_effect["连续上涨天数"]
    print(f"  赚钱效应: {up_ratio:.0%} | 连涨{good_days}天 连跌{bad_days}天")

    if market_severity >= 3:
        print()
        print("  " + "!" * 50)
        print(f"  !! 市场熔断: {market_reason}")
        print("  !! 强烈建议空仓！以下推荐仅供参考")
        print("  " + "!" * 50)
    elif market_severity >= 2:
        print(f"  [警告] {market_reason} — 建议观望或极轻仓")
    elif market_severity >= 1:
        print(f"  [注意] {market_reason} — 需谨慎")
    else:
        print(f"  市场环境正常")

    # ── Step 0.x: 多维度增强分析 ──
    print()
    print("[0.3/16] 大盘趋势+日历效应+增强分析...")
    market_regime, regime_adj, regime_label = calc_market_regime()
    calendar_adj, calendar_label = calc_calendar_adjustment()
    if regime_label:
        print(f"  大盘趋势: {regime_label} ({regime_adj:+d}分)")
    if calendar_label:
        print(f"  日历效应: {calendar_label} ({calendar_adj:+d}分)")

    # ── Step 0.5: 国际大宗商品 + 全球市场 + AH溢价 ──
    print()
    print("[0.5/16] 获取国际大宗商品行情...")
    commodity_data = fetch_commodity_prices()
    if commodity_data:
        for cn, info in commodity_data.items():
            chg = info.get("涨跌幅", 0)
            flag = "⚠" if chg <= -3 else ""
            print(f"  {cn}: {info.get('价格', '?')} ({chg:+.1f}%) {flag}")
    else:
        print("  未获取到大宗商品数据")

    print()
    print("[0.6/16] 获取全球市场行情...")
    global_markets = fetch_global_markets()
    global_penalty, global_warn, global_detail = calc_global_risk(global_markets)
    if global_markets:
        for gn, gi in list(global_markets.items())[:6]:
            gchg = gi.get("涨跌幅", 0)
            gflag = "⚠" if gchg <= -1.5 else ("✓" if gchg >= 1 else "")
            print(f"  {gn}: {gchg:+.1f}% {gflag}")
    if global_warn:
        print(f"  ⚠ {global_warn} (扣{abs(global_penalty)}分)")

    print()
    print("[0.7/16] 获取AH股溢价数据...")
    ah_data = fetch_ah_premium()
    if ah_data:
        high_premium = [(c, d) for c, d in ah_data.items() if d.get("溢价率", 0) >= 100]
        print(f"  AH股 {len(ah_data)} 只，溢价>100%有 {len(high_premium)} 只")
    else:
        print("  未获取到AH溢价数据")

    # ── Step 1: 新闻/政策分析（含国际新闻）──
    print()
    print("[1/16] 扫描今日新闻/政策...")
    news = fetch_news()
    global_news = fetch_global_news()
    all_news = news + global_news
    news_score, hot_concepts, key_headlines, veto_industries = analyze_news_sentiment(all_news)
    global_risk_score, trend_industries, global_headlines = analyze_global_news(global_news)
    print(f"  获取 {len(news)} 条国内新闻 + {len(global_news)} 条国际新闻")
    if veto_industries:
        print(f"  ⚠ 新闻一票否决行业: {', '.join(veto_industries)}")
    if global_risk_score < -5:
        print(f"  ⚠ 国际宏观风险: {global_risk_score}")
    if trend_industries:
        pos = {k: v for k, v in trend_industries.items() if v > 0}
        neg = {k: v for k, v in trend_industries.items() if v < 0}
        if pos:
            print(f"  宏观受益行业: {', '.join([f'{k}(+{v})' for k, v in sorted(pos.items(), key=lambda x: -x[1])[:5]])}")
        if neg:
            print(f"  宏观受损行业: {', '.join([f'{k}({v})' for k, v in sorted(neg.items(), key=lambda x: x[1])[:5]])}")
    if global_headlines:
        print("  国际关键新闻:")
        for h in global_headlines[:3]:
            print(f"    {h}")

    if news_score > 3:
        news_mood = "偏多（利好消息较多）"
    elif news_score > 0:
        news_mood = "中性偏多"
    elif news_score > -3:
        news_mood = "中性偏空"
    else:
        news_mood = "偏空（利空消息较多）"
    print(f"  新闻情绪: {news_mood} ({news_score:+.1f})")

    if hot_concepts:
        top_concepts = [f"{c}({s}分)" for c, s in hot_concepts[:5]]
        print(f"  热点概念: {', '.join(top_concepts)}")
    if key_headlines:
        print("  关键新闻:")
        for h in key_headlines[:5]:
            print(f"    {h}")

    # ── Step 2: 大盘情绪 ──
    print()
    print("[2/12] 分析大盘情绪...")
    sentiment_score, sentiment_info = get_sentiment_score()
    print(f"  {sentiment_info} → 情绪分: {sentiment_score}/15")

    # ── Step 3: 板块/龙虎榜/北向/融资/涨停 ──
    print()
    print("[3/9] 扫描板块资金流向...")
    hot_sectors = get_hot_sectors()
    if hot_sectors:
        print(f"  资金流入板块: {', '.join(list(hot_sectors)[:6])}")

    print()
    print("[4/9] 获取龙虎榜数据...")
    billboard_data = fetch_billboard()
    bb_buy = sum(1 for v in billboard_data.values() if v["龙虎榜净买入"] > 0)
    print(f"  龙虎榜 {len(billboard_data)} 只，净买入 {bb_buy} 只")

    print()
    print("[5/9] 获取北向资金...")
    northbound_total, northbound_info = fetch_northbound_flow()
    if northbound_info:
        print(f"  {northbound_info}")

    print()
    print("[6/9] 获取融资融券...")
    margin_data = fetch_margin_data_top()
    print(f"  融资数据 {len(margin_data)} 只")

    print()
    print("[7/12] 统计涨跌停+涨停池...")
    limit_up, limit_down = count_limit_up()
    limit_up_pool = fetch_limit_up_pool()
    print(f"  涨停 {limit_up} 只，跌停 {limit_down} 只，涨停池 {len(limit_up_pool)} 只")
    # 连板统计
    boards_count = {}
    for v in limit_up_pool.values():
        b = v.get("连板数", 1)
        boards_count[b] = boards_count.get(b, 0) + 1
    if boards_count:
        board_info = " ".join([f"{b}板:{c}只" for b, c in sorted(boards_count.items())])
        print(f"  连板分布: {board_info}")

    print()
    print("[8/12] 获取龙虎榜席位明细...")
    billboard_detail = fetch_billboard_detail()
    inst_count = sum(1 for v in billboard_detail.values() if v.get("机构席位数", 0) > 0)
    print(f"  席位明细 {len(billboard_detail)} 只，机构参与 {inst_count} 只")

    print()
    print("[9/12] 获取股东增持数据...")
    shareholder_data = fetch_shareholder_increase()
    print(f"  近30日增持 {len(shareholder_data)} 只")

    print()
    print("[10/12] 获取行业景气度...")
    industry_data = fetch_industry_prosperity()
    if industry_data:
        top3 = list(industry_data.items())[:3]
        print("  热门行业: " + ", ".join([f"{n}({d['涨跌幅']:+.1f}%)" for n, d in top3]))

    # ── Step 11: 主力资金 + 基本面 + 股票列表 ──
    print()
    print("[11/12] 获取主力资金+基本面+股票列表...")
    capital_rank = fetch_capital_flow_rank(top_n=300)
    fund_batch = fetch_fundamentals_batch()
    stock_list = fetch_stock_list_sina()
    if stock_list.empty:
        print("[错误] 无法获取股票列表")
        return
    print(f"  资金TOP {len(capital_rank)} | 基本面 {len(fund_batch)} | 全市场 {len(stock_list)}")

    # 预筛选
    stock_list = stock_list[
        (stock_list["最新价"] >= 5) & (stock_list["最新价"] <= 80) &
        (stock_list["涨跌幅"] > -3) & (stock_list["涨跌幅"] < 8) &
        (stock_list["换手率"] >= 2) & (stock_list["量比"] >= 0.8)
    ]

    capital_codes = set(capital_rank.keys())
    bb_codes = set(billboard_data.keys())
    # 优先级：龙虎榜净买入 > 资金流入 > 其他
    priority_1 = stock_list[stock_list["代码"].isin(bb_codes & capital_codes)]["代码"].tolist()
    priority_2 = stock_list[stock_list["代码"].isin(capital_codes - bb_codes)]["代码"].tolist()
    priority_3 = stock_list[stock_list["代码"].isin(bb_codes - capital_codes)]["代码"].tolist()
    others = stock_list[~stock_list["代码"].isin(capital_codes | bb_codes)]["代码"].tolist()[:200]
    analyze_list = priority_1 + priority_2 + priority_3 + others
    code_to_row = stock_list.set_index("代码").to_dict("index")

    print(f"  预筛选 {len(analyze_list)} 只")

    # ── Step 12: 逐只评分 ──
    print()
    print("[12/12] 十二维度深度分析...")

    # 新闻热点加分映射
    concept_bonus = {}
    for concept, pts in hot_concepts[:5]:
        concept_bonus[concept] = min(pts, 10)

    results = []
    _go_industry_cache = {}

    def analyze_for_go(code):
        kline = fetch_kline(code, days=120)
        if kline.empty or len(kline) < 30:
            return None
        kline = calc_all_indicators(kline)
        cap_info = capital_rank.get(code)
        sec_score = 10 if code in capital_codes else 3

        # 基本面评分
        fund_info = fund_batch.get(code, {})
        fund_s, fund_d, fund_r = evaluate_fundamentals(fund_info)

        # 额外维度评分（龙虎榜席位/连板/增持/行业）
        stock_ind = ""
        if code in billboard_data or code in limit_up_pool or code in shareholder_data:
            stock_ind = get_stock_industry(code)
            _go_industry_cache[code] = stock_ind

        row = code_to_row.get(code, {})
        name = str(row.get("名称", ""))

        # ★ 新闻一票否决：行业或名称匹配被否决的关键词
        if veto_industries:
            if stock_ind and any(vi in stock_ind for vi in veto_industries):
                return None
            if any(vi in name for vi in veto_industries):
                return None

        # ★ 大宗商品风险
        commodity_pen, commodity_warn = check_commodity_risk(stock_ind, name, commodity_data)

        # ★ 追高风险
        rally_pen, rally_warn = check_consecutive_rally(kline)

        # ★ AH溢价风险
        ah_pen, ah_warn = check_ah_premium_risk(code, name, ah_data)

        # ★ 换手率深度分析
        current_turnover = row.get("换手率", 0) or 0
        turnover_adj, turnover_label = analyze_turnover_depth(kline, current_turnover)

        # ★ 宏观趋势匹配
        macro_adj, macro_label = check_macro_trend_fit(stock_ind, name, trend_industries)

        # ★ 周线趋势 + 历史形态
        wt_adj, wt_label = calc_weekly_trend(kline)
        tc_adj, tc_label = classify_trend_context(kline)

        extra_s, extra_d, extra_r = evaluate_extra_dimensions(
            code, billboard_data, margin_data, northbound_total, limit_up, limit_down,
            limit_up_pool, billboard_detail, shareholder_data, industry_data, stock_ind)

        s, d, r = evaluate_signals_v2(kline, cap_info, sec_score, sentiment_score,
                                       fund_s, fund_d, extra_s, extra_d,
                                       calibrated_weights=cal_weights, golden_combos=cal_combos,
                                       commodity_penalty=commodity_pen, rally_penalty=rally_pen,
                                       global_risk=global_penalty, ah_penalty=ah_pen,
                                       turnover_adj=turnover_adj, macro_adj=macro_adj,
                                       market_regime_adj=regime_adj, weekly_trend_adj=wt_adj,
                                       trend_context_adj=tc_adj, calendar_adj=calendar_adj)

        # 新闻热点加分：如果股票名称匹配热门概念
        news_bonus = 0
        matched_concept = ""
        for concept, bonus in concept_bonus.items():
            if concept in name:
                news_bonus = bonus
                matched_concept = concept
                break

        total_score = s + news_bonus

        # 合并买入理由
        all_reasons = r
        if fund_r:
            all_reasons = r + "；" + fund_r if r != "信号不足" else fund_r
        if extra_r:
            all_reasons += "；" + extra_r if all_reasons != "信号不足" else extra_r
        # 附加风险警告
        risk_warns = [w for w in [commodity_warn, rally_warn, ah_warn, global_warn, turnover_label, macro_label] if w]
        if risk_warns:
            all_reasons += "；⚠" + "，".join(risk_warns)

        # ★ 大盘股提高门槛
        mktcap_yi = row.get("总市值", 0)
        if isinstance(mktcap_yi, (int, float)) and mktcap_yi > 0:
            mktcap_yi = mktcap_yi / 1e8  # API返回元，转亿
        else:
            mktcap_yi = fund_info.get("总市值", 0) or 0
        lc_bonus, lc_adj = apply_largecap_adjustments(mktcap_yi)

        sig_q = d.get("信号质量", "C级")
        has_combo = d.get("黄金组合匹配", False)

        # 判断推荐等级（黄金组合是强推荐的硬门槛）
        go_threshold = (cal_threshold if cal_weights else 100) + lc_bonus
        if total_score >= go_threshold and has_combo:
            rec_level = "强推荐"
        elif total_score >= go_threshold and "C级" not in sig_q:
            rec_level = "推荐"
        elif total_score >= go_threshold:
            rec_level = "弱推荐"
        elif total_score >= go_threshold * 0.7:
            rec_level = "弱推荐"
        else:
            rec_level = "仅参考"

        latest = kline.iloc[-1]
        price = row.get("最新价", latest["收盘"])
        risk = calc_position_and_risk(total_score, sentiment_score,
                                       northbound_total, limit_up, limit_down, price, kline,
                                       largecap_adj=lc_adj)
        return {
            "代码": code, "名称": name,
            "最新价": price,
            "涨跌幅": row.get("涨跌幅", latest["涨跌幅"]),
            "换手率": row.get("换手率", 0),
            "评分": total_score,
            "信号质量": sig_q,
            "推荐等级": rec_level,
            "技术分": s - fund_s - extra_s,
            "基本面分": fund_s,
            "聪明钱分": extra_s,
            "新闻加分": news_bonus,
            "匹配概念": matched_concept,
            "信号": d, "理由": all_reasons,
            "主力净流入占比": cap_info.get("主力净流入占比", 0) if cap_info else 0,
            "kline": kline,
            "基本面": fund_info,
            "行业": _go_industry_cache.get(code, stock_ind),
            "市值亿": mktcap_yi,
            "risk": risk,
        }

    done = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(analyze_for_go, c): c for c in analyze_list}
        for f in as_completed(futures):
            done += 1
            if done % 200 == 0:
                print(f"  进度: {done}/{len(analyze_list)} | 候选: {len(results)}")
            try:
                r = f.result()
                if r:
                    results.append(r)
            except Exception:
                pass

    print(f"  完成！{len(results)} 只候选")

    if not results:
        print()
        print("  今日未扫描到任何候选，市场可能极端低迷。")
        return

    # ── 排序选股 ──
    results.sort(key=lambda x: x["评分"], reverse=True)

    # 选3只：优先不同行业分散风险
    selected = [results[0]]
    used_ind = {results[0].get("行业", "")}
    for r in results[1:]:
        if len(selected) >= 3:
            break
        r_ind = r.get("行业", "")
        if r_ind and r_ind not in used_ind:
            selected.append(r)
            used_ind.add(r_ind)
    for r in results[1:]:
        if len(selected) >= 3:
            break
        if r not in selected:
            selected.append(r)

    # 判断整体推荐强度
    top_score = selected[0]["评分"]
    go_threshold = cal_threshold if cal_weights else 100
    has_strong = any(s.get("推荐等级", "") == "强推荐" for s in selected)

    # ── 输出决策 ──
    print()
    print("=" * 65)
    if has_strong:
        print("  ★★★ 今日操作建议 ★★★")
    elif top_score >= go_threshold:
        print("  ★★ 今日操作建议（信号一般，轻仓参与）★★")
    else:
        print("  ★ 今日 TOP3 参考（信号偏弱，谨慎操作）★")
    print("=" * 65)

    for i, stock in enumerate(selected):
        kline = stock["kline"]
        latest = kline.iloc[-1]
        price = stock["最新价"]

        # 使用预计算的风控结果（已包含大盘股调整）
        risk = stock.get("risk") or calc_position_and_risk(
            stock["评分"], sentiment_score,
            northbound_total, limit_up, limit_down, price, kline)

        rec_level = stock.get("推荐等级", "仅参考")
        print()
        print(f"  ━━━ 第{i+1}只: {stock['名称']}({stock['代码']}) [{rec_level}] ━━━")
        print()
        print(f"  现价: {price:.2f}   今日涨幅: {stock['涨跌幅']:+.2f}%")
        if stock.get("市值亿") and stock["市值亿"] >= 500:
            print(f"  市值: {stock['市值亿']:.0f}亿 (大盘股风控已启用)")
        if stock.get("行业"):
            print(f"  行业: {stock['行业']}")
        print(f"  评分: {stock['评分']}/280")
        score_parts = [f"技术面{stock['技术分']}分", f"基本面{stock['基本面分']}/50分",
                       f"聪明钱{stock['聪明钱分']}/80分"]
        if stock["新闻加分"] > 0:
            score_parts.append(f"新闻+{stock['新闻加分']}分[{stock['匹配概念']}]")
        print(f"        ({', '.join(score_parts)})")

        # 基本面摘要
        fund = stock.get("基本面", {})
        fund_parts = []
        for k in ["营收增长", "利润增长"]:
            v = fund.get(k)
            if v is not None:
                fund_parts.append(f"{k}:{v:+.0f}%")
        for k in ["PE", "PB", "ROE"]:
            v = fund.get(k)
            if v is not None:
                if k == "ROE":
                    fund_parts.append(f"{k}:{v:.1f}%")
                else:
                    fund_parts.append(f"{k}:{v:.1f}")
        if fund_parts:
            print(f"  基本面: {' | '.join(fund_parts)}")

        print(f"  主力资金: {stock['主力净流入占比']:+.1f}%")
        print(f"  买入理由: {stock['理由']}")
        print()

        # 信号详情
        sigs = " | ".join([f"{k}:{v}" for k, v in stock["信号"].items()])
        print(f"  信号: {sigs}")

        print()
        print(f"  ── 智能风控 ──")
        print(f"  风险等级: {risk['风险等级']}  |  ATR波动率: {risk['ATR']:.1f}%")
        print()

        print(f"  ── 买入计划 ──")
        print(f"  买入价:   {price:.2f} (尾盘买入)")
        print(f"  建议仓位: 总资金的 {risk['仓位']}%")
        print()

        print(f"  ── 明日卖出策略（ATR自适应） ──")
        print(f"  止损价:   {risk['止损价']:.2f} ({risk['止损幅度']:+.1f}%，跌破立即卖)")
        print(f"  目标一:   {risk['止盈一']:.2f} (+{risk['止盈一幅度']:.1f}%，卖出1/3仓位)")
        print(f"  目标二:   {risk['止盈二']:.2f} (+{risk['止盈二幅度']:.1f}%，再卖1/3)")
        print(f"  目标三:   {risk['止盈三']:.2f} (+{risk['止盈三幅度']:.1f}%，清仓)")
        print(f"  压力位:   {risk['压力位']:.2f} (近5日最高，临近减仓)")
        print(f"  支撑位:   {risk['支撑位']:.2f} (跌破支撑考虑止损)")
        print()

        print(f"  ── 明日分时卖出节奏 ──")
        if risk['ATR'] > 4:
            print(f"  (该股振幅较大，平均{risk['ATR']:.1f}%，注意波动)")
        print(f"  09:30-09:45  观察开盘: 高开{risk['止盈一幅度']:.0f}%+直接挂单止盈")
        print(f"  09:45-10:30  冲高阶段: 涨{risk['止盈一幅度']:.0f}%卖1/3, 涨{risk['止盈二幅度']:.0f}%再卖1/3")
        print(f"  10:30-13:00  如果横盘: 设好止损价{risk['止损价']:.2f}等待")
        print(f"  13:00-14:30  下午观察: 没涨到目标且开始走弱则卖")
        print(f"  14:30-15:00  尾盘兜底: 无论盈亏，必须清仓！T+1不留隔夜")

    # ── 风险提示 ──
    print()
    print("=" * 65)
    print("  ── 综合研判 ──")
    print()
    print(f"  大盘情绪:   {sentiment_info}")
    print(f"  新闻面:     {news_mood}")
    if hot_concepts:
        print(f"  今日热点:   {', '.join([c for c,_ in hot_concepts[:5]])}")
    if hot_sectors:
        print(f"  资金流入:   {', '.join(list(hot_sectors)[:5])}")
    print()

    # 北向/涨停/融资汇总
    if northbound_info:
        print(f"  北向资金:   {northbound_info}")
    print(f"  涨跌停:     涨停{limit_up}只 跌停{limit_down}只")
    if boards_count:
        board_info = " ".join([f"{b}板:{c}只" for b, c in sorted(boards_count.items())])
        print(f"  连板分布:   {board_info}")
    print()

    # 综合信心度
    confidence = 50
    confidence += min(sentiment_score, 15)
    confidence += min(news_score * 2, 15)
    if selected[0]["评分"] >= 140:
        confidence += 15
    elif selected[0]["评分"] >= 110:
        confidence += 10
    # 聪明钱加分
    if selected[0].get("聪明钱分", 0) >= 30:
        confidence += 5
    # 信号质量加分
    top_sig_q = selected[0].get("信号质量", "")
    if "A级" in top_sig_q:
        confidence += 10
    elif "B级" in top_sig_q:
        confidence += 3
    # 全球风险减分
    if global_penalty < -10:
        confidence -= 10
    elif global_penalty < -5:
        confidence -= 5
    confidence = max(20, min(confidence, 95))

    confidence = int(confidence)
    if confidence >= 75:
        print(f"  信心指数: {confidence}% - 条件较好，可以操作")
    elif confidence >= 55:
        print(f"  信心指数: {confidence}% - 条件一般，轻仓参与")
    else:
        print(f"  信心指数: {confidence}% - 条件较差，建议观望或极轻仓")

    print()
    print("  风险提示:")
    print("  - 以上仅为技术+资金+新闻综合分析，不构成投资建议")
    print("  - 任何策略都无法保证100%盈利")
    print("  - 严格执行止损纪律，-2%必须走")
    print("  - 总仓位不超过可承受亏损的金额")
    print("=" * 65)
    print()

    # 记录推荐
    try:
        from trade_tracker import record_recommendation
        for s in selected:
            record_recommendation("T+1", s["代码"], s["名称"], s["最新价"],
                                   s["评分"], s.get("推荐等级", ""), s["risk"])
    except Exception:
        pass


# ============================================================
# T+5 波段决策（持有2-5天，技术面独立门槛）
# ============================================================

def go5_decision():
    """T+5 波段决策：技术面独立达标 + 基本面/聪明钱调仓位"""

    print()
    print("=" * 65)
    print("  T+5 波段决策系统")
    print("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 65)

    cal_weights, cal_threshold, cal_combos = load_calibrated_weights()
    if cal_weights:
        print(f"  [校准模式] 技术门槛={TECH_GATE_THRESHOLD}")
    else:
        print(f"  [默认模式] 技术门槛={TECH_GATE_THRESHOLD}")

    # ── 市场环境 ──
    print()
    print("[1/8] 市场环境检测...")
    money_effect = calc_money_effect()
    sentiment_score, sentiment_info = get_sentiment_score()
    market_regime, regime_adj, regime_label = calc_market_regime()
    calendar_adj, calendar_label = calc_calendar_adjustment()
    print(f"  {sentiment_info} → 情绪分: {sentiment_score}/15")
    if regime_label:
        print(f"  大盘趋势: {regime_label}")
    if calendar_label:
        print(f"  日历效应: {calendar_label}")

    # ── 新闻 ──
    print()
    print("[2/8] 扫描新闻...")
    news = fetch_news()
    global_news = fetch_global_news()
    all_news = news + global_news
    news_score, hot_concepts, key_headlines, veto_industries = analyze_news_sentiment(all_news)
    global_risk_score, trend_industries, global_headlines = analyze_global_news(global_news)
    news_mood = "偏多" if news_score > 3 else "中性偏多" if news_score > 0 else "中性偏空" if news_score > -3 else "偏空"
    print(f"  新闻情绪: {news_mood}")

    # ── 板块/资金 ──
    print()
    print("[3/8] 获取板块/龙虎榜/北向/融资...")
    hot_sectors = get_hot_sectors()
    billboard_data = fetch_billboard()
    nb_total, nb_info = fetch_northbound_flow()
    margin_data = fetch_margin_data_top()
    limit_up, limit_down = count_limit_up()
    limit_up_pool = fetch_limit_up_pool()
    shareholder_data = fetch_shareholder_increase()
    industry_data = fetch_industry_prosperity()
    print(f"  板块: {', '.join(list(hot_sectors)[:5])}" if hot_sectors else "  板块: 无数据")
    print(f"  龙虎榜 {len(billboard_data)} 只 | 涨停 {limit_up} 跌停 {limit_down}")

    # ── 股票列表+资金 ──
    print()
    print("[4/8] 获取股票列表+资金...")
    capital_rank = fetch_capital_flow_rank(top_n=300)
    fund_batch = fetch_fundamentals_batch()
    stock_list = fetch_stock_list_sina()
    if stock_list.empty:
        print("[错误] 无法获取股票列表")
        return

    # T+5 预筛选（比T+1更宽）
    stock_list = stock_list[
        (stock_list["最新价"] >= 5) & (stock_list["最新价"] <= 100) &
        (stock_list["涨跌幅"] > -5) & (stock_list["涨跌幅"] < 5) &  # 不追涨杀跌
        (stock_list["换手率"] >= 1.5) & (stock_list["量比"] >= 0.6)
    ]
    capital_codes = set(capital_rank.keys())
    bb_codes = set(billboard_data.keys())
    p1 = stock_list[stock_list["代码"].isin(capital_codes & bb_codes)]["代码"].tolist()
    p2 = stock_list[stock_list["代码"].isin(capital_codes - bb_codes)]["代码"].tolist()
    p3 = stock_list[~stock_list["代码"].isin(capital_codes | bb_codes)]["代码"].tolist()[:300]
    analyze_list = p1 + p2 + p3
    code_to_row = stock_list.set_index("代码").to_dict("index")
    print(f"  全市场 {len(stock_list)} → 预筛选 {len(analyze_list)} 只")

    # ── 逐只评分 ──
    print()
    print("[5/8] T+5 门控评分分析...")

    results = []

    def analyze_one_t5(code):
        kl = fetch_kline(code, days=120)
        if kl.empty or len(kl) < 60:
            return None
        kl = calc_all_indicators(kl)
        cap_info = capital_rank.get(code)
        sec_score = 10 if code in capital_codes else 3
        fi = fund_batch.get(code, {})
        fs, fd, fr = evaluate_fundamentals(fi)
        stock_ind = ""
        if code in billboard_data or code in limit_up_pool or code in shareholder_data:
            stock_ind = get_stock_industry(code)
        row = code_to_row.get(code, {})
        name = str(row.get("名称", ""))

        # 新闻否决
        if veto_industries:
            if stock_ind and any(vi in stock_ind for vi in veto_industries):
                return None
            if any(vi in name for vi in veto_industries):
                return None

        # 周线趋势 + 历史形态
        wt_adj, _ = calc_weekly_trend(kl)
        tc_adj, _ = classify_trend_context(kl)

        es, ed, er = evaluate_extra_dimensions(
            code, billboard_data, margin_data, nb_total, limit_up, limit_down,
            limit_up_pool, {}, shareholder_data, industry_data, stock_ind)

        tech_score, pos_mult, d, r, passed = evaluate_signals_gated(
            kl, cap_info, sec_score, sentiment_score, fs, fd, es, ed,
            calibrated_weights=cal_weights, golden_combos=cal_combos,
            market_regime_adj=regime_adj, weekly_trend_adj=wt_adj,
            trend_context_adj=tc_adj, calendar_adj=calendar_adj)

        if not passed:
            return None

        has_combo = d.get("黄金组合匹配", False)
        # 推荐等级
        if has_combo and tech_score >= TECH_GATE_THRESHOLD + 15:
            rec_level = "强推荐"
        elif has_combo or tech_score >= TECH_GATE_THRESHOLD + 10:
            rec_level = "推荐"
        else:
            rec_level = "弱推荐"

        latest = kl.iloc[-1]
        price = float(row.get("最新价", latest["收盘"]))
        mktcap_yi = row.get("总市值", 0)
        if isinstance(mktcap_yi, (int, float)) and mktcap_yi > 0:
            mktcap_yi = mktcap_yi / 1e8
        else:
            mktcap_yi = fi.get("总市值", 0) or 0
        _, lc_adj = apply_largecap_adjustments(mktcap_yi)

        risk = calc_position_and_risk_t5(
            tech_score, pos_mult, sentiment_score, nb_total,
            limit_up, limit_down, price, kl, largecap_adj=lc_adj)

        return {
            "代码": code, "名称": name, "最新价": price,
            "涨跌幅": row.get("涨跌幅", latest.get("涨跌幅", 0)),
            "换手率": row.get("换手率", 0),
            "技术分": tech_score, "仓位倍率": pos_mult,
            "基本面分": fs, "聪明钱分": es,
            "推荐等级": rec_level,
            "信号": d, "理由": r,
            "行业": stock_ind, "市值亿": mktcap_yi,
            "risk": risk, "kline": kl,
            "黄金组合": has_combo,
        }

    done = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(analyze_one_t5, c): c for c in analyze_list}
        for f in as_completed(futures):
            done += 1
            if done % 200 == 0:
                print(f"  进度: {done}/{len(analyze_list)} | 候选: {len(results)}")
            try:
                r = f.result()
                if r:
                    results.append(r)
            except Exception:
                pass

    print(f"  完成！{len(results)} 只通过技术门槛")

    if not results:
        print()
        print("  今日无股票通过技术面独立门槛，暂无波段推荐。")
        return

    # ── 排序选股 ──
    results.sort(key=lambda x: x["技术分"], reverse=True)

    # 优先黄金组合，再分散行业
    combo_results = [r for r in results if r["黄金组合"]]
    other_results = [r for r in results if not r["黄金组合"]]

    selected = []
    used_ind = set()
    for pool in [combo_results, other_results]:
        for r in pool:
            if len(selected) >= 3:
                break
            r_ind = r.get("行业", "")
            if r_ind and r_ind in used_ind:
                continue
            selected.append(r)
            if r_ind:
                used_ind.add(r_ind)
    # 补齐3只
    for r in results:
        if len(selected) >= 3:
            break
        if r not in selected:
            selected.append(r)

    # ── 输出 ──
    print()
    print("=" * 65)
    has_strong = any(s["推荐等级"] == "强推荐" for s in selected)
    if has_strong:
        print("  ★★★ T+5 波段操作建议 ★★★")
    else:
        print("  ★★ T+5 波段参考（无黄金组合强推荐）★★")
    print("=" * 65)

    for i, stock in enumerate(selected):
        rk = stock["risk"]
        rec = stock["推荐等级"]
        print()
        print(f"  ━━━ 第{i+1}只: {stock['名称']}({stock['代码']}) [{rec}] ━━━")
        print()
        print(f"  现价: {stock['最新价']:.2f}   今日涨幅: {stock['涨跌幅']:+.2f}%")
        if stock.get("行业"):
            print(f"  行业: {stock['行业']}")
        d = stock["信号"]
        print(f"  技术评分: {stock['技术分']:.0f}分 {d.get('技术门槛', '')}")
        print(f"  仓位调节: {d.get('仓位倍率', '')} (基本面{stock['基本面分']}/50 聪明钱{stock['聪明钱分']}/80)")
        if stock["黄金组合"]:
            print(f"  黄金组合: {d.get('组合', '')} ★")
        print(f"  买入理由: {stock['理由']}")

        print()
        print(f"  ── T+5 波段风控 ({rk['风险等级']} | ATR {rk['ATR']:.1f}%) ──")
        print(f"  建议仓位: {rk['仓位']}%")
        print(f"  止损价:   {rk['止损价']:.2f} ({rk['止损幅度']:.1f}%)")
        print(f"  目标一:   {rk['止盈一']:.2f} (+{rk['止盈一幅度']:.1f}%) → 卖1/3")
        print(f"  目标二:   {rk['止盈二']:.2f} (+{rk['止盈二幅度']:.1f}%) → 再卖1/3")
        print(f"  目标三:   {rk['止盈三']:.2f} (+{rk['止盈三幅度']:.1f}%) → 清仓")
        print(f"  移动止损: {rk['移动止损']}")
        print(f"  时间止损: {rk['时间止损']}")

        print()
        print(f"  ── 持仓5天操作计划 ──")
        print(f"  Day 1: 尾盘买入，设好止损价 {rk['止损价']:.2f}")
        print(f"  Day 2: 观察趋势确认，不急于操作")
        print(f"  Day 3: 若达目标一 {rk['止盈一']:.2f}，卖出1/3，止损移至成本价")
        print(f"  Day 4: 若达目标二 {rk['止盈二']:.2f}，再卖1/3，止损移至目标一")
        print(f"  Day 5: 收盘前无条件清仓所有剩余仓位！")

    # 市场综合
    print()
    print("=" * 65)
    print(f"  ── T+5 综合研判 ──")
    print()
    print(f"  大盘趋势:   {regime_label or '中性'}")
    print(f"  大盘情绪:   {sentiment_info}")
    print(f"  新闻面:     {news_mood}")
    if hot_concepts:
        top_c = [f"{c}({s}分)" for c, s in hot_concepts[:5]]
        print(f"  今日热点:   {', '.join(top_c)}")
    print(f"  通过门槛:   {len(results)} 只 (黄金组合 {len(combo_results)} 只)")
    print()
    print("  风险提示:")
    print("  - T+5波段持有2-5天，比T+1承受更大波动")
    print("  - 技术面独立达标才推荐，基本面仅调仓位大小")
    print("  - 严格执行止损和第5天清仓纪律")
    print("=" * 65)
    print()

    # 记录推荐
    try:
        from trade_tracker import record_recommendation
        for s in selected:
            record_recommendation("T+5", s["代码"], s["名称"], s["最新价"],
                                   s["技术分"], s["推荐等级"], s["risk"])
    except Exception:
        pass


def backtest_t5(code=None, days=250):
    """T+5 波段回测：持有5天卖出"""
    print("=" * 55)
    print("  T+5 波段策略回测")
    print("=" * 55)
    print()

    cal_weights, cal_threshold, cal_combos = load_calibrated_weights()
    print(f"  技术门槛: {TECH_GATE_THRESHOLD}")

    if code:
        codes = [str(code)]
        print(f"  回测标的: {code}")
    else:
        print("  选取活跃股进行回测...")
        stock_list = fetch_stock_list_sina()
        if stock_list.empty:
            print("[错误] 无法获取股票列表")
            return
        active = stock_list[(stock_list["最新价"] >= 5) & (stock_list["最新价"] <= 100)]
        if active.empty:
            active = stock_list.head(50)
        codes = active.sample(min(50, len(active)), random_state=42)["代码"].tolist()
        print(f"  选取 {len(codes)} 只活跃股")

    print(f"  回测天数: ~{days} 个交易日")
    print(f"  策略: 技术门槛>={TECH_GATE_THRESHOLD} + 高胜率信号 → 买入，持有5天卖出")
    print()

    all_trades = []
    cap_sim = {"主力净流入占比": 3, "主力净流入": 1e7,
               "超大单净流入": 0, "超大单占比": 0, "大单净流入": 0, "大单占比": 0}

    for i, c in enumerate(codes):
        if (i + 1) % 10 == 0:
            print(f"  回测进度: {i+1}/{len(codes)}")

        kline = fetch_kline_long(c, days=days)
        if kline.empty or len(kline) < 60:
            continue
        kline = calc_all_indicators(kline)

        for j in range(30, len(kline) - 5):
            window = kline.iloc[:j+1].copy()
            tech_sc, pos_mult, details, _, passed = evaluate_signals_gated(
                window, cap_sim, 8, 8, 25, {}, 0, {},
                calibrated_weights=cal_weights, golden_combos=cal_combos)

            if not passed:
                continue

            buy_price = kline.iloc[j]["收盘"]
            # 模拟5天持有，检查止损
            atr = kline.iloc[max(0,j-20):j+1]["振幅"].mean()
            stop_pct = -3.0 if atr > 6 else (-4.0 if atr > 4 else -5.0)
            stop_price = buy_price * (1 + stop_pct / 100)

            exit_price = None
            exit_day = 5
            for d in range(1, 6):
                if j + d >= len(kline):
                    break
                day_low = kline.iloc[j + d]["最低"]
                day_close = kline.iloc[j + d]["收盘"]
                if day_low <= stop_price:
                    exit_price = stop_price
                    exit_day = d
                    break
            if exit_price is None:
                idx = min(j + 5, len(kline) - 1)
                exit_price = kline.iloc[idx]["收盘"]

            pnl = (exit_price - buy_price) / buy_price * 100
            all_trades.append({
                "代码": c, "买入日": kline.iloc[j]["日期"],
                "买入价": buy_price, "卖出价": exit_price,
                "收益率": pnl, "持有天数": exit_day,
                "黄金组合": details.get("黄金组合匹配", False),
            })

        time.sleep(0.15)

    if not all_trades:
        print("  回测期间无交易信号")
        return

    df_trades = pd.DataFrame(all_trades)
    total = len(df_trades)
    wins = len(df_trades[df_trades["收益率"] > 0])
    wr = wins / total * 100
    avg_ret = df_trades["收益率"].mean()
    avg_win = df_trades[df_trades["收益率"] > 0]["收益率"].mean() if wins > 0 else 0
    avg_loss = df_trades[df_trades["收益率"] <= 0]["收益率"].mean() if (total - wins) > 0 else 0
    avg_days = df_trades["持有天数"].mean()

    print("  ── T+5 回测结果 ──")
    print()
    print(f"  总交易次数:  {total}")
    print(f"  胜率:        {wr:.1f}%")
    print(f"  平均收益:    {avg_ret:+.2f}%")
    print(f"  平均盈利:    {avg_win:+.2f}%")
    print(f"  平均亏损:    {avg_loss:+.2f}%")
    if avg_loss != 0:
        print(f"  盈亏比:      {abs(avg_win/avg_loss):.2f}")
    print(f"  平均持有:    {avg_days:.1f}天")
    print()

    # 黄金组合子集
    combo_trades = df_trades[df_trades["黄金组合"] == True]
    if len(combo_trades) > 0:
        cw = len(combo_trades[combo_trades["收益率"] > 0])
        print(f"  ── 黄金组合子集 ──")
        print(f"  交易次数:    {len(combo_trades)}")
        print(f"  胜率:        {cw / len(combo_trades) * 100:.1f}%")
        print(f"  平均收益:    {combo_trades['收益率'].mean():+.2f}%")
    print()

    # 分布
    bins = [-999, -5, -3, -1, 0, 2, 5, 8, 999]
    labels = ["<-5%", "-5~-3%", "-3~-1%", "-1~0%", "0~2%", "2~5%", "5~8%", ">8%"]
    df_trades["区间"] = pd.cut(df_trades["收益率"], bins=bins, labels=labels)
    dist = df_trades["区间"].value_counts().sort_index()
    print("  ── 收益分布 ──")
    table = []
    max_count = dist.max() if len(dist) > 0 else 1
    for interval, count in dist.items():
        bar = "#" * int(count / max_count * 15)
        table.append([interval, count, f"{count/total*100:.1f}%", bar])
    print(tabulate(table, headers=["区间", "次数", "占比", "分布"], tablefmt="grid"))
    print()


# ============================================================
# ETF 波段分析（持有1-5天）
# ============================================================

# 场内ETF市场代码
FS_ETF = "b:MK0021,b:MK0022,b:MK0023,b:MK0024"

# ETF主题分类（名称关键词 -> 板块）
ETF_THEME_MAP = {
    "沪深300": "大盘", "上证50": "大盘", "中证500": "中盘", "中证1000": "小盘",
    "创业板": "成长", "科创": "成长", "双创": "成长",
    "半导体": "半导体", "芯片": "半导体", "集成电路": "半导体",
    "新能源": "新能源", "光伏": "新能源", "锂电": "新能源", "碳中和": "新能源", "电力": "新能源",
    "军工": "军工", "国防": "军工",
    "医药": "医药", "医疗": "医药", "生物": "医药", "创新药": "医药", "中药": "医药",
    "消费": "消费", "食品": "消费", "白酒": "消费", "家电": "消费",
    "金融": "金融", "银行": "金融", "券商": "金融", "证券": "金融", "保险": "金融",
    "地产": "地产", "房地产": "地产", "基建": "地产",
    "科技": "科技", "人工智能": "AI", "AI": "AI", "机器人": "机器人",
    "通信": "通信", "5G": "通信",
    "有色": "资源", "煤炭": "资源", "钢铁": "资源", "石油": "资源", "黄金": "黄金",
    "农业": "农业", "养殖": "农业",
    "港股": "港股", "恒生": "港股", "H股": "港股",
    "纳斯达克": "美股", "标普": "美股", "中概": "美股",
    "债": "债券", "国债": "债券",
    "红利": "红利", "高股息": "红利", "央企": "央企", "国企": "央企",
}


def get_etf_theme(name):
    """根据ETF名称识别主题"""
    for keyword, theme in ETF_THEME_MAP.items():
        if keyword in name:
            return theme
    return "其他"


def fetch_etf_list():
    """获取场内ETF列表（含实时行情）"""
    url = "http://push2delay.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": 1, "pz": 2000, "po": 1, "np": 1, "fltt": 2, "invt": 2,
        "fid": "f3", "fs": FS_ETF, "ut": UT,
        "fields": "f12,f14,f2,f3,f5,f6,f7,f8,f10,f15,f16,f17,f18,f20,f21"
    }
    try:
        resp = _get(url, params=params, headers={"Referer": "http://data.eastmoney.com/"})
        data = resp.json()
        if not (data.get("data") and data["data"].get("diff")):
            return pd.DataFrame()
        df = pd.DataFrame(data["data"]["diff"])
        col_map = {
            "f12": "代码", "f14": "名称", "f2": "最新价", "f3": "涨跌幅",
            "f5": "成交量", "f6": "成交额", "f7": "振幅", "f8": "换手率",
            "f10": "量比", "f15": "最高", "f16": "最低",
            "f17": "今开", "f18": "昨收", "f20": "总市值", "f21": "流通市值"
        }
        df = df.rename(columns=col_map)
        df["代码"] = df["代码"].astype(str)
        df["名称"] = df["名称"].astype(str)
        df = df[df["最新价"] != "-"]
        num_cols = ["最新价", "涨跌幅", "成交量", "成交额", "振幅", "换手率", "量比",
                    "最高", "最低", "今开", "昨收", "总市值", "流通市值"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["最新价"])
        df = df[df["最新价"] > 0]
        # 过滤掉规模太小和成交太低的
        df = df[df["成交额"] > 1e7]  # 日成交额 > 1000万
        # 添加主题分类
        df["主题"] = df["名称"].apply(get_etf_theme)
        # 过滤债券/货币ETF（不适合波段）
        df = df[~df["主题"].isin(["债券"])]
        return df
    except Exception as e:
        print(f"[错误] 获取ETF列表失败: {e}")
        return pd.DataFrame()


def evaluate_etf_trend(df):
    """
    ETF趋势评分（满分100），波段策略核心：追强势趋势
    1. 趋势强度 (0-30): MA排列 + 价格位置
    2. 动量信号 (0-25): MACD/KDJ/RSI
    3. 量价配合 (0-20): 放量突破 vs 缩量回调
    4. 波段位置 (0-15): 是否在合适的买入位
    5. 波动率 (0-10): 适中波动最佳
    """
    if len(df) < 30:
        return 0, {}, ""

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    details = {}
    reasons = []

    # === 1. 趋势强度 (0-30) ===
    trend_s = 0
    ma5, ma10, ma20 = latest.get("MA5", 0), latest.get("MA10", 0), latest.get("MA20", 0)
    ma60 = latest.get("MA60", 0) if "MA60" in df.columns else 0
    price = latest["收盘"]

    if ma5 > ma10 > ma20 and (ma60 == 0 or ma20 > ma60):
        trend_s = 30; details["趋势"] = "强势多头排列"
        reasons.append("多头排列")
    elif ma5 > ma10 > ma20:
        trend_s = 25; details["趋势"] = "多头排列"
        reasons.append("多头排列")
    elif ma5 > ma10 and price > ma20:
        trend_s = 20; details["趋势"] = "短期多头+站上MA20"
    elif ma5 > ma10:
        trend_s = 15; details["趋势"] = "短期多头"
    elif price > ma5 and prev["收盘"] <= prev.get("MA5", 0):
        trend_s = 18; details["趋势"] = "突破MA5"
        reasons.append("突破MA5")
    elif price > ma20:
        trend_s = 10; details["趋势"] = "站上MA20"
    else:
        trend_s = 3; details["趋势"] = "偏弱"

    # 近5日涨幅加成
    if len(df) >= 5:
        chg_5d = (price - df.iloc[-5]["收盘"]) / df.iloc[-5]["收盘"] * 100
        if 2 < chg_5d < 8:
            trend_s = min(trend_s + 5, 30)
            details["5日涨幅"] = f"+{chg_5d:.1f}%"
        elif chg_5d > 8:
            trend_s = max(trend_s - 3, 0)  # 短期涨太多扣分
            details["5日涨幅"] = f"+{chg_5d:.1f}%(偏高)"
    score += trend_s

    # === 2. 动量信号 (0-25) ===
    momentum_s = 0
    # MACD
    if latest["DIF"] > latest["DEA"] and prev["DIF"] <= prev["DEA"]:
        momentum_s += 12; details["MACD"] = "金叉"
        reasons.append("MACD金叉")
    elif latest["DIF"] > latest["DEA"] and latest["MACD柱"] > prev["MACD柱"]:
        momentum_s += 8; details["MACD"] = "红柱放大"
    elif latest["DIF"] > latest["DEA"]:
        momentum_s += 5; details["MACD"] = "多头"
    elif (latest["DIF"] - latest["DEA"]) > (prev["DIF"] - prev["DEA"]):
        momentum_s += 3; details["MACD"] = "即将金叉"
    else:
        details["MACD"] = "空头"

    # KDJ
    if latest["K"] > latest["D"] and latest["K"] < 80:
        momentum_s += 6; details["KDJ"] = f"多头(K={latest['K']:.0f})"
    elif latest["J"] < 20:
        momentum_s += 8; details["KDJ"] = f"超卖(J={latest['J']:.0f})"
        reasons.append("KDJ超卖")
    elif latest["K"] > 80:
        momentum_s += 1; details["KDJ"] = "超买注意"
    else:
        details["KDJ"] = f"K={latest['K']:.0f}"

    # RSI
    rsi = latest.get("RSI6", 50)
    if not pd.isna(rsi):
        if 40 <= rsi <= 65:
            momentum_s += 7; details["RSI"] = f"{rsi:.0f}(健康)"
        elif 30 <= rsi < 40:
            momentum_s += 5; details["RSI"] = f"{rsi:.0f}(偏低)"
        elif rsi < 30:
            momentum_s += 4; details["RSI"] = f"{rsi:.0f}(超卖)"
            reasons.append(f"RSI超卖{rsi:.0f}")
        elif 65 < rsi <= 75:
            momentum_s += 3; details["RSI"] = f"{rsi:.0f}(偏强)"
        else:
            momentum_s += 0; details["RSI"] = f"{rsi:.0f}(超买)"
    score += min(momentum_s, 25)

    # === 3. 量价配合 (0-20) ===
    vol_s = 0
    vr = latest.get("量比计算", 1.0)
    chg = latest.get("涨跌幅", 0)
    if not pd.isna(vr):
        if 1.5 <= vr <= 4.0 and chg > 1:
            vol_s = 20; details["量价"] = f"放量上涨(量比{vr:.1f})"
            reasons.append("放量上涨")
        elif 1.2 <= vr <= 5.0 and chg > 0:
            vol_s = 15; details["量价"] = f"量升价涨(量比{vr:.1f})"
        elif 0.8 <= vr < 1.5 and chg > 0:
            vol_s = 10; details["量价"] = f"温和上涨(量比{vr:.1f})"
        elif vr < 0.6:
            vol_s = 3; details["量价"] = f"缩量(量比{vr:.1f})"
        elif chg < -1 and vr > 1.5:
            vol_s = 2; details["量价"] = f"放量下跌(量比{vr:.1f})"
        else:
            vol_s = 5; details["量价"] = f"量比{vr:.1f}"
    score += vol_s

    # === 4. 波段位置 (0-15) ===
    pos_s = 0
    boll_lower = latest.get("BOLL下", 0)
    boll_mid = latest.get("BOLL中", 0)
    boll_upper = latest.get("BOLL上", 0)
    if boll_mid > 0:
        if boll_lower > 0 and price <= boll_lower * 1.02:
            pos_s = 15; details["位置"] = "布林下轨(超跌)"
            reasons.append("触及布林下轨")
        elif price < boll_mid and price > boll_lower:
            pos_s = 12; details["位置"] = "布林中下轨(偏低)"
        elif price > boll_mid and price < boll_upper * 0.98:
            pos_s = 8; details["位置"] = "布林中上轨"
        elif price >= boll_upper * 0.98:
            pos_s = 2; details["位置"] = "布林上轨(偏高)"
        else:
            pos_s = 6; details["位置"] = "布林中轨附近"
    # 回调买入：从高点回落3-8%是好位置
    if len(df) >= 10:
        high_10 = df.tail(10)["最高"].max()
        pullback = (high_10 - price) / high_10 * 100
        if 3 <= pullback <= 8 and trend_s >= 15:
            pos_s = min(pos_s + 5, 15)
            details["回调"] = f"从高点回调{pullback:.1f}%"
            reasons.append(f"回调{pullback:.1f}%")
    score += pos_s

    # === 5. 波动率 (0-10) ===
    atr = df.tail(20)["振幅"].mean() if len(df) >= 20 else 3.0
    if 1.5 <= atr <= 4.0:
        score += 10; details["波动"] = f"ATR {atr:.1f}%(适中)"
    elif 1.0 <= atr < 1.5:
        score += 6; details["波动"] = f"ATR {atr:.1f}%(偏低)"
    elif 4.0 < atr <= 6.0:
        score += 5; details["波动"] = f"ATR {atr:.1f}%(偏高)"
    else:
        score += 2; details["波动"] = f"ATR {atr:.1f}%"

    reason_str = "；".join(reasons) if reasons else "信号不足"
    return score, details, reason_str


def calc_etf_risk(score, price, kline, holding_days=3):
    """
    ETF波段风控：止损止盈 + 仓位
    持有1-5天，比T+1宽松
    """
    recent_20 = kline.tail(20)
    atr = recent_20["振幅"].mean()
    recent_5 = kline.tail(5)

    # 止损：比T+1宽，给波段空间
    if atr > 4:
        stop_pct = -3.0
    elif atr > 2.5:
        stop_pct = -4.0
    else:
        stop_pct = -5.0

    # 止盈：根据ATR和持有天数
    tp1_pct = max(3.0, atr * 0.8)
    tp2_pct = max(5.0, atr * 1.5)
    tp3_pct = max(8.0, atr * 2.5)

    # 仓位
    if score >= 80:
        position = 50
    elif score >= 65:
        position = 40
    elif score >= 50:
        position = 30
    else:
        position = 20

    return {
        "仓位": position,
        "止损价": round(price * (1 + stop_pct / 100), 3),
        "止损幅度": stop_pct,
        "止盈一": round(price * (1 + tp1_pct / 100), 3),
        "止盈一幅度": tp1_pct,
        "止盈二": round(price * (1 + tp2_pct / 100), 3),
        "止盈二幅度": tp2_pct,
        "止盈三": round(price * (1 + tp3_pct / 100), 3),
        "止盈三幅度": tp3_pct,
        "ATR": atr,
        "支撑位": recent_5["最低"].min(),
        "压力位": recent_5["最高"].max(),
        "建议持有": f"{holding_days}天",
        "风险等级": "低风险" if atr < 2.5 else "中等风险" if atr < 4 else "较高风险",
    }


def scan_etf(top_n=10):
    """扫描场内ETF，筛选波段机会"""
    print("=" * 65)
    print("  ETF 波段扫描（持有1-5天）")
    print(f"  扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print()

    # 市场环境
    print("[1/4] 检测市场环境...")
    sentiment_score, sentiment_info = get_sentiment_score()
    print(f"  {sentiment_info}")
    money_effect = calc_money_effect()
    up_ratio = money_effect["今日上涨比例"]
    print(f"  赚钱效应: {up_ratio:.0%}")

    # 板块资金
    print("[2/4] 获取板块资金流向...")
    concept_flow = fetch_sector_flow("concept", top_n=10)
    industry_flow = fetch_sector_flow("industry", top_n=10)
    hot_concepts = [s["板块名称"] for s in concept_flow if s.get("主力净流入", 0) > 0][:5]
    hot_industries = [s["板块名称"] for s in industry_flow if s.get("主力净流入", 0) > 0][:5]
    if hot_concepts:
        print(f"  热门概念: {', '.join(hot_concepts)}")
    if hot_industries:
        print(f"  热门行业: {', '.join(hot_industries)}")

    # ETF列表
    print("[3/4] 获取ETF列表...")
    etf_list = fetch_etf_list()
    if etf_list.empty:
        print("[错误] 无法获取ETF列表")
        return
    print(f"  共 {len(etf_list)} 只活跃ETF")

    # 逐只分析
    print(f"[4/4] 分析ETF趋势...")
    results = []
    code_to_row = etf_list.set_index("代码").to_dict("index")

    def analyze_etf(code):
        kline = fetch_kline(code, days=120)
        if kline.empty or len(kline) < 30:
            return None
        kline = calc_all_indicators(kline)
        s, d, r = evaluate_etf_trend(kline)

        # 板块热度加分
        row = code_to_row.get(code, {})
        name = str(row.get("名称", ""))
        theme = row.get("主题", get_etf_theme(name))
        theme_bonus = 0
        # 检查ETF主题是否在热门板块中
        for hot in hot_concepts + hot_industries:
            if theme in hot or hot in name:
                theme_bonus = 8
                break
        total = s + theme_bonus + min(sentiment_score, 10)

        if total >= 45:
            return {
                "代码": code, "名称": name, "主题": theme,
                "最新价": row.get("最新价", 0), "涨跌幅": row.get("涨跌幅", 0),
                "成交额": row.get("成交额", 0), "换手率": row.get("换手率", 0),
                "评分": total, "趋势分": s, "板块加分": theme_bonus,
                "信号": d, "理由": r, "kline": kline,
            }
        return None

    done = 0
    total = len(etf_list)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(analyze_etf, c): c for c in etf_list["代码"].tolist()}
        for future in as_completed(futures):
            done += 1
            if done % 50 == 0:
                print(f"  进度: {done}/{total}")
            try:
                r = future.result()
                if r:
                    results.append(r)
            except Exception:
                pass

    if not results:
        print("\n  今日未发现合适的ETF波段机会，建议观望。")
        return

    results.sort(key=lambda x: x["评分"], reverse=True)
    results = results[:top_n]

    print(f"\n  分析完成！{len(results)} 只候选")
    print()
    print("=" * 65)
    print(f"  ETF 波段候选 TOP {len(results)}（满分120）")
    print("=" * 65)
    print()

    table_data = []
    for r in results:
        vol_str = f"{r['成交额']/1e8:.1f}亿" if r["成交额"] >= 1e8 else f"{r['成交额']/1e4:.0f}万"
        table_data.append([
            r["代码"], r["名称"], r["主题"],
            f"{r['最新价']:.3f}", f"{r['涨跌幅']:+.2f}%",
            vol_str, r["评分"], r["理由"][:28],
        ])
    print(tabulate(
        table_data,
        headers=["代码", "名称", "主题", "现价", "涨跌幅", "成交额", "评分", "买入理由"],
        tablefmt="simple_grid", stralign="center",
    ))

    # 详细信号
    print()
    print("── 详细信号 ──")
    for r in results[:5]:
        sigs = " | ".join([f"{k}:{v}" for k, v in r["信号"].items()])
        print(f"  {r['代码']} {r['名称']}: {sigs}")

    print()
    print("提示：")
    print("  - 评分>=80 强烈关注，>=60 可关注，>=45 谨慎")
    print("  - ETF波段持有1-5天，止损-3%~-5%，止盈+5%~+10%")
    print("  - 优先选成交额大、趋势明确的ETF")
    print("  - 跟踪板块热度，热点退潮及时离场")


def etf_go(top_n=3):
    """ETF波段一键决策：推荐3只ETF + 波段操作计划"""
    print()
    print("=" * 65)
    print("  ETF 波段决策系统")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # 市场环境
    print()
    print("[1/5] 检测市场环境...")
    sentiment_score, sentiment_info = get_sentiment_score()
    money_effect = calc_money_effect()
    nb_total, nb_info = fetch_northbound_flow()
    limit_up, limit_down = count_limit_up()
    print(f"  {sentiment_info}")
    if nb_info:
        print(f"  {nb_info}")
    print(f"  涨停 {limit_up} 跌停 {limit_down}")

    # 新闻
    print()
    print("[2/5] 扫描新闻热点...")
    news = fetch_news()
    news_score, hot_concepts, key_headlines, _ = analyze_news_sentiment(news)
    news_mood = "偏多" if news_score > 3 else "中性偏多" if news_score > 0 else "中性偏空" if news_score > -3 else "偏空"
    print(f"  新闻情绪: {news_mood}")
    if hot_concepts:
        print(f"  热点: {', '.join([c for c,_ in hot_concepts[:5]])}")

    # 板块
    print()
    print("[3/5] 获取板块资金流向...")
    concept_flow = fetch_sector_flow("concept", top_n=15)
    industry_flow = fetch_sector_flow("industry", top_n=15)
    hot_concepts_list = [s["板块名称"] for s in concept_flow if s.get("主力净流入", 0) > 0][:6]
    hot_ind_list = [s["板块名称"] for s in industry_flow if s.get("主力净流入", 0) > 0][:6]
    if hot_concepts_list:
        print(f"  概念资金流入: {', '.join(hot_concepts_list)}")
    if hot_ind_list:
        print(f"  行业资金流入: {', '.join(hot_ind_list)}")
    all_hot = hot_concepts_list + hot_ind_list

    # ETF扫描
    print()
    print("[4/5] 获取ETF列表...")
    etf_list = fetch_etf_list()
    if etf_list.empty:
        print("[错误] 无法获取ETF列表")
        return
    print(f"  {len(etf_list)} 只活跃ETF")

    print()
    print("[5/5] 分析ETF趋势...")
    results = []
    code_to_row = etf_list.set_index("代码").to_dict("index")

    def analyze(code):
        kline = fetch_kline(code, days=120)
        if kline.empty or len(kline) < 30:
            return None
        kline = calc_all_indicators(kline)
        s, d, r = evaluate_etf_trend(kline)
        row = code_to_row.get(code, {})
        name = str(row.get("名称", ""))
        theme = row.get("主题", get_etf_theme(name))
        theme_bonus = 0
        for hot in all_hot:
            if theme in hot or hot in name:
                theme_bonus = 8
                break
        # 新闻热点加分
        news_bonus = 0
        for concept, pts in (hot_concepts or [])[:5]:
            if concept in name or concept in theme:
                news_bonus = min(pts // 3, 5)
                break
        total = s + theme_bonus + news_bonus + min(sentiment_score, 10)
        if total >= 45:
            price = float(kline.iloc[-1]["收盘"])
            risk = calc_etf_risk(total, price, kline)
            return {
                "代码": code, "名称": name, "主题": theme,
                "最新价": row.get("最新价", 0), "涨跌幅": row.get("涨跌幅", 0),
                "成交额": row.get("成交额", 0),
                "评分": total, "趋势分": s,
                "板块加分": theme_bonus, "新闻加分": news_bonus,
                "信号": d, "理由": r, "risk": risk,
            }
        return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(analyze, c): c for c in etf_list["代码"].tolist()}
        for f in as_completed(futures):
            try:
                r = f.result()
                if r:
                    results.append(r)
            except Exception:
                pass

    if not results:
        print("\n  今日无合适ETF，建议观望。")
        return

    results.sort(key=lambda x: x["评分"], reverse=True)

    # 选3只，优先不同主题
    selected = [results[0]]
    used_themes = {results[0]["主题"]}
    for r in results[1:]:
        if len(selected) >= top_n:
            break
        if r["主题"] not in used_themes:
            selected.append(r)
            used_themes.add(r["主题"])
    for r in results[1:]:
        if len(selected) >= top_n:
            break
        if r not in selected:
            selected.append(r)

    print()
    print("=" * 65)
    print("  ★★★ ETF 波段操作建议（持有1-5天）★★★")
    print("=" * 65)

    for i, etf in enumerate(selected):
        rk = etf["risk"]
        print()
        print(f"  ━━━ 第{i+1}只: {etf['名称']}({etf['代码']}) ━━━")
        print()
        print(f"  现价: {etf['最新价']:.3f}   今日: {etf['涨跌幅']:+.2f}%   主题: {etf['主题']}")
        vol = f"{etf['成交额']/1e8:.1f}亿" if etf['成交额'] >= 1e8 else f"{etf['成交额']/1e4:.0f}万"
        print(f"  成交额: {vol}")
        print(f"  评分: {etf['评分']}/120 (趋势{etf['趋势分']}分"
              + (f" +板块{etf['板块加分']}" if etf['板块加分'] else "")
              + (f" +新闻{etf['新闻加分']}" if etf['新闻加分'] else "") + ")")
        print(f"  买入理由: {etf['理由']}")
        print()

        # 信号详情
        sigs = " | ".join([f"{k}:{v}" for k, v in etf["信号"].items()])
        print(f"  信号: {sigs}")
        print()

        print(f"  ── 波段风控 ──")
        print(f"  风险等级: {rk['风险等级']}  |  ATR: {rk['ATR']:.1f}%  |  建议持有: {rk['建议持有']}")
        print()
        print(f"  ── 买入计划 ──")
        print(f"  买入价:   {etf['最新价']:.3f}")
        print(f"  建议仓位: 总资金的 {rk['仓位']}%")
        print()
        print(f"  ── 持仓期间操作 ──")
        print(f"  止损价:   {rk['止损价']:.3f} ({rk['止损幅度']:+.1f}%，跌破必须卖)")
        print(f"  目标一:   {rk['止盈一']:.3f} (+{rk['止盈一幅度']:.1f}%，卖出1/3)")
        print(f"  目标二:   {rk['止盈二']:.3f} (+{rk['止盈二幅度']:.1f}%，再卖1/3)")
        print(f"  目标三:   {rk['止盈三']:.3f} (+{rk['止盈三幅度']:.1f}%，清仓)")
        print(f"  支撑位:   {rk['支撑位']:.3f}")
        print(f"  压力位:   {rk['压力位']:.3f}")
        print()
        print(f"  ── 每日检查清单 ──")
        print(f"  Day 1: 买入后设好止损{rk['止损价']:.3f}，观察趋势是否延续")
        print(f"  Day 2: 涨到目标一{rk['止盈一']:.3f}卖1/3，移动止损到成本价")
        print(f"  Day 3: 继续持有或到目标二{rk['止盈二']:.3f}再卖1/3")
        print(f"  Day 4-5: 趋势减弱或到期 → 清仓，不恋战")

    # 综合
    print()
    print("=" * 65)
    print("  ── 综合研判 ──")
    print()
    print(f"  大盘: {sentiment_info}")
    print(f"  新闻面: {news_mood}")
    if all_hot:
        print(f"  资金流入: {', '.join(all_hot[:6])}")
    if nb_info:
        print(f"  北向: {nb_info}")
    print()
    print("  风险提示:")
    print("  - ETF波段持有1-5天，不同于T+1当天进出")
    print("  - 严格止损，跌破止损价必须卖")
    print("  - 板块热度退潮时及时离场，不要死扛")
    print("  - 以上仅为技术分析，不构成投资建议")
    print("=" * 65)
    print()


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    top_n = 15
    args = sys.argv[1:]

    if not args:
        scan_market_v2(top_n)
    elif args[0] in ("--help", "-h"):
        print(__doc__)
    elif args[0] == "--go":
        go_decision()
    elif args[0] == "--backtest":
        code = args[1] if len(args) > 1 and args[1].isdigit() else None
        backtest(code)
    elif args[0] == "--go5":
        go5_decision()
    elif args[0] == "--backtest5":
        code = args[1] if len(args) > 1 and args[1].isdigit() else None
        backtest_t5(code)
    elif args[0] == "--track":
        try:
            from trade_tracker import show_trade_report
            show_trade_report()
        except Exception as e:
            print(f"[错误] {e}")
    elif args[0] == "--update-trades":
        try:
            from trade_tracker import update_outcomes
            update_outcomes()
        except Exception as e:
            print(f"[错误] {e}")
    elif args[0] == "--calibrate":
        calibrate()
    elif args[0] == "--market":
        show_market_sentiment()
    elif args[0] == "--sector":
        show_sector_flow()
    elif args[0] == "--etf":
        scan_etf()
    elif args[0] == "--etf-go":
        etf_go()
    elif args[0] == "--top":
        top_n = int(args[1]) if len(args) > 1 else 15
        scan_market_v2(top_n)
    elif args[0].isdigit() and len(args[0]) == 6:
        analyze_single_v2(args[0])
    else:
        print(f"未识别的参数: {args[0]}")
        print(__doc__)
