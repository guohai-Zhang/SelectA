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
    python3 t1_trader.py --go           # 一键决策：直接推荐2只股票+明日卖出策略
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
    """通过新浪接口获取A股列表（含实时行情）"""
    # 使用东方财富delay接口获取列表（它返回完整的股票列表）
    url = "http://push2delay.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": 1, "pz": 6000, "po": 1, "np": 1, "fltt": 2, "invt": 2,
        "fid": "f3", "fs": FS_ALL_A, "ut": UT,
        "fields": "f12,f14,f2,f3,f5,f6,f7,f8,f9,f10,f15,f16,f17,f18,f20,f21"
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
            "f9": "市盈率", "f10": "量比", "f15": "最高", "f16": "最低",
            "f17": "今开", "f18": "昨收", "f20": "总市值", "f21": "流通市值"
        }
        df = df.rename(columns=col_map)
        df["代码"] = df["代码"].astype(str)
        df["名称"] = df["名称"].astype(str)
        df = df[df["最新价"] != "-"]
        df = df[~df["名称"].str.contains("ST|退市|N |C ", na=False)]
        num_cols = ["最新价","涨跌幅","成交量","成交额","振幅","换手率","量比","最高","最低","今开","昨收"]
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
        return 15, {}, ""  # 无数据给中性分

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
        growth_s = 8; details["成长"] = "无数据"
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
        profit_s += 4; details["ROE"] = "无数据"

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
        val_s += 5; details["PE"] = "无数据"

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

def calc_position_and_risk(stock_score, sentiment_score, northbound_total, limit_up, limit_down, price, kline):
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

    # 动态止损：根据波动率调整
    if atr > 6:
        stop_pct = -1.5  # 高波动股票紧止损
    elif atr > 4:
        stop_pct = -2.0
    else:
        stop_pct = -2.5  # 低波动可以宽一点

    stop_loss = price * (1 + stop_pct / 100)

    # 动态止盈：三级止盈
    tp1_pct = max(2.0, atr * 0.5)   # 第一目标：半个ATR
    tp2_pct = max(3.5, atr * 0.8)   # 第二目标
    tp3_pct = max(5.0, atr * 1.2)   # 激进目标

    tp1 = price * (1 + tp1_pct / 100)
    tp2 = price * (1 + tp2_pct / 100)
    tp3 = price * (1 + tp3_pct / 100)

    # 支撑/压力
    support = recent_5["最低"].min()
    resistance = recent_5["最高"].max()

    # 仓位计算
    base_position = 40  # 基础仓位40%
    if stock_score >= 180:
        base_position = 50
    elif stock_score >= 150:
        base_position = 45
    elif stock_score >= 120:
        base_position = 35
    else:
        base_position = 25

    # 市场环境调整
    if sentiment_score >= 10:
        base_position += 10
    elif sentiment_score <= 3:
        base_position -= 15

    if northbound_total > 30e4:
        base_position += 5
    elif northbound_total < -30e4:
        base_position -= 10

    if limit_down > limit_up and limit_down > 30:
        base_position -= 10

    position = max(10, min(base_position, 50))  # 限制在10%-50%

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
    rsv = (df["收盘"] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    df["K"] = rsv.ewm(com=m1-1, adjust=False).mean()
    df["D"] = df["K"].ewm(com=m2-1, adjust=False).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]
    return df

def calc_rsi(df, period=6):
    delta = df["收盘"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"RSI{period}"] = 100 - (100 / (1 + rs))
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
    signals["RSI_回升"] = (not pd.isna(rsi) and 30 <= rsi <= 50)
    signals["RSI_超卖"] = (not pd.isna(rsi) and 20 <= rsi < 30)
    signals["RSI_强势"] = (not pd.isna(rsi) and 50 < rsi <= 70)
    signals["RSI_极度超卖"] = (not pd.isna(rsi) and rsi < 20)
    signals["RSI_偏高"] = (not pd.isna(rsi) and rsi > 70)
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

    return signals


# 默认权重（后续会被校准回测覆盖）
DEFAULT_WEIGHTS = {
    # MACD — 互斥，取最高
    "MACD_金叉": 18, "MACD_红柱放大": 12, "MACD_即将金叉": 7, "MACD_红柱": 4,
    # KDJ — 互斥
    "KDJ_超卖金叉": 15, "KDJ_J超卖": 12, "KDJ_多头": 7,
    # RSI — 互斥
    "RSI_回升": 12, "RSI_超卖": 10, "RSI_强势": 8, "RSI_极度超卖": 8, "RSI_偏高": 2,
    # 均线 — 互斥
    "MA_多头排列": 10, "MA_短期多头": 7, "MA_突破MA5": 8, "MA_站上MA5": 4,
    # 布林 — 互斥
    "BOLL_触及下轨": 5, "BOLL_中轨下方": 2,
    # 量价 — 互斥
    "量价_放量上涨": 20, "量价_量升价涨": 15, "量价_放量横盘": 10, "量价_缩量上涨": 8, "量价_严重缩量": 2,
    # 形态 — 互斥（阳包阴 > 锤子线 > 早晨之星 > 阳线）
    "形态_阳包阴": 10, "形态_锤子线": 8, "形态_早晨之星": 7, "形态_阳线": 2,
    # 资金 — 互斥
    "资金_大幅流入": 30, "资金_明显流入": 24, "资金_温和流入": 18, "资金_小幅流入": 10,
}

# 互斥组：同组内只取第一个触发的信号
SIGNAL_GROUPS = {
    "MACD": ["MACD_金叉", "MACD_红柱放大", "MACD_即将金叉", "MACD_红柱"],
    "KDJ": ["KDJ_超卖金叉", "KDJ_J超卖", "KDJ_多头"],
    "RSI": ["RSI_回升", "RSI_超卖", "RSI_强势", "RSI_极度超卖", "RSI_偏高"],
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
# 综合评分引擎 v3.0 — 信号驱动
# ============================================================

def evaluate_signals_v2(df, capital_info=None, sector_score=0, sentiment_score=0,
                        fundamental_score=0, fundamental_details=None,
                        extra_score=0, extra_details=None,
                        chip_data=None, tail_flow=None, leader_score=0, leader_label="",
                        research_data=None, calibrated_weights=None, golden_combos=None):
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

    # === 黄金组合加分 ===
    combo_bonus = 0
    matched_combo = ""
    if golden_combos and fired:
        fired_set = set(fired)
        for combo in golden_combos:
            combo_sigs = set(combo.get("signals", []))
            if combo_sigs.issubset(fired_set):
                wr = combo.get("win_rate", 0)
                bonus = int((wr - 0.55) * 50)  # 58%→1.5, 65%→5, 70%→7.5
                if bonus > combo_bonus:
                    combo_bonus = min(bonus, 15)
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
        reasons.append(matched_combo)

    reason_str = "；".join(reasons) if reasons else "信号不足"
    return score, details, reason_str


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
            sc, _, _ = evaluate_signals_v2(window, cap_sim, 8, 8, 25, {},
                                           calibrated_weights=cal_weights,
                                           golden_combos=cal_combos)
            if sc >= bt_threshold:
                buy_price = kline.iloc[j]["收盘"]
                sell_price = kline.iloc[j+1]["收盘"]
                pnl = (sell_price - buy_price) / buy_price * 100
                all_trades.append({
                    "代码": c, "买入日": kline.iloc[j]["日期"],
                    "买入价": buy_price, "卖出价": sell_price,
                    "收益率": pnl,
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


def calibrate(days=250, sample_size=80):
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
    for code in codes[:50]:  # 用子集
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
    for code in codes[:30]:  # 用子集加速
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
}

# 利空关键词
NEWS_NEGATIVE_KEYWORDS = [
    "暴跌", "崩盘", "大跌", "跌停", "利空", "制裁", "打压",
    "退市", "爆雷", "违规", "处罚", "下调", "减持", "清仓",
    "战争", "冲突", "加息",  # 美联储加息
]


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
    """
    concept_scores = {}  # 概念 -> 累计分数
    positive_count = 0
    negative_count = 0
    key_headlines = []

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

    # 整体情绪分 (-10 到 +10)
    total = positive_count + negative_count
    if total == 0:
        overall = 0
    else:
        overall = (positive_count - negative_count) / total * 10

    # 排序热门概念
    hot_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)

    return overall, hot_concepts, list(dict.fromkeys(key_headlines))[:10]


# ============================================================
# --go 一键决策
# ============================================================

def go_decision():
    """一键决策：综合所有维度，推荐2只股票 + 明日卖出策略"""

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
    print("[0/12] 市场环境检测...")
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
        print("  !! 今日禁止操作，空仓观望！")
        print("  " + "!" * 50)
        return
    elif market_severity >= 2:
        print(f"  [警告] {market_reason} — 建议观望或极轻仓")
    elif market_severity >= 1:
        print(f"  [注意] {market_reason} — 需谨慎")
    else:
        print(f"  市场环境正常")

    # ── Step 1: 新闻/政策分析 ──
    print()
    print("[1/12] 扫描今日新闻/政策...")
    news = fetch_news()
    news_score, hot_concepts, key_headlines = analyze_news_sentiment(news)
    print(f"  获取 {len(news)} 条新闻")

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
        extra_s, extra_d, extra_r = evaluate_extra_dimensions(
            code, billboard_data, margin_data, northbound_total, limit_up, limit_down,
            limit_up_pool, billboard_detail, shareholder_data, industry_data, stock_ind)

        s, d, r = evaluate_signals_v2(kline, cap_info, sec_score, sentiment_score,
                                       fund_s, fund_d, extra_s, extra_d,
                                       calibrated_weights=cal_weights, golden_combos=cal_combos)

        # 新闻热点加分：如果股票名称匹配热门概念
        row = code_to_row.get(code, {})
        name = str(row.get("名称", ""))
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

        go_threshold = cal_threshold if cal_weights else 100
        if total_score >= go_threshold:
            latest = kline.iloc[-1]
            return {
                "代码": code, "名称": name,
                "最新价": row.get("最新价", latest["收盘"]),
                "涨跌幅": row.get("涨跌幅", latest["涨跌幅"]),
                "换手率": row.get("换手率", 0),
                "评分": total_score,
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
            }
        return None

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
        print("  今日无合适标的，建议观望。")
        return

    # ── 排序选股 ──
    results.sort(key=lambda x: x["评分"], reverse=True)

    # 选2只：优先不同行业分散风险
    selected = [results[0]]
    first_industry = results[0].get("行业", "")
    for r in results[1:]:
        r_ind = r.get("行业", "")
        if r_ind and r_ind != first_industry:
            selected.append(r)
            break
    if len(selected) < 2 and len(results) >= 2:
        selected.append(results[1])

    # ── 输出决策 ──
    print()
    print("=" * 65)
    print("  ★★★ 今日操作建议 ★★★")
    print("=" * 65)

    for i, stock in enumerate(selected):
        kline = stock["kline"]
        latest = kline.iloc[-1]
        price = stock["最新价"]

        # 使用智能风控系统计算动态仓位和止损止盈
        risk = calc_position_and_risk(stock["评分"], sentiment_score,
                                       northbound_total, limit_up, limit_down, price, kline)

        print()
        print(f"  ━━━ 第{i+1}只: {stock['名称']}({stock['代码']}) ━━━")
        print()
        print(f"  现价: {price:.2f}   今日涨幅: {stock['涨跌幅']:+.2f}%")
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
    elif args[0] == "--calibrate":
        calibrate()
    elif args[0] == "--market":
        show_market_sentiment()
    elif args[0] == "--sector":
        show_sector_flow()
    elif args[0] == "--top":
        top_n = int(args[1]) if len(args) > 1 else 15
        scan_market_v2(top_n)
    elif args[0].isdigit() and len(args[0]) == 6:
        analyze_single_v2(args[0])
    else:
        print(f"未识别的参数: {args[0]}")
        print(__doc__)
