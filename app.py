"""
A股 T+1 短线分析工具 - Web API
FastAPI 后端，包装 t1_trader.py 的分析功能
"""

import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import t1_trader as t1

# ── 低资源模式（0.1 CPU / 512MB）──
# 线程数和候选上限适配低配机器
WORKERS = 8          # 线程池大小（与CLI一致）
GO_OTHERS_CAP = 200  # go决策中无信号股票上限（与CLI一致）
SCAN_OTHERS_CAP = 100  # scan中非优先股上限

# ── 简单缓存 ──
_cache = {}
_cache_ttl = {}

def cached_call(key, func, ttl=120, *args, **kwargs):
    """缓存函数调用结果，ttl秒后过期。异常时返回旧缓存（如有）"""
    now = time.time()
    if key in _cache and now - _cache_ttl.get(key, 0) < ttl:
        return _cache[key]
    try:
        result = func(*args, **kwargs)
        _cache[key] = result
        _cache_ttl[key] = now
        return result
    except Exception:
        if key in _cache:
            return _cache[key]
        raise


def _get_shared_data(include_news=False):
    """获取共享市场数据（缓存2分钟，避免重复请求）"""
    data = {}
    data["cal_weights"], data["cal_threshold"], data["cal_combos"] = (
        cached_call("cal_weights", t1.load_calibrated_weights, 600))
    data["sentiment_score"], data["sentiment_info"] = (
        cached_call("sentiment", t1.get_sentiment_score, 120))
    data["hot_sectors"] = cached_call("hot_sectors", t1.get_hot_sectors, 120)
    data["capital_rank"] = cached_call("capital_rank", t1.fetch_capital_flow_rank, 120, 300)
    data["billboard_data"] = cached_call("billboard", t1.fetch_billboard, 120)
    data["billboard_detail"] = cached_call("billboard_detail", t1.fetch_billboard_detail, 120)
    data["nb_total"], data["nb_info"] = cached_call("northbound", t1.fetch_northbound_flow, 120)
    data["margin_data"] = cached_call("margin", t1.fetch_margin_data_top, 120)
    data["limit_up"], data["limit_down"] = cached_call("limits", t1.count_limit_up, 120)
    data["limit_up_pool"] = cached_call("limit_up_pool", t1.fetch_limit_up_pool, 120)
    data["shareholder_data"] = cached_call("shareholder", t1.fetch_shareholder_increase, 120)
    data["industry_data"] = cached_call("industry", t1.fetch_industry_prosperity, 120)
    data["fund_batch"] = cached_call("fund_batch", t1.fetch_fundamentals_batch, 180)
    data["stock_list"] = cached_call("stock_list", t1.fetch_stock_list_sina, 120)
    data["commodity_data"] = cached_call("commodity", t1.fetch_commodity_prices, 300)
    data["global_markets"] = cached_call("global_markets", t1.fetch_global_markets, 300)
    data["global_penalty"], data["global_warn"], _ = t1.calc_global_risk(data["global_markets"])
    data["ah_data"] = cached_call("ah_premium", t1.fetch_ah_premium, 300)
    if include_news:
        news = cached_call("news", t1.fetch_news, 300)
        global_news = cached_call("global_news", t1.fetch_global_news, 300)
        all_news = news + global_news
        data["news_score"], data["hot_concepts"], data["key_headlines"], data["veto_industries"] = (
            t1.analyze_news_sentiment(all_news))
        _, data["trend_industries"], data["global_headlines"] = t1.analyze_global_news(global_news)
    return data


app = FastAPI(title="A股 T+1 短线分析")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── 工具函数 ──

def jsonable(obj):
    """递归转换 numpy/pandas 类型为 Python 原生类型"""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def sse(data):
    """格式化 SSE 事件"""
    return f"data: {json.dumps(jsonable(data), ensure_ascii=False)}\n\n"


# ── 页面入口 ──

@app.get("/")
def index():
    return FileResponse("static/index.html")


# ── API: 大盘情绪 ──

@app.get("/api/market")
def api_market():
    try:
        sentiments = t1.fetch_market_sentiment()
        score, info = t1.get_sentiment_score()
        money = t1.calc_money_effect()
        limit_up, limit_down = t1.count_limit_up()
        nb_total, nb_info = t1.fetch_northbound_flow()
        can_trade, reason, severity = t1.market_go_nogo(
            money, limit_up=limit_up, limit_down=limit_down)

        # 全球市场 + 大宗商品（缓存5分钟）
        global_markets = cached_call("global_markets", t1.fetch_global_markets, 300)
        commodity_data = cached_call("commodity", t1.fetch_commodity_prices, 300)

        return jsonable({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "indices": sentiments,
            "sentiment": {"score": score, "info": info},
            "money_effect": money,
            "limits": {"up": limit_up, "down": limit_down},
            "northbound": {"total": nb_total, "info": nb_info},
            "market_status": {"can_trade": can_trade, "reason": reason, "severity": severity},
            "global_markets": global_markets,
            "commodity": commodity_data,
        })
    except Exception as e:
        return {"error": str(e)}


# ── API: 板块资金 ──

@app.get("/api/sector")
def api_sector():
    try:
        concept = t1.fetch_sector_flow("concept", top_n=20)
        industry = t1.fetch_sector_flow("industry", top_n=20)
        return jsonable({"concept": concept, "industry": industry})
    except Exception as e:
        return {"error": str(e)}


# ── API: 单股分析 ──

@app.get("/api/stock/{code}")
def api_stock(code: str):
    try:
        code = str(code).zfill(6)
        kline = t1.fetch_kline(code, days=120)
        if kline.empty or len(kline) < 30:
            return {"error": f"数据不足，无法分析 {code}"}

        kline = t1.calc_all_indicators(kline)
        latest = kline.iloc[-1]

        # 使用共享缓存 + 仅个股专属数据单独获取
        sd = _get_shared_data()
        cap_rank = sd["capital_rank"]
        sentiment_score = sd["sentiment_score"]
        fund_batch = sd["fund_batch"]
        billboard_data, billboard_detail = sd["billboard_data"], sd["billboard_detail"]
        nb_total, nb_info = sd["nb_total"], sd["nb_info"]
        margin_data = sd["margin_data"]
        limit_up, limit_down = sd["limit_up"], sd["limit_down"]
        limit_up_pool = sd["limit_up_pool"]
        shareholder_data = sd["shareholder_data"]
        industry_data = sd["industry_data"]
        commodity_data = sd["commodity_data"]
        global_penalty = sd["global_penalty"]
        ah_data = sd["ah_data"]
        cal_weights, cal_threshold, cal_combos = sd["cal_weights"], sd["cal_threshold"], sd["cal_combos"]

        # 个股专属数据（并行获取）
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            f_cap_flow = ex.submit(t1.fetch_stock_capital_flow, code)
            f_fund_detail = ex.submit(t1.fetch_fundamental_detail, code)
            f_ind_code = ex.submit(t1.get_stock_industry, code)
            f_rt = ex.submit(t1.fetch_realtime_sina, [code])

        cap_flow = f_cap_flow.result()
        fund_detail = f_fund_detail.result()
        stock_industry = f_ind_code.result()
        rt = f_rt.result()

        name = rt.get(code, {}).get("名称", code)

        # 基本面
        fund_info = fund_batch.get(code, {})
        if fund_detail:
            fund_info.update({k: v for k, v in fund_detail.items() if v is not None})
        fund_s, fund_d, fund_r = t1.evaluate_fundamentals(fund_info)

        # 多维度风险检测
        commodity_pen, commodity_warn = t1.check_commodity_risk(stock_industry, name, commodity_data)
        rally_pen, rally_warn = t1.check_consecutive_rally(kline)
        ah_pen, ah_warn = t1.check_ah_premium_risk(code, name, ah_data)
        global_warn_str = sd["global_warn"]
        turnover_adj_val, turnover_label = t1.analyze_turnover_depth(kline, 0)

        # 额外维度
        extra_s, extra_d, extra_r = t1.evaluate_extra_dimensions(
            code, billboard_data, margin_data, nb_total, limit_up, limit_down,
            limit_up_pool, billboard_detail, shareholder_data, industry_data, stock_industry)

        # 综合评分
        cap_info = cap_rank.get(code)
        score, details, reasons = t1.evaluate_signals_v2(
            kline, cap_info, 8, sentiment_score, fund_s, fund_d, extra_s, extra_d,
            calibrated_weights=cal_weights, golden_combos=cal_combos,
            commodity_penalty=commodity_pen, rally_penalty=rally_pen,
            global_risk=global_penalty, ah_penalty=ah_pen,
            turnover_adj=turnover_adj_val)

        # ★ 大盘股风控
        mktcap_yi = fund_info.get("总市值", 0) or 0
        _, lc_adj = t1.apply_largecap_adjustments(mktcap_yi)

        # 风控
        price = float(latest["收盘"])
        risk = t1.calc_position_and_risk(
            score, sentiment_score, nb_total, limit_up, limit_down, price, kline,
            largecap_adj=lc_adj)

        # 近10日K线
        recent_kline = []
        for _, r in kline.tail(10).iterrows():
            recent_kline.append({
                "日期": r["日期"], "开盘": r["开盘"], "收盘": r["收盘"],
                "最高": r["最高"], "最低": r["最低"], "涨跌幅": r["涨跌幅"],
            })

        # 近5日资金流
        recent_flow = []
        if not cap_flow.empty and len(cap_flow) >= 3:
            for _, r in cap_flow.tail(5).iterrows():
                recent_flow.append({"日期": r["日期"], "主力净流入": r["主力净流入"]})

        # 风险警告
        risk_warnings = [w for w in [commodity_warn, rally_warn, ah_warn, global_warn_str, turnover_label] if w]

        return jsonable({
            "code": code, "name": name,
            "price": price,
            "change": float(latest["涨跌幅"]) if not pd.isna(latest["涨跌幅"]) else 0,
            "score": round(score, 1), "max_score": 280,
            "signals": details,
            "reasons": reasons,
            "fundamentals": fund_info,
            "fund_score": fund_s,
            "extra_score": extra_s,
            "risk": risk,
            "risk_warnings": risk_warnings,
            "sentiment": {"score": sentiment_score, "info": sd["sentiment_info"]},
            "northbound": {"total": nb_total, "info": nb_info},
            "industry": stock_industry,
            "commodity": commodity_data,
            "global_markets": sd["global_markets"],
            "kline": recent_kline,
            "capital_flow": recent_flow,
            "capital_info": cap_info,
        })
    except Exception as e:
        return {"error": str(e)}


# ── API: 全市场扫描 (SSE) ──

@app.get("/api/scan")
def api_scan(top: int = Query(15, ge=1, le=50)):
    def generate():
        try:
            yield sse({"type": "progress", "msg": "检测市场环境...", "pct": 5})
            money_effect = t1.calc_money_effect()
            limit_up_pre, limit_down_pre = t1.count_limit_up()
            can_trade, market_reason, severity = t1.market_go_nogo(
                money_effect, limit_up=limit_up_pre, limit_down=limit_down_pre)

            if severity >= 3:
                yield sse({"type": "blocked", "msg": f"市场熔断: {market_reason}，今日禁止操作！"})
                return

            yield sse({"type": "progress", "msg": "获取市场数据（缓存加速）...", "pct": 15})
            sd = _get_shared_data()
            cal_weights, cal_threshold, cal_combos = sd["cal_weights"], sd["cal_threshold"], sd["cal_combos"]
            sentiment_score, sentiment_info = sd["sentiment_score"], sd["sentiment_info"]
            capital_rank = sd["capital_rank"]
            billboard_data, billboard_detail = sd["billboard_data"], sd["billboard_detail"]
            nb_total, nb_info = sd["nb_total"], sd["nb_info"]
            margin_data = sd["margin_data"]
            limit_up, limit_down = sd["limit_up"], sd["limit_down"]
            limit_up_pool = sd["limit_up_pool"]
            shareholder_data = sd["shareholder_data"]
            industry_data = sd["industry_data"]
            fund_batch = sd["fund_batch"]
            commodity_data, global_penalty, ah_data = sd["commodity_data"], sd["global_penalty"], sd["ah_data"]
            stock_list = sd["stock_list"]
            if stock_list.empty:
                yield sse({"type": "error", "msg": "无法获取股票列表"})
                return

            # 预筛选
            stock_list = t1._prefilter(stock_list, t1.PREFILTER_SCAN)
            capital_codes = set(capital_rank.keys())
            priority = stock_list[stock_list["代码"].isin(capital_codes)]["代码"].tolist()
            others = stock_list[~stock_list["代码"].isin(capital_codes)]["代码"].tolist()[:SCAN_OTHERS_CAP]
            ordered = priority + others
            code_to_row = stock_list.set_index("代码").to_dict("index")
            total = len(ordered)

            yield sse({"type": "progress", "msg": f"分析 {total} 只股票...", "pct": 35})

            results = []
            done = [0]

            def analyze_one(code):
                kl = t1.fetch_kline(code, days=120)
                if kl.empty or len(kl) < 30:
                    return None
                kl = t1.calc_all_indicators(kl)
                cap_info = capital_rank.get(code)
                sec_score = 8 if code in capital_codes else 3
                fi = fund_batch.get(code, {})
                fs, fd, fr = t1.evaluate_fundamentals(fi)
                stock_ind = ""
                if code in billboard_data or code in limit_up_pool or code in shareholder_data:
                    stock_ind = t1.get_stock_industry(code)
                row = code_to_row.get(code, {})
                name = str(row.get("名称", ""))
                commodity_pen, commodity_warn = t1.check_commodity_risk(stock_ind, name, commodity_data)
                rally_pen, rally_warn = t1.check_consecutive_rally(kl)
                ah_pen, ah_warn = t1.check_ah_premium_risk(code, name, ah_data)
                current_turnover = row.get("换手率", 0) or 0
                turnover_adj_val, turnover_label = t1.analyze_turnover_depth(kl, current_turnover)
                es, ed, er = t1.evaluate_extra_dimensions(
                    code, billboard_data, margin_data, nb_total, limit_up, limit_down,
                    limit_up_pool, billboard_detail, shareholder_data, industry_data, stock_ind)
                s, d, r = t1.evaluate_signals_v2(
                    kl, cap_info, sec_score, sentiment_score, fs, fd, es, ed,
                    calibrated_weights=cal_weights, golden_combos=cal_combos,
                    commodity_penalty=commodity_pen, rally_penalty=rally_pen,
                    global_risk=global_penalty, ah_penalty=ah_pen,
                    turnover_adj=turnover_adj_val)
                combined = r
                if fr:
                    combined = r + "；" + fr if r != "信号不足" else fr
                if er:
                    combined += "；" + er if combined != "信号不足" else er
                risk_warns = [w for w in [commodity_warn, rally_warn, ah_warn, turnover_label] if w]
                if risk_warns:
                    combined += "；⚠" + "，".join(risk_warns)
                mktcap_yi = row.get("总市值", 0)
                if isinstance(mktcap_yi, (int, float)) and mktcap_yi > 0:
                    mktcap_yi = mktcap_yi / 1e8
                else:
                    mktcap_yi = fi.get("总市值", 0) or 0
                lc_bonus, _ = t1.apply_largecap_adjustments(mktcap_yi)
                scan_th = (cal_threshold if cal_weights else 110) + lc_bonus
                if s >= scan_th:
                    return {
                        "代码": code, "名称": name,
                        "最新价": row.get("最新价", 0), "涨跌幅": row.get("涨跌幅", 0),
                        "换手率": row.get("换手率", 0), "评分": round(s, 1),
                        "信号": d, "理由": combined,
                        "主力净流入占比": cap_info.get("主力净流入占比", 0) if cap_info else 0,
                    }
                return None

            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                futures = {executor.submit(analyze_one, c): c for c in ordered}
                for future in as_completed(futures):
                    done[0] += 1
                    try:
                        r = future.result()
                        if r:
                            results.append(r)
                    except Exception:
                        pass
                    if done[0] % 50 == 0:
                        pct = 35 + int(done[0] / total * 60)
                        yield sse({"type": "progress",
                                   "msg": f"进度 {done[0]}/{total}，已发现 {len(results)} 只候选",
                                   "pct": min(pct, 95)})

            results.sort(key=lambda x: x["评分"], reverse=True)
            results = results[:top]

            yield sse({"type": "result", "data": results,
                       "market_status": {"can_trade": can_trade, "reason": market_reason,
                                         "severity": severity},
                       "sentiment": {"score": sentiment_score, "info": sentiment_info},
                       "total_analyzed": total, "total_found": len(results)})
        except Exception as e:
            yield sse({"type": "error", "msg": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── API: 一键决策 (SSE) ──

@app.get("/api/go")
def api_go():
    def generate():
        try:
            yield sse({"type": "progress", "msg": "检测市场环境...", "pct": 3})
            money_effect = t1.calc_money_effect()
            limit_up_pre, limit_down_pre = t1.count_limit_up()
            can_trade, market_reason, severity = t1.market_go_nogo(
                money_effect, limit_up=limit_up_pre, limit_down=limit_down_pre)

            market_blocked = severity >= 3

            # 多维度增强分析
            market_regime, regime_adj, regime_label = t1.calc_market_regime()
            calendar_adj, calendar_label = t1.calc_calendar_adjustment()

            yield sse({"type": "progress", "msg": "获取市场数据（缓存加速）...", "pct": 10})
            sd = _get_shared_data(include_news=True)
            cal_weights, cal_threshold, cal_combos = sd["cal_weights"], sd["cal_threshold"], sd["cal_combos"]
            sentiment_score, sentiment_info = sd["sentiment_score"], sd["sentiment_info"]
            hot_sectors = sd["hot_sectors"]
            capital_rank = sd["capital_rank"]
            billboard_data, billboard_detail = sd["billboard_data"], sd["billboard_detail"]
            nb_total, nb_info = sd["nb_total"], sd["nb_info"]
            margin_data = sd["margin_data"]
            limit_up, limit_down = sd["limit_up"], sd["limit_down"]
            limit_up_pool = sd["limit_up_pool"]
            shareholder_data = sd["shareholder_data"]
            industry_data = sd["industry_data"]
            fund_batch = sd["fund_batch"]
            commodity_data = sd["commodity_data"]
            global_markets = sd["global_markets"]
            global_penalty, global_warn = sd["global_penalty"], sd["global_warn"]
            ah_data = sd["ah_data"]
            news_score = sd["news_score"]
            hot_concepts = sd["hot_concepts"]
            key_headlines = sd["key_headlines"]
            veto_industries = sd["veto_industries"]
            trend_industries = sd.get("trend_industries", {})
            global_headlines = sd.get("global_headlines", [])
            news_mood = "偏多" if news_score > 3 else "中性偏多" if news_score > 0 else "中性偏空" if news_score > -3 else "偏空"

            stock_list = sd["stock_list"]
            if stock_list.empty:
                yield sse({"type": "error", "msg": "无法获取股票列表"})
                return

            # 预筛选
            stock_list = t1._prefilter(stock_list, t1.PREFILTER_GO)
            capital_codes = set(capital_rank.keys())
            bb_codes = set(billboard_data.keys())
            p1 = stock_list[stock_list["代码"].isin(bb_codes & capital_codes)]["代码"].tolist()
            p2 = stock_list[stock_list["代码"].isin(capital_codes - bb_codes)]["代码"].tolist()
            p3 = stock_list[stock_list["代码"].isin(bb_codes - capital_codes)]["代码"].tolist()
            p4 = stock_list[~stock_list["代码"].isin(capital_codes | bb_codes)]["代码"].tolist()[:GO_OTHERS_CAP]
            analyze_list = p1 + p2 + p3 + p4
            code_to_row = stock_list.set_index("代码").to_dict("index")

            # 新闻热点加分
            concept_bonus = {}
            for concept, pts in hot_concepts[:5]:
                concept_bonus[concept] = min(pts, 10)

            yield sse({"type": "progress", "msg": f"深度分析 {len(analyze_list)} 只股票...", "pct": 35})

            results = []
            done = [0]
            total = len(analyze_list)

            def analyze_for_go(code):
                kl = t1.fetch_kline(code, days=120)
                if kl.empty or len(kl) < 30:
                    return None
                kl = t1.calc_all_indicators(kl)
                cap_info = capital_rank.get(code)
                sec_score = 10 if code in capital_codes else 3
                fi = fund_batch.get(code, {})
                fs, fd, fr = t1.evaluate_fundamentals(fi)
                stock_ind = ""
                if code in billboard_data or code in limit_up_pool or code in shareholder_data:
                    stock_ind = t1.get_stock_industry(code)
                row = code_to_row.get(code, {})
                name = str(row.get("名称", ""))
                # ★ 新闻一票否决
                if veto_industries:
                    if stock_ind and any(vi in stock_ind for vi in veto_industries):
                        return None
                    if any(vi in name for vi in veto_industries):
                        return None
                # ★ 多维度风险检测
                commodity_pen, commodity_warn = t1.check_commodity_risk(stock_ind, name, commodity_data)
                rally_pen, rally_warn = t1.check_consecutive_rally(kl)
                ah_pen, ah_warn = t1.check_ah_premium_risk(code, name, ah_data)
                current_turnover = row.get("换手率", 0) or 0
                turnover_adj_val, turnover_label = t1.analyze_turnover_depth(kl, current_turnover)
                macro_adj_val, macro_label = t1.check_macro_trend_fit(stock_ind, name, trend_industries)
                wt_adj, _ = t1.calc_weekly_trend(kl)
                tc_adj, _ = t1.classify_trend_context(kl)
                es, ed, er = t1.evaluate_extra_dimensions(
                    code, billboard_data, margin_data, nb_total, limit_up, limit_down,
                    limit_up_pool, billboard_detail, shareholder_data, industry_data, stock_ind)
                s, d, r = t1.evaluate_signals_v2(
                    kl, cap_info, sec_score, sentiment_score, fs, fd, es, ed,
                    calibrated_weights=cal_weights, golden_combos=cal_combos,
                    commodity_penalty=commodity_pen, rally_penalty=rally_pen,
                    global_risk=global_penalty, ah_penalty=ah_pen,
                    turnover_adj=turnover_adj_val, macro_adj=macro_adj_val,
                    market_regime_adj=regime_adj, weekly_trend_adj=wt_adj,
                    trend_context_adj=tc_adj, calendar_adj=calendar_adj)
                news_bonus = 0
                matched_concept = ""
                for concept, bonus in concept_bonus.items():
                    if concept in name:
                        news_bonus = bonus
                        matched_concept = concept
                        break
                total_score = s + news_bonus
                all_reasons = r
                if fr:
                    all_reasons = r + "；" + fr if r != "信号不足" else fr
                if er:
                    all_reasons += "；" + er if all_reasons != "信号不足" else er
                risk_warns = [w for w in [commodity_warn, rally_warn, ah_warn, global_warn, turnover_label, macro_label] if w]
                if risk_warns:
                    all_reasons += "；⚠" + "，".join(risk_warns)
                # ★ 大盘股提高门槛
                mktcap_yi = row.get("总市值", 0)
                if isinstance(mktcap_yi, (int, float)) and mktcap_yi > 0:
                    mktcap_yi = mktcap_yi / 1e8
                else:
                    mktcap_yi = fi.get("总市值", 0) or 0
                lc_bonus, lc_adj = t1.apply_largecap_adjustments(mktcap_yi)
                go_th = (cal_threshold if cal_weights else 100) + lc_bonus
                sig_q = d.get("信号质量", "C级")
                has_combo = d.get("黄金组合匹配", False)
                # 推荐等级（与CLI go_decision一致）
                if total_score >= go_th and has_combo:
                    rec_level = "强推荐"
                elif total_score >= go_th and "C级" not in sig_q:
                    rec_level = "推荐"
                elif total_score >= go_th:
                    rec_level = "弱推荐"
                elif total_score >= go_th * 0.8 and "C级" not in sig_q:
                    rec_level = "观望"
                else:
                    rec_level = "仅参考"
                latest = kl.iloc[-1]
                price = float(row.get("最新价", latest["收盘"]))
                risk = t1.calc_position_and_risk(
                    total_score, sentiment_score, nb_total, limit_up, limit_down, price, kl,
                    largecap_adj=lc_adj)
                return {
                    "代码": code, "名称": name,
                    "最新价": price,
                    "涨跌幅": row.get("涨跌幅", latest.get("涨跌幅", 0)),
                    "换手率": row.get("换手率", 0),
                    "评分": round(total_score, 1),
                    "信号质量": sig_q,
                    "推荐等级": rec_level,
                    "技术分": round(s - fs - es, 1),
                    "基本面分": fs,
                    "聪明钱分": es,
                    "新闻加分": news_bonus,
                    "匹配概念": matched_concept,
                    "信号": d, "理由": all_reasons,
                    "主力净流入占比": cap_info.get("主力净流入占比", 0) if cap_info else 0,
                    "基本面": fi,
                    "行业": stock_ind,
                    "risk": risk,
                }

            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                futures = {executor.submit(analyze_for_go, c): c for c in analyze_list}
                for f in as_completed(futures):
                    done[0] += 1
                    try:
                        r = f.result()
                        if r:
                            results.append(r)
                    except Exception:
                        pass
                    if done[0] % 100 == 0:
                        pct = 35 + int(done[0] / total * 55)
                        yield sse({"type": "progress",
                                   "msg": f"进度 {done[0]}/{total}，候选 {len(results)} 只",
                                   "pct": min(pct, 92)})

            if not results:
                yield sse({"type": "result", "data": {
                    "selected": [], "confidence": 20,
                    "msg": "今日未扫描到任何候选，市场可能极端低迷。"
                }})
                return

            # ── 排序选股（与CLI go_decision一致：纯评分排序+行业分散）──
            results.sort(key=lambda x: x["评分"], reverse=True)

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

            # 信心指数（与CLI go_decision一致）
            confidence = 50
            confidence += min(sentiment_score, 15)
            confidence += min(int(news_score * 2), 15)
            if selected[0]["评分"] >= 140:
                confidence += 15
            elif selected[0]["评分"] >= 110:
                confidence += 10
            if selected[0].get("聪明钱分", 0) >= 30:
                confidence += 5
            top_sig_q = selected[0].get("信号质量", "")
            if "A级" in top_sig_q:
                confidence += 10
            elif "B级" in top_sig_q:
                confidence += 3
            # market_blocked惩罚已移除，与CLI一致
            if global_penalty < -10:
                confidence -= 10
            elif global_penalty < -5:
                confidence -= 5
            confidence = max(20, min(confidence, 95))

            # 连板统计
            boards_count = {}
            for v in limit_up_pool.values():
                b = v.get("连板数", 1)
                boards_count[b] = boards_count.get(b, 0) + 1

            # 记录推荐
            try:
                from trade_tracker import record_recommendation
                for s in selected:
                    record_recommendation("T+1", s["代码"], s["名称"], s["最新价"],
                                           s["评分"], s["推荐等级"], s.get("信号"), s.get("risk"))
            except Exception:
                pass

            yield sse({"type": "result", "data": {
                "selected": selected,
                "all_candidates": [r for r in results[:10]],
                "confidence": confidence,
                "market_info": {
                    "sentiment": {"score": sentiment_score, "info": sentiment_info},
                    "news_mood": news_mood,
                    "news_score": news_score,
                    "hot_concepts": [{"name": c, "score": s} for c, s in hot_concepts[:5]],
                    "key_headlines": key_headlines[:5],
                    "hot_sectors": list(hot_sectors)[:6],
                    "northbound": {"total": nb_total, "info": nb_info},
                    "limits": {"up": limit_up, "down": limit_down},
                    "boards": boards_count,
                    "market_status": {"can_trade": can_trade, "reason": market_reason,
                                      "severity": severity},
                    "money_effect": money_effect,
                    "commodity": commodity_data,
                    "global_markets": global_markets,
                    "global_risk": global_penalty,
                    "global_headlines": global_headlines[:5] if global_headlines else [],
                    "trend_industries": trend_industries,
                    "veto_industries": list(veto_industries) if veto_industries else [],
                },
            }})
        except Exception as e:
            yield sse({"type": "error", "msg": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── API: T+5 波段决策 (SSE) ──

@app.get("/api/go5")
def api_go5():
    def generate():
        try:
            yield sse({"type": "progress", "msg": "检测市场环境...", "pct": 3})
            market_regime, regime_adj, regime_label = t1.calc_market_regime()
            calendar_adj, calendar_label = t1.calc_calendar_adjustment()

            yield sse({"type": "progress", "msg": "获取市场数据（缓存加速）...", "pct": 10})
            sd = _get_shared_data(include_news=True)
            cal_weights, cal_threshold, cal_combos = sd["cal_weights"], sd["cal_threshold"], sd["cal_combos"]
            sentiment_score, sentiment_info = sd["sentiment_score"], sd["sentiment_info"]
            hot_sectors = sd["hot_sectors"]
            capital_rank = sd["capital_rank"]
            billboard_data = sd["billboard_data"]
            nb_total, nb_info = sd["nb_total"], sd["nb_info"]
            margin_data = sd["margin_data"]
            limit_up, limit_down = sd["limit_up"], sd["limit_down"]
            limit_up_pool = sd["limit_up_pool"]
            shareholder_data = sd["shareholder_data"]
            industry_data = sd["industry_data"]
            fund_batch = sd["fund_batch"]
            commodity_data = sd["commodity_data"]
            global_penalty, global_warn = sd["global_penalty"], sd["global_warn"]
            ah_data = sd["ah_data"]
            news_score = sd["news_score"]
            hot_concepts = sd["hot_concepts"]
            veto_industries = sd["veto_industries"]
            trend_industries = sd.get("trend_industries", {})

            stock_list = sd["stock_list"]
            if stock_list.empty:
                yield sse({"type": "error", "msg": "无法获取股票列表"})
                return

            # T+5 宽筛选
            stock_list = t1._prefilter(stock_list, t1.PREFILTER_GO5)
            capital_codes = set(capital_rank.keys())
            bb_codes = set(billboard_data.keys())
            p1 = stock_list[stock_list["代码"].isin(capital_codes & bb_codes)]["代码"].tolist()
            p2 = stock_list[stock_list["代码"].isin(capital_codes - bb_codes)]["代码"].tolist()
            p3 = stock_list[~stock_list["代码"].isin(capital_codes | bb_codes)]["代码"].tolist()[:GO_OTHERS_CAP]
            analyze_list = p1 + p2 + p3
            code_to_row = stock_list.set_index("代码").to_dict("index")

            yield sse({"type": "progress", "msg": f"T+5门控分析 {len(analyze_list)} 只...", "pct": 30})

            results = []
            done = [0]
            total = len(analyze_list)

            def analyze_one(code):
                kl = t1.fetch_kline(code, days=120)
                if kl.empty or len(kl) < 60:
                    return None
                kl = t1.calc_all_indicators(kl)
                cap_info = capital_rank.get(code)
                sec_score = 10 if code in capital_codes else 3
                fi = fund_batch.get(code, {})
                fs, fd, fr = t1.evaluate_fundamentals(fi)
                stock_ind = ""
                if code in billboard_data or code in limit_up_pool or code in shareholder_data:
                    stock_ind = t1.get_stock_industry(code)
                row = code_to_row.get(code, {})
                name = str(row.get("名称", ""))
                if veto_industries:
                    if stock_ind and any(vi in stock_ind for vi in veto_industries):
                        return None
                    if any(vi in name for vi in veto_industries):
                        return None
                # ★ 多维度风险检测（与T+1一致）
                commodity_pen, commodity_warn = t1.check_commodity_risk(stock_ind, name, commodity_data)
                rally_pen, rally_warn = t1.check_consecutive_rally(kl)
                ah_pen, ah_warn = t1.check_ah_premium_risk(code, name, ah_data)
                current_turnover = row.get("换手率", 0) or 0
                turnover_adj_val, turnover_label = t1.analyze_turnover_depth(kl, current_turnover)
                macro_adj_val, macro_label = t1.check_macro_trend_fit(stock_ind, name, trend_industries)
                wt_adj, _ = t1.calc_weekly_trend(kl)
                tc_adj, _ = t1.classify_trend_context(kl)
                es, ed, er = t1.evaluate_extra_dimensions(
                    code, billboard_data, margin_data, nb_total, limit_up, limit_down,
                    limit_up_pool, {}, shareholder_data, industry_data, stock_ind)
                tech_sc, pos_mult, d, r, passed = t1.evaluate_signals_gated(
                    kl, cap_info, sec_score, sentiment_score, fs, fd, es, ed,
                    calibrated_weights=cal_weights, golden_combos=cal_combos,
                    market_regime_adj=regime_adj, weekly_trend_adj=wt_adj,
                    trend_context_adj=tc_adj, calendar_adj=calendar_adj)
                if not passed:
                    return None
                # ★ 风险惩罚扣分
                risk_penalty = commodity_pen + rally_pen + ah_pen + min(global_penalty, 0)
                tech_sc += risk_penalty
                # 合并风险警告
                risk_warns = [w for w in [commodity_warn, rally_warn, ah_warn, global_warn, turnover_label, macro_label] if w]
                if risk_warns:
                    r = r + "；⚠" + "，".join(risk_warns) if r else "⚠" + "，".join(risk_warns)
                has_combo = d.get("黄金组合匹配", False)
                if has_combo and tech_sc >= t1.TECH_GATE_THRESHOLD + 15:
                    rec_level = "强推荐"
                elif has_combo or tech_sc >= t1.TECH_GATE_THRESHOLD + 10:
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
                _, lc_adj = t1.apply_largecap_adjustments(mktcap_yi)
                risk = t1.calc_position_and_risk_t5(
                    tech_sc, pos_mult, sentiment_score, nb_total,
                    limit_up, limit_down, price, kl, largecap_adj=lc_adj)
                return {
                    "代码": code, "名称": name, "最新价": price,
                    "涨跌幅": row.get("涨跌幅", latest.get("涨跌幅", 0)),
                    "技术分": tech_sc, "仓位倍率": pos_mult,
                    "基本面分": fs, "聪明钱分": es,
                    "推荐等级": rec_level, "信号": d, "理由": r,
                    "行业": stock_ind, "risk": risk, "黄金组合": has_combo,
                }

            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                futures = {executor.submit(analyze_one, c): c for c in analyze_list}
                for f in as_completed(futures):
                    done[0] += 1
                    try:
                        r = f.result()
                        if r:
                            results.append(r)
                    except Exception:
                        pass
                    if done[0] % 100 == 0:
                        pct = 30 + int(done[0] / total * 60)
                        yield sse({"type": "progress",
                                   "msg": f"进度 {done[0]}/{total}，候选 {len(results)} 只",
                                   "pct": min(pct, 92)})

            if not results:
                yield sse({"type": "result", "data": {
                    "selected": [], "msg": "今日无股票通过技术面独立门槛。"
                }})
                return

            results.sort(key=lambda x: x["技术分"], reverse=True)
            combo_results = [r for r in results if r["黄金组合"]]
            other_results = [r for r in results if not r["黄金组合"]]
            selected = []
            used_ind = set()
            for pool in [combo_results, other_results]:
                for r in pool:
                    if len(selected) >= 3: break
                    r_ind = r.get("行业", "")
                    if r_ind and r_ind in used_ind: continue
                    selected.append(r)
                    if r_ind: used_ind.add(r_ind)
            for r in results:
                if len(selected) >= 3: break
                if r not in selected: selected.append(r)

            yield sse({"type": "result", "data": {
                "selected": selected,
                "total_passed": len(results),
                "combo_count": len(combo_results),
                "market_info": {
                    "sentiment": {"score": sentiment_score, "info": sentiment_info},
                    "regime": {"label": regime_label, "adj": regime_adj},
                    "news_mood": "偏多" if news_score > 3 else "中性偏多" if news_score > 0 else "中性偏空" if news_score > -3 else "偏空",
                    "hot_concepts": [{"name": c, "score": s} for c, s in hot_concepts[:5]],
                },
            }})

            # 记录推荐
            try:
                from trade_tracker import record_recommendation
                for s in selected:
                    record_recommendation("T+5", s["代码"], s["名称"], s["最新价"],
                                           s["技术分"], s["推荐等级"], s["risk"])
            except Exception:
                pass

        except Exception as e:
            yield sse({"type": "error", "msg": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── API: 交易记录 ──

@app.get("/api/trades")
def api_trades():
    try:
        from trade_tracker import get_trade_history, get_performance_summary
        history = get_trade_history(days=30)
        summary = get_performance_summary()
        return {"trades": history, "summary": summary}
    except Exception as e:
        return {"error": str(e), "trades": [], "summary": {}}


# ── API: ETF 波段扫描 (SSE) ──

@app.get("/api/etf")
def api_etf():
    def generate():
        try:
            yield sse({"type": "progress", "msg": "检测市场环境...", "pct": 5})
            sentiment_score, sentiment_info = t1.get_sentiment_score()
            money_effect = t1.calc_money_effect()

            yield sse({"type": "progress", "msg": "获取板块资金流向...", "pct": 15})
            concept_flow = t1.fetch_sector_flow("concept", top_n=15)
            industry_flow = t1.fetch_sector_flow("industry", top_n=15)
            hot_concepts_list = [s["板块名称"] for s in concept_flow if s.get("主力净流入", 0) > 0][:6]
            hot_ind_list = [s["板块名称"] for s in industry_flow if s.get("主力净流入", 0) > 0][:6]
            all_hot = hot_concepts_list + hot_ind_list

            yield sse({"type": "progress", "msg": "获取ETF列表...", "pct": 25})
            etf_list = t1.fetch_etf_list()
            if etf_list.empty:
                yield sse({"type": "error", "msg": "无法获取ETF列表"})
                return

            yield sse({"type": "progress", "msg": f"分析 {len(etf_list)} 只ETF...", "pct": 35})
            results = []
            code_to_row = etf_list.set_index("代码").to_dict("index")
            total = len(etf_list)
            done = [0]

            # 新闻热点
            news = t1.fetch_news()
            news_score, hot_concepts, _, _ = t1.analyze_news_sentiment(news)

            def analyze(code):
                kl = t1.fetch_kline(code, days=120)
                if kl.empty or len(kl) < 30:
                    return None
                kl = t1.calc_all_indicators(kl)
                s, d, r = t1.evaluate_etf_trend(kl)
                row = code_to_row.get(code, {})
                name = str(row.get("名称", ""))
                theme = row.get("主题", t1.get_etf_theme(name))
                theme_bonus = 0
                for hot in all_hot:
                    if theme in hot or hot in name:
                        theme_bonus = 8
                        break
                news_bonus = 0
                for concept, pts in (hot_concepts or [])[:5]:
                    if concept in name or concept in theme:
                        news_bonus = min(pts // 3, 5)
                        break
                total_score = s + theme_bonus + news_bonus + min(sentiment_score, 10)
                if total_score >= 45:
                    price = float(kl.iloc[-1]["收盘"])
                    risk = t1.calc_etf_risk(total_score, price, kl)
                    return {
                        "代码": code, "名称": name, "主题": theme,
                        "最新价": row.get("最新价", 0), "涨跌幅": row.get("涨跌幅", 0),
                        "成交额": row.get("成交额", 0),
                        "评分": total_score, "趋势分": s,
                        "板块加分": theme_bonus, "新闻加分": news_bonus,
                        "信号": d, "理由": r, "risk": risk,
                    }
                return None

            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                futures = {executor.submit(analyze, c): c for c in etf_list["代码"].tolist()}
                for f in as_completed(futures):
                    done[0] += 1
                    try:
                        r = f.result()
                        if r:
                            results.append(r)
                    except Exception:
                        pass
                    if done[0] % 30 == 0:
                        pct = 35 + int(done[0] / total * 55)
                        yield sse({"type": "progress",
                                   "msg": f"进度 {done[0]}/{total}，候选 {len(results)} 只",
                                   "pct": min(pct, 92)})

            results.sort(key=lambda x: x["评分"], reverse=True)

            # 选3只，优先不同主题
            selected = [results[0]] if results else []
            used = {results[0]["主题"]} if results else set()
            for r in results[1:]:
                if len(selected) >= 3:
                    break
                if r["主题"] not in used:
                    selected.append(r)
                    used.add(r["主题"])
            for r in results[1:]:
                if len(selected) >= 3:
                    break
                if r not in selected:
                    selected.append(r)

            yield sse({"type": "result", "data": {
                "selected": selected,
                "all_candidates": results[:10],
                "market_info": {
                    "sentiment": {"score": sentiment_score, "info": sentiment_info},
                    "hot_sectors": all_hot[:6],
                    "money_effect": money_effect,
                },
            }})
        except Exception as e:
            yield sse({"type": "error", "msg": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── API: 模拟实盘回测 (SSE) ──

@app.get("/api/simulate")
def api_simulate():
    def generate():
        import re as _re
        try:
            cal_weights, cal_threshold, cal_combos = t1.load_calibrated_weights()
            bt_threshold = cal_threshold if cal_weights else 110

            yield sse({"type": "progress", "msg": "获取股票列表...", "pct": 2})
            stock_list = t1.fetch_stock_list_sina()
            if stock_list.empty:
                yield sse({"type": "error", "msg": "无法获取股票列表"})
                return

            sample_size = 200
            active = stock_list[(stock_list["最新价"] >= 5) & (stock_list["最新价"] <= 80) &
                                (stock_list["换手率"] >= 1)]
            if len(active) < sample_size:
                active = stock_list.head(sample_size)
            codes = active.sample(min(sample_size, len(active)), random_state=42)["代码"].tolist()

            yield sse({"type": "progress", "msg": f"获取 {len(codes)} 只股票K线...", "pct": 5})

            # 批量获取K线
            all_klines = {}
            for i, c in enumerate(codes):
                kline = t1.fetch_kline_long(c, days=250)
                if kline.empty or len(kline) < 60:
                    continue
                kline = t1.calc_all_indicators(kline)
                all_klines[c] = kline
                if (i + 1) % 40 == 0:
                    pct = 5 + int(i / len(codes) * 40)
                    yield sse({"type": "progress",
                               "msg": f"K线 {i+1}/{len(codes)}，有效 {len(all_klines)} 只",
                               "pct": pct})
                import time as _time
                _time.sleep(0.1)

            if not all_klines:
                yield sse({"type": "error", "msg": "无法获取K线数据"})
                return

            # 收集所有日期
            all_dates = set()
            for kline in all_klines.values():
                all_dates.update(kline["日期"].tolist())
            all_dates = sorted(all_dates)[-250:]

            cap_sim = {"主力净流入占比": 3, "主力净流入": 1e7,
                       "超大单净流入": 0, "超大单占比": 0, "大单净流入": 0, "大单占比": 0}

            yield sse({"type": "progress", "msg": f"逐日评分 {len(all_dates)} 个交易日...", "pct": 50})

            # 逐日选TOP1
            trades = []
            no_signal_days = 0
            monthly_stats = {}

            for di, date in enumerate(all_dates):
                if (di + 1) % 50 == 0:
                    pct = 50 + int(di / len(all_dates) * 40)
                    yield sse({"type": "progress",
                               "msg": f"评分 {di+1}/{len(all_dates)}，已选 {len(trades)} 笔",
                               "pct": pct})

                best_code = None
                best_score = -999
                best_details = {}
                best_pos = 0
                best_kline = None

                for code, kline in all_klines.items():
                    date_mask = kline["日期"] == date
                    if not date_mask.any():
                        continue
                    idx = kline.index[date_mask][-1]
                    pos = kline.index.get_loc(idx)
                    if pos < 30 or pos >= len(kline) - 1:
                        continue
                    window = kline.iloc[:pos+1].copy()
                    sc, details, _ = t1.evaluate_signals_v2(
                        window, cap_sim, 8, 8, 25, {},
                        calibrated_weights=cal_weights, golden_combos=cal_combos)
                    if sc > best_score:
                        best_score = sc
                        best_code = code
                        best_details = details
                        best_pos = pos
                        best_kline = kline

                if best_score < bt_threshold or best_code is None:
                    no_signal_days += 1
                    continue

                buy_price = float(best_kline.iloc[best_pos]["收盘"])
                sell_price = float(best_kline.iloc[best_pos + 1]["收盘"])
                pnl = (sell_price - buy_price) / buy_price * 100

                sig_q = best_details.get("信号质量", "C级")
                has_combo = best_details.get("黄金组合匹配", False)
                hq_match = _re.search(r"(\d+)个高胜率", sig_q)
                high_count = int(hq_match.group(1)) if hq_match else 0

                row = best_kline.iloc[best_pos]
                today_chg = float(row.get("涨跌幅", 0))
                amplitude = float((row["最高"] - row["最低"]) / row["最低"] * 100) if row["最低"] > 0 else 0
                start = max(0, best_pos - 2)
                cum_3d = float((best_kline.iloc[best_pos]["收盘"] / best_kline.iloc[start]["收盘"] - 1) * 100)

                trades.append({
                    "日期": date, "代码": best_code, "评分": round(best_score, 1),
                    "买入价": round(buy_price, 2), "卖出价": round(sell_price, 2),
                    "收益率": round(pnl, 2), "信号质量": sig_q, "黄金组合": has_combo,
                    "高胜率数": high_count, "当日涨幅": round(today_chg, 2),
                    "振幅": round(amplitude, 2), "近3日涨幅": round(cum_3d, 2),
                })

                month_key = date[:7]
                if month_key not in monthly_stats:
                    monthly_stats[month_key] = {"wins": 0, "total": 0, "returns": []}
                monthly_stats[month_key]["total"] += 1
                if pnl > 0:
                    monthly_stats[month_key]["wins"] += 1
                monthly_stats[month_key]["returns"].append(pnl)

            yield sse({"type": "progress", "msg": "统计分析...", "pct": 95})

            if not trades:
                yield sse({"type": "error", "msg": "回测期间无达标信号"})
                return

            # 统计
            df = pd.DataFrame(trades)
            total = len(df)
            wins = int(len(df[df["收益率"] > 0]))
            losses = total - wins
            win_rate = wins / total * 100
            avg_ret = float(df["收益率"].mean())
            avg_win = float(df[df["收益率"] > 0]["收益率"].mean()) if wins > 0 else 0
            avg_loss = float(df[df["收益率"] <= 0]["收益率"].mean()) if losses > 0 else 0

            df["止损收益"] = df["收益率"].clip(lower=-2)
            stop_avg = float(df["止损收益"].mean())

            df["累计"] = (1 + df["收益率"] / 100).cumprod()
            final_return = float((df["累计"].iloc[-1] - 1) * 100)
            cummax = df["累计"].cummax()
            max_dd = float(((df["累计"] - cummax) / cummax * 100).min())

            # 连胜连亏
            max_ws = max_ls = cur = 0
            for r in df["收益率"]:
                if r <= 0:
                    cur += 1
                    max_ls = max(max_ls, cur)
                else:
                    cur = 0
            cur = 0
            for r in df["收益率"]:
                if r > 0:
                    cur += 1
                    max_ws = max(max_ws, cur)
                else:
                    cur = 0

            # 策略优化扫描
            def _wr(sub):
                if len(sub) == 0:
                    return {"n": 0, "wr": 0, "avg": 0}
                w = int(len(sub[sub["收益率"] > 0]))
                return {"n": len(sub), "wr": round(w / len(sub) * 100, 1),
                        "avg": round(float(sub["收益率"].mean()), 2)}

            strategies = []
            base = _wr(df)
            strategies.append({"name": "基准(无过滤)", **base, "delta": "-"})

            filters = [
                ("评分>=135", df[df["评分"] >= 135]),
                ("评分>=140", df[df["评分"] >= 140]),
                ("评分>=145", df[df["评分"] >= 145]),
                ("高胜率>=5", df[df["高胜率数"] >= 5]),
                ("高胜率>=6", df[df["高胜率数"] >= 6]),
                ("高胜率>=7", df[df["高胜率数"] >= 7]),
                ("涨幅<=3%", df[df["当日涨幅"] <= 3]),
                ("当日下跌才买", df[df["当日涨幅"] <= 0]),
                ("当日上涨才买", df[df["当日涨幅"] > 0]),
                ("振幅<=5%", df[df["振幅"] <= 5]),
                ("近3日<=5%", df[df["近3日涨幅"] <= 5]),
                ("评分>=135+高胜率>=6", df[(df["评分"] >= 135) & (df["高胜率数"] >= 6)]),
                ("高胜率>=6+涨幅<=3%", df[(df["高胜率数"] >= 6) & (df["当日涨幅"] <= 3)]),
                ("高胜率>=6+振幅<=5%+3日<=5%",
                 df[(df["高胜率数"] >= 6) & (df["振幅"] <= 5) & (df["近3日涨幅"] <= 5)]),
            ]
            for name, sub in filters:
                s = _wr(sub)
                s["name"] = name
                s["delta"] = round(s["wr"] - base["wr"], 1) if s["n"] > 0 else 0
                strategies.append(s)

            # 月度
            month_list = []
            for m in sorted(monthly_stats.keys()):
                ms = monthly_stats[m]
                mwr = round(ms["wins"] / ms["total"] * 100) if ms["total"] > 0 else 0
                mavg = round(float(np.mean(ms["returns"])), 2)
                mcum = round(float((np.prod([1 + r/100 for r in ms["returns"]]) - 1) * 100), 1)
                month_list.append({"month": m, "n": ms["total"], "wr": mwr, "avg": mavg, "cum": mcum})

            # 累计收益曲线数据
            equity_curve = []
            for _, t in df.iterrows():
                equity_curve.append({"date": t["日期"], "cum": round(float(t["累计"]) * 100 - 100, 1)})

            yield sse({"type": "result", "data": {
                "summary": {
                    "total": total, "total_days": len(all_dates),
                    "wins": wins, "losses": losses,
                    "win_rate": round(win_rate, 1),
                    "avg_ret": round(avg_ret, 2),
                    "stop_avg": round(stop_avg, 2),
                    "avg_win": round(avg_win, 2),
                    "avg_loss": round(avg_loss, 2),
                    "profit_ratio": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
                    "max_win": round(float(df["收益率"].max()), 2),
                    "max_loss": round(float(df["收益率"].min()), 2),
                    "final_return": round(final_return, 1),
                    "max_drawdown": round(max_dd, 1),
                    "max_win_streak": max_ws,
                    "max_lose_streak": max_ls,
                    "threshold": bt_threshold,
                    "sample_size": len(all_klines),
                },
                "strategies": strategies,
                "monthly": month_list,
                "equity_curve": equity_curve,
                "trades": trades[-30:],
            }})
        except Exception as e:
            yield sse({"type": "error", "msg": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── 启动入口 ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
