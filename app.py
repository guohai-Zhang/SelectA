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

# ── 简单缓存 ──
_cache = {}
_cache_ttl = {}

def cached_call(key, func, ttl=120, *args, **kwargs):
    """缓存函数调用结果，ttl秒后过期"""
    now = time.time()
    if key in _cache and now - _cache_ttl.get(key, 0) < ttl:
        return _cache[key]
    result = func(*args, **kwargs)
    _cache[key] = result
    _cache_ttl[key] = now
    return result


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

        # 并行获取多维度数据
        with ThreadPoolExecutor(max_workers=6) as ex:
            f_cap_rank = ex.submit(t1.fetch_capital_flow_rank, 200)
            f_cap_flow = ex.submit(t1.fetch_stock_capital_flow, code)
            f_sentiment = ex.submit(t1.get_sentiment_score)
            f_fund_detail = ex.submit(t1.fetch_fundamental_detail, code)
            f_fund_batch = ex.submit(t1.fetch_fundamentals_batch)
            f_billboard = ex.submit(t1.fetch_billboard)
            f_bb_detail = ex.submit(t1.fetch_billboard_detail)
            f_nb = ex.submit(t1.fetch_northbound_flow)
            f_margin = ex.submit(t1.fetch_margin_data_top)
            f_limits = ex.submit(t1.count_limit_up)
            f_zt_pool = ex.submit(t1.fetch_limit_up_pool)
            f_sh_inc = ex.submit(t1.fetch_shareholder_increase)
            f_ind = ex.submit(t1.fetch_industry_prosperity)
            f_ind_code = ex.submit(t1.get_stock_industry, code)
            f_rt = ex.submit(t1.fetch_realtime_sina, [code])
            f_commodity = ex.submit(t1.fetch_commodity_prices)

        cap_rank = f_cap_rank.result()
        cap_flow = f_cap_flow.result()
        sentiment_score, sentiment_info = f_sentiment.result()
        fund_detail = f_fund_detail.result()
        fund_batch = f_fund_batch.result()
        billboard_data = f_billboard.result()
        billboard_detail = f_bb_detail.result()
        nb_total, nb_info = f_nb.result()
        margin_data = f_margin.result()
        limit_up, limit_down = f_limits.result()
        limit_up_pool = f_zt_pool.result()
        shareholder_data = f_sh_inc.result()
        industry_data = f_ind.result()
        stock_industry = f_ind_code.result()
        rt = f_rt.result()
        commodity_data = f_commodity.result()

        name = rt.get(code, {}).get("名称", code)

        # 基本面
        fund_info = fund_batch.get(code, {})
        if fund_detail:
            fund_info.update({k: v for k, v in fund_detail.items() if v is not None})
        fund_s, fund_d, fund_r = t1.evaluate_fundamentals(fund_info)

        # ★ 多维度风险检测
        commodity_pen, commodity_warn = t1.check_commodity_risk(stock_industry, name, commodity_data)
        rally_pen, rally_warn = t1.check_consecutive_rally(kline)
        ah_data = t1.fetch_ah_premium()
        ah_pen, ah_warn = t1.check_ah_premium_risk(code, name, ah_data)
        global_markets = t1.fetch_global_markets()
        global_penalty, global_warn_str, _ = t1.calc_global_risk(global_markets)
        # 换手率深度（从实时数据获取）
        rt_info = rt.get(code, {})
        turnover_adj_val, turnover_label = t1.analyze_turnover_depth(kline, 0)

        # 额外维度
        extra_s, extra_d, extra_r = t1.evaluate_extra_dimensions(
            code, billboard_data, margin_data, nb_total, limit_up, limit_down,
            limit_up_pool, billboard_detail, shareholder_data, industry_data, stock_industry)

        # 综合评分
        cap_info = cap_rank.get(code)
        cal_weights, cal_threshold, cal_combos = t1.load_calibrated_weights()
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
            "score": score, "max_score": 280,
            "signals": details,
            "reasons": reasons,
            "fundamentals": fund_info,
            "fund_score": fund_s,
            "extra_score": extra_s,
            "risk": risk,
            "risk_warnings": risk_warnings,
            "sentiment": {"score": sentiment_score, "info": sentiment_info},
            "northbound": {"total": nb_total, "info": nb_info},
            "industry": stock_industry,
            "commodity": commodity_data,
            "global_markets": global_markets,
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
            yield sse({"type": "progress", "msg": "加载校准权重...", "pct": 0})
            cal_weights, cal_threshold, cal_combos = t1.load_calibrated_weights()

            yield sse({"type": "progress", "msg": "检测市场环境...", "pct": 5})
            money_effect = t1.calc_money_effect()
            limit_up_pre, limit_down_pre = t1.count_limit_up()
            can_trade, market_reason, severity = t1.market_go_nogo(
                money_effect, limit_up=limit_up_pre, limit_down=limit_down_pre)

            if severity >= 3:
                yield sse({"type": "blocked", "msg": f"市场熔断: {market_reason}，今日禁止操作！"})
                return

            yield sse({"type": "progress", "msg": "获取市场情绪+全球数据...", "pct": 10})
            sentiment_score, sentiment_info = t1.get_sentiment_score()
            commodity_data = cached_call("commodity", t1.fetch_commodity_prices, 300)
            global_markets = cached_call("global_markets", t1.fetch_global_markets, 300)
            global_penalty, _, _ = t1.calc_global_risk(global_markets)
            ah_data = cached_call("ah_premium", t1.fetch_ah_premium, 300)

            yield sse({"type": "progress", "msg": "获取板块/资金/龙虎榜...", "pct": 15})
            hot_sectors = t1.get_hot_sectors()
            capital_rank = t1.fetch_capital_flow_rank(top_n=200)
            billboard_data = t1.fetch_billboard()
            billboard_detail = t1.fetch_billboard_detail()

            yield sse({"type": "progress", "msg": "获取北向/融资/涨停...", "pct": 25})
            nb_total, nb_info = t1.fetch_northbound_flow()
            margin_data = t1.fetch_margin_data_top()
            limit_up, limit_down = t1.count_limit_up()
            limit_up_pool = t1.fetch_limit_up_pool()
            shareholder_data = t1.fetch_shareholder_increase()
            industry_data = t1.fetch_industry_prosperity()

            yield sse({"type": "progress", "msg": "获取股票列表+基本面...", "pct": 35})
            stock_list = t1.fetch_stock_list_sina()
            if stock_list.empty:
                yield sse({"type": "error", "msg": "无法获取股票列表"})
                return
            fund_batch = t1.fetch_fundamentals_batch()

            # 预筛选
            stock_list = stock_list[
                (stock_list["最新价"] >= 3) & (stock_list["最新价"] <= 100) &
                (stock_list["涨跌幅"] > -5) & (stock_list["涨跌幅"] < 9.8) &
                (stock_list["换手率"] >= 2) & (stock_list["量比"] >= 0.8)
            ]
            capital_codes = set(capital_rank.keys())
            priority = stock_list[stock_list["代码"].isin(capital_codes)]["代码"].tolist()
            others = stock_list[~stock_list["代码"].isin(capital_codes)]["代码"].tolist()
            ordered = priority + others
            code_to_row = stock_list.set_index("代码").to_dict("index")
            total = len(ordered)

            yield sse({"type": "progress", "msg": f"分析 {total} 只股票...", "pct": 40})

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
                # ★ 多维度风险检测
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
                # ★ 大盘股提高门槛
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
                        "换手率": row.get("换手率", 0), "评分": s,
                        "信号": d, "理由": combined,
                        "主力净流入占比": cap_info.get("主力净流入占比", 0) if cap_info else 0,
                    }
                return None

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(analyze_one, c): c for c in ordered}
                for future in as_completed(futures):
                    done[0] += 1
                    try:
                        r = future.result()
                        if r:
                            results.append(r)
                    except Exception:
                        pass
                    if done[0] % 100 == 0:
                        pct = 40 + int(done[0] / total * 55)
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
            yield sse({"type": "progress", "msg": "加载校准权重...", "pct": 0})
            cal_weights, cal_threshold, cal_combos = t1.load_calibrated_weights()

            yield sse({"type": "progress", "msg": "检测市场环境...", "pct": 3})
            money_effect = t1.calc_money_effect()
            limit_up_pre, limit_down_pre = t1.count_limit_up()
            can_trade, market_reason, severity = t1.market_go_nogo(
                money_effect, limit_up=limit_up_pre, limit_down=limit_down_pre)

            market_blocked = severity >= 3

            # 多维度增强分析
            market_regime, regime_adj, regime_label = t1.calc_market_regime()
            calendar_adj, calendar_label = t1.calc_calendar_adjustment()

            yield sse({"type": "progress", "msg": "获取大宗商品+全球市场+AH溢价...", "pct": 4})
            commodity_data = t1.fetch_commodity_prices()
            global_markets = t1.fetch_global_markets()
            global_penalty, global_warn, global_detail = t1.calc_global_risk(global_markets)
            ah_data = t1.fetch_ah_premium()

            yield sse({"type": "progress", "msg": "扫描国内+国际新闻...", "pct": 5})
            news = t1.fetch_news()
            global_news = t1.fetch_global_news()
            all_news_combined = news + global_news
            news_score, hot_concepts, key_headlines, veto_industries = t1.analyze_news_sentiment(all_news_combined)
            global_risk_score, trend_industries, global_headlines = t1.analyze_global_news(global_news)
            news_mood = "偏多" if news_score > 3 else "中性偏多" if news_score > 0 else "中性偏空" if news_score > -3 else "偏空"

            yield sse({"type": "progress", "msg": "分析大盘情绪...", "pct": 8})
            sentiment_score, sentiment_info = t1.get_sentiment_score()

            yield sse({"type": "progress", "msg": "获取板块/龙虎榜/北向/融资...", "pct": 12})
            hot_sectors = t1.get_hot_sectors()
            billboard_data = t1.fetch_billboard()
            billboard_detail = t1.fetch_billboard_detail()
            nb_total, nb_info = t1.fetch_northbound_flow()
            margin_data = t1.fetch_margin_data_top()
            limit_up, limit_down = t1.count_limit_up()
            limit_up_pool = t1.fetch_limit_up_pool()

            yield sse({"type": "progress", "msg": "获取增持/行业/资金...", "pct": 20})
            shareholder_data = t1.fetch_shareholder_increase()
            industry_data = t1.fetch_industry_prosperity()
            capital_rank = t1.fetch_capital_flow_rank(top_n=300)
            fund_batch = t1.fetch_fundamentals_batch()

            yield sse({"type": "progress", "msg": "获取股票列表...", "pct": 28})
            stock_list = t1.fetch_stock_list_sina()
            if stock_list.empty:
                yield sse({"type": "error", "msg": "无法获取股票列表"})
                return

            # 预筛选
            stock_list = stock_list[
                (stock_list["最新价"] >= 5) & (stock_list["最新价"] <= 80) &
                (stock_list["涨跌幅"] > -3) & (stock_list["涨跌幅"] < 8) &
                (stock_list["换手率"] >= 2) & (stock_list["量比"] >= 0.8)
            ]
            capital_codes = set(capital_rank.keys())
            bb_codes = set(billboard_data.keys())
            p1 = stock_list[stock_list["代码"].isin(bb_codes & capital_codes)]["代码"].tolist()
            p2 = stock_list[stock_list["代码"].isin(capital_codes - bb_codes)]["代码"].tolist()
            p3 = stock_list[stock_list["代码"].isin(bb_codes - capital_codes)]["代码"].tolist()
            p4 = stock_list[~stock_list["代码"].isin(capital_codes | bb_codes)]["代码"].tolist()[:200]
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
                high_q = "A级" in sig_q  # ★ A级=2+高胜率信号
                mid_q = "B级" in sig_q   # ★ B级=1个高胜率信号

                # ★ 推荐等级：回测验证，黄金组合64%胜率，>=120分62%胜率
                if total_score >= go_th + 20 and has_combo:
                    rec_level = "强推荐"  # ★ 黄金组合+高分=强推荐（回测64%+）
                elif total_score >= go_th + 10 and (has_combo or high_q):
                    rec_level = "推荐"    # ★ 高分+A级信号=推荐（回测57%+）
                elif total_score >= go_th and (high_q or mid_q):
                    rec_level = "弱推荐"  # ★ 达标+至少1个高胜率信号
                elif total_score >= go_th:
                    rec_level = "仅参考"  # ★ 达标但无高胜率信号=仅参考（回测53%，不赚钱）
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
                    "评分": total_score,
                    "信号质量": sig_q,
                    "推荐等级": rec_level,
                    "技术分": s - fs - es,
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

            with ThreadPoolExecutor(max_workers=8) as executor:
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

            results.sort(key=lambda x: x["评分"], reverse=True)

            # ★ 过滤：只保留"推荐"及以上等级（仅参考的不进入决策）
            qualified = [r for r in results if r["推荐等级"] in ("强推荐", "推荐", "弱推荐")]
            if not qualified:
                qualified = results[:3]  # 实在没有就取前3但降低信心

            # 选3只，优先不同行业分散风险，优先高推荐等级
            level_order = {"强推荐": 0, "推荐": 1, "弱推荐": 2, "仅参考": 3}
            qualified.sort(key=lambda x: (level_order.get(x["推荐等级"], 3), -x["评分"]))
            selected = [qualified[0]]
            used_ind = {qualified[0].get("行业", "")}
            for r in qualified[1:]:
                if len(selected) >= 3:
                    break
                r_ind = r.get("行业", "")
                if r_ind and r_ind not in used_ind:
                    selected.append(r)
                    used_ind.add(r_ind)
            for r in qualified[1:]:
                if len(selected) >= 3:
                    break
                if r not in selected:
                    selected.append(r)

            # ★ 信心指数 — 更保守的计算
            confidence = 40  # ★ 基础值从50降到40
            confidence += min(sentiment_score, 10)  # ★ 情绪最多+10（从15降）
            confidence += min(int(news_score * 1.5), 10)  # ★ 新闻最多+10（从15降）
            # ★ 基于推荐等级而非纯评分
            top_level = selected[0].get("推荐等级", "仅参考")
            if top_level == "强推荐":
                confidence += 20
            elif top_level == "推荐":
                confidence += 12
            elif top_level == "弱推荐":
                confidence += 5
            # ★ 多只候选信心加成
            strong_count = sum(1 for s in selected if s["推荐等级"] in ("强推荐", "推荐"))
            confidence += strong_count * 3
            # 市场熔断时降低信心
            if market_blocked:
                confidence -= 30
            # ★ 全球风险
            if global_penalty < -5:
                confidence -= 10
            confidence = max(15, min(confidence, 85))  # ★ 上限从95降到85（永远不要太自信）

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
            cal_weights, cal_threshold, cal_combos = t1.load_calibrated_weights()
            money_effect = t1.calc_money_effect()
            sentiment_score, sentiment_info = t1.get_sentiment_score()
            market_regime, regime_adj, regime_label = t1.calc_market_regime()
            calendar_adj, calendar_label = t1.calc_calendar_adjustment()

            yield sse({"type": "progress", "msg": "扫描新闻+板块...", "pct": 8})
            news = t1.fetch_news()
            global_news = t1.fetch_global_news()
            all_news = news + global_news
            news_score, hot_concepts, key_headlines, veto_industries = t1.analyze_news_sentiment(all_news)
            _, trend_industries, _ = t1.analyze_global_news(global_news)
            hot_sectors = t1.get_hot_sectors()
            billboard_data = t1.fetch_billboard()
            nb_total, nb_info = t1.fetch_northbound_flow()
            margin_data = t1.fetch_margin_data_top()
            limit_up, limit_down = t1.count_limit_up()
            limit_up_pool = t1.fetch_limit_up_pool()
            shareholder_data = t1.fetch_shareholder_increase()
            industry_data = t1.fetch_industry_prosperity()

            yield sse({"type": "progress", "msg": "获取股票列表+资金...", "pct": 20})
            capital_rank = t1.fetch_capital_flow_rank(top_n=300)
            fund_batch = t1.fetch_fundamentals_batch()
            stock_list = t1.fetch_stock_list_sina()
            if stock_list.empty:
                yield sse({"type": "error", "msg": "无法获取股票列表"})
                return

            # T+5 宽筛选
            stock_list = stock_list[
                (stock_list["最新价"] >= 5) & (stock_list["最新价"] <= 100) &
                (stock_list["涨跌幅"] > -5) & (stock_list["涨跌幅"] < 5) &
                (stock_list["换手率"] >= 1.5) & (stock_list["量比"] >= 0.6)
            ]
            capital_codes = set(capital_rank.keys())
            bb_codes = set(billboard_data.keys())
            p1 = stock_list[stock_list["代码"].isin(capital_codes & bb_codes)]["代码"].tolist()
            p2 = stock_list[stock_list["代码"].isin(capital_codes - bb_codes)]["代码"].tolist()
            p3 = stock_list[~stock_list["代码"].isin(capital_codes | bb_codes)]["代码"].tolist()[:300]
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

            with ThreadPoolExecutor(max_workers=8) as executor:
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

            with ThreadPoolExecutor(max_workers=8) as executor:
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


# ── 启动入口 ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
