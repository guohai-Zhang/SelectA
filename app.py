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
        return jsonable({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "indices": sentiments,
            "sentiment": {"score": score, "info": info},
            "money_effect": money,
            "limits": {"up": limit_up, "down": limit_down},
            "northbound": {"total": nb_total, "info": nb_info},
            "market_status": {"can_trade": can_trade, "reason": reason, "severity": severity},
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

        name = rt.get(code, {}).get("名称", code)

        # 基本面
        fund_info = fund_batch.get(code, {})
        if fund_detail:
            fund_info.update({k: v for k, v in fund_detail.items() if v is not None})
        fund_s, fund_d, fund_r = t1.evaluate_fundamentals(fund_info)

        # 额外维度
        extra_s, extra_d, extra_r = t1.evaluate_extra_dimensions(
            code, billboard_data, margin_data, nb_total, limit_up, limit_down,
            limit_up_pool, billboard_detail, shareholder_data, industry_data, stock_industry)

        # 综合评分
        cap_info = cap_rank.get(code)
        cal_weights, cal_threshold, cal_combos = t1.load_calibrated_weights()
        score, details, reasons = t1.evaluate_signals_v2(
            kline, cap_info, 8, sentiment_score, fund_s, fund_d, extra_s, extra_d,
            calibrated_weights=cal_weights, golden_combos=cal_combos)

        # 风控
        price = float(latest["收盘"])
        risk = t1.calc_position_and_risk(
            score, sentiment_score, nb_total, limit_up, limit_down, price, kline)

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
            "sentiment": {"score": sentiment_score, "info": sentiment_info},
            "northbound": {"total": nb_total, "info": nb_info},
            "industry": stock_industry,
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

            yield sse({"type": "progress", "msg": "获取市场情绪...", "pct": 10})
            sentiment_score, sentiment_info = t1.get_sentiment_score()

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
                es, ed, er = t1.evaluate_extra_dimensions(
                    code, billboard_data, margin_data, nb_total, limit_up, limit_down,
                    limit_up_pool, billboard_detail, shareholder_data, industry_data, stock_ind)
                s, d, r = t1.evaluate_signals_v2(
                    kl, cap_info, sec_score, sentiment_score, fs, fd, es, ed,
                    calibrated_weights=cal_weights, golden_combos=cal_combos)
                combined = r
                if fr:
                    combined = r + "；" + fr if r != "信号不足" else fr
                if er:
                    combined += "；" + er if combined != "信号不足" else er
                scan_th = cal_threshold if cal_weights else 90
                if s >= scan_th:
                    row = code_to_row.get(code, {})
                    return {
                        "代码": code, "名称": row.get("名称", ""),
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

            if severity >= 3:
                yield sse({"type": "blocked", "msg": f"市场熔断: {market_reason}，今日禁止操作！"})
                return

            yield sse({"type": "progress", "msg": "扫描新闻/政策...", "pct": 5})
            news = t1.fetch_news()
            news_score, hot_concepts, key_headlines = t1.analyze_news_sentiment(news)
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
                es, ed, er = t1.evaluate_extra_dimensions(
                    code, billboard_data, margin_data, nb_total, limit_up, limit_down,
                    limit_up_pool, billboard_detail, shareholder_data, industry_data, stock_ind)
                s, d, r = t1.evaluate_signals_v2(
                    kl, cap_info, sec_score, sentiment_score, fs, fd, es, ed,
                    calibrated_weights=cal_weights, golden_combos=cal_combos)
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
                all_reasons = r
                if fr:
                    all_reasons = r + "；" + fr if r != "信号不足" else fr
                if er:
                    all_reasons += "；" + er if all_reasons != "信号不足" else er
                go_th = cal_threshold if cal_weights else 100
                if total_score >= go_th:
                    latest = kl.iloc[-1]
                    price = float(latest["收盘"])
                    risk = t1.calc_position_and_risk(
                        total_score, sentiment_score, nb_total, limit_up, limit_down, price, kl)
                    return {
                        "代码": code, "名称": name,
                        "最新价": row.get("最新价", price),
                        "涨跌幅": row.get("涨跌幅", latest.get("涨跌幅", 0)),
                        "换手率": row.get("换手率", 0),
                        "评分": total_score,
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
                return None

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
                    "selected": [], "confidence": 30,
                    "msg": "今日无合适标的，建议观望。"
                }})
                return

            results.sort(key=lambda x: x["评分"], reverse=True)

            # 选2只，优先不同行业
            selected = [results[0]]
            first_ind = results[0].get("行业", "")
            for r in results[1:]:
                if r.get("行业", "") and r.get("行业", "") != first_ind:
                    selected.append(r)
                    break
            if len(selected) < 2 and len(results) >= 2:
                selected.append(results[1])

            # 信心指数
            confidence = 50
            confidence += min(sentiment_score, 15)
            confidence += min(int(news_score * 2), 15)
            if selected[0]["评分"] >= 140:
                confidence += 15
            elif selected[0]["评分"] >= 110:
                confidence += 10
            if selected[0].get("聪明钱分", 0) >= 30:
                confidence += 5
            confidence = max(20, min(confidence, 95))

            # 连板统计
            boards_count = {}
            for v in limit_up_pool.values():
                b = v.get("连板数", 1)
                boards_count[b] = boards_count.get(b, 0) + 1

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
