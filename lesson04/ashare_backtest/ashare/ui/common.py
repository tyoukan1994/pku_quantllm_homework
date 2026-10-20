from dataclasses import replace
from datetime import date
import io
import json
import zipfile

import altair as alt
import pandas as pd
import streamlit as st

from ..config import BacktestConfig


def navigate(page):
    st.session_state["pending_page"] = page
    st.rerun()


def json_object(value):
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError("请输入JSON对象")
    json.dumps(result, allow_nan=False)
    return result


def strategy_picker(store, key):
    specs = store.list()
    mapping = {spec.identifier: spec for spec in specs}
    preferred = st.session_state.get("preferred_strategy")
    identifiers = list(mapping)
    index = identifiers.index(preferred) if preferred in mapping else 0
    identifier = st.selectbox("策略", identifiers, index=index,
                              format_func=lambda value: mapping[value].name, key=key)
    return mapping[identifier]


def base_config(preset="标准日频", training=False):
    config = BacktestConfig(start="2024-01-02", end="2024-12-31", initial_cash=1_000_000,
                             commission_rate=0.0001, stamp_tax_rate=0.0005, transfer_fee_rate=0,
                             minimum_commission=0, slippage_bps=0, require_st_history=True,
                             seed_previous_close=True, split_orders=True)
    if preset == "课程作业":
        config = replace(config, start="2025-01-01", end="2026-09-30", initial_cash=1_000_000_000,
                         liquidate_on_end=True, liquidation_window=20, write_off_research_dust=True)
    if training:
        config = replace(config, start="2020-01-02", end="2024-12-31", seed_previous_close=False)
    return config


def configuration(prefix, defaults, include_dates=True, compact=False):
    def key(name):
        return f"{prefix}_{name}"

    if include_dates:
        columns = st.columns(2 if compact else 3)
        start_column, end_column = columns[:2]
        cash_column = st if compact else columns[2]
        start = start_column.date_input("开始日期", date.fromisoformat(defaults.start), key=key("start"))
        end = end_column.date_input("结束日期", date.fromisoformat(defaults.end), key=key("end"))
        cash = cash_column.number_input("初始资金（元）", min_value=1000.0, value=float(defaults.initial_cash), step=100_000.0, key=key("cash"))
    else:
        start, end, cash = defaults.start, defaults.end, defaults.initial_cash
    with st.expander("费用与成交"):
        columns = st.columns(2 if compact else 3)
        third = columns[0] if compact else columns[2]
        participation_column = columns[1] if compact else columns[0]
        minimum_column = columns[0] if compact else columns[1]
        transfer_column = columns[1] if compact else columns[2]
        commission = columns[0].number_input("佣金（基点）", 0.0, 100.0, defaults.commission_rate * 10000, 0.5, key=key("commission"))
        stamp = columns[1].number_input("卖出印花税（基点）", 0.0, 100.0, float(defaults.stamp_tax_rate or 0) * 10000, 0.5, key=key("stamp"))
        slippage = third.number_input("单边滑点（基点）", 0.0, 500.0, float(defaults.slippage_bps), 1.0, key=key("slippage"))
        participation = participation_column.number_input("上日成交额上限（%）", 0.01, 100.0, defaults.participation_rate * 100, 0.5, key=key("participation"))
        minimum = minimum_column.number_input("最低佣金（元）", 0.0, 1000.0, float(defaults.minimum_commission), 1.0, key=key("minimum"))
        transfer = transfer_column.number_input("双边过户费（基点）", 0.0, 10.0, float(defaults.transfer_fee_rate or 0) * 10000, 0.1, key=key("transfer"))
    with st.expander("股票池与账户"):
        scope = st.text_area("股票代码（留空为全量）", height=70, placeholder="000001.SZ,600000.SH", key=key("scope"))
        columns = st.columns(2 if compact else 3)
        mode_column = st if compact else columns[2]
        history = columns[0].number_input("最低历史天数", 1, 1500, defaults.min_history, key=key("history"))
        max_weight = columns[1].number_input("单股权重上限（%）", 0.1, 100.0, defaults.max_weight * 100, key=key("max_weight"))
        mode = mode_column.selectbox("账户模式", ["复权研究", "严格实股"], index=int(defaults.accounting == "strict"), key=key("mode"))
        exclude_st = st.checkbox("排除ST", value=defaults.exclude_st, key=key("exclude_st"))
        require_st = st.checkbox("缺少历史ST状态时不买入", value=defaults.require_st_history, disabled=not exclude_st, key=key("require_st"))
        strict_rules = st.checkbox("要求精确逐日交易状态", value=defaults.strict_rules, key=key("strict_rules"))
        multiplier = st.number_input("成交额转为元的倍率", min_value=0.0001, value=float(defaults.amount_multiplier), key=key("amount"))
        annual_days = st.number_input("年化交易日数", 1, 366, defaults.annual_days, key=key("annual_days"))
        risk_free = st.number_input("无风险年利率（%）", -99.0, 100.0, defaults.risk_free_rate * 100, key=key("risk_free"))
        dividend_tax = st.number_input("严格账户股息税率（%）", 0.0, 100.0, defaults.dividend_tax_rate * 100, key=key("dividend_tax"))
    with st.expander("首末日处理"):
        seed = st.checkbox("使用前一交易日信号，首日建仓", value=defaults.seed_previous_close, key=key("seed"))
        liquidate = st.checkbox("期末清仓", value=defaults.liquidate_on_end, key=key("liquidate"))
        window = st.number_input("退出窗口（交易日，只卖不买）", 1, 120, defaults.liquidation_window, key=key("window"))
        dust = st.checkbox("末日将不足1股的研究残余核销为损失", value=defaults.write_off_research_dust, key=key("dust"))
    symbols = list(dict.fromkeys(value.strip() for value in scope.replace("，", ",").replace("\n", ",").split(",") if value.strip())) or None
    settings = {**defaults.to_dict(), "start": str(start), "end": str(end), "initial_cash": cash,
                "commission_rate": commission / 10000, "stamp_tax_rate": stamp / 10000,
                "transfer_fee_rate": transfer / 10000, "slippage_bps": slippage,
                "participation_rate": participation / 100, "minimum_commission": minimum,
                "min_history": int(history), "max_weight": max_weight / 100,
                "accounting": "research" if mode == "复权研究" else "strict",
                "exclude_st": exclude_st, "require_st_history": exclude_st and require_st,
                "strict_rules": strict_rules, "amount_multiplier": multiplier,
                "annual_days": int(annual_days), "risk_free_rate": risk_free / 100,
                "dividend_tax_rate": dividend_tax / 100, "seed_previous_close": seed,
                "liquidate_on_end": liquidate, "liquidation_window": int(window),
                "write_off_research_dust": dust}
    return settings, symbols


def result_view(result, directory, key):
    metrics = result.metrics
    st.caption(f"{result.strategy_name} · {metrics['start']} — {metrics['end']} · 初始 ¥{result.config['initial_cash']:,.0f}")
    values = [("累计收益", f"{metrics['total_return']:.2%}"), ("年化收益", f"{metrics['annual_return']:.2%}"),
              ("最大回撤", f"{metrics['max_drawdown']:.2%}"),
              ("夏普", f"{metrics['sharpe']:.2f}" if metrics['sharpe'] is not None else "—")]
    for column, (label, value) in zip(st.columns(4), values):
        column.metric(label, value)
    curve_tab, trades_tab, orders_tab, holdings_tab, config_tab = st.tabs(["净值", "成交", "订单", "持仓", "配置与口径"])
    with curve_tab:
        chart = result.equity.reset_index()
        st.altair_chart(alt.Chart(chart).mark_line(color="#2563eb", strokeWidth=2).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y-%m", tickCount=7)),
            y=alt.Y("net_value:Q", title="净值", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("date:T", title="日期"), alt.Tooltip("net_value:Q", format=".4f", title="净值")]).properties(height=290), width="stretch")
        st.altair_chart(alt.Chart(chart).mark_area(color="#d99583", opacity=0.65).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y-%m", tickCount=7)),
            y=alt.Y("drawdown:Q", title="回撤", axis=alt.Axis(format=".0%")),
            tooltip=["date:T", alt.Tooltip("drawdown:Q", format=".2%")]).properties(height=120), width="stretch")
        st.caption(f"成交 {metrics['fills']:,} 笔 · 费用 ¥{metrics['total_fees']:,.2f} · 平均仓位 {metrics['average_exposure']:.1%}")
        if result.config.get("liquidate_on_end") and not metrics.get("liquidation_complete", False):
            st.warning(f"期末仍有股票 ¥{metrics.get('terminal_market_value', result.equity.market_value.iloc[-1]):,.2f}，未完成清仓。")
    with trades_tab:
        st.dataframe(result.trades, hide_index=True, width="stretch")
    with orders_tab:
        st.dataframe(result.orders, hide_index=True, width="stretch")
        if not result.orders.empty:
            st.dataframe(result.orders.loc[result.orders.reason != "", "reason"].value_counts().rename("次数"))
    with holdings_tab:
        dates = result.equity.index.strftime("%Y-%m-%d").tolist()
        selected = st.select_slider("持仓日期", options=dates, value=dates[-1], key=key + "_holdings")
        positions = result.positions[result.positions.date == selected]
        if positions.empty:
            st.caption("当日无股票持仓")
        else:
            st.dataframe(positions, hide_index=True, width="stretch")
    with config_tab:
        st.json(result.config)
        for note in result.assumptions:
            st.write(note)
        with st.expander("数据与事件"):
            st.json(result.data_summary)
            st.dataframe(result.events, hide_index=True, width="stretch")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    st.download_button("下载结果", buffer.getvalue(), file_name=f"{directory.name}.zip", mime="application/zip", key=key + "_download")
