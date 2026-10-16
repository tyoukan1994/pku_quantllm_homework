from datetime import datetime
import io
import json
from pathlib import Path
import sys
import zipfile

import altair as alt
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ashare import BacktestConfig
from ashare.cli import DEFAULT_DATA, run_backtest


st.set_page_config(page_title="沪深策略实验室", page_icon="📈", layout="wide")
st.markdown("""<style>
.stApp {background: #f5f7fc;} .block-container {max-width:1400px;padding-top:2.5rem;padding-left:1.5rem;padding-right:1.5rem;}
h1 {letter-spacing:-1.5px;} [data-testid="stMetric"] {background:white;border:1px solid #e3e8f0;border-radius:14px;padding:14px;}
[data-testid="stMetricValue"] {font-size:22px!important;}
[data-testid="stSidebar"] {background:#eef2f9;} .eyebrow {color:#4765a0;font-weight:700;letter-spacing:3px;font-size:12px;}
textarea {font-family: ui-monospace,SFMono-Regular,monospace!important;font-size:13px!important;}
</style>""", unsafe_allow_html=True)
st.markdown('<p class="eyebrow">A-SHARE · DAILY RESEARCH</p>', unsafe_allow_html=True)
st.title("沪深策略实验室")
st.caption("收盘形成信号 → 下一交易日开盘尝试成交 → 跟踪真实持仓变化与费用后净值")

with st.sidebar:
    st.header("回测设置")
    data_path = st.text_input("Feather 数据目录", DEFAULT_DATA)
    start = st.date_input("开始日期", datetime(2024, 1, 2))
    end = st.date_input("结束日期", datetime(2025, 12, 31))
    initial_cash = st.number_input("初始资金（元）", min_value=1000.0, value=1_000_000.0, step=100_000.0)
    scope = st.text_area("股票池（留空为全部沪深代码）", "", height=80,
                         placeholder="000001.SZ,600000.SH,300750.SZ")
    accounting_label = st.selectbox("账户口径", ["复权研究模式", "严格账户模式"])
    accounting = "research" if accounting_label == "复权研究模式" else "strict"
    strict_rules = st.checkbox("要求完整逐日交易状态", value=False)
    if accounting == "research":
        st.info("以复权权益份额研究策略，包含分红再投资近似；实际现金分红与到账股数请使用严格账户模式。")
    else:
        st.info("持仓发生公司行为时需要 corporate_actions.csv；缺少明细会停止回测。")
    with st.expander("费用、流动性与历史窗口"):
        commission = st.number_input("佣金（基点，1 基点=万分之一）", 0.0, 100.0, 3.0, 0.5)
        minimum_fee = st.number_input("每笔最低佣金（元，假定值）", 0.0, 1000.0, 5.0, 1.0)
        slippage = st.number_input("单边滑点（基点）", 0.0, 500.0, 5.0, 1.0)
        participation = st.number_input("上日成交额参与上限（%）", 0.01, 100.0, 1.0, 0.5)
        amount_multiplier = st.number_input("成交额换算为元的倍率", min_value=0.0001, value=1.0)
        min_history = st.number_input("最低有效历史天数", 1, 1500, 60)
        maximum_weight = st.number_input("单股票目标权重上限（%）", 0.1, 100.0, 100.0, 5.0)
        annual_days = st.number_input("年化交易日数", 1, 366, 252)
        risk_free = st.number_input("无风险年利率（%）", 0.0, 30.0, 0.0, 0.5)
        dividend_tax = st.number_input("严格模式固定股息税率（%）", 0.0, 100.0, 0.0, 1.0)
        exclude_st = st.checkbox("排除有历史标记的 ST 股票", value=True)
        benchmark = st.text_input("参考股票代码（可留空；不是市场指数）", "")
    st.caption("印花税、过户费按历史日期切换；默认金额单位按元，需与老师确认。")

left, right = st.columns([1.4, 1], gap="large")
examples = {"动量选股 · 每 5 日调仓": "momentum.py", "均线趋势 · 每 5 日调仓": "moving_average.py",
            "固定股票篮子 · 每 20 日等权": "equal_weight.py", "自定义策略模板": "custom_template.py"}
with left:
    st.subheader("01 / 编写策略")
    example = st.selectbox("策略起点", list(examples))
    filename = examples[example]
    source = st.text_area("Python 策略代码", (ROOT / "strategies" / filename).read_text(encoding="utf-8"),
                          height=450, key=f"source_{filename}")
    default_parameters = {"momentum.py": '{"lookback":60,"skip":5,"top_n":10,"rebalance_days":5}',
                          "moving_average.py": '{"fast":20,"slow":60,"top_n":20}',
                          "equal_weight.py": '{}', "custom_template.py": '{"lookback":20,"top_n":5}'}
    parameters_text = st.text_area("策略参数（JSON）", default_parameters[filename], height=90, key=f"params_{filename}")
    submitted = st.button("运行回测", type="primary", width="stretch")
    st.caption("策略在本机 Python 中执行，请运行你信任的代码。")
with right:
    st.subheader("02 / 数据与成交口径")
    st.markdown("""
    - 历史接口只返回当前收盘时点及以前的数据。
    - 返回权重字典：下一观测日尝试调仓；遗漏股票目标为零。
    - 返回 `None`：继续持仓，不重新等权。
    - 返回 `{}`：下一观测日尝试清仓。
    - T+1、数量单位、资金和成交规模约束均在成交层检查。
    - 开盘涨停不买、跌停不卖；订单当日失效，不保证成交。
    """)
    st.warning("老师的数据没有历史 ST、精确停复牌/涨跌停状态和公司行为明细。当前默认是研究近似；勾选严格检查后，数据不足会停止。")
    with st.expander("最小策略接口"):
        st.code('''from ashare import Strategy

class UserStrategy(Strategy):
    def on_close(self, context):
        eligible = context.universe()
        if "000001.SZ" in eligible:
            return {"000001.SZ": 0.5}
        return {}''', language="python")

if submitted:
    try:
        parameters = json.loads(parameters_text)
        if not isinstance(parameters, dict):
            raise ValueError("策略参数必须是 JSON 对象")
        symbols = [symbol.strip() for symbol in scope.replace("，", ",").replace("\n", ",").split(",") if symbol.strip()] or None
        config = BacktestConfig(start=str(start), end=str(end), initial_cash=initial_cash,
                                commission_rate=commission / 10000, minimum_commission=minimum_fee,
                                slippage_bps=slippage, participation_rate=participation / 100,
                                amount_multiplier=amount_multiplier, min_history=int(min_history),
                                annual_days=int(annual_days), risk_free_rate=risk_free / 100,
                                max_weight=maximum_weight / 100, accounting=accounting,
                                strict_rules=strict_rules, exclude_st=exclude_st,
                                dividend_tax_rate=dividend_tax / 100, benchmark=benchmark.strip())
        destination = ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        destination.mkdir(parents=True, exist_ok=False)
        strategy_path = destination / "strategy.py"
        strategy_path.write_text(source, encoding="utf-8")
        progress_bar = st.progress(0, text="正在读取数据与核对表结构…")
        result = run_backtest(data_path, strategy_path, parameters, config, destination, symbols,
                              lambda fraction, date: progress_bar.progress(fraction, text=f"回测至 {date}"))
        st.session_state["result"] = result
        st.session_state["destination"] = str(destination)
        progress_bar.empty()
        st.success("回测完成，策略源码、参数和报告已保存。")
    except Exception as error:
        st.error(f"回测未完成：{error}")
        with st.expander("错误详情"):
            st.exception(error)

if "result" in st.session_state:
    result = st.session_state["result"]
    metrics = result.metrics
    st.divider()
    st.subheader("03 / 回测结果")
    st.caption(f"上次完成的运行：{metrics['start']} — {metrics['end']} · {result.strategy_name} · {result.config['accounting']} · {metrics['observations']} 个观测日")
    columns = st.columns(3) + st.columns(3)
    values = [("累计收益", f"{metrics['total_return']:.2%}"), ("年化收益", f"{metrics['annual_return']:.2%}"),
              ("最大回撤", f"{metrics['max_drawdown']:.2%}"),
              ("夏普比率", f"{metrics['sharpe']:.2f}" if metrics['sharpe'] is not None else "—"),
              ("成交笔数", str(metrics['fills'])), ("费用合计", f"¥{metrics['total_fees']:,.0f}")]
    for column, (label, value) in zip(columns, values):
        column.metric(label, value)
    net_tab, trades_tab, orders_tab, holdings_tab, assumptions_tab = st.tabs(["净值与回撤", "成交记录", "订单及拒单原因", "持仓变化", "口径与配置"])
    with net_tab:
        curve_columns = ["net_value"] + (["benchmark_net_value"] if "benchmark_net_value" in result.equity else [])
        chart_data = result.equity[curve_columns].rename(columns={"net_value": "策略净值", "benchmark_net_value": "指定参考股票"}).reset_index()
        net_chart = alt.Chart(chart_data).transform_fold(chart_data.columns[1:].tolist(), as_=["series", "value"]).mark_line().encode(
            x=alt.X("date:T", title="日期", axis=alt.Axis(format="%Y-%m", tickCount=6)), y=alt.Y("value:Q", title="净值", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title=""), tooltip=[alt.Tooltip("date:T", title="日期"), alt.Tooltip("series:N", title="序列"), alt.Tooltip("value:Q", title="净值", format=".4f")]).properties(height=320)
        drawdown_chart = alt.Chart(result.equity.reset_index()).mark_area(color="#e07059", opacity=0.6).encode(
            x=alt.X("date:T", title="日期", axis=alt.Axis(format="%Y-%m", tickCount=6)), y=alt.Y("drawdown:Q", title="回撤", axis=alt.Axis(format=".0%")),
            tooltip=[alt.Tooltip("date:T", title="日期"), alt.Tooltip("drawdown:Q", title="回撤", format=".2%")]).properties(height=180)
        st.altair_chart(net_chart, width="stretch")
        st.altair_chart(drawdown_chart, width="stretch")
        st.caption(f"双边累计换手 {metrics['two_way_turnover']:.2f} 倍 · 平均股票仓位 {metrics['average_exposure']:.1%} · 期末现金 ¥{metrics['final_cash']:,.2f}")
    with trades_tab:
        st.dataframe(result.trades, width="stretch", hide_index=True)
    with orders_tab:
        st.dataframe(result.orders, width="stretch", hide_index=True)
        if not result.orders.empty:
            st.dataframe(result.orders[result.orders.reason != ""].reason.value_counts().rename("次数"))
    with holdings_tab:
        dates = sorted(result.positions.date.unique()) if not result.positions.empty else []
        if dates:
            selected_date = st.select_slider("持仓日期", options=dates, value=dates[-1])
            st.dataframe(result.positions[result.positions.date == selected_date], width="stretch", hide_index=True)
        else:
            st.info("本次没有持仓。")
    with assumptions_tab:
        for note in result.assumptions:
            st.write("• " + note)
        st.json({"config": result.config, "data": result.data_summary})
    destination = Path(st.session_state["destination"])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(destination.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    st.download_button("下载报告、CSV 与策略源码", buffer.getvalue(), file_name=f"回测结果-{destination.name}.zip", mime="application/zip")
    st.caption(f"结果目录：{destination}")
