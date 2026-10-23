from datetime import date
import io
import json
import zipfile

import pandas as pd
import streamlit as st

from ..evaluation import EvaluationConfig
from ..factors import FACTOR_NAMES


def render(store, settings):
    st.header("因子评价")
    st.markdown('<div class="page-subtitle">全期表现 · 时间稳定性 · 阶段反转</div>', unsafe_allow_html=True)
    create_tab, records_tab = st.tabs(["新建评价", "评价记录"])
    with create_tab:
        with st.form("factor_evaluation_form", border=False):
            left, right = st.columns([1.2, 1], gap="large")
            with left:
                st.markdown('<div class="panel-heading">因子与样本 <span>FACTOR</span></div>', unsafe_allow_html=True)
                factors = st.multiselect("评价因子", list(FACTOR_NAMES), default=["reversal5", "momentum20"],
                                         format_func=FACTOR_NAMES.get, key="factor_selected")
                columns = st.columns(2)
                start = columns[0].date_input("评价开始", date(2020, 1, 2), key="factor_start")
                end = columns[1].date_input("评价结束", date(2024, 12, 31), key="factor_end")
                horizons = st.multiselect("未来收益期（交易日）", [1, 5, 10, 20, 60], default=[1, 5, 20], key="factor_horizons")
                symbols_text = st.text_area("股票代码（留空为数据全量）", key="factor_symbols", height=70)
                known_st = st.checkbox("只纳入历史状态明确非ST的股票", value=True, key="factor_st")
            with right:
                st.markdown('<div class="panel-heading">动态诊断 <span>STABILITY</span></div>', unsafe_allow_html=True)
                fast = st.selectbox("近期窗口（交易日）", [21, 63, 126], index=1, key="factor_fast")
                reference = st.selectbox("历史参考窗口（交易日）", [126, 252, 504], index=1, key="factor_reference")
                direction = st.selectbox("预先指定的因子方向", [1, -1], format_func=lambda value: "高值预期高收益" if value == 1 else "低值预期高收益", key="factor_direction")
                floor = st.number_input("IC实用阈值", 0.0, 0.5, 0.01, 0.005, format="%.3f", key="factor_floor")
                with st.expander("样本与流动性"):
                    min_stocks = st.number_input("每日最少股票数", 5, 5000, 30, key="factor_min_stocks")
                    minimum_amount = st.number_input("20日平均成交额下限（万元）", 0, 100000, 5000, key="factor_amount")
                    liquid_pool = st.number_input("流动性候选上限", 5, 10000, 1500, key="factor_pool")
                st.caption("滚动曲线：21 / 63 / 126 / 252日；窗口不以样本总长机械确定。")
            submitted = st.form_submit_button("运行因子评价", icon=":material/monitoring:", type="primary", width="stretch")
        if submitted:
            try:
                config = EvaluationConfig(start=str(start), end=str(end), factors=tuple(factors), horizons=tuple(horizons),
                    fast_window=fast, reference_window=reference, direction=direction, effect_floor=floor,
                    min_stocks=int(min_stocks), minimum_amount=minimum_amount * 10000,
                    liquid_pool=int(liquid_pool), require_st_history=known_st)
                symbols = [symbol.strip() for symbol in symbols_text.replace("，", ",").replace("\n", ",").split(",") if symbol.strip()]
                with st.spinner("读取数据与计算因子"):
                    progress = st.progress(0, text="准备评价")
                    identifier = store.run(settings["data_path"], settings.get("metadata_path"), config, symbols or None,
                                           lambda fraction, message: progress.progress(fraction, text=message))
                    progress.empty()
                st.session_state["factor_active"] = identifier
                st.session_state.pop("factor_record", None)
                st.success("评价已保存，请在评价记录查看")
            except Exception as error:
                st.error(str(error))
    with records_tab:
        records = store.list()
        completed = {record["id"]: record for record in records if record["status"] == "completed"}
        if not completed:
            st.info("暂无评价记录。可先在2020—2024区间评价，不会自动读取后续考核结果。")
        else:
            identifiers = list(completed)
            active = st.session_state.get("factor_active", identifiers[0])
            identifier = st.selectbox("评价记录", identifiers, index=identifiers.index(active) if active in identifiers else 0,
                format_func=lambda value: completed[value]["config"]["start"] + " — " + completed[value]["config"]["end"] + " · " + value,
                key="factor_record")
            try:
                request, tables = store.load(identifier)
                render_result(request, tables)
            except Exception as error:
                st.error(str(error))
        failures = [record for record in records if record["status"] == "failed"]
        if failures:
            with st.expander("失败记录"):
                st.dataframe(pd.DataFrame([{"记录": row["id"], "错误": row["error"]} for row in failures]), hide_index=True)


def render_result(request, tables):
    columns = st.columns(2)
    factor = columns[0].selectbox("因子", list(tables["static"].factor.unique()), format_func=lambda value: FACTOR_NAMES.get(value, value), key="factor_result_name")
    horizons = list(tables["static"].loc[tables["static"].factor == factor, "horizon"].unique())
    horizon = columns[1].selectbox("预测期", horizons, format_func=lambda value: f"{value}个交易日", key="factor_result_horizon")
    subset = {name: table[(table.factor == factor) & (table.horizon == horizon)].copy() for name, table in tables.items()}
    summary = subset["static"].iloc[0]
    latest = subset["dynamic"].iloc[-1]
    columns = st.columns(4)
    columns[0].metric("全期 RankIC", f"{summary['mean']:.3f}" if pd.notna(summary["mean"]) else "—")
    fast = request["config"]["fast_window"]
    recent = latest[f"mean_{fast}"]
    columns[1].metric(f"近期{fast}日 RankIC", f"{recent:.3f}" if pd.notna(recent) else "—")
    columns[2].metric("最新诊断", latest["state"])
    transition = subset["transitions"].iloc[0]
    columns[3].metric("正向阶段后反转", f"{int(transition.next_negative_blocks)} / {int(transition.positive_previous_blocks)}")
    st.caption(f"{request['symbol_count']}个输入代码 · 信息截至 {latest.as_of} 收盘 · 方向预设 {request['config']['direction']:+d} · 状态依据已兑现收益，不是未来业绩预测")
    static_tab, dynamic_tab, stages_tab, details_tab = st.tabs(["静态评价", "动态稳定性", "分段与反转", "口径与导出"])
    with static_tab:
        st.dataframe(subset["static"].rename(columns={"mean": "RankIC均值", "std": "IC标准差", "ir": "ICIR（不年化）", "count": "有效IC日", "lower": "95%下界", "upper": "95%上界", "coverage": "配对覆盖率", "mean_spread": "Q5-Q1毛收益"}), hide_index=True, width="stretch")
        quantiles = pd.DataFrame({"分组": [f"Q{group}" for group in range(1, 6)], "平均未来收益": [summary[f"q{group}"] for group in range(1, 6)]})
        st.bar_chart(quantiles.set_index("分组"))
        st.caption("分组为原始因子从低到高；收益未扣费、未模拟涨跌停与容量，不是可成交组合收益。")
    with dynamic_tab:
        frame = subset["dynamic"].set_index("as_of")
        frame.index = pd.to_datetime(frame.index)
        windows = request["config"]["windows"]
        st.line_chart(frame[[f"mean_{window}" for window in windows]].rename(columns={f"mean_{window}": f"{window}日IC" for window in windows}))
        st.line_chart(frame[[f"mean_{fast}", f"lower_{fast}", f"upper_{fast}"]].rename(columns={f"mean_{fast}": "近期IC", f"lower_{fast}": "95%下界", f"upper_{fast}": "95%上界"}))
        st.line_chart(frame[["drift", "drift_upper"]].rename(columns={"drift": "近期－历史参考IC", "drift_upper": "差值95%上界"}))
        st.caption("历史参考与近期窗口不重叠，并按预测期留间隔。95%区间为逐点HAC近似；连续报警不等于独立重复验证，也不控制全程误报率。")
        st.dataframe(subset["dynamic"][["as_of", "state", "reference_mean", "drift", "alert_streak", "persistent_alert"]].tail(30), hide_index=True, width="stretch")
    with stages_tab:
        st.dataframe(subset["monthly"].pivot(index="factor", columns="month", values="mean_ic"), width="stretch")
        st.dataframe(subset["blocks"], hide_index=True, width="stretch")
        st.dataframe(subset["transitions"], hide_index=True, width="stretch")
        st.caption("63交易日非重叠分段；只保留段内起止均完整的标签。尾段或样本不足段不参与相邻阶段反转统计；样本数是阶段对，不是股票数。")
    with details_tab:
        st.warning("稳定的低相关也可能没有价值；正向支持也不保证下一阶段有效。窗口、方向和阈值应在验证前固定；反复选择最漂亮的窗口属于多重尝试。")
        st.json(request["config"])
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("request.json", json.dumps(request, ensure_ascii=False, indent=2))
            for name, table in tables.items():
                archive.writestr(name + ".csv", table.to_csv(index=False))
        st.download_button("下载评价结果", buffer.getvalue(), "factor_evaluation.zip", "application/zip", key="factor_results_download")
