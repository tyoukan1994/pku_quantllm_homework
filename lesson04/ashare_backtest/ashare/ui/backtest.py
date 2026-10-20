import json

import streamlit as st

from ..config import BacktestConfig
from .common import base_config, configuration, json_object, navigate, result_view, strategy_picker
from .design import empty_results


def render(strategies, runs, settings):
    st.header("回测")
    st.markdown('<div class="page-subtitle">策略配置与绩效分析</div>', unsafe_allow_html=True)
    configuration_column, result_column = st.columns([1, 2.45], gap="large")
    with configuration_column:
        with st.container(key="config_panel"):
            st.markdown('<div class="panel-heading">回测设置 <span>CONFIGURATION</span></div>', unsafe_allow_html=True)
            strategy_column, preset_column = st.columns([1.2, 1])
            with strategy_column:
                strategy = strategy_picker(strategies, "backtest_strategy")
            with preset_column:
                preset = st.selectbox("配置预设", ["标准日频", "课程作业"], key="backtest_preset")
            parameters_text = st.text_area("策略参数（JSON）", json.dumps(strategy.parameters, ensure_ascii=False),
                                           height=85, key=f"backtest_params_{strategy.identifier}")
            config_values, symbols = configuration("backtest_" + preset, base_config(preset), compact=True)
            run = st.button("运行回测", icon=":material/play_arrow:", type="primary", width="stretch")
            if st.button("编辑策略", icon=":material/code:", width="stretch"):
                st.session_state["preferred_strategy"] = strategy.identifier
                st.session_state.pop("library_strategy", None)
                navigate("策略库")
            st.markdown('<div class="strategy-meta"><span>收盘信号 / 次日开盘</span><span>T+1</span></div>', unsafe_allow_html=True)
    if run:
        try:
            parameters = json_object(parameters_text)
            config = BacktestConfig(**config_values)
            with result_column:
                bar = st.progress(0, text="读取数据")
                identifier, result = runs.run(settings["data_path"], settings.get("metadata_path"),
                                               strategy, parameters, config, symbols,
                                               progress=lambda fraction, day: bar.progress(fraction, text=day))
                bar.empty()
            st.session_state["backtest_active_run"] = identifier
        except Exception as error:
            with result_column:
                st.error(str(error))
    with result_column:
        identifier = st.session_state.get("backtest_active_run")
        if identifier:
            st.markdown('<div class="result-heading">回测结果 <span class="neutral-tag">已完成</span></div>', unsafe_allow_html=True)
            with st.container(key="result_live_panel"):
                result_view(runs.load(identifier), runs.path(identifier), "backtest_result_" + identifier)
        else:
            st.markdown('<div class="result-heading">回测结果 <span class="neutral-tag">未运行</span></div>', unsafe_allow_html=True)
            empty_results()
