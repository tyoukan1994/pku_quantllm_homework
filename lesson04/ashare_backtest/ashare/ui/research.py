from dataclasses import replace
import json

import pandas as pd
import streamlit as st

from ..config import BacktestConfig
from ..factors import FACTOR_NAMES
from ..platform.runs import read_json
from ..platform.studies import expand_grid
from .common import base_config, configuration, json_object, navigate, strategy_picker
from .discovery import render_campaigns


def render(strategies, studies, settings):
    st.header("策略研究")
    st.markdown('<div class="page-subtitle">参数搜索、因子比较与候选验证</div>', unsafe_allow_html=True)
    create_tab, studies_tab, campaigns_tab = st.tabs(["新建研究", "研究记录", "策略探索"])
    with campaigns_tab:
        render_campaigns(studies.runs)
    with create_tab:
        candidates_column, training_column = st.columns([1.1, 1], gap="large")
        with candidates_column:
            with st.container(key="research_form"):
                st.markdown('<div class="panel-heading">候选范围 <span>CANDIDATES</span></div>', unsafe_allow_html=True)
                mode = st.radio("研究方式", ["参数搜索", "量价因子比较"], horizontal=True)
                if mode == "参数搜索":
                    strategy = strategy_picker(strategies, "research_strategy")
                    parameters_text = st.text_area("基础参数（JSON）", json.dumps(strategy.parameters, ensure_ascii=False),
                                                   height=100, key="research_params_" + strategy.identifier)
                    grid_text = st.text_area("参数网格（JSON）", json.dumps(strategy.grid or {"lookback": [10, 20]}, ensure_ascii=False),
                                             height=130, key="research_grid_" + strategy.identifier)
                else:
                    strategy = strategies.get("builtin:price_amount")
                    factors = st.multiselect("因子", list(FACTOR_NAMES), default=list(FACTOR_NAMES), format_func=FACTOR_NAMES.get)
                    counts = st.multiselect("持仓数量", [50, 100, 200, 300], default=[100, 200])
                    rebalance = st.number_input("调仓间隔（交易日）", 1, 120, 5)
                    parameters_text = json.dumps({**strategy.parameters, "rebalance_days": int(rebalance)})
                    grid_text = json.dumps({"factor": factors, "top_n": counts})
                label = st.text_input("研究名称", strategy.name)
                metrics = {"sharpe": "夏普", "total_return": "累计收益", "max_drawdown": "最大回撤"}
                metric = st.selectbox("训练排序指标", list(metrics), format_func=metrics.get)
                try:
                    count = len(expand_grid(json_object(parameters_text), json_object(grid_text)))
                    st.caption(f"{count} 个候选 / 最多24个组合")
                except (ValueError, TypeError) as error:
                    st.caption(str(error))
        with training_column:
            with st.container(key="config_panel"):
                st.markdown('<div class="panel-heading">训练配置 <span>TRAINING</span></div>', unsafe_allow_html=True)
                preset = st.selectbox("账户预设", ["标准日频", "课程作业"])
                config_values, symbols = configuration("research_" + preset, base_config(preset, training=True), compact=True)
                submitted = st.button("运行训练搜索", icon=":material/play_arrow:", type="primary", width="stretch")
                st.caption("本步骤仅运行训练区间。验证在研究记录中单独发起。")
        if submitted:
            try:
                bar = st.progress(0, text="训练搜索")
                identifier = studies.search(settings["data_path"], settings.get("metadata_path"), strategy,
                                              json_object(parameters_text), json_object(grid_text),
                                              BacktestConfig(**config_values), symbols, label, metric,
                                              lambda fraction, message: bar.progress(fraction, text=message))
                bar.empty()
                st.session_state["study_active"] = identifier
                st.session_state.pop("research_study_selector", None)
                st.success("已保存，可在研究记录中选择候选并验证")
            except Exception as error:
                st.error(str(error))
    with studies_tab:
        records = studies.list()
        if not records:
            st.caption("暂无研究记录；原有实验可在实验记录中查看。")
            return
        lookup = {record["id"]: record for record in records}
        identifiers = list(lookup)
        active = st.session_state.get("study_active", identifiers[0])
        index = identifiers.index(active) if active in identifiers else 0
        identifier = st.selectbox("研究", identifiers, index=index,
                                  format_func=lambda value: lookup[value]["label"] + " · " + value,
                                  key="research_study_selector")
        study = lookup[identifier]
        st.caption(f"训练：{study['train_config']['start']} — {study['train_config']['end']}")
        rows = [{"运行": record.get("run_id", ""), "参数": json.dumps(record["parameters"], ensure_ascii=False),
                 "状态": record["status"], "收益(%)": record.get("metrics", {}).get("total_return", float("nan")) * 100,
                 "回撤(%)": record.get("metrics", {}).get("max_drawdown", float("nan")) * 100,
                 "夏普": record.get("metrics", {}).get("sharpe"), "错误": record.get("error", "")}
                for record in study["candidates"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        successful = [record for record in study["candidates"] if record["status"] == "completed"]
        if not successful:
            return
        frozen_path = studies.path(identifier) / "frozen.json"
        frozen = read_json(frozen_path) if frozen_path.exists() else None
        if frozen and frozen["status"] == "completed":
            st.caption("验证已完成 · " + ("区间曾查看" if frozen["previously_viewed"] else "使用者标记区间未查看"))
            if st.button("查看验证结果"):
                st.session_state["records_selection"] = [frozen["run_id"]]
                navigate("实验记录")
            return
        st.subheader("冻结候选并验证")
        if frozen:
            candidate_id = frozen["candidate_id"]
            validation = BacktestConfig(**frozen["validation_config"])
            viewed = frozen["previously_viewed"]
            st.caption(f"重试原冻结配置 · {candidate_id} · {validation.start} — {validation.end}")
        else:
            metric = study["metric"]
            successful.sort(key=lambda record: record["metrics"].get(metric) if record["metrics"].get(metric) is not None else float("-inf"), reverse=True)
            mapping = {record["run_id"]: record for record in successful}
            candidate_id = st.selectbox("训练候选", list(mapping),
                                        format_func=lambda value: json.dumps(mapping[value]["parameters"], ensure_ascii=False),
                                        key="candidate_" + identifier)
            columns = st.columns(2)
            default_start = pd.Timestamp(study["train_config"]["end"]) + pd.Timedelta(days=1)
            start = columns[0].date_input("验证开始", default_start.date(), key="validate_start_" + identifier)
            end = columns[1].date_input("验证结束", (default_start + pd.DateOffset(years=1) - pd.Timedelta(days=1)).date(), key="validate_end_" + identifier)
            viewed = st.checkbox("该验证区间已在此前研究中查看", value=True, key="viewed_" + identifier)
            validation = replace(BacktestConfig(**study["train_config"]), start=str(start), end=str(end), seed_previous_close=True)
        if st.button("重试冻结验证" if frozen else "冻结并运行验证", type="primary"):
            try:
                bar = st.progress(0, text="验证")
                run_id, result = studies.freeze_and_validate(identifier, candidate_id, validation, viewed,
                    lambda fraction, day: bar.progress(fraction, text=day))
                bar.empty()
                st.rerun()
            except Exception as error:
                st.error(str(error))
