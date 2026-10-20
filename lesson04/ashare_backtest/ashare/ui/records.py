import altair as alt
import pandas as pd
import streamlit as st

from .common import result_view


KINDS = {"backtest": "回测", "training": "训练候选", "validation": "验证", "legacy": "历史运行"}


def render(runs):
    st.header("实验记录")
    st.markdown('<div class="page-subtitle">运行归档与同口径比较</div>', unsafe_allow_html=True)
    records = runs.list()
    if not records:
        st.info("暂无运行记录")
        return
    columns = st.columns(4)
    columns[0].metric("已保存运行", len(records))
    columns[1].metric("完成", sum(record["status"] == "completed" for record in records))
    columns[2].metric("失败", sum(record["status"] == "failed" for record in records))
    columns[3].metric("训练候选", sum(record["kind"] == "training" for record in records))
    st.markdown('<div class="panel-label">运行列表</div>', unsafe_allow_html=True)
    kinds = st.multiselect("类型", list(KINDS), format_func=KINDS.get, default=list(KINDS))
    records = [record for record in records if record["kind"] in kinds]
    rows = [{"运行": record["id"], "策略": record["label"], "类型": KINDS.get(record["kind"], record["kind"]),
             "状态": {"completed": "完成", "failed": "失败", "running": "运行中"}.get(record["status"], record["status"]),
             "开始": record["metrics"].get("start", record["config"].get("start")),
             "结束": record["metrics"].get("end", record["config"].get("end")),
             "初始资金": record["config"].get("initial_cash"),
             "收益(%)": record["metrics"].get("total_return", float("nan")) * 100,
             "回撤(%)": record["metrics"].get("max_drawdown", float("nan")) * 100,
             "夏普": record["metrics"].get("sharpe")} for record in records]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "收益(%)": st.column_config.NumberColumn("累计收益", format="%.2f%%"),
        "回撤(%)": st.column_config.NumberColumn("最大回撤", format="%.2f%%"),
        "夏普": st.column_config.NumberColumn("夏普", format="%.2f"),
        "初始资金": st.column_config.NumberColumn("初始资金（元）", format="%.0f")})
    completed = {record["id"]: record for record in records if record["status"] == "completed"}
    existing = st.session_state.get("records_selection", [])
    if any(identifier not in completed for identifier in existing):
        st.session_state["records_selection"] = [identifier for identifier in existing if identifier in completed]
    selected = st.multiselect("查看或比较", list(completed),
                              format_func=lambda identifier: f"{completed[identifier]['label']} · {identifier}", key="records_selection")
    if len(selected) == 1:
        identifier = selected[0]
        result_view(runs.load(identifier), runs.path(identifier), "record_" + identifier)
    elif len(selected) > 1:
        table, curves, differences = runs.comparison(selected)
        if differences:
            st.warning("比较口径不同，仅并列展示：" + "、".join(differences))
        st.dataframe(table, hide_index=True, width="stretch")
        st.altair_chart(alt.Chart(curves).mark_line().encode(
            x=alt.X("date:T", title=None), y=alt.Y("net_value:Q", title="净值", scale=alt.Scale(zero=False)),
            color=alt.Color("run:N", title="运行"), tooltip=["run:N", "date:T", alt.Tooltip("net_value:Q", format=".4f")]).properties(height=360), width="stretch")
        st.download_button("下载比较表", table.to_csv(index=False).encode("utf-8-sig"), "comparison.csv", "text/csv")
    failed = [record for record in records if record["status"] == "failed"]
    if failed:
        with st.expander("失败记录"):
            for record in failed:
                st.write(record["id"], record["error"])
