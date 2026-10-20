from pathlib import Path

import pandas as pd
import streamlit as st

from ..discovery import table_rows
from ..platform.runs import read_json
from .common import navigate


def render_campaigns(runs):
    paths = sorted(runs.root.glob("*/campaign.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not paths:
        st.caption("暂无批量策略探索")
        return
    lookup = {path.parent.name: path for path in paths}
    identifier = st.selectbox("探索记录", list(lookup), format_func=lambda value: read_json(lookup[value])["name"])
    path = lookup[identifier]
    record = read_json(path)
    protocol = read_json(path.parent / "protocol.json")
    config = protocol["training_config"]
    confirmation = protocol["confirmation"]
    st.caption(f"开发 {config['start']} — {config['end']}　·　确认 {confirmation[0]} — {confirmation[1]}　·　资金 {config['initial_cash'] / 1e8:g}亿元")
    columns = st.columns(4)
    columns[0].metric("候选", len(protocol["candidates"]))
    columns[1].metric("完成开发", len(record["calibration"]))
    columns[2].metric("通过开发门槛", sum(row.get("qualified", False) for row in record["calibration"]))
    columns[3].metric("确认运行", len(record["confirmation"]))
    if record["status"] != "completed":
        st.progress(len(record["calibration"]) / len(protocol["candidates"]), text="运行中；刷新查看进展")
    sections = [("开发比较", "calibration"), ("确认表现", "confirmation"), ("滑点压力", "sensitivity")]
    if record.get("evaluation"):
        sections.append(("考核结果", "evaluation"))
    for tab, (_, key) in zip(st.tabs([name for name, key in sections]), sections):
        with tab:
            rows = record.get(key, [])
            if not rows:
                st.caption("尚无结果")
                continue
            if key == "calibration":
                rows = sorted(rows, key=lambda row: (-int(row.get("qualified", False)), -row.get("robustness", 0), row["id"]))
            table = pd.DataFrame(table_rows(rows))
            visible = [name for name in ("策略", "类别", "收益(%)", "回撤(%)", "夏普", "平均仓位(%)", "费用(万元)", "训练门槛通过") if name in table]
            if key == "evaluation":
                st.caption(f"考核 {rows[0]['metrics']['start']} — {rows[0]['metrics']['end']}；末日清仓、20日退出及5bp滑点分别列示，不据考核结果调参。")
                for note in record.get("evaluation_notes", []):
                    st.warning(note)
                table["清仓完成"] = [row["metrics"]["liquidation_complete"] for row in rows]
                table["剩余股票市值(万元)"] = [row["metrics"]["terminal_market_value"] / 1e4 for row in rows]
                visible = ["策略", "收益(%)", "年化(%)", "回撤(%)", "夏普", "清仓完成", "剩余股票市值(万元)"]
            st.dataframe(table[visible], hide_index=True, width="stretch", column_config={
                "收益(%)": st.column_config.NumberColumn("累计收益", format="%.2f%%"),
                "年化(%)": st.column_config.NumberColumn("年化收益", format="%.2f%%"),
                "回撤(%)": st.column_config.NumberColumn("最大回撤", format="%.2f%%"),
                "夏普": st.column_config.NumberColumn("夏普", format="%.2f"),
                "平均仓位(%)": st.column_config.NumberColumn("平均仓位", format="%.1f%%"),
                "费用(万元)": st.column_config.NumberColumn("费用（万元）", format="%.2f"),
                "剩余股票市值(万元)": st.column_config.NumberColumn("剩余股票市值（万元）", format="%.2f")})
            if key == "evaluation":
                report = runs.path(Path(record["evaluation_protocol"]).parent.name) / "考核报告.md"
                if report.exists():
                    st.download_button("下载考核报告", report.read_bytes(), "2025—2026策略考核.md", "text/markdown")
            selected = st.selectbox("查看运行", range(len(rows)), format_func=lambda index, options=rows: options[index]["name"], key="campaign_candidate_" + identifier + key)
            row = rows[selected]
            run_id = row.get("run_id", f"{identifier}/{key}/{row['id']}")
            if st.button("打开回测明细", key="campaign_open_" + key):
                st.session_state["records_selection"] = [run_id]
                navigate("实验记录")
            st.download_button("下载比较表", table.to_csv(index=False).encode("utf-8-sig"),
                               f"{identifier}-{key}.csv", "text/csv", key="campaign_download_" + key)
    if record.get("baseline"):
        with st.expander("同口径参照"):
            st.dataframe(pd.DataFrame(table_rows(record["baseline"])), hide_index=True, width="stretch")
    with st.expander("年度表现与参数"):
        yearly = [{"策略": row["name"], "年度": year["year"], "收益(%)": year["return"] * 100,
                   "回撤(%)": year["drawdown"] * 100, "夏普": year["sharpe"]}
                  for row in record["calibration"] for year in row["years"]]
        st.dataframe(pd.DataFrame(yearly), hide_index=True, width="stretch")
        st.json({row["name"]: row["parameters"] for row in record.get("confirmation", [])})
    report = path.parent / "研究报告.md"
    if report.exists():
        st.download_button("下载研究报告", report.read_bytes(), f"{identifier}-研究报告.md", "text/markdown")
    st.caption("确认结果按事前冻结顺序展示；后续考核不回流选参。开发不达标的诊断候选也完整保留。" if record.get("evaluation") else "确认结果按事前冻结顺序展示。2025—2026未参与本次比较；开发不达标的诊断候选也完整保留。")
