import pandas as pd
import streamlit as st

from ..data import DataBundle
from ..platform.runs import write_json


def render(settings, settings_path):
    st.header("数据与规则")
    st.markdown('<div class="page-subtitle">数据源连接、字段检查与撮合口径</div>', unsafe_allow_html=True)
    connection_column, rules_column = st.columns([1, 1.5], gap="large")
    with connection_column:
        with st.container(key="data_form"):
            st.markdown('<div class="panel-heading">数据源 <span>LOCAL FILES</span></div>', unsafe_allow_html=True)
            with st.form("data_settings", border=False):
                source = st.text_input("行情目录", settings["data_path"])
                metadata = st.text_input("历史状态目录", settings.get("metadata_path") or "")
                if st.form_submit_button("保存数据设置", type="primary", width="stretch"):
                    settings.update(data_path=source.strip(), metadata_path=metadata.strip() or None)
                    write_json(settings_path, settings)
                    st.session_state["platform_data_settings"] = settings
                    st.success("已保存")
            check = st.button("检查数据", icon=":material/fact_check:", width="stretch")
            with st.expander("字段口径"):
                st.write("价格复权：原始价格×cumadj。amt为成交额，num_trades为成交笔数；不将其当作股数成交量。")
    with rules_column:
        with st.container(key="rule_table"):
            st.markdown('<div class="panel-heading">撮合规则 <span>EXECUTION</span></div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([
                ("信号", "当日收盘生成，下一交易日开盘执行"),
                ("涨跌停", "开盘涨停不买、跌停不卖；精确限价优先，否则按历史板块规则推算"),
                ("T+1", "当日买入，下一个交易观测日才可卖出"),
                ("申报", "按板块数量单位及单笔上限处理，可拆单"),
                ("资金与容量", "先卖后买；子订单共享上一日成交额参与上限"),
                ("非ST", "可配置；要求历史标记时，未知状态不买"),
                ("期末", "可选清仓及退出窗口；受阻持仓保留并报告"),
                ("账户", "复权研究为份额近似；严格实股模式需要公司行为明细"),
            ], columns=["项目", "规则"]), hide_index=True, width="stretch")
            st.caption("日频不能复现盘口排队和开盘冲击；精确状态、公司行为不足时仍存在研究近似。")
    if check:
        try:
            with st.spinner("读取数据"):
                data = DataBundle.from_feathers(settings["data_path"], metadata_dir=settings.get("metadata_path"))
                summary = data.summary()
            st.subheader("数据检查")
            columns = st.columns(3)
            columns[0].metric("股票代码", summary["symbols"])
            columns[1].metric("交易观测日", summary["dates"])
            columns[2].metric("最新日期", summary["end"])
            st.caption(f"{summary['start']} — {summary['end']}")
            with st.expander("完整检查信息"):
                st.json(summary)
        except Exception as error:
            st.error(str(error))
