import json

import streamlit as st

from .common import json_object, navigate, strategy_picker


def render(store):
    st.header("策略库")
    st.markdown('<div class="page-subtitle">策略源码、参数与版本管理</div>', unsafe_allow_html=True)
    settings_column, editor_column = st.columns([1, 2.45], gap="large")
    with settings_column:
        with st.container(key="config_panel"):
            st.markdown('<div class="panel-heading">策略信息 <span>STRATEGY</span></div>', unsafe_allow_html=True)
            strategy = strategy_picker(store, "library_strategy")
            st.caption("内置模板 · 另存后可修改" if strategy.builtin else f"自定义策略 · 版本 {strategy.revision}")
            name = st.text_input("名称", strategy.name, key="strategy_name_" + strategy.identifier)
            parameters = st.text_area("默认参数（JSON）", json.dumps(strategy.parameters, ensure_ascii=False), height=130,
                                      key=f"library_params_{strategy.identifier}_{strategy.revision}")
            use = st.button("用于回测", icon=":material/play_arrow:", type="primary", width="stretch")
            with st.expander("导入文件"):
                upload = st.file_uploader("导入Python文件", type=["py"])
                import_file = upload is not None and st.button("从文件创建策略")
            with st.expander("策略接口"):
                st.code('''from ashare import Strategy

class UserStrategy(Strategy):
    def on_close(self, context):
        symbols = context.universe()
        return {symbol: 0.9 / len(symbols) for symbol in symbols} if symbols else {}
''', language="python")
                st.caption("history()截止当日；返回权重在下一日执行；None不下单，{}尝试清仓。")
    with editor_column:
        with st.container(key="strategy_editor"):
            st.markdown('<div class="panel-heading">策略源码 <span>PYTHON / UserStrategy</span></div>', unsafe_allow_html=True)
            source = st.text_area("策略源码", strategy.path.read_text(encoding="utf-8"), height=510,
                                  label_visibility="collapsed", key=f"library_source_{strategy.identifier}_{strategy.revision}")
            save_column, new_column = st.columns(2)
            update = save_column.button("保存修改", disabled=strategy.builtin, width="stretch")
            create = new_column.button("另存为新策略", width="stretch")
            st.caption("仅执行可信Python源码。每次运行保留独立源码快照。")
    if import_file:
        try:
            saved = store.save(upload.name.removesuffix(".py"), upload.getvalue().decode("utf-8-sig"), {})
            st.session_state["preferred_strategy"] = saved.identifier
            st.session_state.pop("library_strategy", None)
            st.rerun()
        except Exception as error:
            st.error(str(error))
    if update or create:
        try:
            saved = store.save(name, source, json_object(parameters), None if create else strategy.identifier)
            st.session_state["preferred_strategy"] = saved.identifier
            st.session_state.pop("library_strategy", None)
            st.session_state["strategy_saved"] = True
            st.rerun()
        except Exception as error:
            st.error(str(error))
    if st.session_state.pop("strategy_saved", False):
        st.success("已保存")
    if use:
        st.session_state["preferred_strategy"] = strategy.identifier
        st.session_state.pop("backtest_strategy", None)
        navigate("回测")
