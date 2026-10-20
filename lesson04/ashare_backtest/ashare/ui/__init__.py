import os
from pathlib import Path

import streamlit as st

from ..cli import DEFAULT_DATA
from ..platform import RunStore, StrategyStore, StudyStore
from ..platform.runs import read_json
from . import backtest, data, records, research, strategies
from .common import navigate
from .design import theme, topbar


def launch(project_root):
    for key in list(st.session_state):
        if (key.startswith(("backtest_", "library_", "research_", "strategy_name_", "candidate_", "validate_", "viewed_"))
                or key == "records_selection") and not key.endswith("_download"):
            st.session_state[key] = st.session_state[key]
    project_root = Path(project_root)
    root = Path(os.environ.get("ASHARE_WORKSPACE_DIR", project_root))
    strategy_store = StrategyStore(root / "workspace" / "strategies", project_root / "strategies")
    run_store = RunStore(root / "runs")
    study_store = StudyStore(root / "workspace" / "studies", run_store)
    settings_path = root / "workspace" / "settings.json"
    defaults = {"data_path": DEFAULT_DATA,
                "metadata_path": os.environ.get("ASHARE_METADATA_DIR", str(project_root.parents[1] / "work" / "market_metadata"))}
    settings = st.session_state.setdefault("platform_data_settings", read_json(settings_path) if settings_path.exists() else defaults)
    st.set_page_config(page_title="A股策略平台", page_icon="📊", layout="wide")
    theme()
    if "pending_page" in st.session_state:
        st.session_state["platform_page"] = st.session_state.pop("pending_page")
    page = st.session_state.get("platform_page", "回测")
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark">A</div><div><div class="brand-name">A股策略平台</div><div class="brand-sub">STRATEGY PLATFORM</div></div></div><div class="nav-group">工作空间</div>', unsafe_allow_html=True)
        icons = {"回测": "candlestick_chart", "策略库": "code_blocks", "策略研究": "science",
                 "实验记录": "query_stats", "数据与规则": "database"}
        for name, icon in icons.items():
            if st.button(name, icon=f":material/{icon}:", type="primary" if page == name else "secondary",
                         width="stretch", key="nav_" + name):
                navigate(name)
        source = Path(settings["data_path"]).expanduser()
        available = sum((source / (field + ".feather")).exists()
                        for field in ("open", "high", "low", "close", "amt", "cumadj", "num_trades"))
        metadata = settings.get("metadata_path")
        known_st = bool(metadata and (Path(metadata).expanduser() / "is_st.feather").exists())
        status_class = "status-dot" if available >= 6 else "status-dot off"
        st.markdown(f'<div class="sidebar-data"><div class="sidebar-data-title"><span class="{status_class}"></span>数据连接</div><div class="sidebar-data-line">行情文件<b>{available} / 7</b></div><div class="sidebar-data-line">历史 ST 状态<b>{"已连接" if known_st else "未连接"}</b></div><div class="sidebar-data-line">市场范围<b>沪深 A 股</b></div></div><div class="sidebar-note">本地研究环境　v0.3</div>', unsafe_allow_html=True)
    topbar(page)
    if page == "回测":
        backtest.render(strategy_store, run_store, settings)
    elif page == "策略库":
        strategies.render(strategy_store)
    elif page == "策略研究":
        research.render(strategy_store, study_store, settings)
    elif page == "实验记录":
        records.render(run_store)
    else:
        data.render(settings, settings_path)
