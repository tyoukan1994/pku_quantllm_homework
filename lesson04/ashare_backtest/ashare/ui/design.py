from html import escape

import streamlit as st


def theme():
    st.markdown("""<style>
:root{--ink:#17263d;--muted:#7b879a;--accent:#2879f4;--line:#e5eaf2}
.stApp{background:#f3f5f9;color:var(--ink)}
.block-container{max-width:1680px;padding:3.7rem 2.1rem 2.2rem}
[data-testid="stAppDeployButton"]{display:none}
[data-testid="stSidebar"]{background:#111e32;border-right:0;min-width:220px;max-width:250px}
[data-testid="stSidebarUserContent"]{padding:2.8rem 1.1rem 1.5rem}
.brand{display:flex;align-items:center;gap:11px;color:#f5f8ff;margin:0 5px 30px}
.brand-mark{display:grid;place-items:center;background:#2879f4;width:34px;height:34px;border-radius:10px;font-size:20px;font-weight:700;box-shadow:0 5px 18px #0c1328}
.brand-name{font-size:18px;font-weight:650;letter-spacing:.5px}.brand-sub{color:#7e91ad;font-size:9px;letter-spacing:1.9px;margin-top:3px}
.nav-group{color:#60758f;font-size:10px;letter-spacing:1.5px;margin:3px 14px 10px}
[data-testid="stSidebar"] [data-testid="stButton"] button{border:0;box-shadow:none;color:#9aabc2;background:transparent;border-radius:8px;min-height:43px;justify-content:flex-start;padding-left:16px;font-size:14px}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover{background:#192b45;color:#edf4ff}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]{background:#213c60;color:#f3f7ff;border-left:3px solid #58a4ff;border-radius:3px 8px 8px 3px;padding-left:13px}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:8px}
.sidebar-data{border:1px solid #26374f;border-radius:10px;padding:14px;margin:40px 4px 0;font-size:12px;color:#a6b7ce;background:#142339}
.sidebar-data-title{font-weight:600;color:#d7e4f4;margin-bottom:10px}.status-dot{display:inline-block;height:6px;width:6px;border-radius:50%;background:#39ba97;margin-right:6px}
.status-dot.off{background:#77869b}
.sidebar-data-line{display:flex;justify-content:space-between;font-size:11px;color:#8196b2;margin-top:7px}.sidebar-data-line b{color:#bdcde0;font-weight:500}
.sidebar-note{font-size:10px;color:#617791;margin:16px 10px}
.workspace-topbar{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #e0e6ef;padding-bottom:15px;margin-bottom:23px;color:#8692a4;font-size:12px}
.workspace-topbar strong{font-weight:500;color:#435672;margin-left:9px}.env-pill{font-size:11px;background:#e7edf6;border:1px solid #dbe4f0;border-radius:5px;padding:4px 8px;color:#5b7090}
h2{font-size:26px!important;font-weight:680!important;letter-spacing:-.4px;padding-top:0!important}h3{font-size:17px!important;font-weight:650!important}
.page-subtitle{font-size:12px;color:#8692a4;margin:-12px 0 24px}
[data-testid="stMetric"]{border:1px solid #e1e7f0;border-radius:10px;padding:16px 17px;background:white;box-shadow:0 2px 4px #182d4a03}
[data-testid="stMetricLabel"]{color:#8190a5;font-size:12px}
[data-testid="stMetricValue"]{font-size:25px!important;font-weight:600;letter-spacing:-.7px;color:#243c60}
[data-testid="stBaseButton-primary"]{background:#2879f4;border-color:#2879f4;border-radius:7px;box-shadow:0 3px 8px #2879f41a}
[data-testid="stBaseButton-primary"]:hover{background:#206ce0;border-color:#206ce0}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input{font-size:13px}
[data-baseweb="input"],[data-baseweb="select"]>div,textarea{border-radius:6px!important}
textarea{font-family:ui-monospace,SFMono-Regular,monospace!important;font-size:12px!important;line-height:1.7!important}
[data-baseweb="tab-list"]{gap:24px;border-bottom:1px solid #e8edf5}
[data-baseweb="tab"]{font-size:13px;font-weight:550}
[data-baseweb="tab-highlight"]{background:#2879f4}
[aria-selected="true"][role="tab"]{color:#2879f4}
[data-testid="stExpander"]{background:#fff;border-color:#e6ebf3;border-radius:8px}
[data-testid="stExpander"] summary p{font-size:12px!important;color:#5c6f89}
.st-key-config_panel{background:#fff;border:1px solid #e1e7f0;border-radius:11px;padding:19px 18px 17px;box-shadow:0 3px 10px #1f3d6404}
.st-key-config_panel [data-testid="stVerticalBlock"]{gap:12px}
.panel-heading{display:flex;align-items:center;justify-content:space-between;font-size:14px;font-weight:650;color:#29415e;margin-bottom:18px}
.panel-heading span{font-size:10px;color:#8c9caf;font-weight:500;letter-spacing:.7px}
.panel-label{font-size:11px;font-weight:600;color:#6b809b;margin:12px 0 8px;letter-spacing:.5px}
.result-heading{display:flex;align-items:center;justify-content:space-between;font-size:14px;font-weight:650;color:#29415e;margin:1px 0 20px}
.neutral-tag{font-size:10px;background:#e9eef6;color:#71849f;border-radius:5px;padding:4px 8px;font-weight:500}
.chart-empty{height:286px;background-color:#fbfcfe;background-image:linear-gradient(#eaf0f7 1px,transparent 1px),linear-gradient(90deg,#eef3f8 1px,transparent 1px);background-size:100% 57px,12.5% 100%;border:1px solid #edf1f7;border-radius:8px;display:flex;align-items:center;justify-content:center;text-align:center;margin:15px 0 14px}
.empty-symbol{background:#f0f5fd;border:1px solid #e3ebf8;display:inline-flex;justify-content:center;align-items:center;color:#7197ce;height:46px;width:46px;border-radius:12px;margin-bottom:14px}
.empty-title{font-size:14px;color:#71859f;font-weight:550}.empty-detail{font-size:11px;color:#9aa8bb;margin-top:8px}
.empty-details{display:flex;gap:30px;color:#97a5b8;font-size:11px;border-top:1px solid #edf1f7;padding-top:14px}
.empty-details b{font-size:12px;color:#637995;font-weight:500;margin-left:5px}
.st-key-result_empty_panel,.st-key-result_live_panel{border:1px solid #e1e7f0;border-radius:11px;padding:18px 20px;background:#fff}
.st-key-result_empty_panel [data-testid="stTabs"]{margin-top:4px}
.strategy-meta{font-size:11px;color:#8b9bb0;display:flex;justify-content:space-between;border-top:1px solid #edf1f7;padding-top:13px;margin-top:6px}
.st-key-strategy_editor,.st-key-research_form,.st-key-research_history,.st-key-data_form,.st-key-rule_table,.st-key-factor_evaluation_form{border:1px solid #e1e7f0;border-radius:11px;padding:20px;background:#fff}
.st-key-strategy_editor textarea{background:#18283e!important;color:#d9e4f4!important;border:1px solid #233a56!important;border-radius:8px!important;padding:18px!important;line-height:1.9!important}
[data-testid="stDataFrame"]{border:1px solid #e4eaf2;border-radius:8px;overflow:hidden}
@media(max-width:1000px){.block-container{padding:3rem 1rem 1.5rem}.st-key-config_panel{padding:14px}}
</style>""", unsafe_allow_html=True)


def topbar(page):
    st.markdown(f'<div class="workspace-topbar"><div>工作空间 <strong>/　{escape(page)}</strong></div><span class="env-pill">本地环境 · 日频</span></div>', unsafe_allow_html=True)


def empty_results():
    columns = st.columns(4)
    for column, name in zip(columns, ["累计收益", "年化收益", "最大回撤", "夏普"]):
        column.metric(name, "—")
    with st.container(key="result_empty_panel"):
        st.markdown('<div class="panel-heading">绩效分析 <span>BACKTEST RESULT</span></div>', unsafe_allow_html=True)
        chart_tab, trades_tab, orders_tab, holdings_tab, config_tab = st.tabs(["净值", "成交", "订单", "持仓", "配置与口径"])
        with chart_tab:
            st.markdown('''<div class="chart-empty"><div><div class="empty-symbol"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 4v16h16M8 14l4-4 3 3 5-7"/></svg></div><div class="empty-title">尚未运行回测</div><div class="empty-detail">完成左侧配置后，运行回测查看净值与回撤。</div></div></div><div class="empty-details">成交笔数 <b>—</b>　 交易费用 <b>—</b>　 平均仓位 <b>—</b></div>''', unsafe_allow_html=True)
        for tab, message in [(trades_tab, "暂无成交"), (orders_tab, "暂无订单"), (holdings_tab, "暂无持仓"), (config_tab, "运行后保存完整配置与数据版本")]:
            with tab:
                st.caption(message)
