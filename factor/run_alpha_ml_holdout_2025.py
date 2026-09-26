"""Run the one-time frozen 2025 holdout evaluation.

The model specification and hyperparameters come exclusively from the
2015-2024 development experiment.  Calendar-year 2025 returns are used only
for this final evaluation and never for model or ensemble selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIR = PROJECT_ROOT / "data" / "private" / "rqdata"
RESULT_ROOT = PROJECT_ROOT / "factor" / "results" / "alpha_ml"
RESULT_DIR = RESULT_ROOT / "holdout_2025"
DEFAULT_LICENSE_SCRIPT = Path.home() / "Downloads" / "20250911081305_RQData" / "make.sh"
DEV_MONTHLY = PRIVATE_DIR / "ff3_combo_monthly_raw.parquet"
DEV_FEATURES = PRIVATE_DIR / "alpha158_monthly_features.parquet"
HOLDOUT_MONTHLY = PRIVATE_DIR / "ff3_combo_2025_holdout_raw.parquet"
HOLDOUT_DAILY = PRIVATE_DIR / "alpha_ml_daily_holdout_2025.parquet"
HOLDOUT_FEATURES = PRIVATE_DIR / "alpha158_holdout_2025_features.parquet"

FACTOR_FIELDS = [
    "market_cap",
    "book_to_market_ratio_lf",
    "return_on_equity_ttm",
    "RSI10",
]
ALL_START = "2024-12-01"
ALL_END = "2025-12-31"
FEATURE_START = "2024-09-01"
FEATURE_END = "2025-11-30"
RIDGE4_ALPHA = 1.0
RIDGE_ALL_ALPHA = 100.0
LIGHTGBM_ROUNDS = 59
MLP_STEPS = 110
RANDOM_SEED = 42

os.environ.setdefault("MPLCONFIGDIR", str(PRIVATE_DIR / ".matplotlib"))

from alpha_ml import (  # noqa: E402
    ALPHA158_FEATURES,
    COURSE_FEATURES,
    FUNDAMENTAL_FEATURES,
    adjusted_vwap,
    calculate_monthly_features,
    robust_scale,
)
from check_rqdata import load_license_uri  # noqa: E402
from run_alpha_ml import evaluate_scores, prepare_model_panel  # noqa: E402
from research_policy import (  # noqa: E402
    require_final_feature_window,
    require_final_return_window,
    require_final_signal_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2025冻结参数最终样本外检验")
    parser.add_argument("--license-script", type=Path, default=DEFAULT_LICENSE_SCRIPT)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    return parser.parse_args()


def call_with_retry(function, *args, **kwargs):
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                break
            print(f"数据请求失败，第{attempt}次重试：{type(exc).__name__}", flush=True)
            time.sleep(attempt * 2)
    assert last_error is not None
    raise last_error


def trading_dates_by_month(rqdatac) -> dict[pd.Period, list[pd.Timestamp]]:
    require_final_return_window("2025-01-01", ALL_END)
    dates = call_with_retry(rqdatac.get_trading_dates, ALL_START, ALL_END)
    grouped: dict[pd.Period, list[pd.Timestamp]] = {}
    for value in dates:
        timestamp = pd.Timestamp(value)
        grouped.setdefault(timestamp.to_period("M"), []).append(timestamp)
    return grouped


def download_holdout_monthly(rqdatac, refresh: bool) -> pd.DataFrame:
    if HOLDOUT_MONTHLY.exists() and not refresh:
        print(f"读取2025收益缓存：{HOLDOUT_MONTHLY}", flush=True)
        return pd.read_parquet(HOLDOUT_MONTHLY)

    grouped = trading_dates_by_month(rqdatac)
    months = pd.period_range("2024-12", "2025-12", freq="M")
    frames: list[pd.DataFrame] = []
    for number, signal_month in enumerate(months[:-1], start=1):
        return_month = months[number]
        signal_date = grouped[signal_month][-1]
        entry_date = grouped[return_month][0]
        exit_date = grouped[return_month][-1]
        require_final_signal_window(signal_date.date(), signal_date.date())
        require_final_return_window(entry_date.date(), exit_date.date())

        csi300 = call_with_retry(rqdatac.index_components, "000300.XSHG", date=signal_date)
        csi500 = call_with_retry(rqdatac.index_components, "000905.XSHG", date=signal_date)
        symbols = sorted(set(csi300) | set(csi500))
        factors = call_with_retry(
            rqdatac.get_factor,
            symbols,
            FACTOR_FIELDS,
            start_date=signal_date,
            end_date=signal_date,
        )
        prices = call_with_retry(
            rqdatac.get_price,
            symbols,
            start_date=entry_date,
            end_date=exit_date,
            frequency="1d",
            fields=["open", "close", "volume"],
            adjust_type="pre",
            skip_suspended=False,
        )
        factors = factors.reset_index().rename(columns={"order_book_id": "symbol"})
        prices = prices.reset_index().rename(columns={"order_book_id": "symbol"})
        prices["date"] = pd.to_datetime(prices["date"])
        entry = prices.loc[
            prices["date"].eq(entry_date), ["symbol", "open", "volume"]
        ].rename(columns={"open": "entry_open", "volume": "entry_volume"})
        exit_frame = prices.loc[
            prices["date"].eq(exit_date), ["symbol", "close", "volume"]
        ].rename(columns={"close": "exit_close", "volume": "exit_volume"})
        panel = factors.merge(entry, on="symbol", how="inner").merge(
            exit_frame, on="symbol", how="inner"
        )
        panel = panel[
            (panel["entry_open"] > 0)
            & (panel["exit_close"] > 0)
            & (panel["entry_volume"] > 0)
            & (panel["exit_volume"] > 0)
        ].copy()
        panel["signal_date"] = signal_date
        panel["return_date"] = exit_date
        panel["forward_return"] = panel["exit_close"] / panel["entry_open"] - 1
        frames.append(panel)
        print(
            f"[{number:02d}/12] {signal_date.date()} -> {exit_date.date()}：{len(panel)}只",
            flush=True,
        )

    result = pd.concat(frames, ignore_index=True)
    HOLDOUT_MONTHLY.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(HOLDOUT_MONTHLY, index=False)
    return result


def download_holdout_daily(rqdatac, monthly: pd.DataFrame, refresh: bool) -> pd.DataFrame:
    if HOLDOUT_DAILY.exists() and not refresh:
        print(f"读取2025特征行情缓存：{HOLDOUT_DAILY}", flush=True)
        return pd.read_parquet(HOLDOUT_DAILY)

    require_final_feature_window(FEATURE_START, FEATURE_END)
    symbols = sorted(monthly["symbol"].drop_duplicates().tolist())
    actual_end = pd.to_datetime(monthly["signal_date"]).max().strftime("%Y-%m-%d")
    print(f"下载最终特征窗口：{len(symbols)}只，{FEATURE_START}至{actual_end}", flush=True)
    prices = call_with_retry(
        rqdatac.get_price,
        symbols,
        start_date=FEATURE_START,
        end_date=actual_end,
        frequency="1d",
        fields=["open", "high", "low", "close", "volume", "total_turnover"],
        adjust_type="pre",
        skip_suspended=False,
    )
    raw = call_with_retry(
        rqdatac.get_price,
        symbols,
        start_date=FEATURE_START,
        end_date=actual_end,
        frequency="1d",
        fields=["close", "volume", "total_turnover"],
        adjust_type="none",
        skip_suspended=False,
    )
    prices = prices.reset_index().rename(columns={"order_book_id": "symbol", "date": "datetime"})
    raw = raw.reset_index().rename(
        columns={
            "order_book_id": "symbol",
            "date": "datetime",
            "close": "raw_close",
            "volume": "raw_volume",
            "total_turnover": "raw_total_turnover",
        }
    )
    prices = prices.merge(
        raw[["symbol", "datetime", "raw_close", "raw_volume", "raw_total_turnover"]],
        on=["symbol", "datetime"],
        how="left",
        validate="one_to_one",
    )
    prices["datetime"] = pd.to_datetime(prices["datetime"])
    prices["vwap"] = adjusted_vwap(
        prices["close"],
        prices["raw_close"],
        prices["raw_volume"],
        prices["raw_total_turnover"],
    )
    result = prices[
        [
            "datetime", "symbol", "open", "high", "low", "close", "volume",
            "total_turnover", "vwap",
        ]
    ].sort_values(["symbol", "datetime"])
    result.to_parquet(HOLDOUT_DAILY, index=False)
    return result


def build_holdout_features(rqdatac, monthly: pd.DataFrame, refresh: bool) -> pd.DataFrame:
    if HOLDOUT_FEATURES.exists() and not refresh:
        print(f"读取2025因子缓存：{HOLDOUT_FEATURES}", flush=True)
        return pd.read_parquet(HOLDOUT_FEATURES)
    daily = download_holdout_daily(rqdatac, monthly, refresh)
    members = monthly[["symbol", "signal_date"]]
    started = time.time()
    result = calculate_monthly_features(daily, members)
    result.to_parquet(HOLDOUT_FEATURES, index=False)
    print(f"2025特征完成：{len(result):,}行，耗时{time.time()-started:.1f}秒", flush=True)
    return result


def fit_frozen_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    all_features: list[str],
) -> tuple[pd.DataFrame, str]:
    ridge4 = Ridge(alpha=RIDGE4_ALPHA, fit_intercept=False)
    ridge4.fit(train[list(FUNDAMENTAL_FEATURES)], train["label"])
    ridge_all = Ridge(alpha=RIDGE_ALL_ALPHA, fit_intercept=False)
    ridge_all.fit(train[all_features], train["label"])

    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "feature_fraction_seed": RANDOM_SEED,
        "bagging_seed": RANDOM_SEED,
    }
    train_set = lgb.Dataset(train[all_features], label=train["label"])
    tree = lgb.train(params, train_set, num_boost_round=LIGHTGBM_ROUNDS)

    import torch
    from torch import nn

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"冻结MLP训练设备：{device}；固定步数：{MLP_STEPS}", flush=True)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    mlp = nn.Sequential(
        nn.Linear(len(all_features), 64),
        nn.SiLU(),
        nn.Dropout(0.05),
        nn.Linear(64, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=0.001, weight_decay=0.001)
    loss_function = nn.MSELoss()
    x_train = torch.as_tensor(train[all_features].to_numpy(dtype=np.float32), device=device)
    y_train = torch.as_tensor(train["label"].to_numpy(dtype=np.float32), device=device)
    batch_size = min(4096, len(train))
    for step in range(1, MLP_STEPS + 1):
        mlp.train()
        indices = torch.randint(0, len(train), (batch_size,), device=device)
        loss = loss_function(mlp(x_train[indices]).reshape(-1), y_train[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 20 == 0 or step == MLP_STEPS:
            print(f"MLP {step:03d}/{MLP_STEPS}，训练损失={loss.item():.6f}", flush=True)

    output = test[["symbol", "signal_date", "return_date", "forward_return"]].copy()
    output["ridge4"] = ridge4.predict(test[list(FUNDAMENTAL_FEATURES)])
    output["ridge_all"] = ridge_all.predict(test[all_features])
    output["lightgbm"] = tree.predict(test[all_features])
    x_test = torch.as_tensor(test[all_features].to_numpy(dtype=np.float32), device=device)
    mlp.eval()
    with torch.no_grad():
        output["mlp"] = mlp(x_test).reshape(-1).detach().cpu().numpy()
    ranks = [
        output.groupby("signal_date")[name].rank(method="average", pct=True)
        for name in ("ridge_all", "lightgbm", "mlp")
    ]
    output["ensemble_equal"] = sum(ranks) / 3
    return output, device


def pct(value: float) -> str:
    return f"{value:.2%}"


def metric_table(metrics: pd.DataFrame, names: list[str]) -> str:
    labels = {
        "grid_reference": "开发期四因子网格参考",
        "pv_anti_bare": "量价反共振（裸）",
        "pv_anti_pair": "量价反共振×换手",
        "pv_anti_triple": "量价反共振×换手×振幅",
        "gap_diff4_bare": "隔夜—日内分解",
        "ridge4": "Ridge四因子",
        "ridge_all": "Ridge全特征",
        "lightgbm": "LightGBM全特征",
        "mlp": "MLP全特征",
        "ensemble_equal": "三模型等权秩集成",
    }
    indexed = metrics.set_index("model")
    rows = []
    for name in names:
        row = indexed.loc[name]
        rows.append(
            f"| {labels[name]} | {row.mean_rank_ic:.4f} | {row.annualized_icir:.3f} | "
            f"{row.positive_rate:.1%} | {pct(row.q5_net_annual_return)} | "
            f"{pct(row.ls_net_annual_return)} | {row.ls_net_sharpe:.3f} | "
            f"{pct(row.ls_net_max_drawdown)} |"
        )
    return "\n".join(rows)


def write_final_report(
    dev_panel: pd.DataFrame,
    holdout_panel: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
    monthly_tables: dict[str, pd.DataFrame],
    device: str,
    cost_bps: float,
) -> Path:
    from scipy import stats

    dev_metrics = pd.read_csv(RESULT_ROOT / "model_metrics.csv")
    selected = ["ridge_all", "lightgbm", "mlp", "ensemble_equal"]
    comparison_rows = []
    dev_indexed = dev_metrics.set_index("model")
    out_indexed = holdout_metrics.set_index("model")
    for name, label in (
        ("ridge_all", "Ridge全特征"),
        ("lightgbm", "LightGBM全特征"),
        ("mlp", "MLP全特征"),
        ("ensemble_equal", "三模型等权秩集成"),
    ):
        dev = dev_indexed.loc[name]
        out = out_indexed.loc[name]
        comparison_rows.append(
            f"| {label} | {dev.mean_rank_ic:.4f} | {out.mean_rank_ic:.4f} | "
            f"{pct(dev.ls_net_annual_return)} | {pct(out.ls_net_annual_return)} | "
            f"{dev.ls_net_sharpe:.3f} | {out.ls_net_sharpe:.3f} |"
        )
    comparison = "\n".join(comparison_rows)

    ensemble = monthly_tables["ensemble_equal"].copy()
    predictions = pd.read_csv(RESULT_DIR / "holdout_predictions.csv", parse_dates=["signal_date"])
    monthly_ic = predictions.groupby("signal_date").apply(
        lambda group: group["ensemble_equal"].corr(group["forward_return"], method="spearman"),
        include_groups=False,
    )
    ic_test = stats.ttest_1samp(monthly_ic, 0)
    ic_interval = stats.t.interval(
        0.95,
        len(monthly_ic) - 1,
        loc=float(monthly_ic.mean()),
        scale=float(stats.sem(monthly_ic)),
    )
    benchmark_annual = float(
        (1 + ensemble["benchmark"]).prod() ** (12 / len(ensemble)) - 1
    )
    monthly_lines = []
    for row in ensemble.itertuples():
        signal = pd.Timestamp(row.signal_date)
        monthly_lines.append(
            f"| {pd.Timestamp(row.return_date):%Y-%m} | {monthly_ic.loc[signal]:.4f} | "
            f"{pct(row.q5_net)} | {pct(row.long_short_net)} | "
            f"{int(row.stock_count)} |"
        )
    monthly_table = "\n".join(monthly_lines)

    final_row = out_indexed.loc["ensemble_equal"]
    if final_row.mean_rank_ic > 0 and final_row.ls_net_annual_return > 0:
        conclusion = (
            "2025年独立检验中，组合因子的平均Rank IC和扣费多空收益均为正，"
            "方向上支持开发期发现；但只有12个月，统计精度仍有限。"
        )
    elif final_row.mean_rank_ic > 0:
        conclusion = (
            "2025年平均Rank IC点估计仍为正，但统计上不能与零区分，扣费多空收益"
            "也未保持为正。这个结果不能支持因子具有稳定的样本外预测力，亦不能宣称"
            "策略通过经济检验。"
        )
    else:
        conclusion = (
            "2025年平均Rank IC未保持为正，冻结模型没有在独立样本中复现开发期优势；"
            "应如实视为样本外失效证据，不允许回看2025后再改参数并继续称其为同一次检验。"
        )

    report = f"""# 最终因子挖掘报告：Alpha158与机器学习复合因子

**课程：**《当代量化交易系统的原理与实现》  
**作者：**张欢（2401220063）  
**院系年级：**心理与认知科学学院，2024级硕士  
**最终检验执行日：**2026-09-26

## 摘要

本研究在每月末的历史沪深300与中证500成分股并集上，组合vn.py Alpha158价量特征、价值/规模/ROE/RSI特征及课程公式因子，比较Ridge、LightGBM和适配MacBook Air M3的PyTorch MLP。2015—2024年仅用于开发；在模型、超参数和集成方式全部冻结后，才一次性读取2025年收益完成最终样本外检验。

**最终因子定义：**对每个信号月的Ridge全特征、LightGBM和MLP预测分别转为截面百分位秩，再等权平均：

```text
Score(i,t) = [RankPct(Ridge(i,t)) + RankPct(LightGBM(i,t)) + RankPct(MLP(i,t))] / 3
```

{conclusion}

2025年组合因子平均Rank IC为 **{final_row.mean_rank_ic:.4f}**，年化ICIR为 **{final_row.annualized_icir:.3f}**，IC为正月份占 **{final_row.positive_rate:.1%}**。按12个月做的描述性单样本t检验为 *t*(11)={ic_test.statistic:.3f}、*p*={ic_test.pvalue:.3f}，均值95%置信区间为[{ic_interval[0]:.4f}, {ic_interval[1]:.4f}]。最高组扣除换手成本后的年化收益为 **{pct(final_row.q5_net_annual_return)}**，但同期股票池等权基准为 **{pct(benchmark_annual)}**；多空扣费年化收益为 **{pct(final_row.ls_net_annual_return)}**，多空Sharpe为 **{final_row.ls_net_sharpe:.3f}**，最大回撤为 **{pct(final_row.ls_net_max_drawdown)}**。

## 一、冻结方案与防止样本泄漏

| 项目 | 冻结设置 |
|---|---|
| 开发数据 | 2015-01至2024-12的可实现收益；开发面板{len(dev_panel):,}个股票—月份 |
| 最终检验收益 | 2025-01至2025-12，共12个月 |
| 最终信号 | 2024-12至2025-11月末信号 |
| 股票池 | 各信号月末当时的沪深300与中证500成分股并集 |
| 特征 | 158项Alpha158 + 4项基本面/技术面 + 4项课程公式，共166项 |
| Ridge | 四因子α={RIDGE4_ALPHA:g}；全特征α={RIDGE_ALL_ALPHA:g} |
| LightGBM | 固定{LIGHTGBM_ROUNDS}轮，其余参数沿用开发期 |
| MLP | 166→64→1，SiLU，Dropout 5%，AdamW，固定{MLP_STEPS}步 |
| 集成 | 三模型月度截面百分位秩等权，不搜索权重 |
| 交易成本 | 单边{cost_bps:.1f} bps，按月度持仓换手计提 |
| 运行设备 | {device.upper()} |

预处理的中位数和MAD只由开发集拟合，然后原样应用于2025。2025真实收益没有用于特征筛选、正则选择、树轮数、MLP训练步数、模型选择或集成权重选择。模型最终在全部开发数据上重拟合一次，随后对2025做一次性预测。由于训练标签使用下一月收益，开发集最后一个信号月为2024年11月；2024年12月信号是2025年1月收益的第一个最终检验点。

## 二、数据与因子构造

- 数据源：米筐RQData。
- 最终检验面板：{holdout_panel['signal_date'].nunique()}个月、{len(holdout_panel):,}个股票—月份。
- 收益定义：下一自然月首个交易日开盘买入，月末收盘卖出。
- Alpha158只使用信号日及此前日频行情；最长回看60个交易日。
- VWAP先用未复权成交额/成交量计算，再映射到前复权价格尺度。
- 财务/风格输入为价值、规模、ROE和RSI；每月截面先缩尾、标准化。
- 预测标签为下一月收益的月度截面秩；标签仅用于训练或事后评价，不进入特征。

## 三、开发期结果（2020—2024滚动开发检验）

| 方法 | 平均Rank IC | 年化ICIR | IC为正月份 | 最高组扣费年化 | 多空扣费年化 | 多空Sharpe | 多空最大回撤 |
|---|---:|---:|---:|---:|---:|---:|---:|
{metric_table(dev_metrics, selected)}

开发期结果是模型选择证据，不是最终样本外证据。最终选择三模型等权秩集成，是因为其开发期平均Rank IC与扣费多空Sharpe在候选中领先，同时不再额外搜索集成权重。

## 四、2025年最终样本外结果

| 方法 | 平均Rank IC | 年化ICIR | IC为正月份 | 最高组扣费年化 | 多空扣费年化 | 多空Sharpe | 多空最大回撤 |
|---|---:|---:|---:|---:|---:|---:|---:|
{metric_table(holdout_metrics, selected)}

### 开发期与最终检验对照

| 方法 | 开发期Rank IC | 2025 Rank IC | 开发期多空年化 | 2025多空年化 | 开发期多空Sharpe | 2025多空Sharpe |
|---|---:|---:|---:|---:|---:|---:|
{comparison}

### 三模型等权秩集成逐月表现

| 收益月份 | Rank IC | 最高组扣费收益 | 多空扣费收益 | 股票数 |
|---|---:|---:|---:|---:|
{monthly_table}

## 五、结论与解释边界

1. 本报告把2025定义为真正的最终样本外区间，所有参数在查看2025结果前已固定；因此无论结果好坏，都不进行事后调参。
2. 2025平均Rank IC虽为正，但置信区间宽且包含零，正负月份各半；这是弱且不确定的点估计，不能表述为“已经验证有效”。
3. 最高组净收益{pct(final_row.q5_net_annual_return)}低于股票池等权基准{pct(benchmark_annual)}，同时多空组合亏损{pct(-final_row.ls_net_annual_return)}；最终经济检验没有通过。
4. Rank IC检验的是股票截面排序能力；最高组或多空收益还受收益分布、组合构造、换手与交易成本影响，两者不应混为一谈。
5. 2025只有12个独立月度截面，ICIR、t检验和Sharpe都对少数月份敏感，不能据此推断长期稳定性，也不能等同于实盘可交易性。
6. 回测计入了按持仓变化估算的单边成本，但未完整模拟涨跌停、冲击成本、容量、印花税差异及无法成交情形。
7. 历史指数成分降低了幸存者偏差；财务字段的历史时点可用性仍依赖RQData供应商口径。
8. MLP在M3设备上使用小网络是为控制样本容量相对有限时的过拟合，而非追求更深模型；深度学习并不天然意味着更可靠的因子。

## 六、可复现文件

- `holdout_2025/holdout_settings.csv`：冻结参数和样本边界。
- `holdout_2025/holdout_metrics.csv`：2025最终评价汇总。
- `holdout_2025/holdout_predictions.csv`：股票—月份预测与真实收益。
- `holdout_2025/monthly_*.csv`：各方法月度组合表现。
- `model_metrics.csv`、`fold_settings.csv`：开发期结果与当时确定的超参数。
- 完整授权行情与股票级特征保存在`data/private/rqdata/`，不应提交到公开GitHub。

本报告仅用于课程研究与方法展示，不构成投资建议。
"""
    path = RESULT_ROOT / "final_factor_mining_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    require_final_signal_window("2024-12-01", "2025-11-30")
    require_final_return_window("2025-01-01", "2025-12-31")
    require_final_feature_window(FEATURE_START, FEATURE_END)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["RQDATAC2_CONF"] = load_license_uri(args.license_script)
    import rqdatac

    try:
        rqdatac.init()
        holdout_monthly = download_holdout_monthly(rqdatac, args.refresh)
        holdout_features = build_holdout_features(rqdatac, holdout_monthly, args.refresh)
    finally:
        try:
            rqdatac.reset()
        finally:
            os.environ.pop("RQDATAC2_CONF", None)

    dev_monthly = pd.read_parquet(DEV_MONTHLY)
    dev_features = pd.read_parquet(DEV_FEATURES)
    dev_panel = prepare_model_panel(dev_monthly, dev_features)
    holdout_panel = prepare_model_panel(holdout_monthly, holdout_features)
    all_features = [*FUNDAMENTAL_FEATURES, *ALPHA158_FEATURES, *COURSE_FEATURES]
    dev_scaled, holdout_scaled = robust_scale(
        dev_panel, holdout_panel, columns=all_features
    )
    print(
        f"冻结训练：{len(dev_scaled):,}行；最终检验：{len(holdout_scaled):,}行、"
        f"{holdout_scaled['signal_date'].nunique()}个月",
        flush=True,
    )
    predictions, device = fit_frozen_models(dev_scaled, holdout_scaled, all_features)
    predictions.to_csv(RESULT_DIR / "holdout_predictions.csv", index=False)
    metrics, monthly_tables = evaluate_scores(predictions, holdout_panel, args.cost_bps)
    metrics.to_csv(RESULT_DIR / "holdout_metrics.csv", index=False)
    for name, table in monthly_tables.items():
        table.to_csv(RESULT_DIR / f"monthly_{name}.csv", index=False)
    settings = pd.DataFrame(
        [
            {
                "development_start": "2015-01",
                "development_last_signal": "2024-11",
                "holdout_return_start": "2025-01",
                "holdout_return_end": "2025-12",
                "ridge4_alpha": RIDGE4_ALPHA,
                "ridge_all_alpha": RIDGE_ALL_ALPHA,
                "lightgbm_rounds": LIGHTGBM_ROUNDS,
                "mlp_steps": MLP_STEPS,
                "ensemble": "equal monthly percentile ranks",
                "cost_bps_one_way": args.cost_bps,
                "device": device,
                "random_seed": RANDOM_SEED,
            }
        ]
    )
    settings.to_csv(RESULT_DIR / "holdout_settings.csv", index=False)
    report = write_final_report(
        dev_panel,
        holdout_panel,
        metrics,
        monthly_tables,
        device,
        args.cost_bps,
    )
    print(metrics.to_string(index=False), flush=True)
    print(f"最终Markdown报告：{report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
