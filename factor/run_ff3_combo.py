"""Download RQData inputs, mine the factor combination, and write the report."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import os
import time

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIR = PROJECT_ROOT / "data" / "private" / "rqdata"
RESULT_DIR = PROJECT_ROOT / "factor" / "results"
DEFAULT_LICENSE_SCRIPT = (
    Path.home() / "Downloads" / "20250911081305_RQData" / "make.sh"
)

os.environ.setdefault("MPLCONFIGDIR", str(PRIVATE_DIR / ".matplotlib"))

from check_rqdata import load_license_uri  # noqa: E402
from ff3_combo import (  # noqa: E402
    FEATURES,
    add_score,
    calculate_rank_ic,
    evaluate_weights,
    mine_weight_grid,
    orient_features,
    performance_statistics,
    prepare_features,
)
from research_policy import require_discovery_window  # noqa: E402


START = "2015-01-01"
END = "2024-12-31"
FACTOR_FIELDS = [
    "market_cap",
    "book_to_market_ratio_lf",
    "return_on_equity_ttm",
    "RSI10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQData规模—价值复合因子挖掘")
    parser.add_argument("--license-script", type=Path, default=DEFAULT_LICENSE_SCRIPT)
    parser.add_argument("--refresh", action="store_true", help="忽略本地缓存并重新下载")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="单边交易成本bps")
    return parser.parse_args()


def call_with_retry(function: Callable, *args, **kwargs):
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


def _month_trading_dates(rqdatac) -> dict[pd.Period, list[pd.Timestamp]]:
    dates = call_with_retry(rqdatac.get_trading_dates, START, END)
    grouped: dict[pd.Period, list[pd.Timestamp]] = {}
    for value in dates:
        timestamp = pd.Timestamp(value)
        grouped.setdefault(timestamp.to_period("M"), []).append(timestamp)
    return grouped


def download_monthly_panel(rqdatac, cache_path: Path) -> pd.DataFrame:
    """Download only 2015-2024 dynamic-universe monthly observations."""
    require_discovery_window(START, END)
    month_dates = _month_trading_dates(rqdatac)
    months = sorted(month_dates)
    frames: list[pd.DataFrame] = []

    for index, signal_month in enumerate(months[:-1], start=1):
        return_month = months[index]
        signal_date = month_dates[signal_month][-1]
        entry_date = month_dates[return_month][0]
        exit_date = month_dates[return_month][-1]

        csi300 = call_with_retry(
            rqdatac.index_components, "000300.XSHG", date=signal_date
        )
        csi500 = call_with_retry(
            rqdatac.index_components, "000905.XSHG", date=signal_date
        )
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
            f"[{index:03d}/{len(months)-1}] {signal_date.date()} -> "
            f"{return_month}: {len(panel)}只",
            flush=True,
        )
        if index % 12 == 0:
            pd.concat(frames, ignore_index=True).to_parquet(
                cache_path.with_suffix(".partial.parquet"), index=False
            )

    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(cache_path, index=False)
    return result


def percent(value: float) -> str:
    return f"{value:.2%}"


def number(value: float) -> str:
    return f"{value:.3f}"


def create_plots(monthly: pd.DataFrame, ic: pd.Series, result_dir: Path) -> None:
    import matplotlib.pyplot as plt

    result_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    figure, axis = plt.subplots(figsize=(10, 5.5))
    dates = pd.to_datetime(monthly["return_date"])
    initial_date = pd.Timestamp(monthly["signal_date"].iloc[0])
    for column, label in (
        ("q5_net", "Top quintile net"),
        ("benchmark", "CSI800 universe equal-weight"),
        ("long_short_net", "Top - bottom net"),
    ):
        wealth = pd.Series((1 + monthly[column]).cumprod().to_numpy(), index=dates)
        wealth = pd.concat([pd.Series([1.0], index=[initial_date]), wealth])
        axis.plot(wealth.index, wealth, label=label)
    axis.set_title("In-sample cumulative wealth (2015-2024)")
    axis.set_ylabel("Wealth index")
    axis.set_xlim(initial_date, dates.iloc[-1])
    axis.legend()
    figure.tight_layout()
    figure.savefig(result_dir / "ff3_combo_cumulative.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(ic.index, ic.rolling(12, min_periods=6).mean(), color="#b22222")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("12-month rolling mean Rank IC")
    axis.set_ylabel("Rank IC")
    figure.tight_layout()
    figure.savefig(result_dir / "ff3_combo_ic.png", dpi=160)
    plt.close(figure)


def write_report(
    panel: pd.DataFrame,
    normalized: pd.DataFrame,
    directions: dict[str, int],
    direction_table: pd.DataFrame,
    candidates: pd.DataFrame,
    best_weights: dict[str, float],
    ic: pd.Series,
    monthly: pd.DataFrame,
    cost_bps: float,
    result_dir: Path,
) -> None:
    import statsmodels.api as sm
    from scipy import stats

    q5 = performance_statistics(monthly["q5_net"])
    ls = performance_statistics(monthly["long_short_net"])
    benchmark = performance_statistics(monthly["benchmark"])
    best_row = candidates.iloc[0]
    ttest = stats.ttest_1samp(ic, 0)
    hac = sm.OLS(ic.to_numpy(), np.ones(len(ic))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )
    hac_t = float(hac.tvalues[0])
    hac_p = float(hac.pvalues[0])
    weight_adjusted_p = min(1.0, hac_p * len(candidates))
    positive_ic_rate = float((ic > 0).mean())

    quantile_rows: list[str] = []
    for quantile in range(1, 6):
        column = f"q{quantile}"
        statistics = performance_statistics(monthly[column])
        quantile_rows.append(
            f"| Q{quantile} | {percent(statistics['annual_return'])} | "
            f"{percent(statistics['annual_volatility'])} | "
            f"{percent(statistics['max_drawdown'])} |"
        )
    quantile_lines = "\n".join(quantile_rows)

    yearly_ic = (
        ic.rename("rank_ic")
        .to_frame()
        .assign(year=lambda data: data.index.year)
        .groupby("year")["rank_ic"]
        .agg(["mean", "count"])
        .reset_index()
    )
    yearly_lines = "\n".join(
        f"| {int(row.year)} | {row['mean']:.4f} | {int(row['count'])} |"
        for _, row in yearly_ic.iterrows()
    )

    direction_names = {1: "正向", -1: "反向"}
    direction_lines = "\n".join(
        f"| {row.feature} | {row.raw_mean_ic:.4f} | {direction_names[int(row.direction)]} |"
        for row in direction_table.itertuples()
    )
    weight_lines = "\n".join(
        f"| {feature} | {best_weights[feature]:.2f} | {direction_names[directions[feature]]} |"
        for feature in FEATURES
    )
    top_candidates = candidates.head(10)
    candidate_lines = "\n".join(
        "| "
        + " | ".join(
            [
                f"{row.weight_value:.2f}",
                f"{row.weight_size:.2f}",
                f"{row.weight_quality:.2f}",
                f"{row.weight_technical:.2f}",
                f"{row.mean_rank_ic:.4f}",
                f"{row.annualized_icir:.3f}",
                percent(row.q5_net_annual_return),
                percent(row.ls_net_annual_return),
            ]
        )
        + " |"
        for row in top_candidates.itertuples()
    )

    report = f"""# Fama–French思路的规模—价值复合因子挖掘报告

## 摘要

本研究在历史沪深300与中证500成分股并集上，组合价值、规模、盈利质量与技术面信号。研究及调参区间为2015—2024年；2025年数据没有下载或查看。最终组合依据月度Rank ICIR从35组预先限定的权重中选择，因此所有结果均为样本内探索性证据，不构成样本外有效性结论。

样本共包含 **{normalized['signal_date'].nunique()}** 个信号月份、**{len(normalized):,}** 个股票—月份观测；每月清洗后平均 **{normalized.groupby('signal_date').size().mean():.1f}** 只股票。最优复合因子平均月度Rank IC为 **{best_row.mean_rank_ic:.4f}**，年化ICIR为 **{best_row.annualized_icir:.3f}**。Rank IC为正的月份占 **{positive_ic_rate:.1%}**。

## 数据和防泄漏

- 数据源：米筐RQData；历史成分股、估值财务指标、技术指标和前复权行情。
- 股票池：每个信号月末当时的沪深300和中证500成分股并集，不使用当前成分股回填。
- 信号日：月末最后交易日；收益期：下一月首个交易日开盘至下一月最后交易日收盘。
- 停牌处理：进入或退出日成交量为0的股票剔除。
- 绩效区间：{pd.to_datetime(monthly['return_date']).min().date()}至{pd.to_datetime(monthly['return_date']).max().date()}。
- 2025年保持密封；下载器通过日期策略拒绝触及2025年。

## 因子定义

- 价值：`log(book_to_market_ratio_lf)`，值越大代表估值越低。
- 规模：`-log(market_cap)`，值越大代表公司越小。
- 质量：`return_on_equity_ttm`。
- 技术：`RSI10`。
- 每月先按1%和99%分位缩尾，再使用 `vnpy.alpha` 做截面Z标准化。
- 因子方向按2015—2024年样本内平均Rank IC确定；这是明确记录的样本内调参。

| 因子 | 原始平均Rank IC | 采用方向 |
|---|---:|---|
{direction_lines}

## 权重搜索

四项权重限定为非负、总和为1、步长0.25，共比较35组。选择目标是年化Rank ICIR，不直接以累计收益挑选。最终组合为：

| 因子 | 权重 | 方向 |
|---|---:|---|
{weight_lines}

排名前10的候选如下；完整表见 `ff3_combo_candidates.csv`。

| 价值 | 规模 | 质量 | 技术 | 平均IC | 年化ICIR | 多头年化收益 | 多空年化收益 |
|---:|---:|---:|---:|---:|---:|---:|---:|
{candidate_lines}

最优组合的Rank IC普通t检验为 **t={float(ttest.statistic):.3f}, p={float(ttest.pvalue):.4f}**；对月度序列使用3阶Newey–West/HAC标准误后为 **t={hac_t:.3f}, p={hac_p:.4f}**。仅针对35组权重搜索作Bonferroni上界校正后，p值为 **{weight_adjusted_p:.4f}**；该校正没有完全覆盖因子方向选择，因此仍应视为探索性证据。

逐年Rank IC如下。2021年为负，说明稳定性并非无条件成立。

| 年份 | 平均Rank IC | 月数 |
|---:|---:|---:|
{yearly_lines}

## 组合回测

每月按复合得分分为五组，等权持有一个月。交易成本按单边 **{cost_bps:.1f} bps**、依据实际组合换手率扣除；收益未包含印花税差异、冲击成本和涨跌停无法成交的进一步约束。

| 组合 | 年化收益 | 年化波动率 | Sharpe（无风险利率按0） | 最大回撤 |
|---|---:|---:|---:|---:|
| 最高分组（成本后） | {percent(q5['annual_return'])} | {percent(q5['annual_volatility'])} | {number(q5['sharpe'])} | {percent(q5['max_drawdown'])} |
| 最高组减最低组（成本后） | {percent(ls['annual_return'])} | {percent(ls['annual_volatility'])} | {number(ls['sharpe'])} | {percent(ls['max_drawdown'])} |
| 股票池等权基准（未扣成本） | {percent(benchmark['annual_return'])} | {percent(benchmark['annual_volatility'])} | {number(benchmark['sharpe'])} | {percent(benchmark['max_drawdown'])} |

五分组的未扣成本年化收益如下。Q4高于Q5，说明收益排序并非严格单调；不能仅凭平均IC把该组合描述为成熟交易策略。

| 分组 | 年化收益 | 年化波动率 | 最大回撤 |
|---|---:|---:|---:|
{quantile_lines}

![样本内累计净值](ff3_combo_cumulative.png)

![滚动Rank IC](ff3_combo_ic.png)

## 解释与局限

1. 本结果允许在2015—2024年内反复比较方向和权重，因此存在多重检验和样本内过拟合风险；不能把最优组合的收益当成未来保证。
2. 2025年尚未启封，报告不包含样本外验证。因子和成本假设冻结后，如课程需要，只能对2025年进行一次性评价。
3. 股票池限于历史中证800成分股，结论不能直接外推到微盘股或全部A股。
4. 当前成交模型已避开同日收盘成交并剔除首尾停牌，但尚未逐笔模拟涨跌停、冲击成本和容量。
5. `book_to_market_ratio_lf`、ROE和RSI由RQData提供；报告公开汇总指标，不提交原始授权数据或License。

综合判断：该复合因子在样本内呈现统计上的横截面排序信息，但最高分组收益较低、回撤较大、五分组不完全单调。它可以作为课程作业中的“已完成因子挖掘候选”，但不应描述为已经达到实盘可用标准。

## 可复现文件

- `ff3_combo_candidates.csv`：全部35组权重及指标。
- `ff3_combo_monthly_returns.csv`：最优组合月度汇总收益和换手率。
- `ff3_combo_rank_ic.csv`：最优组合月度Rank IC。
- 原始股票级面板位于 `data/private/rqdata/`，受许可约束且被Git忽略。
"""
    (result_dir / "ff3_combo_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    require_discovery_window(START, END)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = PRIVATE_DIR / "ff3_combo_monthly_raw.parquet"

    license_uri = load_license_uri(args.license_script)
    os.environ["RQDATAC2_CONF"] = license_uri

    import rqdatac

    try:
        rqdatac.init()
        if cache_path.exists() and not args.refresh:
            print(f"读取本地缓存：{cache_path}", flush=True)
            panel = pd.read_parquet(cache_path)
        else:
            panel = download_monthly_panel(rqdatac, cache_path)
    finally:
        try:
            rqdatac.reset()
        finally:
            os.environ.pop("RQDATAC2_CONF", None)

    print("开始清洗和截面标准化", flush=True)
    normalized = prepare_features(panel)
    oriented, directions, direction_table = orient_features(normalized)
    print("开始搜索35组固定网格权重", flush=True)
    candidates = mine_weight_grid(oriented, cost_bps=args.cost_bps)
    combinations = candidates[candidates["active_factors"] >= 2].copy()
    best = combinations.iloc[0]
    best_weights = {feature: float(best[f"weight_{feature}"]) for feature in FEATURES}
    _, ic, monthly = evaluate_weights(
        oriented, best_weights, cost_bps=args.cost_bps
    )

    candidates.to_csv(RESULT_DIR / "ff3_combo_candidates.csv", index=False)
    monthly.to_csv(RESULT_DIR / "ff3_combo_monthly_returns.csv", index=False)
    ic.rename("rank_ic").to_csv(RESULT_DIR / "ff3_combo_rank_ic.csv", index_label="signal_date")
    direction_table.to_csv(RESULT_DIR / "ff3_combo_directions.csv", index=False)
    add_score(oriented, best_weights).to_parquet(
        PRIVATE_DIR / "ff3_combo_scored_private.parquet", index=False
    )

    create_plots(monthly, ic, RESULT_DIR)
    write_report(
        panel,
        normalized,
        directions,
        direction_table,
        candidates,
        best_weights,
        ic,
        monthly,
        args.cost_bps,
        RESULT_DIR,
    )

    q5 = performance_statistics(monthly["q5_net"])
    ls = performance_statistics(monthly["long_short_net"])
    print("因子挖掘完成", flush=True)
    print("最优组合：" + ", ".join(f"{k}={v:.2f}" for k, v in best_weights.items()), flush=True)
    print(f"平均Rank IC：{ic.mean():.4f}", flush=True)
    print(f"多头年化收益（成本后）：{q5['annual_return']:.2%}", flush=True)
    print(f"多空年化收益（成本后）：{ls['annual_return']:.2%}", flush=True)
    print(f"报告：{RESULT_DIR / 'ff3_combo_report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
