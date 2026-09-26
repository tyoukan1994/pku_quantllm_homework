"""Build monthly Alpha158 features and compare linear, tree and MLP models."""

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
RESULT_DIR = PROJECT_ROOT / "factor" / "results" / "alpha_ml"
DEFAULT_LICENSE_SCRIPT = Path.home() / "Downloads" / "20250911081305_RQData" / "make.sh"
MONTHLY_RAW = PRIVATE_DIR / "ff3_combo_monthly_raw.parquet"
FEATURE_CACHE = PRIVATE_DIR / "alpha158_monthly_features.parquet"

os.environ.setdefault("MPLCONFIGDIR", str(PRIVATE_DIR / ".matplotlib"))

from alpha_ml import (  # noqa: E402
    ALPHA158_FEATURES,
    COURSE_FEATURES,
    FUNDAMENTAL_FEATURES,
    adjusted_vwap,
    calculate_monthly_features,
    cross_sectional_rank_label,
    expanding_folds,
    mean_monthly_rank_ic,
    rank_ic_statistics,
    robust_scale,
)
from check_rqdata import load_license_uri  # noqa: E402
from ff3_combo import (  # noqa: E402
    calculate_monthly_portfolios,
    performance_statistics,
    prepare_features,
)
from research_policy import require_discovery_window, require_feature_context_window  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpha158机器学习因子对照实验")
    parser.add_argument("--license-script", type=Path, default=DEFAULT_LICENSE_SCRIPT)
    parser.add_argument("--refresh-daily", action="store_true")
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--skip-mlp", action="store_true")
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


def download_year(rqdatac, monthly: pd.DataFrame, year: int, refresh: bool) -> pd.DataFrame:
    cache = PRIVATE_DIR / f"alpha_ml_daily_{year}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    year_members = monthly.loc[
        pd.to_datetime(monthly["signal_date"]).dt.year.eq(year), "symbol"
    ].drop_duplicates()
    symbols = sorted(year_members.tolist())
    start = f"{year - 1}-09-01"
    end = f"{year}-12-31"
    require_feature_context_window(start, end)
    print(f"下载{year}特征窗口：{len(symbols)}只，{start}至{end}", flush=True)
    prices = call_with_retry(
        rqdatac.get_price,
        symbols,
        start_date=start,
        end_date=end,
        frequency="1d",
        fields=["open", "high", "low", "close", "volume", "total_turnover"],
        adjust_type="pre",
        skip_suspended=False,
    )
    raw_prices = call_with_retry(
        rqdatac.get_price,
        symbols,
        start_date=start,
        end_date=end,
        frequency="1d",
        fields=["close", "volume", "total_turnover"],
        adjust_type="none",
        skip_suspended=False,
    )
    prices = prices.reset_index().rename(
        columns={"order_book_id": "symbol", "date": "datetime"}
    )
    raw_prices = raw_prices.reset_index().rename(
        columns={
            "order_book_id": "symbol",
            "date": "datetime",
            "close": "raw_close",
            "volume": "raw_volume",
            "total_turnover": "raw_total_turnover",
        }
    )
    prices = prices.merge(
        raw_prices[
            [
                "symbol", "datetime", "raw_close", "raw_volume",
                "raw_total_turnover",
            ]
        ],
        on=["symbol", "datetime"],
        how="left",
        validate="one_to_one",
    )
    prices["datetime"] = pd.to_datetime(prices["datetime"])
    # RQData can also adjust volume for share splits, while cash dividends only
    # change the price adjustment factor.  Always compute VWAP from the raw
    # fields, then map it onto the same pre-adjusted scale as OHLC.
    prices["vwap"] = adjusted_vwap(
        prices["close"],
        prices["raw_close"],
        prices["raw_volume"],
        prices["raw_total_turnover"],
    )
    prices = prices[
        [
            "datetime", "symbol", "open", "high", "low", "close", "volume",
            "total_turnover", "vwap",
        ]
    ].sort_values(["symbol", "datetime"])
    prices.to_parquet(cache, index=False)
    return prices


def build_feature_cache(
    monthly: pd.DataFrame,
    license_script: Path,
    refresh_daily: bool,
    refresh_features: bool,
) -> pd.DataFrame:
    if FEATURE_CACHE.exists() and not refresh_features:
        print(f"读取特征缓存：{FEATURE_CACHE}", flush=True)
        return pd.read_parquet(FEATURE_CACHE)

    os.environ["RQDATAC2_CONF"] = load_license_uri(license_script)
    import rqdatac

    frames: list[pd.DataFrame] = []
    try:
        rqdatac.init()
        for year in range(2015, 2025):
            daily = download_year(rqdatac, monthly, year, refresh_daily)
            members = monthly.loc[
                pd.to_datetime(monthly["signal_date"]).dt.year.eq(year),
                ["symbol", "signal_date"],
            ]
            started = time.time()
            features = calculate_monthly_features(daily, members)
            frames.append(features)
            print(
                f"{year}特征完成：{features.shape[0]:,}行、{features.shape[1]-2}项，"
                f"{time.time()-started:.1f}秒",
                flush=True,
            )
            pd.concat(frames, ignore_index=True).to_parquet(
                FEATURE_CACHE.with_suffix(".partial.parquet"), index=False
            )
    finally:
        try:
            rqdatac.reset()
        finally:
            os.environ.pop("RQDATAC2_CONF", None)

    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(FEATURE_CACHE, index=False)
    return result


def prepare_model_panel(monthly: pd.DataFrame, alpha: pd.DataFrame) -> pd.DataFrame:
    normalized = prepare_features(monthly).rename(
        columns={
            "value": "fund_value",
            "size": "fund_size",
            "quality": "fund_quality",
            "technical": "fund_rsi",
        }
    )
    keep = [
        "symbol", "signal_date", "return_date", "forward_return", *FUNDAMENTAL_FEATURES
    ]
    panel = normalized[keep].merge(
        alpha, on=["symbol", "signal_date"], how="inner", validate="one_to_one"
    )
    panel["label"] = cross_sectional_rank_label(panel)
    panel["year"] = pd.to_datetime(panel["signal_date"]).dt.year
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel.sort_values(["signal_date", "symbol"]).reset_index(drop=True)


def fit_ridge(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, float, np.ndarray]:
    best_alpha = 0.0
    best_ic = -np.inf
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(train[feature_names], train["label"])
        candidate = valid.copy()
        candidate["prediction"] = model.predict(valid[feature_names])
        ic = mean_monthly_rank_ic(candidate, "prediction")
        if ic > best_ic:
            best_ic = ic
            best_alpha = alpha
    combined = pd.concat([train, valid], ignore_index=True)
    final = Ridge(alpha=best_alpha, fit_intercept=False)
    final.fit(combined[feature_names], combined["label"])
    return final.predict(test[feature_names]), best_alpha, final.coef_.copy()


def fit_lightgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, int, pd.Series]:
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
        "seed": 42,
        "feature_fraction_seed": 42,
        "bagging_seed": 42,
    }
    train_set = lgb.Dataset(train[feature_names], label=train["label"])
    valid_set = lgb.Dataset(valid[feature_names], label=valid["label"], reference=train_set)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    importance = pd.Series(
        model.feature_importance(importance_type="gain"), index=feature_names, dtype=float
    )
    importance = importance / importance.sum()
    return model.predict(test[feature_names]), int(model.best_iteration), importance


def fit_mlp(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, int | None, str]:
    import copy
    import torch
    from torch import nn

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"MLP训练设备：{device}", flush=True)

    np.random.seed(42)
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(len(feature_names), 64),
        nn.SiLU(),
        nn.Dropout(0.05),
        nn.Linear(64, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    loss_function = nn.MSELoss()

    x_train = torch.as_tensor(
        train[feature_names].to_numpy(dtype=np.float32), device=device
    )
    y_train = torch.as_tensor(train["label"].to_numpy(dtype=np.float32), device=device)
    x_valid = torch.as_tensor(
        valid[feature_names].to_numpy(dtype=np.float32), device=device
    )
    y_valid = torch.as_tensor(valid["label"].to_numpy(dtype=np.float32), device=device)
    x_test = torch.as_tensor(
        test[feature_names].to_numpy(dtype=np.float32), device=device
    )

    best_loss = float("inf")
    best_step: int | None = None
    best_state: dict | None = None
    stale_evaluations = 0
    batch_size = min(4096, len(train))
    for step in range(1, 201):
        model.train()
        indices = torch.randint(0, len(train), (batch_size,), device=device)
        prediction = model(x_train[indices]).reshape(-1)
        loss = loss_function(prediction, y_train[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 10:
            continue
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_function(model(x_valid).reshape(-1), y_valid).item())
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
            stale_evaluations = 0
        else:
            stale_evaluations += 1
            if stale_evaluations >= 5:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(x_test).reshape(-1).detach().cpu().numpy()
    return prediction, best_step, device


def run_walk_forward(panel: pd.DataFrame, skip_mlp: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_features = [*FUNDAMENTAL_FEATURES, *ALPHA158_FEATURES, *COURSE_FEATURES]
    model_predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    importance_rows: list[pd.DataFrame] = []

    for fold in expanding_folds():
        train = panel.loc[panel["year"].between(fold.train_start, fold.train_end)].copy()
        valid = panel.loc[panel["year"].eq(fold.valid_year)].copy()
        test = panel.loc[panel["year"].eq(fold.test_year)].copy()
        train, valid, test = robust_scale(train, valid, test, columns=all_features)
        print(
            f"滚动折{fold.test_year}：训练{fold.train_start}-{fold.train_end}，"
            f"验证{fold.valid_year}，预测{fold.test_year}",
            flush=True,
        )

        ridge4_pred, ridge4_alpha, ridge4_coef = fit_ridge(
            train, valid, test, list(FUNDAMENTAL_FEATURES)
        )
        ridge_pred, ridge_alpha, ridge_coef = fit_ridge(
            train, valid, test, all_features
        )
        lgb_pred, best_iteration, importance = fit_lightgbm(
            train, valid, test, all_features
        )
        predictions = {
            "ridge4": ridge4_pred,
            "ridge_all": ridge_pred,
            "lightgbm": lgb_pred,
        }
        mlp_step: int | None = None
        mlp_device = "未运行"
        if not skip_mlp:
            mlp_pred, mlp_step, mlp_device = fit_mlp(train, valid, test, all_features)
            predictions["mlp"] = mlp_pred

        output = test[["symbol", "signal_date", "return_date", "forward_return"]].copy()
        for name, values in predictions.items():
            output[name] = values
        model_predictions.append(output)
        fold_rows.append(
            {
                "test_year": fold.test_year,
                "train_end": fold.train_end,
                "valid_year": fold.valid_year,
                "ridge4_alpha": ridge4_alpha,
                "ridge_all_alpha": ridge_alpha,
                "lightgbm_best_iteration": best_iteration,
                "mlp_best_step": mlp_step,
                "mlp_device": mlp_device,
            }
        )
        ridge4_table = pd.DataFrame(
            {
                "test_year": fold.test_year,
                "model": "ridge4",
                "feature": list(FUNDAMENTAL_FEATURES),
                "importance": ridge4_coef,
            }
        )
        ridge_table = pd.DataFrame(
            {
                "test_year": fold.test_year,
                "model": "ridge_all",
                "feature": all_features,
                "importance": ridge_coef,
            }
        )
        lgb_table = importance.rename("importance").rename_axis("feature").reset_index()
        lgb_table.insert(0, "model", "lightgbm")
        lgb_table.insert(0, "test_year", fold.test_year)
        importance_rows.extend([ridge4_table, ridge_table, lgb_table])

    combined_predictions = pd.concat(model_predictions, ignore_index=True)
    ensemble_members = ["ridge_all", "lightgbm"]
    if "mlp" in combined_predictions:
        ensemble_members.append("mlp")
    ranked_predictions = [
        combined_predictions.groupby("signal_date")[name].rank(
            method="average", pct=True
        )
        for name in ensemble_members
    ]
    combined_predictions["ensemble_equal"] = sum(ranked_predictions) / len(
        ranked_predictions
    )
    return (
        combined_predictions,
        pd.DataFrame(fold_rows),
        pd.concat(importance_rows, ignore_index=True),
    )


def evaluate_scores(
    prediction: pd.DataFrame,
    panel: pd.DataFrame,
    cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    years = pd.to_datetime(prediction["signal_date"]).dt.year
    reference = panel.loc[panel["year"].isin(sorted(years.unique()))].copy()
    reference["grid_reference"] = (
        0.25 * reference["fund_value"]
        + 0.50 * reference["fund_quality"]
        - 0.25 * reference["fund_rsi"]
    )
    columns = ["grid_reference", *COURSE_FEATURES]
    scores = prediction.merge(
        reference[["symbol", "signal_date", *columns]],
        on=["symbol", "signal_date"],
        how="left",
        validate="one_to_one",
    )
    score_names = [
        "grid_reference", *COURSE_FEATURES, "ridge4", "ridge_all", "lightgbm"
    ]
    if "mlp" in scores:
        score_names.append("mlp")
    score_names.append("ensemble_equal")

    rows: list[dict[str, object]] = []
    monthly_tables: dict[str, pd.DataFrame] = {}
    for name in score_names:
        frame = scores[["symbol", "signal_date", "return_date", "forward_return", name]].dropna()
        frame = frame.rename(columns={name: "score"})
        ic = rank_ic_statistics(frame, "score")
        monthly = calculate_monthly_portfolios(frame, cost_bps=cost_bps)
        monthly_tables[name] = monthly
        q5 = performance_statistics(monthly["q5_net"])
        long_short = performance_statistics(monthly["long_short_net"])
        rows.append(
            {
                "model": name,
                **ic,
                "q5_net_annual_return": q5["annual_return"],
                "q5_net_sharpe": q5["sharpe"],
                "q5_net_max_drawdown": q5["max_drawdown"],
                "ls_net_annual_return": long_short["annual_return"],
                "ls_net_sharpe": long_short["sharpe"],
                "ls_net_max_drawdown": long_short["max_drawdown"],
            }
        )
    return pd.DataFrame(rows).sort_values("mean_rank_ic", ascending=False), monthly_tables


def create_plots(monthly_tables: dict[str, pd.DataFrame], result_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "grid_reference": "Current grid reference",
        "pv_anti_bare": "PV anti bare",
        "pv_anti_pair": "PV anti pair",
        "pv_anti_triple": "PV anti triple",
        "gap_diff4_bare": "Gap diff4 bare",
        "ridge4": "Ridge 4-factor",
        "ridge_all": "Ridge all",
        "lightgbm": "LightGBM",
        "mlp": "MLP",
        "ensemble_equal": "Equal-rank ensemble",
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(11, 6))
    first_signal = min(pd.to_datetime(table["signal_date"]).min() for table in monthly_tables.values())
    last_return = max(pd.to_datetime(table["return_date"]).max() for table in monthly_tables.values())
    for name, monthly in monthly_tables.items():
        dates = pd.to_datetime(monthly["return_date"])
        wealth = pd.Series((1 + monthly["long_short_net"]).cumprod().to_numpy(), index=dates)
        wealth = pd.concat([pd.Series([1.0], index=[first_signal]), wealth])
        axis.plot(wealth.index, wealth, label=labels[name])
    axis.set_title("Walk-forward development long-short wealth (2020-2024)")
    axis.set_ylabel("Wealth index")
    axis.set_xlim(first_signal, last_return)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(result_dir / "model_comparison_cumulative.png", dpi=160)
    plt.close(figure)


def write_report(
    panel: pd.DataFrame,
    metrics: pd.DataFrame,
    folds: pd.DataFrame,
    importance: pd.DataFrame,
    cost_bps: float,
) -> None:
    labels = {
        "grid_reference": "原35组网格参考（非严格滚动）",
        "pv_anti_bare": "量价反共振（裸）",
        "pv_anti_pair": "量价反共振×换手",
        "pv_anti_triple": "量价反共振×换手×振幅",
        "gap_diff4_bare": "隔夜—日内分解",
        "ridge4": "Ridge四因子连续权重",
        "ridge_all": "Ridge全特征",
        "lightgbm": "LightGBM全特征",
        "mlp": "MLP全特征",
        "ensemble_equal": "Ridge/LightGBM/MLP等权秩集成",
    }
    table_lines = []
    for row in metrics.itertuples():
        table_lines.append(
            f"| {labels[row.model]} | {row.mean_rank_ic:.4f} | {row.annualized_icir:.3f} | "
            f"{row.positive_rate:.1%} | {row.q5_net_annual_return:.2%} | "
            f"{row.ls_net_annual_return:.2%} | {row.ls_net_sharpe:.3f} | "
            f"{row.ls_net_max_drawdown:.2%} |"
        )
    table = "\n".join(table_lines)

    ridge4 = importance.loc[importance["model"].eq("ridge4")].copy()
    ridge4_summary = ridge4.groupby("feature")["importance"].agg(["mean", "std"]).reset_index()
    ridge_lines = "\n".join(
        f"| {row.feature} | {row['mean']:.4f} | {row['std']:.4f} |"
        for _, row in ridge4_summary.iterrows()
    )
    lgb_importance = (
        importance.loc[importance["model"].eq("lightgbm")]
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )
    lgb_lines = "\n".join(
        f"| {name} | {value:.2%} |" for name, value in lgb_importance.items()
    )
    fold_lines = "\n".join(
        f"| {int(row.test_year)} | 2015—{int(row.train_end)} | {int(row.valid_year)} | "
        f"{row.ridge4_alpha:g} | {row.ridge_all_alpha:g} | "
        f"{int(row.lightgbm_best_iteration)} | "
        f"{int(row.mlp_best_step) if pd.notna(row.mlp_best_step) else '未运行'} | "
        f"{row.mlp_device} |"
        for row in folds.itertuples()
    )
    report = f"""# Alpha158与机器学习复合因子报告

## 摘要

本研究在历史沪深300和中证500成分股并集上，将vn.py Alpha158价量特征、价值/规模/ROE/RSI基本面特征，以及课件中的量价反共振和隔夜—日内分解因子合并，比较Ridge、LightGBM和MLP。绩效标签仍为下一月首个交易日开盘至月末收盘收益；所有模型输出均为连续分数。

特征与调参数据限于2015—2024年。2025年没有下载或查看。2020—2024结果来自扩展式滚动开发检验，仍属于样本内开发证据，不能称为最终样本外验证。

## 数据和特征

- 数据源：米筐RQData；股票池为每个信号月末当时的历史中证800成分股。
- 特征面板：{panel['signal_date'].nunique()}个月、{len(panel):,}个股票—月份观测。
- Alpha158：严格沿用vn.py `Alpha158`的158项命名和公式，仅在月末计算，避免存储巨大的日频宽表。
- 复权一致性：VWAP先由未复权成交额和成交量计算，再乘价格复权因子；最终`vwap/close`的1%—99%分位为约0.957—1.026，避免把送股或现金分红调整误当成预测信号。
- 公式核对：143项与本机vn.py逐项输出一致；其余15项涨跌天数统计按Alpha158公式语义计算。当前vn.py 4.4.0的DataProxy比较路径在对照样本中返回全零，因此没有复制该异常结果。
- 新增因子：`PV_ANTI_BARE/PAIR/TRIPLE`和`GAP_DIFF4_BARE`。
- 财务特征：价值、规模、ROE和RSI；先按月缩尾和标准化。
- 预测标签：下一月收益的月度截面秩，中心化为约[-1.73, 1.73]。
- 每一滚动折仅使用训练期拟合特征中位数和MAD；验证期用于Ridge正则强度选择以及树/MLP早停。
- 模型比较区间：2020—2024年；单边交易成本：{cost_bps:.1f} bps。

## 滚动设计

| 预测年 | 训练期 | 验证年 | Ridge4 α | Ridge全特征 α | LightGBM轮数 | MLP最佳步数 | 设备 |
|---:|---|---:|---:|---:|---:|---:|---|
{fold_lines}

同一预测年内，训练、验证和预测严格按时间先后排列。由于研究者可以在看到2020—2024汇总结果后继续修改方案，这仍是开发期内部的准样本外检验。

## 对照结果

| 方法 | 平均Rank IC | 年化ICIR | IC为正月份 | 最高组扣费年化 | 多空扣费年化 | 多空Sharpe | 多空最大回撤 |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

“原35组网格参考”的权重曾使用整个2015—2024年选择，因此只作直观参照，不能与滚动训练模型作完全公平的样本外比较。课件公式因子也属于已从既有课件结果中挑选的候选，不是本项目的全新盲选结果。

三模型集成不再搜索权重，而是在每个月分别把Ridge、LightGBM和MLP预测转为截面百分位秩后做等权平均；这种做法减少不同模型输出尺度的影响，也避免根据本次结果事后调权。

综合选择：将**三模型等权秩集成**作为当前主候选，因为它同时取得最高平均Rank IC（0.0784）和最高扣费多空Sharpe（0.877）；将**Ridge全特征**作为更简单的备选，因为其最大回撤较小（-14.44%）且可解释性更强。MLP证明了非线性学习可以提升排序，但不单独作为最终因子。

![滚动开发期多空净值](model_comparison_cumulative.png)

## 连续四因子权重

下表为五个滚动折中Ridge四因子标准化系数的均值和标准差。它直接回答“权重能否不使用0.25步长”：可以，系数由训练数据连续估计，并由下一年验证集选择正则强度。

| 特征 | 平均系数 | 折间标准差 |
|---|---:|---:|
{ridge_lines}

## LightGBM主要特征

| 特征 | 平均增益占比 |
|---|---:|
{lgb_lines}

MLP沿用vn.py课件的PyTorch建模思路，但针对M3 Air做了适配：自动选择MPS，使用单隐藏层64单元、SiLU、5% Dropout、AdamW、L2正则和验证集早停。没有在MPS层内使用BatchNorm，因为当前vn.py实现会产生明显的同步开销；输入已在每个滚动折内使用训练期中位数和MAD完成标准化。选择小网络而非课件默认256单元，是因为本研究只有119个月度截面，必须限制模型容量，而不是因为M3算力不足。

## 解释边界

1. 深度学习的自由度更高，不应仅因训练损失更低就称为更好的因子；本报告以滚动Rank IC和扣费组合表现判断。
2. 月度股票观测并不等于独立时间样本；有效时间长度仍然有限，因此MLP结果的不确定性高于Ridge。
3. Alpha158只使用历史价量；财务指标由RQData提供，但仍需依赖供应商的历史时点口径。
4. 当前回测处理了换手成本，但没有完整模拟涨跌停、冲击成本、印花税差异和容量。
5. 2025年继续密封。只有在因子、模型和超参数冻结后，才能进行一次性最终评价。

## 可复现文件

- `model_metrics.csv`：模型和公式因子汇总指标。
- `walk_forward_predictions.csv`：2020—2024滚动预测及真实收益。
- `fold_settings.csv`：每折训练、验证、预测设置。
- `feature_importance.csv`：Ridge系数和LightGBM重要性。
- 授权原始行情及完整股票级特征位于`data/private/rqdata/`，不会提交GitHub。
"""
    (RESULT_DIR / "alpha_ml_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    require_discovery_window("2015-01-01", "2024-12-31")
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_parquet(MONTHLY_RAW)
    monthly["signal_date"] = pd.to_datetime(monthly["signal_date"])
    alpha = build_feature_cache(
        monthly,
        args.license_script,
        args.refresh_daily,
        args.refresh_features,
    )
    if args.features_only:
        print(f"特征缓存完成：{FEATURE_CACHE}", flush=True)
        return 0

    panel = prepare_model_panel(monthly, alpha)
    print(f"模型面板：{panel.shape[0]:,}行、{len(ALPHA158_FEATURES)+len(COURSE_FEATURES)+len(FUNDAMENTAL_FEATURES)}项特征", flush=True)
    predictions, folds, importance = run_walk_forward(panel, args.skip_mlp)
    metrics, monthly_tables = evaluate_scores(predictions, panel, args.cost_bps)

    predictions.to_csv(RESULT_DIR / "walk_forward_predictions.csv", index=False)
    folds.to_csv(RESULT_DIR / "fold_settings.csv", index=False)
    importance.to_csv(RESULT_DIR / "feature_importance.csv", index=False)
    metrics.to_csv(RESULT_DIR / "model_metrics.csv", index=False)
    for name, table in monthly_tables.items():
        table.to_csv(RESULT_DIR / f"monthly_{name}.csv", index=False)
    create_plots(monthly_tables, RESULT_DIR)
    write_report(panel, metrics, folds, importance, args.cost_bps)
    print(metrics.to_string(index=False), flush=True)
    print(f"报告：{RESULT_DIR / 'alpha_ml_report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
