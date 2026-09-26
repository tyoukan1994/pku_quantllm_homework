"""Cross-sectional mining utilities for the Fama-style factor combination."""

from __future__ import annotations

from itertools import product
from math import sqrt

import numpy as np
import pandas as pd
import polars as pl


FEATURES = ("value", "size", "quality", "technical")


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean raw RQData fields and normalize each monthly cross-section."""
    required = {
        "signal_date",
        "symbol",
        "market_cap",
        "book_to_market_ratio_lf",
        "return_on_equity_ttm",
        "RSI10",
        "forward_return",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"因子面板缺少字段：{', '.join(sorted(missing))}")

    clean = frame.copy()
    clean["signal_date"] = pd.to_datetime(clean["signal_date"])
    numeric = [
        "market_cap",
        "book_to_market_ratio_lf",
        "return_on_equity_ttm",
        "RSI10",
        "forward_return",
    ]
    for column in numeric:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean[
        (clean["market_cap"] > 0)
        & (clean["book_to_market_ratio_lf"] > 0)
    ].dropna(subset=numeric)

    clean["value"] = np.log(clean["book_to_market_ratio_lf"])
    clean["size"] = -np.log(clean["market_cap"])
    clean["quality"] = clean["return_on_equity_ttm"]
    clean["technical"] = clean["RSI10"]

    # Pre-specified 1%/99% cross-sectional clipping.
    for feature in FEATURES:
        grouped = clean.groupby("signal_date")[feature]
        lower = grouped.transform(lambda values: values.quantile(0.01))
        upper = grouped.transform(lambda values: values.quantile(0.99))
        clean[feature] = clean[feature].clip(lower=lower, upper=upper)

    # Use vnpy.alpha's cross-sectional processor for z-score normalization.
    from vnpy.alpha.dataset.processor import process_cs_norm

    alpha_frame = clean.rename(
        columns={"signal_date": "datetime", "symbol": "vt_symbol"}
    )
    normalized = process_cs_norm(
        pl.from_pandas(alpha_frame), method="zscore", names=list(FEATURES)
    ).to_pandas()
    normalized = normalized.rename(
        columns={"datetime": "signal_date", "vt_symbol": "symbol"}
    )
    return normalized.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[*FEATURES, "forward_return"]
    )


def calculate_rank_ic(
    frame: pd.DataFrame, score_column: str
) -> pd.Series:
    """Calculate one Spearman rank IC for every signal month."""
    values: dict[pd.Timestamp, float] = {}
    for signal_date, group in frame.groupby("signal_date"):
        values[pd.Timestamp(signal_date)] = group[score_column].corr(
            group["forward_return"], method="spearman"
        )
    return pd.Series(values, name="rank_ic").sort_index().dropna()


def orient_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    """Orient each feature by its in-sample mean IC and retain the audit table."""
    oriented = frame.copy()
    rows: list[dict] = []
    directions: dict[str, int] = {}
    for feature in FEATURES:
        ic = calculate_rank_ic(oriented, feature)
        direction = 1 if ic.mean() >= 0 else -1
        directions[feature] = direction
        oriented[feature] *= direction
        rows.append(
            {
                "feature": feature,
                "raw_mean_ic": float(ic.mean()),
                "direction": direction,
                "oriented_mean_ic": float(ic.mean() * direction),
                "months": int(ic.size),
            }
        )
    return oriented, directions, pd.DataFrame(rows)


def generate_weight_grid(step: float = 0.25) -> list[dict[str, float]]:
    """Generate non-negative weights summing to one."""
    units = round(1 / step)
    weights: list[dict[str, float]] = []
    for combination in product(range(units + 1), repeat=len(FEATURES)):
        if sum(combination) != units:
            continue
        weights.append(
            {
                feature: combination[index] / units
                for index, feature in enumerate(FEATURES)
            }
        )
    return weights


def add_score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    scored = frame.copy()
    scored["score"] = sum(scored[name] * weights[name] for name in FEATURES)
    return scored


def _turnover(current_symbols: list[str], previous_symbols: list[str]) -> float:
    if not current_symbols:
        return 0.0
    current_weight = 1 / len(current_symbols)
    previous_weight = 1 / len(previous_symbols) if previous_symbols else 0.0
    symbols = set(current_symbols) | set(previous_symbols)
    return 0.5 * sum(
        abs(
            (current_weight if symbol in current_symbols else 0.0)
            - (previous_weight if symbol in previous_symbols else 0.0)
        )
        for symbol in symbols
    )


def calculate_monthly_portfolios(
    frame: pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Build equal-weight quintile portfolios and charge one-way turnover cost."""
    rows: list[dict] = []
    previous_top: list[str] = []
    previous_bottom: list[str] = []
    cost_rate = cost_bps / 10_000

    for signal_date, group in frame.sort_values("signal_date").groupby("signal_date"):
        group = group.copy()
        group["quintile"] = pd.qcut(
            group["score"].rank(method="first"), 5, labels=False
        ) + 1
        quantile_returns = group.groupby("quintile")["forward_return"].mean()
        top_symbols = group.loc[group["quintile"] == 5, "symbol"].tolist()
        bottom_symbols = group.loc[group["quintile"] == 1, "symbol"].tolist()
        top_turnover = _turnover(top_symbols, previous_top)
        bottom_turnover = _turnover(bottom_symbols, previous_bottom)

        top = float(quantile_returns.loc[5])
        bottom = float(quantile_returns.loc[1])
        row = {
            "signal_date": pd.Timestamp(signal_date),
            "return_date": pd.Timestamp(group["return_date"].iloc[0]),
            "stock_count": int(len(group)),
            "benchmark": float(group["forward_return"].mean()),
            "q1": bottom,
            "q2": float(quantile_returns.loc[2]),
            "q3": float(quantile_returns.loc[3]),
            "q4": float(quantile_returns.loc[4]),
            "q5": top,
            "long_short_gross": top - bottom,
            "top_turnover": top_turnover,
            "bottom_turnover": bottom_turnover,
            "q5_net": top - cost_rate * top_turnover,
            "long_short_net": (
                top - bottom - cost_rate * (top_turnover + bottom_turnover)
            ),
        }
        rows.append(row)
        previous_top = top_symbols
        previous_bottom = bottom_symbols

    return pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)


def performance_statistics(returns: pd.Series) -> dict[str, float]:
    """Calculate annualized monthly-return statistics."""
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return {
            "annual_return": float("nan"),
            "annual_volatility": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }
    wealth = (1 + values).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    annual_return = float(wealth.iloc[-1] ** (12 / len(values)) - 1)
    annual_volatility = float(values.std(ddof=1) * sqrt(12))
    sharpe = annual_return / annual_volatility if annual_volatility else float("nan")
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def evaluate_weights(
    frame: pd.DataFrame,
    weights: dict[str, float],
    cost_bps: float = 10.0,
) -> tuple[dict[str, float], pd.Series, pd.DataFrame]:
    scored = add_score(frame, weights)
    ic = calculate_rank_ic(scored, "score")
    portfolios = calculate_monthly_portfolios(scored, cost_bps=cost_bps)
    ic_std = float(ic.std(ddof=1))
    q5_stats = performance_statistics(portfolios["q5_net"])
    ls_stats = performance_statistics(portfolios["long_short_net"])
    metrics = {
        **{f"weight_{name}": weights[name] for name in FEATURES},
        "active_factors": float(sum(weight > 0 for weight in weights.values())),
        "mean_rank_ic": float(ic.mean()),
        "monthly_icir": float(ic.mean() / ic_std) if ic_std else float("nan"),
        "annualized_icir": float(ic.mean() / ic_std * sqrt(12)) if ic_std else float("nan"),
        "q5_net_annual_return": q5_stats["annual_return"],
        "q5_net_sharpe": q5_stats["sharpe"],
        "q5_net_max_drawdown": q5_stats["max_drawdown"],
        "ls_net_annual_return": ls_stats["annual_return"],
        "ls_net_sharpe": ls_stats["sharpe"],
        "ls_net_max_drawdown": ls_stats["max_drawdown"],
    }
    return metrics, ic, portfolios


def mine_weight_grid(
    frame: pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for weights in generate_weight_grid():
        metrics, _, _ = evaluate_weights(frame, weights, cost_bps=cost_bps)
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values(
        ["annualized_icir", "mean_rank_ic"], ascending=False
    ).reset_index(drop=True)
