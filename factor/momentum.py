"""Transparent candidate factor and leakage-aware daily backtest."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_volatility_adjusted_momentum(
    bars: pd.DataFrame, lookback: int = 20, cost_bps: float = 2.0
) -> pd.DataFrame:
    """Compute a candidate trend factor and a next-bar executable backtest.

    The factor at close t is the lookback log return divided by realized
    volatility.  Its sign becomes the position for bar t+1.  Costs are charged
    on absolute position changes.  This is a candidate, not validated alpha.
    """
    if lookback < 2:
        raise ValueError("lookback 必须至少为 2。")
    if cost_bps < 0:
        raise ValueError("cost_bps 不能为负数。")

    result = bars.copy()
    log_close = np.log(result["close"])
    log_return = log_close.diff()
    momentum = log_close - log_close.shift(lookback)
    realized_vol = log_return.rolling(lookback).std(ddof=1) * math.sqrt(lookback)

    result["factor"] = momentum / realized_vol.replace(0, np.nan)
    result["signal"] = np.sign(result["factor"]).fillna(0.0)
    result["position"] = result["signal"].shift(1).fillna(0.0)
    result["bar_return"] = result["close"].pct_change().fillna(0.0)
    result["turnover"] = result["position"].diff().abs().fillna(result["position"].abs())
    result["cost"] = result["turnover"] * cost_bps / 10_000
    result["strategy_return"] = result["position"] * result["bar_return"] - result["cost"]
    result["equity"] = (1.0 + result["strategy_return"]).cumprod()
    return result


def summarize_daily_backtest(result: pd.DataFrame) -> dict[str, float | int]:
    """Return compact daily-frequency metrics with explicit assumptions."""
    returns = result["strategy_return"].dropna()
    if returns.empty:
        raise ValueError("没有可汇总的策略收益。")

    volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / volatility * math.sqrt(252)) if volatility else 0.0
    equity = result["equity"]
    drawdown = equity / equity.cummax() - 1.0
    return {
        "observations": int(len(result)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "annualized_volatility": float(volatility * math.sqrt(252)),
        "sharpe_0rf": sharpe,
        "max_drawdown": float(drawdown.min()),
        "turnover": float(result["turnover"].sum()),
    }
