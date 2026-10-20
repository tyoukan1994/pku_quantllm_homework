import numpy as np
import pandas as pd


FACTOR_NAMES = {
    "reversal5": "五日反转",
    "momentum20": "二十日动量",
    "momentum60": "六十日动量",
    "low_volatility20": "二十日低波动",
    "amount_reversal5": "成交额放大反转",
    "close_pressure5": "收盘位置量价反转",
}


def factor_scores(close, amount, high, low):
    if len(close) < 61 or len(amount) < 20:
        return pd.DataFrame(index=close.columns, columns=list(FACTOR_NAMES), dtype=float)
    daily = close.pct_change(fill_method=None)
    shock = (amount.tail(5).mean() / amount.tail(20).mean()).clip(0.5, 2.0)
    location = (2 * close.tail(5) - high.tail(5) - low.tail(5)) / (high.tail(5) - low.tail(5)).replace(0, np.nan)
    reversal = -(close.iloc[-1] / close.iloc[-6] - 1)
    result = pd.DataFrame({
        "reversal5": reversal.where(close.tail(6).notna().all()),
        "momentum20": (close.iloc[-1] / close.iloc[-21] - 1).where(close.tail(21).notna().all()),
        "momentum60": (close.iloc[-1] / close.iloc[-61] - 1).where(close.tail(61).notna().all()),
        "low_volatility20": -daily.tail(20).std(ddof=1).where(daily.tail(20).notna().all()),
        "amount_reversal5": reversal.where(close.tail(6).notna().all()) * shock,
        "close_pressure5": -(location * amount.tail(5).div(amount.tail(20).mean())).mean().where(location.notna().all()),
    })
    return result.replace([np.inf, -np.inf], np.nan)


def eligible_scores(context, liquid_pool=1500, minimum_amount=50_000_000):
    universe = context.universe()
    if not universe:
        return pd.DataFrame(columns=list(FACTOR_NAMES), dtype=float)
    close = context.history("close", 61, universe, adjusted=True)
    amount = context.history("amt", 20, universe)
    high = context.history("high", 5, universe, adjusted=True)
    low = context.history("low", 5, universe, adjusted=True)
    scores = factor_scores(close, amount, high, low)
    liquidity = amount.mean().where(amount.notna().all() & (amount > 0).all())
    liquid = liquidity[liquidity >= minimum_amount].sort_index().nlargest(liquid_pool).index
    return scores.reindex(liquid)
