import numpy as np
import pandas as pd

from .strategy import Strategy


FACTORS = {
    "high252", "high252_lowvol", "intraday_reversal", "overnight_reversal",
    "amihud_low", "amihud_high", "alpha33", "alpha101", "alpha101_reverse",
    "range_reversal", "amount_expansion", "price_amount_divergence",
}


def literature_scores(close, opening, high, low, amount):
    latest = close.iloc[-1]
    missing = latest * np.nan
    returns = close.pct_change(fill_method=None)
    intraday = np.log(close / opening)
    overnight = np.log(opening / close.shift(1))
    volatility = returns.tail(20).std(ddof=1).where(returns.tail(20).notna().all()) if len(close) >= 21 else missing
    high252 = latest / close.tail(252).max().where(close.tail(252).notna().all()) if len(close) >= 252 else missing
    illiquidity = (returns.abs() / amount.replace(0, np.nan)).tail(20)
    amihud = illiquidity.mean().where(illiquidity.notna().all()) if len(close) >= 21 else missing
    previous = close.shift(1)
    true_range = pd.DataFrame(np.maximum.reduce([(high - low).to_numpy(), (high - previous).abs().to_numpy(),
                                                (low - previous).abs().to_numpy()]), index=close.index, columns=close.columns)
    atr = true_range.tail(20).mean().where(true_range.tail(20).notna().all()) / latest if len(close) >= 21 else missing
    reversal = -(latest / close.iloc[-6] - 1) if len(close) >= 6 else missing
    amount20 = amount.tail(20).mean().where(amount.tail(20).gt(0).all()) if len(amount) >= 20 else missing
    intraday5 = -intraday.tail(5).sum().where(intraday.tail(5).notna().all()) if len(close) >= 5 else missing
    overnight5 = -overnight.tail(5).sum().where(overnight.tail(5).notna().all()) if len(close) >= 6 else missing
    changes = np.log(amount.replace(0, np.nan)).diff()
    correlation = returns.tail(20).corrwith(changes.tail(20)).where(returns.tail(20).notna().all() & changes.tail(20).notna().all()) if len(close) >= 21 else missing
    return pd.DataFrame({"high252": high252,
                         "volatility": volatility,
                         "intraday_reversal": intraday5, "overnight_reversal": overnight5,
                         "amihud_low": -amihud, "amihud_high": amihud,
                         "alpha33": opening.iloc[-1] / latest - 1,
                         "alpha101": (latest - opening.iloc[-1]) / (high.iloc[-1] - low.iloc[-1] + 0.001),
                         "range_reversal": reversal / atr.replace(0, np.nan),
                         "amount_expansion": amount.tail(5).mean().where(amount.tail(5).notna().all()) / amount20,
                         "price_amount_divergence": -correlation, "amount20": amount20,
                         "history120": close.tail(120).notna().all().astype(float) if len(close) >= 120 else missing,
                         }).replace([np.inf, -np.inf], np.nan)


class LiteratureStrategy(Strategy):
    name = "文献量价选股"

    def __init__(self, factor="high252", top_n=100, rebalance_days=5,
                 investment=0.95, minimum_amount=100_000_000, liquid_pool=1500):
        if factor not in FACTORS:
            raise ValueError("不支持的文献因子")
        for value in (top_n, rebalance_days, liquid_pool):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or int(value) != value or value < 1:
                raise ValueError("数量和调仓间隔必须为正整数")
        if not np.isfinite(investment) or not np.isfinite(minimum_amount) or not 0 < investment <= 1 or minimum_amount < 0 or top_n > liquid_pool:
            raise ValueError("仓位或流动性参数无效")
        self.factor, self.top_n = factor, int(top_n)
        self.rebalance_days, self.liquid_pool = int(rebalance_days), int(liquid_pool)
        self.investment, self.minimum_amount = investment, minimum_amount
        self.counter = 0

    def on_close(self, context):
        rebalance = self.counter % self.rebalance_days == 0
        self.counter += 1
        if not rebalance:
            return None
        symbols = context.universe()
        if not symbols:
            return {}
        close = context.history("close", 253, symbols, adjusted=True)
        opening = context.history("open", 253, symbols, adjusted=True)
        high = context.history("high", 253, symbols, adjusted=True)
        low = context.history("low", 253, symbols, adjusted=True)
        amount = context.history("amt", 253, symbols)
        frame = literature_scores(close, opening, high, low, amount)
        eligible = frame.loc[(frame.history120 == 1) & (frame.amount20 >= self.minimum_amount)]
        liquid = eligible.amount20.sort_index().nlargest(self.liquid_pool).index
        eligible = eligible.reindex(liquid)
        if self.factor == "high252_lowvol":
            score = 0.5 * eligible.high252.rank(pct=True) + 0.5 * (-eligible.volatility).rank(pct=True)
        elif self.factor == "alpha101_reverse":
            score = -eligible.alpha101
        elif self.factor == "alpha101":
            raw_close = context.history("close", 1, eligible.index).iloc[-1]
            raw_open = context.history("open", 1, eligible.index).iloc[-1]
            raw_high = context.history("high", 1, eligible.index).iloc[-1]
            raw_low = context.history("low", 1, eligible.index).iloc[-1]
            score = (raw_close - raw_open) / (raw_high - raw_low + 0.001)
        else:
            score = eligible[self.factor]
        selected = score.replace([np.inf, -np.inf], np.nan).dropna().sort_index().nlargest(self.top_n).index
        return {symbol: self.investment / self.top_n for symbol in selected}
