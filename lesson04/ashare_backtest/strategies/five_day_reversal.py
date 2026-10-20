import numpy as np

from ashare import Strategy


class UserStrategy(Strategy):
    name = "5DR · 五日反转"

    def __init__(self, top_n=200, rebalance_days=5, investment=0.95,
                 liquid_pool=1500, minimum_amount=50_000_000, lookback=5):
        for value in (top_n, rebalance_days, liquid_pool, lookback):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError("窗口、调仓间隔及股票数量必须为正整数")
        if top_n > liquid_pool or not 0 < investment <= 1 or minimum_amount < 0:
            raise ValueError("反转策略参数超出范围")
        self.top_n = int(top_n)
        self.rebalance_days = int(rebalance_days)
        self.investment = float(investment)
        self.liquid_pool = int(liquid_pool)
        self.minimum_amount = float(minimum_amount)
        self.lookback = int(lookback)
        self.counter = 0

    def on_close(self, context):
        rebalance = self.counter % self.rebalance_days == 0
        self.counter += 1
        if not rebalance:
            return None
        universe = context.universe()
        if not universe:
            return {}
        prices = context.history("close", self.lookback + 1, universe, adjusted=True)
        amounts = context.history("amt", 20, universe)
        if len(prices) < self.lookback + 1 or len(amounts) < 20:
            return {}
        liquid = amounts.mean().where(amounts.notna().all() & (amounts > 0).all())
        liquid = liquid[liquid >= self.minimum_amount].sort_index().nlargest(self.liquid_pool)
        scores = -(prices.iloc[-1] / prices.iloc[0] - 1)
        scores = scores.where(prices.notna().all()).replace([np.inf, -np.inf], np.nan)
        selected = scores.reindex(liquid.index).dropna().sort_index().nlargest(self.top_n)
        return {symbol: self.investment / len(selected) for symbol in selected.index} if len(selected) else {}
