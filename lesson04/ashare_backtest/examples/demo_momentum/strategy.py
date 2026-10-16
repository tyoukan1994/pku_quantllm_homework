from ashare import Strategy


class UserStrategy(Strategy):
    name = "动量选股"

    def __init__(self, lookback=60, skip=5, top_n=10, rebalance_days=5, investment=0.95):
        if lookback < 1 or skip < 0 or top_n < 1 or rebalance_days < 1 or not 0 < investment <= 1:
            raise ValueError("动量策略参数无效")
        self.lookback = int(lookback)
        self.skip = int(skip)
        self.top_n = int(top_n)
        self.rebalance_days = int(rebalance_days)
        self.investment = float(investment)
        self.counter = 0

    def on_close(self, context):
        should_rebalance = self.counter % self.rebalance_days == 0
        self.counter += 1
        if not should_rebalance:
            return None
        universe = context.universe()
        if not universe:
            return {}
        window = self.lookback + self.skip + 1
        prices = context.history("close", window, universe, adjusted=True)
        if len(prices) < window:
            return None
        complete = prices.notna().all()
        momentum = prices.iloc[-self.skip - 1] / prices.iloc[0] - 1
        selected = momentum[complete & (momentum > 0)].sort_index().nlargest(self.top_n).index
        if len(selected) == 0:
            return {}
        return {symbol: self.investment / len(selected) for symbol in selected}
