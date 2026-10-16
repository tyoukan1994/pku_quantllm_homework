from ashare import Strategy


class UserStrategy(Strategy):
    name = "自定义动量模板"

    def __init__(self, lookback=20, top_n=5):
        if lookback < 1 or top_n < 1:
            raise ValueError("lookback 和 top_n 必须为正整数")
        self.lookback = int(lookback)
        self.top_n = int(top_n)

    def on_close(self, context):
        universe = context.universe()
        if not universe:
            return {}
        prices = context.history("close", self.lookback + 1, universe, adjusted=True)
        if len(prices) < self.lookback + 1:
            return None
        returns = prices.iloc[-1] / prices.iloc[0] - 1
        selected = returns[prices.notna().all() & (returns > 0)].sort_index().nlargest(self.top_n).index
        return {symbol: 0.9 / len(selected) for symbol in selected} if len(selected) else {}
