from ashare import Strategy


class UserStrategy(Strategy):
    name = "均线趋势"

    def __init__(self, fast=20, slow=60, top_n=20, rebalance_days=5, investment=0.95):
        if not 1 <= fast < slow or top_n < 1 or rebalance_days < 1 or not 0 < investment <= 1:
            raise ValueError("均线策略参数无效")
        self.fast = int(fast)
        self.slow = int(slow)
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
        prices = context.history("close", self.slow, universe, adjusted=True)
        if len(prices) < self.slow:
            return None
        ratio = prices.tail(self.fast).mean() / prices.mean() - 1
        selected = ratio[prices.notna().all() & (ratio > 0)].sort_index().nlargest(self.top_n).index
        return {symbol: self.investment / len(selected) for symbol in selected} if len(selected) else {}
