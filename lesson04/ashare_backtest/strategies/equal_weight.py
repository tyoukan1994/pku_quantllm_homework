from ashare import Strategy


class UserStrategy(Strategy):
    name = "固定股票篮子等权"

    def __init__(self, symbols=None, rebalance_days=20, investment=0.95):
        self.symbols = symbols or ["000001.SZ", "600000.SH", "600519.SH", "300750.SZ"]
        if rebalance_days < 1 or not 0 < investment <= 1 or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("等权策略参数无效")
        self.rebalance_days = int(rebalance_days)
        self.investment = float(investment)
        self.counter = 0

    def on_close(self, context):
        should_rebalance = self.counter % self.rebalance_days == 0
        self.counter += 1
        if not should_rebalance:
            return None
        eligible = set(context.universe())
        selected = [symbol for symbol in self.symbols if symbol in eligible]
        return {symbol: self.investment / len(selected) for symbol in selected} if selected else {}
