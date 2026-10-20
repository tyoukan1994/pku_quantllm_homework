from ashare import Strategy
from ashare.factors import FACTOR_NAMES, eligible_scores


class UserStrategy(Strategy):
    name = "量价因子 · 非 ST 全市场选股"

    def __init__(self, factor="reversal5", top_n=200, rebalance_days=5,
                 investment=0.95, liquid_pool=1500, minimum_amount=50_000_000):
        if factor not in FACTOR_NAMES:
            raise ValueError(f"未知因子：{factor}")
        for value in (top_n, rebalance_days, liquid_pool):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError("数量和调仓周期须为正整数")
        if top_n > liquid_pool or not 0 < investment <= 1 or minimum_amount < 0:
            raise ValueError("仓位或流动性参数无效")
        self.factor = factor
        self.top_n = int(top_n)
        self.rebalance_days = int(rebalance_days)
        self.investment = float(investment)
        self.liquid_pool = int(liquid_pool)
        self.minimum_amount = float(minimum_amount)
        self.counter = 0
        self.name = f"{FACTOR_NAMES[factor]} · Top {top_n}"

    def on_close(self, context):
        rebalance = self.counter % self.rebalance_days == 0
        self.counter += 1
        if not rebalance:
            return None
        scores = eligible_scores(context, self.liquid_pool, self.minimum_amount)
        selected = scores[self.factor].dropna().sort_index().nlargest(self.top_n)
        return {symbol: self.investment / len(selected) for symbol in selected.index} if len(selected) else {}
