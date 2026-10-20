import numpy as np
import pandas as pd

from .strategy import Strategy


MODES = {
    "rsi_rebound", "rsi_pullback", "bollinger_rebound", "reversal", "momentum",
    "breakout", "ma_trend", "low_volatility", "lowvol_reversal", "lowvol_momentum",
    "amount_reversal", "close_pressure", "defensive_regime", "liquid_equal",
}


class WilderRSI:
    def __init__(self, period, size):
        self.period = period
        self.previous = np.full(size, np.nan)
        self.gain = np.zeros(size)
        self.loss = np.zeros(size)
        self.count = np.zeros(size, dtype=int)
        self.value = np.full(size, np.nan)

    def update(self, prices):
        prices = np.asarray(prices, dtype=float)
        valid = np.isfinite(prices) & np.isfinite(self.previous)
        reset = ~valid
        self.count[reset] = 0
        self.gain[reset] = 0
        self.loss[reset] = 0
        changes = np.where(valid, prices - self.previous, 0)
        gains, losses = np.maximum(changes, 0), np.maximum(-changes, 0)
        seed = valid & (self.count < self.period)
        smooth = valid & ~seed
        self.gain[seed] += gains[seed] / self.period
        self.loss[seed] += losses[seed] / self.period
        self.gain[smooth] = (self.gain[smooth] * (self.period - 1) + gains[smooth]) / self.period
        self.loss[smooth] = (self.loss[smooth] * (self.period - 1) + losses[smooth]) / self.period
        self.count[valid] += 1
        total = self.gain + self.loss
        self.value = np.full(len(prices), np.nan)
        ready = valid & (self.count >= self.period)
        np.divide(100 * self.gain, total, out=self.value, where=ready & (total > 0))
        self.value[ready & (total == 0)] = 50
        self.previous = prices.copy()
        return self.value.copy()


def wilder_rsi(close, period=14):
    if isinstance(period, bool) or not isinstance(period, int) or period < 1:
        raise ValueError("RSI周期必须为正整数")
    state = WilderRSI(period, close.shape[1])
    values = np.vstack([state.update(row) for row in close.to_numpy()])
    return pd.DataFrame(values, index=close.index, columns=close.columns)


def prepare_features(close, amount, high, low):
    features = {"close": close, "history120": close.notna().rolling(120).sum().eq(120).astype(float)}
    for window in (5, 20, 60, 120, 200):
        features[f"ma{window}"] = close.rolling(window).mean()
    returns = close.pct_change(fill_method=None)
    for window in (20, 60):
        features[f"vol{window}"] = returns.rolling(window).std(ddof=1)
    for window in (5, 20):
        features[f"reversal{window}"] = -(close / close.shift(window) - 1)
    features["momentum60"] = close.shift(5) / close.shift(65) - 1
    features["momentum120"] = close.shift(20) / close.shift(140) - 1
    features["amount20"] = amount.rolling(20).mean().where(amount.gt(0).rolling(20).sum().eq(20))
    features["amount_shock"] = amount.rolling(5).mean() / features["amount20"]
    deviation = close.rolling(20).std(ddof=0).replace(0, np.nan)
    features["bollinger"] = (close - features["ma20"]) / deviation
    features["bollinger_previous"] = features["bollinger"].shift(1)
    location = (2 * close - high - low) / (high - low).replace(0, np.nan)
    features["close_pressure"] = -(location * amount).rolling(5).mean() / features["amount20"]
    for window in (20, 60):
        features[f"channel{window}"] = high.shift(1).rolling(window).max()
        features[f"exit{window // 2}"] = low.shift(1).rolling(window // 2).min()
    for period in (2, 14):
        features[f"rsi{period}"] = wilder_rsi(close, period)
        features[f"rsi{period}_previous"] = features[f"rsi{period}"].shift(1)
    return {name: frame.astype(np.float32) for name, frame in features.items()}


class PreparedFeatures:
    def __init__(self, panels):
        self.panels = panels

    def at(self, context):
        return pd.DataFrame({name: frame.loc[context.date] for name, frame in self.panels.items()})


class OnlineFeatures:
    def __init__(self):
        self.date = None
        self.states = {}
        self.previous_values = {}

    def at(self, context):
        close = context.history("close", 100_000 if self.date is None else 253, adjusted=True)
        if self.date is not None and self.date not in close.index:
            raise ValueError("技术指标状态不能跳过超过252个交易日，请重新创建策略")
        fresh = close if self.date is None else close.loc[close.index > self.date]
        if not self.states:
            self.states = {period: WilderRSI(period, close.shape[1]) for period in (2, 14)}
        for prices in fresh.to_numpy():
            for period, state in self.states.items():
                self.previous_values[period] = state.value.copy()
                state.update(prices)
        self.date = context.date
        close = close.tail(253)
        amount = context.history("amt", len(close))
        high = context.history("high", len(close), adjusted=True)
        low = context.history("low", len(close), adjusted=True)
        latest = close.iloc[-1]
        features = {"close": latest, "history120": close.tail(120).notna().all().astype(float) if len(close) >= 120 else latest * np.nan}
        for window in (5, 20, 60, 120, 200):
            features[f"ma{window}"] = close.tail(window).mean().where(close.tail(window).notna().all()) if len(close) >= window else latest * np.nan
        returns = close.pct_change(fill_method=None)
        for window in (20, 60):
            features[f"vol{window}"] = returns.tail(window).std(ddof=1).where(returns.tail(window).notna().all()) if len(close) > window else latest * np.nan
        for window in (5, 20):
            features[f"reversal{window}"] = -(latest / close.iloc[-window - 1] - 1) if len(close) > window else latest * np.nan
        features["momentum60"] = close.iloc[-6] / close.iloc[-66] - 1 if len(close) >= 66 else latest * np.nan
        features["momentum120"] = close.iloc[-21] / close.iloc[-141] - 1 if len(close) >= 141 else latest * np.nan
        features["amount20"] = amount.tail(20).mean().where(amount.tail(20).gt(0).all()) if len(amount) >= 20 else latest * np.nan
        features["amount_shock"] = amount.tail(5).mean().where(amount.tail(5).notna().all()) / features["amount20"] if len(amount) >= 5 else latest * np.nan
        deviation = close.tail(20).std(ddof=0).replace(0, np.nan).where(close.tail(20).notna().all()) if len(close) >= 20 else latest * np.nan
        features["bollinger"] = (latest - features["ma20"]) / deviation
        previous = close.iloc[:-1].tail(20)
        features["bollinger_previous"] = (previous.iloc[-1] - previous.mean()) / previous.std(ddof=0).replace(0, np.nan).where(previous.notna().all()) if len(previous) == 20 else latest * np.nan
        location = (2 * close.tail(5) - high.tail(5) - low.tail(5)) / (high.tail(5) - low.tail(5)).replace(0, np.nan)
        features["close_pressure"] = -(location * amount.tail(5)).mean().where(location.notna().all()) / features["amount20"] if len(close) >= 5 else latest * np.nan
        for window in (20, 60):
            features[f"channel{window}"] = high.iloc[:-1].tail(window).max().where(high.iloc[:-1].tail(window).notna().all()) if len(high) > window else latest * np.nan
            features[f"exit{window // 2}"] = low.iloc[:-1].tail(window // 2).min().where(low.iloc[:-1].tail(window // 2).notna().all()) if len(low) > window // 2 else latest * np.nan
        for period, state in self.states.items():
            features[f"rsi{period}"] = pd.Series(state.value, index=close.columns)
            features[f"rsi{period}_previous"] = pd.Series(self.previous_values[period], index=close.columns)
        return pd.DataFrame(features).astype(np.float32)


class TechnicalStrategy(Strategy):
    name = "技术指标选股"

    def __init__(self, mode="rsi_rebound", top_n=100, rebalance_days=1, investment=0.95,
                 minimum_amount=100_000_000, liquid_pool=1500, lookback=20,
                 entry_threshold=30, exit_threshold=60, max_hold=10,
                 trend_window=0, blend_weight=0.5):
        for value in (top_n, rebalance_days, liquid_pool, lookback, max_hold):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or int(value) != value or value < 1:
                raise ValueError("数量、窗口和持有期必须为正整数")
        numeric = (investment, minimum_amount, entry_threshold, exit_threshold, blend_weight)
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("参数必须为有限数值")
        if mode not in MODES or lookback not in (5, 20, 60, 120) or trend_window not in (0, 60, 120, 200):
            raise ValueError("不支持的技术策略或窗口")
        if not 0 < investment <= 1 or minimum_amount < 0 or top_n > liquid_pool or not 0 <= blend_weight <= 1:
            raise ValueError("仓位、流动性或组合权重无效")
        if not 0 <= entry_threshold < exit_threshold <= 100:
            raise ValueError("RSI阈值必须满足0≤买入阈值<卖出阈值≤100")
        if mode == "breakout" and lookback not in (20, 60):
            raise ValueError("突破窗口只能为20或60")
        if mode in {"momentum", "lowvol_momentum"} and lookback not in (60, 120):
            raise ValueError("动量窗口只能为60或120")
        if mode == "low_volatility" and lookback not in (20, 60):
            raise ValueError("低波动窗口只能为20或60")
        if mode == "ma_trend" and lookback not in (20, 60):
            raise ValueError("均线窗口只能为20或60")
        if mode == "reversal" and lookback not in (5, 20):
            raise ValueError("反转窗口只能为5或20")
        self.mode, self.top_n = mode, int(top_n)
        self.rebalance_days, self.liquid_pool = int(rebalance_days), int(liquid_pool)
        self.investment, self.minimum_amount = float(investment), float(minimum_amount)
        self.lookback, self.max_hold = int(lookback), int(max_hold)
        self.entry_threshold, self.exit_threshold = float(entry_threshold), float(exit_threshold)
        self.trend_window, self.blend_weight = trend_window, blend_weight
        self.counter = 0
        self.ages = {}
        self.features = OnlineFeatures()

    def on_close(self, context):
        held = {symbol for symbol, shares in context.positions.items() if shares >= 1}
        self.ages = {symbol: self.ages.get(symbol, -1) + 1 for symbol in held}
        rebalance = self.counter % self.rebalance_days == 0
        self.counter += 1
        if not rebalance:
            return None
        frame = self.features.at(context)
        eligible = frame.reindex(context.universe())
        eligible = eligible.loc[(eligible.history120 == 1) & (eligible.amount20 >= self.minimum_amount)]
        liquid = eligible.amount20.sort_index().nlargest(self.liquid_pool).index
        eligible = eligible.reindex(liquid)
        if self.trend_window:
            eligible = eligible.loc[eligible.close > eligible[f"ma{self.trend_window}"]]
        if self.mode in {"rsi_rebound", "rsi_pullback", "bollinger_rebound", "breakout"}:
            return self._event_targets(frame, eligible, held)
        if eligible.empty:
            return {}
        volatility = -eligible.vol20
        reversal = eligible.reversal5
        momentum = eligible[f"momentum{self.lookback}"] if self.lookback in (60, 120) else eligible.momentum60
        if self.mode == "reversal":
            score = eligible[f"reversal{self.lookback}"]
        elif self.mode == "momentum":
            score = momentum.where(momentum > 0)
        elif self.mode == "ma_trend":
            slow = 60 if self.lookback == 20 else 120
            score = (eligible[f"ma{self.lookback}"] / eligible[f"ma{slow}"] - 1).where(eligible[f"ma{self.lookback}"] > eligible[f"ma{slow}"])
        elif self.mode == "low_volatility":
            score = -eligible[f"vol{self.lookback}"]
        elif self.mode in {"lowvol_reversal", "defensive_regime"}:
            score = self.blend_weight * volatility.rank(pct=True) + (1 - self.blend_weight) * reversal.rank(pct=True)
        elif self.mode == "lowvol_momentum":
            score = self.blend_weight * volatility.rank(pct=True) + (1 - self.blend_weight) * momentum.rank(pct=True)
        elif self.mode == "amount_reversal":
            score = reversal * eligible.amount_shock.clip(0.5, 2)
        elif self.mode == "close_pressure":
            score = eligible.close_pressure
        else:
            score = eligible.amount20
        selected = score.replace([np.inf, -np.inf], np.nan).dropna().sort_index().nlargest(self.top_n).index
        exposure = self.investment
        if self.mode == "defensive_regime":
            breadth = (eligible.close > eligible.ma60).mean()
            exposure *= 1 if breadth >= 0.5 else 0.5
        return {symbol: exposure / self.top_n for symbol in selected}

    def _event_targets(self, frame, eligible, held):
        survivors = []
        for symbol in sorted(held):
            if symbol not in eligible.index or self.ages[symbol] >= self.max_hold:
                continue
            row = frame.loc[symbol]
            if self.mode.startswith("rsi"):
                period = 2 if self.mode == "rsi_pullback" else 14
                exit_signal = row[f"rsi{period}"] >= self.exit_threshold
                if self.mode == "rsi_pullback":
                    exit_signal = exit_signal or row.close > row.ma5
            elif self.mode == "bollinger_rebound":
                exit_signal = row.close >= row.ma20
            else:
                exit_signal = row.close < row[f"exit{self.lookback // 2}"]
            if not exit_signal:
                survivors.append(symbol)
        if self.mode == "rsi_rebound":
            candidates = eligible.loc[(eligible.rsi14_previous <= self.entry_threshold) & (eligible.rsi14 > self.entry_threshold)]
            score = candidates.amount20
        elif self.mode == "rsi_pullback":
            candidates = eligible.loc[eligible.rsi2 < self.entry_threshold]
            score = -candidates.rsi2
        elif self.mode == "bollinger_rebound":
            candidates = eligible.loc[(eligible.bollinger_previous <= -2) & (eligible.bollinger > -2)]
            score = candidates.amount20
        else:
            candidates = eligible.loc[eligible.close > eligible[f"channel{self.lookback}"]]
            score = candidates.close / candidates[f"channel{self.lookback}"] - 1
        score = score.drop(index=held, errors="ignore").dropna().sort_index()
        slots = max(0, self.top_n - len(survivors))
        selected = survivors[:self.top_n] + score.nlargest(slots).index.tolist()
        return {symbol: self.investment / self.top_n for symbol in selected}
