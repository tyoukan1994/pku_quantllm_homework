"""Pure-Python teaching signal. No broker access and no order placement."""

from collections import deque


class TickSignal:
    """Report price crossings of the *previous* tick window mean."""

    def __init__(self, window=10):
        if window < 2:
            raise ValueError("window must be at least 2")
        self.window = window
        self.prices = {}
        self.last_side = {}

    def update(self, vt_symbol, price):
        if not vt_symbol or price <= 0:
            return None

        history = self.prices.setdefault(vt_symbol, deque(maxlen=self.window))
        signal = None
        if len(history) == self.window:
            mean = sum(history) / self.window
            side = 1 if price > mean else -1 if price < mean else 0
            previous = self.last_side.get(vt_symbol)
            if previous is not None and side and side != previous:
                signal = "上穿" if side > 0 else "下穿"
            if side:
                self.last_side[vt_symbol] = side

        history.append(price)
        return signal

