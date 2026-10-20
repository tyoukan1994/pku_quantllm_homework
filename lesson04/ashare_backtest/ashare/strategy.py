from abc import ABC, abstractmethod
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import MappingProxyType

import numpy as np
import pandas as pd


class Strategy(ABC):
    def __init__(self, **parameters):
        self.parameters = parameters

    @abstractmethod
    def on_close(self, context):
        raise NotImplementedError


class Context:
    def __init__(self, data, position, config, cash, equity, positions):
        self.__data = data
        self.__position = position
        self.__config = config
        self.date = data.dates[position]
        self.cash = cash
        self.equity = equity
        self.positions = MappingProxyType(dict(positions))

    def history(self, field="close", bars=60, symbols=None, adjusted=False):
        return self.__data.history(self.__position, field, bars, symbols, adjusted)

    def universe(self):
        position = self.__position
        data = self.__data
        valid = data.history_counts[position] >= self.__config.min_history
        valid &= np.isfinite(data.arrays["close"][position])
        valid &= np.isfinite(data.arrays["cumadj"][position])
        valid &= data.arrays["amt"][position] > 0
        candidates = [symbol for symbol, usable in zip(data.symbols, valid) if usable]
        result = []
        for symbol in candidates:
            state = data.state(self.date, symbol)
            instrument = data.instruments.get(symbol, {})
            listing = instrument.get("list_date")
            delisting = instrument.get("delist_date")
            if listing and self.date < listing:
                continue
            if delisting and self.date >= delisting:
                continue
            if state.get("suspended") or state.get("can_buy") is False:
                continue
            if self.__config.exclude_st and state.get("is_st"):
                continue
            if self.__config.require_st_history and state.get("is_st") is not False:
                continue
            result.append(symbol)
        return result


def load_strategy(path, parameters=None):
    path = Path(path).expanduser().resolve()
    name = "user_strategy_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("无法加载策略文件")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    strategy_class = getattr(module, "UserStrategy", None)
    if not isinstance(strategy_class, type) or not issubclass(strategy_class, Strategy):
        raise ValueError("策略文件需要定义继承 Strategy 的 UserStrategy")
    return strategy_class(**(parameters or {}))


def validate_targets(targets, symbols, maximum_weight):
    if targets is None:
        return None
    if not isinstance(targets, (dict, pd.Series)):
        raise ValueError("on_close 必须返回权重字典、Series 或 None")
    if isinstance(targets, pd.Series) and targets.index.has_duplicates:
        raise ValueError("目标权重的股票代码不能重复")
    result = {}
    allowed = set(symbols)
    for symbol, weight in dict(targets).items():
        weight = float(weight)
        if symbol not in allowed:
            raise ValueError(f"目标股票不在数据中：{symbol}")
        if not np.isfinite(weight) or weight < 0 or weight > maximum_weight + 1e-10:
            raise ValueError(f"{symbol} 的目标权重无效：{weight}")
        if weight > 0:
            result[symbol] = weight
    if sum(result.values()) > 1 + 1e-10:
        raise ValueError("目标权重之和不能超过 1，不支持融资或做空")
    return result
