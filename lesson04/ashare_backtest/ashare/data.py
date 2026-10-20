from pathlib import Path
import re

import numpy as np
import pandas as pd


REQUIRED_FIELDS = ("open", "high", "low", "close", "amt", "cumadj")
PRICE_FIELDS = {"open", "high", "low", "close"}


def parse_bool(value):
    if pd.isna(value):
        return None
    if str(value).strip().lower() in {"1", "true", "yes"}:
        return True
    if str(value).strip().lower() in {"0", "false", "no"}:
        return False
    raise ValueError(f"布尔值只能为 true/false 或 1/0，收到 {value!r}")


class DataBundle:
    def __init__(self, panels, states=None, instruments=None, actions=None, source="memory"):
        self.source = str(source)
        self._panels = {}
        for field in REQUIRED_FIELDS:
            if field not in panels:
                raise ValueError(f"缺少数据表 {field}")
        for field, frame in panels.items():
            frame = frame.copy()
            if "date" in frame.columns:
                frame = frame.set_index("date")
            frame.index = pd.DatetimeIndex(frame.index, name="date")
            if frame.index.tz or frame.index.hasnans or frame.index.has_duplicates:
                raise ValueError(f"{field} 的日期必须唯一、有效且无时区")
            if not frame.index.is_monotonic_increasing or not frame.index.equals(frame.index.normalize()):
                raise ValueError(f"{field} 必须按日升序排列")
            if frame.columns.has_duplicates or frame.empty:
                raise ValueError(f"{field} 股票列重复或数据为空")
            frame = frame.astype(float)
            values = frame.to_numpy()
            if np.isinf(values).any() or (values < 0).any():
                raise ValueError(f"{field} 存在无穷值或负数")
            if field in PRICE_FIELDS | {"cumadj"} and (values == 0).any():
                raise ValueError(f"{field} 存在零值")
            self._panels[field] = frame
        reference = self._panels["close"]
        self.dates = reference.index
        self.date_index = {date: position for position, date in enumerate(self.dates)}
        self.symbols = list(reference.columns)
        self.symbol_index = {symbol: position for position, symbol in enumerate(self.symbols)}
        for symbol in self.symbols:
            if not re.fullmatch(r"(?:00\d{4}|30\d{4})\.SZ|(?:60\d{4}|68\d{4})\.SH", symbol):
                raise ValueError(f"不支持的沪深 A 股代码：{symbol}")
        for field, frame in self._panels.items():
            if not frame.index.equals(self.dates) or not frame.columns.equals(reference.columns):
                raise ValueError(f"{field} 与 close 的日期或股票列不一致")
        self.arrays = {field: frame.to_numpy() for field, frame in self._panels.items()}
        complete = np.isfinite(self.arrays["open"]) & np.isfinite(self.arrays["high"])
        complete &= np.isfinite(self.arrays["low"]) & np.isfinite(self.arrays["close"])
        invalid = complete & (
            (self.arrays["high"] < np.maximum(self.arrays["open"], self.arrays["close"]))
            | (self.arrays["low"] > np.minimum(self.arrays["open"], self.arrays["close"]))
            | (self.arrays["high"] < self.arrays["low"])
        )
        if invalid.any():
            raise ValueError("价格表存在 OHLC 关系冲突")
        valid = np.isfinite(self.arrays["close"]) & np.isfinite(self.arrays["cumadj"])
        self.history_counts = np.cumsum(valid, axis=0, dtype=np.int32)
        self.states = self._states(states)
        self.status_arrays = {}
        self.metadata_manifest = {}
        self.instruments = self._instruments(instruments)
        self.actions = self._actions(actions)

    @classmethod
    def from_feathers(cls, source, symbols=None, metadata_dir=None, end=None):
        source = Path(source).expanduser().resolve()
        columns = ["date"] + list(symbols) if symbols else None
        panels = {field: pd.read_feather(source / f"{field}.feather", columns=columns)
                  for field in REQUIRED_FIELDS}
        if (source / "num_trades.feather").exists():
            panels["num_trades"] = pd.read_feather(source / "num_trades.feather", columns=columns)
        if end is not None:
            panels = {field: frame.loc[pd.to_datetime(frame["date"]) <= pd.Timestamp(end)]
                      for field, frame in panels.items()}
        optional = {}
        for name in ("states", "instruments", "actions"):
            filename = {"states": "market_state.csv", "instruments": "instruments.csv",
                        "actions": "corporate_actions.csv"}[name]
            path = Path(metadata_dir).expanduser() / filename if metadata_dir else source / filename
            if not path.exists():
                path = source / filename
            optional[name] = pd.read_csv(path) if path.exists() else None
        result = cls(panels, source=source, **optional)
        if metadata_dir:
            result.add_metadata(metadata_dir)
        return result

    def add_metadata(self, directory):
        import json
        directory = Path(directory).expanduser()
        for field in ("is_st", "suspended"):
            path = directory / f"{field}.feather"
            if path.exists():
                frame = pd.read_feather(path).set_index("date")
                frame.index = pd.DatetimeIndex(frame.index)
                if frame.index.has_duplicates or frame.columns.has_duplicates:
                    raise ValueError(f"{field} 状态日期或代码重复")
                values = frame.reindex(index=self.dates, columns=self.symbols).astype(float).to_numpy()
                if (~(np.isnan(values) | (values == 0) | (values == 1))).any():
                    raise ValueError(f"{field} 只能包含布尔状态或缺失")
                self.status_arrays[field] = values
        path = directory / "manifest.json"
        if path.exists():
            self.metadata_manifest = json.loads(path.read_text(encoding="utf-8"))
        return self

    def _states(self, frame):
        if frame is None or frame.empty:
            return {}
        frame = frame.copy()
        if not {"date", "symbol"}.issubset(frame.columns):
            raise ValueError("market_state.csv 必须包含 date,symbol")
        frame["date"] = pd.to_datetime(frame["date"])
        if frame["date"].isna().any() or not frame["date"].equals(frame["date"].dt.normalize()):
            raise ValueError("market_state.csv 的日期必须为有效日日期")
        if frame.duplicated(["date", "symbol"]).any():
            raise ValueError("market_state.csv 的 date,symbol 重复")
        result = {}
        for record in frame.to_dict("records"):
            for field in ("is_st", "suspended", "no_limit", "can_buy", "can_sell"):
                record[field] = parse_bool(record.get(field))
            for field in ("limit_up", "limit_down"):
                value = record.get(field, np.nan)
                if pd.notna(value) and (not np.isfinite(float(value)) or float(value) <= 0):
                    raise ValueError(f"{field} 必须是有效正数")
            result[(record["date"], record["symbol"])] = record
        return result

    def _instruments(self, frame):
        if frame is None or frame.empty:
            return {}
        if "symbol" not in frame or frame["symbol"].duplicated().any():
            raise ValueError("instruments.csv 的 symbol 必须存在且唯一")
        result = {}
        for record in frame.to_dict("records"):
            for field in ("list_date", "delist_date"):
                record[field] = pd.Timestamp(record[field]) if pd.notna(record.get(field)) else None
                if record[field] is not None and (record[field].tz or record[field] != record[field].normalize()):
                    raise ValueError(f"{field} 必须使用无时区日日期，例如 2024-01-02")
            if record["list_date"] and record["delist_date"] and record["list_date"] >= record["delist_date"]:
                raise ValueError("上市日期必须早于退市日期")
            result[record["symbol"]] = record
        return result

    def _actions(self, frame):
        if frame is None or frame.empty:
            return {}
        needed = {"date", "symbol", "cash_per_share", "share_multiplier", "pay_date", "share_available_date"}
        if not needed.issubset(frame.columns):
            raise ValueError(f"corporate_actions.csv 必须包含 {sorted(needed)}")
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        if frame["date"].isna().any() or not frame["date"].equals(frame["date"].dt.normalize()):
            raise ValueError("corporate_actions.csv 的日期必须为有效日日期")
        if frame.duplicated(["date", "symbol"]).any():
            raise ValueError("同一股票同一天的公司行为请先合并为一条")
        result = {}
        for record in frame.to_dict("records"):
            for field in ("pay_date", "share_available_date"):
                record[field] = pd.Timestamp(record[field])
                if pd.isna(record[field]) or record[field] < record["date"]:
                    raise ValueError(f"{field} 不能早于除权除息日")
            for field in ("cash_per_share", "share_multiplier"):
                record[field] = float(record[field])
                if not np.isfinite(record[field]) or record[field] < 0:
                    raise ValueError("公司行为金额或股数倍率无效")
            if record["share_multiplier"] <= 0:
                raise ValueError("股数倍率必须大于零")
            result[(record["date"], record["symbol"])] = record
        return result

    def value(self, field, position, symbol):
        return float(self.arrays[field][position, self.symbol_index[symbol]])

    def state(self, date, symbol):
        date = pd.Timestamp(date)
        result = dict(self.states.get((date, symbol), {}))
        position = self.date_index.get(date, -1) if self.status_arrays else -1
        if position >= 0 and symbol in self.symbol_index:
            for field, values in self.status_arrays.items():
                value = values[position, self.symbol_index[symbol]]
                if np.isfinite(value):
                    result[field] = bool(value)
        return result

    def history(self, position, field, bars, symbols=None, adjusted=False):
        if field not in self._panels or not isinstance(bars, int) or bars < 1:
            raise ValueError("history 的字段或窗口长度无效")
        start = max(0, position - bars + 1)
        frame = self._panels[field].iloc[start:position + 1]
        if symbols is not None:
            frame = frame.loc[:, list(symbols)]
        if adjusted:
            if field not in PRICE_FIELDS:
                raise ValueError("只有价格字段可以复权")
            frame = frame * self._panels["cumadj"].iloc[start:position + 1].loc[:, frame.columns]
        return frame.copy()

    def summary(self):
        result = {"source": self.source, "start": str(self.dates[0].date()),
                  "end": str(self.dates[-1].date()), "dates": len(self.dates),
                  "symbols": len(self.symbols), "state_records": len(self.states),
                  "action_records": len(self.actions), "instrument_records": len(self.instruments)}
        result["latest_missing"] = {field: int(np.isnan(values[-1]).sum())
                                    for field, values in self.arrays.items()}
        valid = np.isfinite(self.arrays["close"])
        result["status_coverage"] = {field: {"known_price_days": int((np.isfinite(values) & valid).sum()),
                                             "missing_price_days": int((np.isnan(values) & valid).sum()),
                                             "true_price_days": int(((values == 1) & valid).sum())}
                                     for field, values in self.status_arrays.items()}
        result["metadata_manifest"] = self.metadata_manifest
        return result
