"""Replaceable historical-bar data sources for factor research."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


REQUIRED_COLUMNS = ("datetime", "symbol", "open", "high", "low", "close", "volume")


class BarDataSource(Protocol):
    """Minimal interface that a CSV or Choice adapter must implement."""

    def load_bars(
        self, symbol: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Return canonical OHLCV bars ordered by datetime."""


def normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize bars without silently fabricating missing fields."""
    renamed = {str(column).strip().lower(): column for column in frame.columns}
    missing = [column for column in REQUIRED_COLUMNS if column not in renamed]
    if missing:
        raise ValueError(f"历史数据缺少字段：{', '.join(missing)}")

    bars = frame[[renamed[column] for column in REQUIRED_COLUMNS]].copy()
    bars.columns = list(REQUIRED_COLUMNS)
    bars["datetime"] = pd.to_datetime(bars["datetime"], errors="raise")
    bars["symbol"] = bars["symbol"].astype(str).str.strip()
    for column in ("open", "high", "low", "close", "volume"):
        bars[column] = pd.to_numeric(bars[column], errors="raise")

    if bars.duplicated(subset=["symbol", "datetime"]).any():
        raise ValueError("历史数据含同一品种、同一 datetime 的重复记录；请先说明去重规则。")
    if (bars[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC 价格必须大于 0。")
    if (bars["volume"] < 0).any():
        raise ValueError("成交量不能为负数。")

    return bars.sort_values("datetime").reset_index(drop=True)


@dataclass(frozen=True)
class CsvBarDataSource:
    """Load canonical daily bars from a local CSV export."""

    path: Path

    def load_bars(
        self, symbol: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        bars = normalize_bars(pd.read_csv(self.path))
        bars = bars[bars["symbol"].str.casefold() == symbol.casefold()]
        if start:
            bars = bars[bars["datetime"] >= pd.Timestamp(start)]
        if end:
            bars = bars[bars["datetime"] <= pd.Timestamp(end)]
        if bars.empty:
            raise ValueError(f"CSV 中没有 {symbol!r} 在指定区间的数据。")
        return bars.reset_index(drop=True)


@dataclass
class JqDataSource:
    """Load historical bars from JQData without persisting credentials."""

    username: str | None = None
    password: str | None = field(default=None, repr=False)
    frequency: str = "daily"
    sdk: Any | None = field(default=None, repr=False)
    _authenticated: bool = field(default=False, init=False, repr=False)

    def _get_sdk(self) -> Any:
        if self.sdk is None:
            try:
                import jqdatasdk
            except ImportError as exc:
                raise RuntimeError(
                    "尚未安装聚宽 SDK，请在项目虚拟环境运行 "
                    "pip install jqdatasdk。"
                ) from exc
            self.sdk = jqdatasdk
        return self.sdk

    def _ensure_authenticated(self) -> Any:
        sdk = self._get_sdk()
        if self._authenticated:
            return sdk

        if self.username and self.password:
            sdk.auth(self.username, self.password)
        elif not sdk.is_auth():
            raise RuntimeError("聚宽尚未登录，请先安全输入账号和密码。")

        self._authenticated = True
        return sdk

    def load_bars(
        self, symbol: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Query one JQData symbol and return canonical unadjusted OHLCV bars."""
        sdk = self._ensure_authenticated()
        raw = sdk.get_price(
            security=symbol,
            start_date=start,
            end_date=end,
            frequency=self.frequency,
            fields=["open", "high", "low", "close", "volume"],
            skip_paused=True,
            fq=None,
            panel=False,
        )
        if raw is None or raw.empty:
            raise ValueError(f"聚宽没有返回 {symbol!r} 在指定区间的数据。")

        frame = raw.copy()
        if "time" in frame.columns:
            datetimes = frame["time"].to_numpy()
        elif "date" in frame.columns:
            datetimes = frame["date"].to_numpy()
        else:
            datetimes = frame.index.to_numpy()
        frame = frame.reset_index(drop=True)
        frame["datetime"] = datetimes

        if "code" in frame.columns:
            frame["symbol"] = frame["code"].astype(str)
        else:
            frame["symbol"] = symbol

        return normalize_bars(frame)
