"""Pure utilities for monthly Alpha158-style machine-learning research.

The feature names and formulas follow :class:`vnpy.alpha.dataset.datasets.Alpha158`,
but are evaluated only at monthly signal dates.  This avoids materialising a
158-column daily panel while preserving the official vn.py definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd


ALPHA_WINDOWS: tuple[int, ...] = (5, 10, 20, 30, 60)
ALPHA_BASE_FEATURES: tuple[str, ...] = (
    "kmid",
    "klen",
    "kmid_2",
    "kup",
    "kup_2",
    "klow",
    "klow_2",
    "ksft",
    "ksft_2",
    "open_0",
    "high_0",
    "low_0",
    "vwap_0",
)
ALPHA_WINDOW_FEATURES: tuple[str, ...] = (
    "roc",
    "ma",
    "std",
    "beta",
    "rsqr",
    "resi",
    "max",
    "min",
    "qtlu",
    "qtld",
    "rank",
    "rsv",
    "imax",
    "imin",
    "imxd",
    "corr",
    "cord",
    "cntp",
    "cntn",
    "cntd",
    "sump",
    "sumn",
    "sumd",
    "vma",
    "vstd",
    "wvma",
    "vsump",
    "vsumn",
    "vsumd",
)
ALPHA158_FEATURES: tuple[str, ...] = ALPHA_BASE_FEATURES + tuple(
    f"{name}_{window}"
    for name in ALPHA_WINDOW_FEATURES
    for window in ALPHA_WINDOWS
)
COURSE_FEATURES: tuple[str, ...] = (
    "pv_anti_bare",
    "pv_anti_pair",
    "pv_anti_triple",
    "gap_diff4_bare",
)
FUNDAMENTAL_FEATURES: tuple[str, ...] = (
    "fund_value",
    "fund_size",
    "fund_quality",
    "fund_rsi",
)


@dataclass(frozen=True)
class Fold:
    """Expanding-window development fold."""

    train_start: int
    train_end: int
    valid_year: int
    test_year: int


def adjusted_vwap(
    pre_close: pd.Series,
    raw_close: pd.Series,
    raw_volume: pd.Series,
    raw_turnover: pd.Series,
) -> pd.Series:
    """Map raw VWAP onto the same pre-adjusted scale as OHLC."""
    factor = pre_close / raw_close.replace(0, np.nan)
    return raw_turnover / raw_volume.replace(0, np.nan) * factor


def expanding_folds(
    first_train_year: int = 2015,
    first_test_year: int = 2020,
    last_test_year: int = 2024,
) -> list[Fold]:
    """Return folds with the year before each test year used for validation."""
    return [
        Fold(first_train_year, test_year - 2, test_year - 1, test_year)
        for test_year in range(first_test_year, last_test_year + 1)
    ]


def _safe_div(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan
    if abs(denominator) < 1e-12:
        return np.nan
    return float(numerator / denominator)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 2:
        return np.nan
    x = left[mask]
    y = right[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _window(values: np.ndarray, end: int, size: int) -> np.ndarray | None:
    start = end - size + 1
    if start < 0:
        return None
    result = values[start : end + 1]
    if result.size != size:
        return None
    return result


def _regression_stats(values: np.ndarray) -> tuple[float, float, float]:
    """Return slope, r-squared and last-point residual."""
    if not np.isfinite(values).all() or values.size < 2:
        return np.nan, np.nan, np.nan
    x = np.arange(values.size, dtype=float)
    x_mean = x.mean()
    y_mean = values.mean()
    denominator = np.square(x - x_mean).sum()
    if denominator < 1e-12:
        return np.nan, np.nan, np.nan
    slope = float(((x - x_mean) * (values - y_mean)).sum() / denominator)
    intercept = float(y_mean - slope * x_mean)
    residual = float(values[-1] - (intercept + slope * x[-1]))
    variance = np.square(values - y_mean).sum()
    if variance < 1e-12:
        r_squared = np.nan
    else:
        fitted = intercept + slope * x
        r_squared = float(1 - np.square(values - fitted).sum() / variance)
    return slope, r_squared, residual


def _percentile_rank_current(values: np.ndarray) -> float:
    """Match scipy percentileofscore(..., kind='rank') for the last value."""
    if not np.isfinite(values).all():
        return np.nan
    score = values[-1]
    less = int(np.sum(values < score))
    equal = int(np.sum(values == score))
    return float((less + (equal + 1) / 2) / len(values))


def _alpha_row(group: pd.DataFrame, position: int) -> dict[str, float]:
    arrays = {
        name: pd.to_numeric(group[name], errors="coerce").to_numpy(dtype=float)
        for name in ("open", "high", "low", "close", "volume", "vwap", "total_turnover")
    }
    open_ = arrays["open"]
    high = arrays["high"]
    low = arrays["low"]
    close = arrays["close"]
    volume = arrays["volume"]
    vwap = arrays["vwap"]

    o = open_[position]
    h = high[position]
    l = low[position]
    c = close[position]
    v = volume[position]
    vw = vwap[position]
    span = h - l
    body_high = max(o, c)
    body_low = min(o, c)

    result: dict[str, float] = {
        "kmid": _safe_div(c - o, o),
        "klen": _safe_div(span, o),
        "kmid_2": _safe_div(c - o, span + 1e-12),
        "kup": _safe_div(h - body_high, o),
        "kup_2": _safe_div(h - body_high, span + 1e-12),
        "klow": _safe_div(body_low - l, o),
        "klow_2": _safe_div(body_low - l, span + 1e-12),
        "ksft": _safe_div(c * 2 - h - l, o),
        "ksft_2": _safe_div(c * 2 - h - l, span + 1e-12),
        "open_0": _safe_div(o, c),
        "high_0": _safe_div(h, c),
        "low_0": _safe_div(l, c),
        "vwap_0": _safe_div(vw, c),
    }

    price_return = np.full_like(close, np.nan)
    volume_ratio_log = np.full_like(volume, np.nan)
    close_ratio = np.full_like(close, np.nan)
    close_delta = np.full_like(close, np.nan)
    volume_delta = np.full_like(volume, np.nan)
    if close.size > 1:
        valid_close = np.isfinite(close[1:]) & np.isfinite(close[:-1]) & (close[:-1] != 0)
        close_ratio[1:][valid_close] = close[1:][valid_close] / close[:-1][valid_close]
        price_return[1:][valid_close] = close_ratio[1:][valid_close] - 1
        close_delta[1:] = close[1:] - close[:-1]
        valid_volume = np.isfinite(volume[1:]) & np.isfinite(volume[:-1]) & (volume[:-1] > 0)
        raw_volume_ratio = np.full(volume.size - 1, np.nan)
        raw_volume_ratio[valid_volume] = volume[1:][valid_volume] / volume[:-1][valid_volume]
        volume_ratio_log[1:] = np.log(raw_volume_ratio + 1)
        volume_delta[1:] = volume[1:] - volume[:-1]

    for size in ALPHA_WINDOWS:
        close_w = _window(close, position, size)
        high_w = _window(high, position, size)
        low_w = _window(low, position, size)
        volume_w = _window(volume, position, size)
        ratio_w = _window(close_ratio, position, size)
        volume_ratio_w = _window(volume_ratio_log, position, size)
        close_delta_w = _window(close_delta, position, size)
        volume_delta_w = _window(volume_delta, position, size)
        return_w = _window(price_return, position, size)

        if position >= size:
            result[f"roc_{size}"] = _safe_div(close[position - size], c)
        else:
            result[f"roc_{size}"] = np.nan

        if close_w is None or not np.isfinite(close_w).all():
            for name in (
                "ma", "std", "beta", "rsqr", "resi", "qtlu", "qtld", "rank"
            ):
                result[f"{name}_{size}"] = np.nan
        else:
            slope, r_squared, residual = _regression_stats(close_w)
            result[f"ma_{size}"] = _safe_div(float(np.mean(close_w)), c)
            result[f"std_{size}"] = _safe_div(float(np.std(close_w, ddof=0)), c)
            result[f"beta_{size}"] = _safe_div(slope, c)
            result[f"rsqr_{size}"] = r_squared
            result[f"resi_{size}"] = _safe_div(residual, c)
            result[f"qtlu_{size}"] = _safe_div(float(np.quantile(close_w, 0.8)), c)
            result[f"qtld_{size}"] = _safe_div(float(np.quantile(close_w, 0.2)), c)
            result[f"rank_{size}"] = _percentile_rank_current(close_w)

        if high_w is None or not np.isfinite(high_w).all():
            result[f"max_{size}"] = np.nan
            result[f"imax_{size}"] = np.nan
        else:
            result[f"max_{size}"] = _safe_div(float(np.max(high_w)), c)
            result[f"imax_{size}"] = float((int(np.argmax(high_w)) + 1) / size)

        if low_w is None or not np.isfinite(low_w).all():
            result[f"min_{size}"] = np.nan
            result[f"imin_{size}"] = np.nan
        else:
            result[f"min_{size}"] = _safe_div(float(np.min(low_w)), c)
            result[f"imin_{size}"] = float((int(np.argmin(low_w)) + 1) / size)

        if high_w is None or low_w is None or not (
            np.isfinite(high_w).all() and np.isfinite(low_w).all()
        ):
            result[f"rsv_{size}"] = np.nan
            result[f"imxd_{size}"] = np.nan
        else:
            result[f"rsv_{size}"] = _safe_div(
                c - float(np.min(low_w)),
                float(np.max(high_w) - np.min(low_w)) + 1e-12,
            )
            result[f"imxd_{size}"] = float(
                (int(np.argmax(high_w)) - int(np.argmin(low_w))) / size
            )

        if close_w is None or volume_w is None:
            result[f"corr_{size}"] = np.nan
        else:
            result[f"corr_{size}"] = _safe_corr(close_w, np.log(volume_w + 1))
        result[f"cord_{size}"] = (
            _safe_corr(ratio_w, volume_ratio_w)
            if ratio_w is not None and volume_ratio_w is not None
            else np.nan
        )

        if close_delta_w is None:
            for name in ("cntp", "cntn", "cntd", "sump", "sumn", "sumd"):
                result[f"{name}_{size}"] = np.nan
        else:
            valid = np.isfinite(close_delta_w)
            if not valid.all():
                for name in ("cntp", "cntn", "cntd", "sump", "sumn", "sumd"):
                    result[f"{name}_{size}"] = np.nan
            else:
                positive = np.maximum(close_delta_w, 0)
                negative = np.maximum(-close_delta_w, 0)
                absolute_sum = float(np.abs(close_delta_w).sum())
                result[f"cntp_{size}"] = float(np.mean(close_delta_w > 0))
                result[f"cntn_{size}"] = float(np.mean(close_delta_w < 0))
                result[f"cntd_{size}"] = result[f"cntp_{size}"] - result[f"cntn_{size}"]
                result[f"sump_{size}"] = _safe_div(float(positive.sum()), absolute_sum + 1e-12)
                result[f"sumn_{size}"] = _safe_div(float(negative.sum()), absolute_sum + 1e-12)
                result[f"sumd_{size}"] = _safe_div(
                    float(positive.sum() - negative.sum()), absolute_sum + 1e-12
                )

        if volume_w is None or not np.isfinite(volume_w).all():
            result[f"vma_{size}"] = np.nan
            result[f"vstd_{size}"] = np.nan
        else:
            result[f"vma_{size}"] = _safe_div(float(np.mean(volume_w)), v)
            result[f"vstd_{size}"] = _safe_div(float(np.std(volume_w, ddof=0)), v)

        if return_w is None or volume_w is None:
            result[f"wvma_{size}"] = np.nan
        else:
            weighted = np.abs(return_w) * volume_w
            result[f"wvma_{size}"] = _safe_div(
                float(np.nanstd(weighted, ddof=0)),
                float(np.nanmean(weighted)) + 1e-12,
            )

        if volume_delta_w is None or not np.isfinite(volume_delta_w).all():
            for name in ("vsump", "vsumn", "vsumd"):
                result[f"{name}_{size}"] = np.nan
        else:
            positive_volume = np.maximum(volume_delta_w, 0)
            negative_volume = np.maximum(-volume_delta_w, 0)
            absolute_volume = float(np.abs(volume_delta_w).sum())
            result[f"vsump_{size}"] = _safe_div(
                float(positive_volume.sum()), absolute_volume + 1e-12
            )
            result[f"vsumn_{size}"] = _safe_div(
                float(negative_volume.sum()), absolute_volume + 1e-12
            )
            result[f"vsumd_{size}"] = _safe_div(
                float(positive_volume.sum() - negative_volume.sum()),
                absolute_volume + 1e-12,
            )

    # Course formula factors, evaluated without any future data.
    ret5 = _window(price_return, position, 5)
    dvol = np.full_like(volume, np.nan)
    if volume.size > 1:
        valid = np.isfinite(volume[1:]) & np.isfinite(volume[:-1]) & (volume[:-1] > 0)
        dvol[1:][valid] = volume[1:][valid] / volume[:-1][valid] - 1
    dvol5 = _window(dvol, position, 5)
    pv_anti = -_safe_corr(ret5, dvol5) if ret5 is not None and dvol5 is not None else np.nan
    amp_high = _window(high, position, 5)
    amp_low = _window(low, position, 5)
    if amp_high is None or amp_low is None:
        amp5 = np.nan
    else:
        amp5 = float(np.nanmean(amp_high / amp_low - 1))

    gap_values = np.full_like(close, np.nan)
    if close.size > 1:
        valid = (
            np.isfinite(open_[1:])
            & np.isfinite(close[:-1])
            & np.isfinite(close[1:])
            & (close[:-1] != 0)
            & (open_[1:] != 0)
        )
        gap_values[1:][valid] = (
            open_[1:][valid] / close[:-1][valid] - 1
        ) - (close[1:][valid] / open_[1:][valid] - 1)
    gap4 = _window(gap_values, position, 4)

    result["pv_anti_bare"] = pv_anti
    result["amp5"] = amp5
    result["gap_diff4_bare"] = (
        float(np.sum(gap4)) if gap4 is not None and np.isfinite(gap4).all() else np.nan
    )
    result["signal_turnover"] = arrays["total_turnover"][position]
    return result


def calculate_monthly_features(
    daily: pd.DataFrame,
    signal_members: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate Alpha158 and course factors for exact member/date pairs."""
    required_daily = {
        "datetime", "symbol", "open", "high", "low", "close", "volume",
        "total_turnover", "vwap",
    }
    missing = required_daily.difference(daily.columns)
    if missing:
        raise ValueError(f"日频行情缺少字段：{sorted(missing)}")
    members = signal_members[["symbol", "signal_date"]].drop_duplicates().copy()
    members["signal_date"] = pd.to_datetime(members["signal_date"])
    wanted = members.groupby("symbol")["signal_date"].apply(set).to_dict()

    rows: list[dict[str, object]] = []
    ordered = daily.copy()
    ordered["datetime"] = pd.to_datetime(ordered["datetime"])
    ordered = ordered.sort_values(["symbol", "datetime"])
    for symbol, group in ordered.groupby("symbol", sort=False):
        dates = wanted.get(symbol)
        if not dates:
            continue
        group = group.reset_index(drop=True)
        positions = {value: index for index, value in enumerate(group["datetime"])}
        for signal_date in sorted(dates):
            position = positions.get(signal_date)
            if position is None:
                continue
            values = _alpha_row(group, position)
            rows.append({"symbol": symbol, "signal_date": signal_date, **values})

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["turnover_rank"] = result.groupby("signal_date")["signal_turnover"].rank(
        method="average", pct=True
    )
    result["pv_anti_pair"] = result["pv_anti_bare"] * result["turnover_rank"]
    result["pv_anti_triple"] = result["pv_anti_pair"] * result["amp5"]
    result = result.drop(columns=["amp5", "signal_turnover", "turnover_rank"])
    expected = {"symbol", "signal_date", *ALPHA158_FEATURES, *COURSE_FEATURES}
    missing_features = expected.difference(result.columns)
    if missing_features:
        raise AssertionError(f"未生成完整特征：{sorted(missing_features)}")
    return result[["symbol", "signal_date", *ALPHA158_FEATURES, *COURSE_FEATURES]]


def cross_sectional_rank_label(frame: pd.DataFrame) -> pd.Series:
    """Create the same centred rank label used in the vn.py Alpha workflow."""
    rank = frame.groupby("signal_date")["forward_return"].rank(
        method="average", pct=True
    )
    return (rank - 0.5) * 3.46


def robust_scale(
    train: pd.DataFrame,
    *others: pd.DataFrame,
    columns: Iterable[str],
) -> tuple[pd.DataFrame, ...]:
    """Fit median/MAD scaling on train only and apply it to all supplied frames."""
    names = list(columns)
    train_values = train[names].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    median = np.nanmedian(train_values, axis=0)
    mad = np.nanmedian(np.abs(train_values - median), axis=0) * 1.4826
    mad = np.where(np.isfinite(mad) & (mad > 1e-12), mad, 1.0)
    median = np.where(np.isfinite(median), median, 0.0)

    transformed: list[pd.DataFrame] = []
    for frame in (train, *others):
        result = frame.copy()
        values = result[names].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        values = np.clip((values - median) / mad, -3, 3)
        result.loc[:, names] = np.nan_to_num(values, nan=0.0, posinf=3.0, neginf=-3.0)
        transformed.append(result)
    return tuple(transformed)


def mean_monthly_rank_ic(frame: pd.DataFrame, score_column: str) -> float:
    values = frame.groupby("signal_date", sort=True).apply(
        lambda group: group[score_column].corr(group["forward_return"], method="spearman"),
        include_groups=False,
    )
    return float(values.mean())


def rank_ic_statistics(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    values = frame.groupby("signal_date", sort=True).apply(
        lambda group: group[score_column].corr(group["forward_return"], method="spearman"),
        include_groups=False,
    ).dropna()
    std = float(values.std(ddof=1))
    return {
        "months": float(len(values)),
        "mean_rank_ic": float(values.mean()),
        "annualized_icir": float(values.mean() / std * sqrt(12)) if std else np.nan,
        "positive_rate": float((values > 0).mean()),
    }
