from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .factors import FACTOR_NAMES, eligible_scores
from .strategy import Context


@dataclass(frozen=True)
class EvaluationConfig:
    start: str = "2020-01-02"
    end: str = "2024-12-31"
    factors: tuple = tuple(FACTOR_NAMES)
    horizons: tuple = (1, 5, 20)
    windows: tuple = (21, 63, 126, 252)
    fast_window: int = 63
    reference_window: int = 252
    block_window: int = 63
    min_fraction: float = 0.8
    min_stocks: int = 30
    min_history: int = 120
    liquid_pool: int = 1500
    minimum_amount: float = 50_000_000
    direction: int = 1
    effect_floor: float = 0.01
    persistence: int = 3
    require_st_history: bool = True

    def __post_init__(self):
        BacktestConfig(start=self.start, end=self.end)
        for name in ("factors", "horizons", "windows"):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} 必须非空且不重复")
            object.__setattr__(self, name, values)
        for value in (*self.horizons, *self.windows, self.fast_window, self.reference_window,
                      self.block_window, self.min_stocks, self.min_history, self.liquid_pool, self.persistence):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("窗口、预测期及样本数必须为正整数")
        if self.fast_window < 20 or self.reference_window <= self.fast_window or min(self.windows) < 20:
            raise ValueError("滚动窗口至少20日，历史参考窗口必须长于近期窗口")
        if self.block_window < 20 or self.min_stocks < 5:
            raise ValueError("分段窗口至少20日，截面至少5只股票")
        if self.direction not in (-1, 1) or isinstance(self.direction, bool):
            raise ValueError("预期方向只能是1或-1，不能根据评价区间自动选向")
        if not 0 < self.min_fraction <= 1:
            raise ValueError("有效观测比例必须在(0,1]内")
        if not math.isfinite(self.minimum_amount) or self.minimum_amount < 0:
            raise ValueError("成交额门槛必须非负且有限")
        if not math.isfinite(self.effect_floor) or not 0 <= self.effect_floor < 1:
            raise ValueError("IC实用阈值必须在[0,1)内")

    def to_dict(self):
        return asdict(self)


def collect_observations(data, config, score_provider=None, progress=None):
    if score_provider is None and not set(config.factors).issubset(FACTOR_NAMES):
        raise ValueError("非内置因子需要提供score_provider(context)，返回股票×因子DataFrame")
    selection = BacktestConfig(start=config.start, end=config.end, min_history=config.min_history,
                               exclude_st=True, require_st_history=config.require_st_history)
    positions = np.flatnonzero((data.dates >= pd.Timestamp(config.start)) & (data.dates <= pd.Timestamp(config.end)))
    if not len(positions):
        raise ValueError("评价区间没有交易观测日")
    last_position = positions[-1]
    rows = []
    for counter, position in enumerate(positions):
        context = Context(data, int(position), selection, 1, 1, {})
        scores = (eligible_scores(context, config.liquid_pool, config.minimum_amount)
                  if score_provider is None else score_provider(context))
        if not isinstance(scores, pd.DataFrame) or scores.index.has_duplicates:
            raise ValueError("因子函数必须返回股票代码不重复的DataFrame")
        scores = scores.reindex(index=scores.index.intersection(context.universe()), columns=config.factors)
        for horizon in config.horizons:
            exit_position = position + 1 + horizon
            if exit_position > last_position:
                continue
            entry = data.arrays["open"][position + 1] * data.arrays["cumadj"][position + 1]
            exit_prices = data.arrays["open"][exit_position] * data.arrays["cumadj"][exit_position]
            with np.errstate(invalid="ignore", divide="ignore"):
                labels = pd.Series(exit_prices / entry - 1, index=data.symbols)
            for factor in config.factors:
                values = scores[factor].replace([np.inf, -np.inf], np.nan).dropna()
                sample = pd.concat([values.rename("score"), labels.reindex(values.index).rename("return")], axis=1)
                sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
                row = {"date": context.date, "factor": factor, "horizon": horizon,
                       "label_start": data.dates[position + 1], "label_end": data.dates[exit_position],
                       "eligible_stocks": len(values), "stocks": len(sample),
                       "coverage": len(sample) / len(values) if len(values) else np.nan,
                       "rank_ic": np.nan, "pearson_ic": np.nan, "spread": np.nan}
                row.update({f"q{group}": np.nan for group in range(1, 6)})
                if len(sample) >= config.min_stocks and sample.score.nunique() > 1 and sample["return"].nunique() > 1:
                    ranks = sample.rank(method="average")
                    row["rank_ic"] = ranks.score.corr(ranks["return"])
                    row["pearson_ic"] = sample.score.corr(sample["return"])
                    quantiles = np.ceil(sample.score.rank(method="average", pct=True) * 5).clip(1, 5)
                    for group in range(1, 6):
                        row[f"q{group}"] = sample.loc[quantiles == group, "return"].mean()
                    row["spread"] = row["q5"] - row["q1"]
                rows.append(row)
        if progress and (counter % 20 == 0 or counter == len(positions) - 1):
            progress((counter + 1) / len(positions), str(context.date.date()))
    if not rows:
        raise ValueError("区间不足以兑现任一预测期，请扩大日期范围")
    return pd.DataFrame(rows)


def hac_mean(values, lags, minimum=20):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    count = int(valid.sum())
    mean = float(values[valid].mean()) if count else np.nan
    std = float(values[valid].std(ddof=1)) if count > 1 else np.nan
    result = {"mean": mean, "std": std, "count": count, "se": np.nan,
              "lower": np.nan, "upper": np.nan, "ir": mean / std if std > 1e-12 else np.nan}
    if count < max(minimum, 3 * (lags + 1)) or not std > 1e-12:
        return result
    errors = np.where(valid, values - mean, 0.0)
    variance = _hac_variance(errors, lags) * count / (count - 1) / count ** 2
    if variance > 1e-20:
        result["se"] = math.sqrt(variance)
        result["lower"] = mean - 1.96 * result["se"]
        result["upper"] = mean + 1.96 * result["se"]
    return result


def _hac_variance(errors, lags):
    bandwidth = min(lags, len(errors) - 1)
    variance = float(errors @ errors)
    for lag in range(1, bandwidth + 1):
        variance += 2 * (1 - lag / (bandwidth + 1)) * float(errors[lag:] @ errors[:-lag])
    return max(variance, 0.0)


def _bandwidth(length, horizon):
    return max(horizon - 1, int(4 * (length / 100) ** (2 / 9)))


def _minimum(window, config):
    return max(20, math.ceil(window * config.min_fraction))


def _drift(reference, recent, gap, lags):
    reference_count = np.isfinite(reference).sum()
    recent_count = np.isfinite(recent).sum()
    difference = float(np.nanmean(recent) - np.nanmean(reference))
    errors = np.r_[np.where(np.isfinite(reference), -(reference - np.nanmean(reference)) / reference_count, 0),
                   np.zeros(gap), np.where(np.isfinite(recent), (recent - np.nanmean(recent)) / recent_count, 0)]
    variance = _hac_variance(errors, lags) * (reference_count + recent_count) / (reference_count + recent_count - 2)
    return difference, difference + 1.96 * math.sqrt(variance) if variance > 1e-20 else np.nan


def dynamic_diagnostics(series, horizon, config):
    rows = []
    alert_streak = 0
    previous_alert = None
    for position, date in enumerate(series.index):
        row = {"as_of": date}
        for window in sorted(set((*config.windows, config.fast_window, config.reference_window))):
            sample = series.iloc[max(0, position - window + 1):position + 1].to_numpy()
            stats = hac_mean(sample, _bandwidth(window, horizon), _minimum(window, config))
            enough = len(sample) == window and stats["count"] >= _minimum(window, config)
            for name in ("mean", "lower", "upper", "ir"):
                row[f"{name}_{window}"] = stats[name] if enough else np.nan
            row[f"count_{window}"] = stats["count"]
        fast = config.fast_window
        recent = series.iloc[max(0, position - fast + 1):position + 1].to_numpy() * config.direction
        stats = hac_mean(recent, _bandwidth(fast, horizon), _minimum(fast, config))
        row.update(drift=np.nan, drift_upper=np.nan, reference_mean=np.nan)
        reference_end = position - fast - horizon + 1
        reference_start = reference_end - config.reference_window + 1
        if reference_start >= 0:
            reference = series.iloc[reference_start:reference_end + 1].to_numpy() * config.direction
            if np.isfinite(reference).sum() >= _minimum(config.reference_window, config):
                row["reference_mean"] = float(np.nanmean(reference))
                if np.isfinite(stats["se"]):
                    row["drift"], row["drift_upper"] = _drift(reference, recent, horizon - 1,
                        _bandwidth(len(reference) + fast + horizon - 1, horizon))
        if len(recent) < fast or stats["count"] < _minimum(fast, config):
            state = "样本不足"
        elif stats["upper"] < -config.effect_floor:
            state = "近期反向"
        elif row["drift_upper"] < -config.effect_floor and row["reference_mean"] > config.effect_floor:
            state = "近期衰减"
        elif stats["lower"] > config.effect_floor:
            state = "正向支持"
        else:
            state = "证据不足"
        alert = state if state in {"近期反向", "近期衰减"} else None
        alert_streak = alert_streak + 1 if alert and alert == previous_alert else int(bool(alert))
        previous_alert = alert
        row.update(state=state, alert_streak=alert_streak, persistent_alert=alert_streak >= config.persistence)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_observations(observations, calendar, config):
    calendar = pd.DatetimeIndex(calendar)
    if calendar.has_duplicates or calendar.hasnans or not calendar.is_monotonic_increasing or calendar.tz:
        raise ValueError("评价日历必须升序、唯一、无时区")
    calendar = calendar[(calendar >= pd.Timestamp(config.start)) & (calendar <= pd.Timestamp(config.end))]
    if calendar.empty:
        raise ValueError("评价日历为空")
    observations = observations.copy()
    for name in ("date", "label_start", "label_end"):
        observations[name] = pd.to_datetime(observations[name])
    if observations.duplicated(["factor", "horizon", "date"]).any():
        raise ValueError("同因子、预测期与信号日只能有一个IC")
    if not ((observations.label_start > observations.date) & (observations.label_end > observations.label_start)).all():
        raise ValueError("未来收益必须从信号日之后开始且在开始日之后兑现")
    for name in ("date", "label_start", "label_end"):
        if not observations[name].isin(calendar).all():
            raise ValueError("信号及收益标签不能越过评价日历边界")
    signal_positions = calendar.get_indexer(observations.date)
    entry_positions = calendar.get_indexer(observations.label_start)
    exit_positions = calendar.get_indexer(observations.label_end)
    if not ((entry_positions == signal_positions + 1) &
            (exit_positions == entry_positions + observations.horizon.to_numpy())).all():
        raise ValueError("标签必须严格匹配次日开盘和指定交易日预测期")
    if not set(observations.factor).issubset(config.factors) or not set(observations.horizon).issubset(config.horizons):
        raise ValueError("观测中的因子或预测期与配置不一致")
    static_rows, dynamic_frames, blocks = [], [], []
    for (factor, horizon), sample in observations.groupby(["factor", "horizon"], sort=True):
        sample = sample.sort_values("date")
        if sample.label_end.duplicated().any():
            raise ValueError("同一预测期的兑现日不能重复")
        values = sample.set_index("label_end").rank_ic.reindex(calendar)
        if ((values.dropna() < -1 - 1e-10) | (values.dropna() > 1 + 1e-10)).any():
            raise ValueError("RankIC必须在[-1,1]内")
        stats = hac_mean(values.to_numpy(), _bandwidth(len(values), horizon))
        row = {"factor": factor, "horizon": horizon, **stats,
               "positive_fraction": (values.dropna() * config.direction > 0).mean(),
               "coverage": sample.coverage.mean(), "pearson_ic": sample.pearson_ic.mean(),
               "mean_spread": sample.spread.mean()}
        row.update({f"q{group}": sample[f"q{group}"].mean() for group in range(1, 6)})
        static_rows.append(row)
        dynamic = dynamic_diagnostics(values, horizon, config)
        dynamic["factor"], dynamic["horizon"] = factor, horizon
        dynamic_frames.append(dynamic)
        for start in range(0, len(calendar), config.block_window):
            dates = calendar[start:start + config.block_window]
            block = sample[(sample.label_start >= dates[0]) & (sample.label_end <= dates[-1])]
            valid = block.rank_ic.dropna()
            complete = len(dates) == config.block_window
            usable = complete and len(valid) >= _minimum(max(20, config.block_window - horizon), config)
            blocks.append({"factor": factor, "horizon": horizon, "start": dates[0], "end": dates[-1],
                           "count": len(valid), "mean_ic": valid.mean(), "complete": complete, "usable": usable})
    if not static_rows:
        raise ValueError("没有可评价的因子观测")
    block_table = pd.DataFrame(blocks)
    transitions = []
    for (factor, horizon), frame in block_table.groupby(["factor", "horizon"]):
        frame = frame.copy()
        previous = frame.mean_ic.shift(1) * config.direction
        current = frame.mean_ic * config.direction
        usable_pair = frame.usable & frame.usable.shift(1, fill_value=False)
        positive = usable_pair & (previous > config.effect_floor)
        reversal = positive & (current < -config.effect_floor)
        transitions.append({"factor": factor, "horizon": horizon, "adjacent_pairs": int(usable_pair.sum()),
                            "positive_previous_blocks": int(positive.sum()), "next_negative_blocks": int(reversal.sum()),
                            "next_negative_fraction": reversal.sum() / positive.sum() if positive.sum() else np.nan})
    monthly = observations.assign(month=observations.date.dt.to_period("M").astype(str)).groupby(
        ["factor", "horizon", "month"], as_index=False).agg(mean_ic=("rank_ic", "mean"), count=("rank_ic", "count"))
    return {"observations": observations, "static": pd.DataFrame(static_rows),
            "dynamic": pd.concat(dynamic_frames, ignore_index=True), "blocks": block_table,
            "transitions": pd.DataFrame(transitions), "monthly": monthly}
