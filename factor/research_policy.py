"""Hard boundaries for the factor-discovery sample.

The untouched 2025 holdout is protected during factor discovery. Code that
downloads or evaluates research data should call ``require_discovery_window``
before issuing a data request.
"""

from __future__ import annotations

from datetime import date


RESEARCH_START = date(2015, 1, 1)
DISCOVERY_END = date(2024, 12, 31)
PROTECTED_START = date(2025, 1, 1)
SEALED_START = date(2025, 1, 1)
SEALED_END = date(2025, 12, 31)
FEATURE_CONTEXT_START = date(2014, 9, 1)
FINAL_SIGNAL_START = date(2024, 12, 1)
FINAL_SIGNAL_END = date(2025, 11, 30)
FINAL_RETURN_START = date(2025, 1, 1)
FINAL_RETURN_END = date(2025, 12, 31)
FINAL_FEATURE_START = date(2024, 9, 1)
FINAL_FEATURE_END = date(2025, 11, 30)


class SampleLeakageError(ValueError):
    """Raised when a request touches the sealed holdout sample."""


def _as_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def require_discovery_window(start: str | date, end: str | date) -> tuple[date, date]:
    """Validate that a requested performance window stays in 2015-2024."""
    start_date = _as_date(start)
    end_date = _as_date(end)

    if start_date > end_date:
        raise ValueError("研究区间的开始日期不能晚于结束日期。")
    if start_date < RESEARCH_START:
        raise ValueError(f"绩效研究不得早于 {RESEARCH_START.isoformat()}。")
    if end_date > DISCOVERY_END:
        raise SampleLeakageError(
            f"请求截止到 {end_date.isoformat()}，触及从 {PROTECTED_START.isoformat()} "
            "开始的受保护区间；因子冻结前禁止查询或评价该区间。"
        )
    return start_date, end_date


def require_feature_context_window(
    start: str | date, end: str | date
) -> tuple[date, date]:
    """Allow historical lookback inputs but never data from the sealed period.

    Alpha158 needs up to 60 prior trading observations.  The pre-2015 portion is
    used only as feature context and is never evaluated as a return sample.
    """
    start_date = _as_date(start)
    end_date = _as_date(end)
    if start_date > end_date:
        raise ValueError("特征窗口的开始日期不能晚于结束日期。")
    if start_date < FEATURE_CONTEXT_START:
        raise ValueError(
            f"特征回看窗口不得早于 {FEATURE_CONTEXT_START.isoformat()}。"
        )
    if end_date > DISCOVERY_END:
        raise SampleLeakageError(
            f"特征请求截止到 {end_date.isoformat()}，触及从 "
            f"{PROTECTED_START.isoformat()} 开始的受保护区间。"
        )
    return start_date, end_date


def _require_bounded_window(
    start: str | date,
    end: str | date,
    lower: date,
    upper: date,
    label: str,
) -> tuple[date, date]:
    start_date = _as_date(start)
    end_date = _as_date(end)
    if start_date > end_date:
        raise ValueError(f"{label}的开始日期不能晚于结束日期。")
    if start_date < lower or end_date > upper:
        raise SampleLeakageError(
            f"{label}必须限制在 {lower.isoformat()} 至 {upper.isoformat()}，"
            f"实际请求为 {start_date.isoformat()} 至 {end_date.isoformat()}。"
        )
    return start_date, end_date


def require_final_signal_window(start: str | date, end: str | date) -> tuple[date, date]:
    """Allow only the frozen 12 signal months for the 2025 final test."""
    return _require_bounded_window(
        start, end, FINAL_SIGNAL_START, FINAL_SIGNAL_END, "最终检验信号窗口"
    )


def require_final_return_window(start: str | date, end: str | date) -> tuple[date, date]:
    """Allow only realised returns whose calendar dates are in 2025."""
    return _require_bounded_window(
        start, end, FINAL_RETURN_START, FINAL_RETURN_END, "最终检验收益窗口"
    )


def require_final_feature_window(start: str | date, end: str | date) -> tuple[date, date]:
    """Allow historical feature context through the last 2025 signal date."""
    return _require_bounded_window(
        start, end, FINAL_FEATURE_START, FINAL_FEATURE_END, "最终检验特征窗口"
    )
