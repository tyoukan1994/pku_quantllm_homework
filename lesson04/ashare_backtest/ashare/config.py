from dataclasses import asdict, dataclass
import math

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    start: str = "2024-01-02"
    end: str = "2025-12-31"
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    stamp_tax_rate: float | None = None
    transfer_fee_rate: float | None = None
    slippage_bps: float = 5.0
    participation_rate: float = 0.01
    amount_multiplier: float = 1.0
    annual_days: int = 252
    risk_free_rate: float = 0.0
    min_history: int = 60
    max_weight: float = 1.0
    accounting: str = "research"
    strict_rules: bool = False
    exclude_st: bool = True
    dividend_tax_rate: float = 0.0
    factor_tolerance: float = 0.0001
    stale_warning_days: int = 20
    benchmark: str = ""
    require_st_history: bool = False
    seed_previous_close: bool = False
    liquidate_on_end: bool = False
    liquidation_window: int = 1
    split_orders: bool = False
    write_off_research_dust: bool = False

    def __post_init__(self):
        start = pd.Timestamp(self.start)
        end = pd.Timestamp(self.end)
        if pd.isna(start) or pd.isna(end) or start > end or start.tz or end.tz:
            raise ValueError("起止日期必须有效、无时区，且开始日期不晚于结束日期")
        if start != start.normalize() or end != end.normalize():
            raise ValueError("日频回测起止日期不能包含日内时间")
        positive = ("initial_cash", "amount_multiplier", "factor_tolerance")
        nonnegative = ("commission_rate", "minimum_commission", "slippage_bps")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name in positive and value == 0):
                raise ValueError(f"参数 {name} 超出有效范围")
        for name in ("participation_rate", "max_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} 必须在 (0, 1] 内")
        if not 0 <= self.dividend_tax_rate <= 1 or not math.isfinite(self.risk_free_rate) or self.risk_free_rate <= -1:
            raise ValueError("股息税率或无风险利率无效")
        for name in ("min_history", "annual_days", "stale_warning_days", "liquidation_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        if self.accounting not in {"research", "strict"}:
            raise ValueError("accounting 只能是 research 或 strict")
        if self.slippage_bps >= 10000:
            raise ValueError("滑点必须小于 10000 基点")
        for name in ("stamp_tax_rate", "transfer_fee_rate"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} 必须为空或在 [0, 1] 内")
        if self.require_st_history and not self.exclude_st:
            raise ValueError("require_st_history 要求同时启用 exclude_st")

    def to_dict(self):
        return asdict(self)
