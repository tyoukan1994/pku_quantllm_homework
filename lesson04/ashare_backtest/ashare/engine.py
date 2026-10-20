from collections import Counter
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .rules import (board, limit_rate, max_order_quantity, order_quantity,
                    slipped_price, tick_round, trading_fees)
from .strategy import Context, validate_targets


class DataCoverageError(ValueError):
    pass


@dataclass
class Lot:
    units: float
    available_date: pd.Timestamp


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    positions: pd.DataFrame
    events: pd.DataFrame
    config: dict
    data_summary: dict
    assumptions: list
    strategy_name: str
    metrics: dict


TRADE_COLUMNS = ["date", "signal_date", "symbol", "side", "quantity", "price", "gross",
                 "commission", "stamp_tax", "transfer_fee", "total_fee", "cash_after"]
ORDER_COLUMNS = ["date", "signal_date", "symbol", "side", "requested_quantity",
                 "filled_quantity", "status", "reason"]
POSITION_COLUMNS = ["date", "symbol", "units", "equivalent_shares", "sellable_shares",
                    "market_value", "weight", "stale_days"]
EVENT_COLUMNS = ["date", "symbol", "event", "value", "shares_before", "shares_after"]


class BacktestEngine:
    def __init__(self, data, config=None):
        self.data = data
        self.config = config or BacktestConfig()
        self.cash = self.config.initial_cash
        self.lots = {}
        self.receivables = []
        self.last_marks = {}
        self.last_factors = {}
        self.stale_days = Counter()
        self.daily_notional = Counter()
        self.daily_buys = Counter()
        self.trade_records = []
        self.order_records = []
        self.event_records = []
        self.warnings = Counter()
        self.has_run = False

    def _units(self, symbol):
        return sum(lot.units for lot in self.lots.get(symbol, []))

    def _scale(self, position, symbol):
        if self.config.accounting == "strict":
            return 1.0
        return self.data.value("cumadj", position, symbol)

    def _available(self, date, position, symbol):
        units = sum(lot.units for lot in self.lots.get(symbol, []) if lot.available_date <= date)
        scale = self._scale(position, symbol)
        return units * scale if np.isfinite(scale) else 0.0

    def _cash_claims(self):
        return sum(record["amount"] for record in self.receivables)

    def _corporate_actions(self, position):
        date = self.data.dates[position]
        for symbol in list(self.lots):
            if self._units(symbol) <= 1e-10:
                continue
            factor = self.data.value("cumadj", position, symbol)
            previous = self.last_factors.get(symbol)
            action = self.data.actions.get((date, symbol))
            if self.config.accounting == "strict":
                if not np.isfinite(factor):
                    raise DataCoverageError(f"{date.date()} {symbol} 持仓复权因子缺失，无法核对公司行为")
                changed = previous is not None and abs(factor / previous - 1) > self.config.factor_tolerance
                if changed and action is None:
                    raise DataCoverageError(
                        f"{date.date()} {symbol} 因子变化但缺少公司行为明细；补充 corporate_actions.csv 或使用复权研究模式"
                    )
                if action is not None:
                    self._apply_action(date, symbol, action)
            if np.isfinite(factor):
                self.last_factors[symbol] = factor
        unsettled = []
        for claim in self.receivables:
            if claim["pay_date"] <= date:
                self.cash += claim["amount"]
                self.event_records.append({"date": str(date.date()), "symbol": claim["symbol"],
                                           "event": "dividend_paid", "value": claim["amount"]})
            else:
                unsettled.append(claim)
        self.receivables = unsettled

    def _apply_action(self, date, symbol, action):
        quantity = self._units(symbol)
        multiplier = action["share_multiplier"]
        new_quantity = quantity * multiplier
        if abs(new_quantity - round(new_quantity)) > 1e-7:
            raise DataCoverageError(f"{symbol} 公司行为产生不足一股，需补充实际到账股数或零股处理明细")
        dividend = quantity * action["cash_per_share"] * (1 - self.config.dividend_tax_rate)
        if dividend > 0:
            self.receivables.append({"pay_date": action["pay_date"], "symbol": symbol,
                                     "amount": tick_round(dividend)})
        if multiplier >= 1:
            bonus = round(new_quantity - quantity)
            if bonus:
                self.lots[symbol].append(Lot(float(bonus), action["share_available_date"]))
        else:
            for lot in self.lots[symbol]:
                lot.units *= multiplier
            if any(abs(lot.units - round(lot.units)) > 1e-7 for lot in self.lots[symbol]):
                raise DataCoverageError("缩股造成分笔零股，需提供按实际持仓合并后的股份变动明细")
        self.event_records.append({"date": str(date.date()), "symbol": symbol,
                                   "event": "corporate_action", "value": dividend,
                                   "shares_before": quantity, "shares_after": round(new_quantity)})

    def _mark(self, position, field, end_of_day=False):
        total = self.cash + self._cash_claims()
        values = {}
        for symbol in list(self.lots):
            units = self._units(symbol)
            if units <= 1e-10:
                continue
            price = self.data.value(field, position, symbol)
            scale = self._scale(position, symbol)
            valid = np.isfinite(price) and np.isfinite(scale)
            if valid:
                mark = price * scale
                self.last_marks[symbol] = mark
                if end_of_day:
                    self.stale_days[symbol] = 0
            else:
                mark = self.last_marks.get(symbol)
                if mark is None:
                    raise DataCoverageError(f"持仓 {symbol} 没有可用估值价格")
                if end_of_day:
                    self.stale_days[symbol] += 1
                    self.warnings["stale_valuation"] += 1
            values[symbol] = units * mark
            total += values[symbol]
        return total, values

    def _limits(self, position, symbol, state):
        date = self.data.dates[position]
        if self.config.strict_rules:
            required = ("is_st", "suspended", "no_limit")
            if any(state.get(field) is None for field in required):
                raise DataCoverageError(f"{date.date()} {symbol} 缺少完整 market_state.csv 状态")
            if not state["no_limit"] and any(pd.isna(state.get(field)) for field in ("limit_up", "limit_down")):
                raise DataCoverageError(f"{date.date()} {symbol} 缺少精确涨跌停价格")
        if state.get("no_limit"):
            return None, None
        upper, lower = state.get("limit_up"), state.get("limit_down")
        if pd.notna(upper) and pd.notna(lower):
            if float(lower) > float(upper):
                raise DataCoverageError("涨跌停价格顺序错误")
            return float(upper), float(lower)
        instrument = self.data.instruments.get(symbol, {})
        listing = instrument.get("list_date")
        if listing is not None and date >= listing:
            age = int(((self.data.dates >= listing) & (self.data.dates <= date)).sum())
            if (board(symbol) == "star" or (board(symbol) == "chinext" and listing >= pd.Timestamp("2020-08-24"))
                    or (board(symbol) == "main" and listing >= pd.Timestamp("2023-04-10"))) and listing >= self.data.dates[0] and age <= 5:
                return None, None
            if board(symbol) != "star" and date == listing:
                issue = instrument.get("issue_price")
                if pd.isna(issue) or float(issue) <= 0:
                    raise DataCoverageError("旧制度新股上市首日需要发行价或显式涨跌停价")
                return tick_round(float(issue) * 1.44), tick_round(float(issue) * 0.64)
        if position == 0:
            return math.nan, math.nan
        previous_close = self.data.value("close", position - 1, symbol)
        previous_factor = self.data.value("cumadj", position - 1, symbol)
        current_factor = self.data.value("cumadj", position, symbol)
        if not all(np.isfinite(value) and value > 0 for value in (previous_close, previous_factor, current_factor)):
            return math.nan, math.nan
        reference = tick_round(previous_close * previous_factor / current_factor)
        rate = limit_rate(symbol, date, state.get("is_st") is True)
        self.warnings["inferred_price_limits"] += 1
        upper = max(tick_round(reference * (1 + rate)), tick_round(reference + 0.01))
        lower = max(0.01, min(tick_round(reference * (1 - rate)), tick_round(reference - 0.01)))
        return upper, lower

    def _order(self, position, symbol, side, desired, signal_date, budget=None):
        date = self.data.dates[position]
        desired = max(0, math.floor(desired + 1e-8))
        record = {"date": str(date.date()), "signal_date": str(pd.Timestamp(signal_date).date()),
                  "symbol": symbol, "side": side, "requested_quantity": desired,
                  "filled_quantity": 0, "status": "rejected", "reason": ""}

        def reject(reason):
            record["reason"] = reason
            self.order_records.append(record)
            return 0

        if desired <= 0:
            return reject("below_one_share")
        state = self.data.state(date, symbol)
        instrument = self.data.instruments.get(symbol, {})
        if instrument.get("delist_date") and date >= instrument["delist_date"]:
            return reject("delisted")
        if instrument.get("list_date") and date < instrument["list_date"]:
            return reject("not_listed")
        if state.get("suspended"):
            return reject("suspended")
        if state.get("can_" + side) is False:
            return reject("direction_disabled")
        if side == "buy" and self.config.exclude_st and state.get("is_st"):
            return reject("st_excluded")
        if side == "buy" and self.config.require_st_history and state.get("is_st") is not False:
            return reject("unknown_st_status")
        prices = {field: self.data.value(field, position, symbol)
                  for field in ("open", "high", "low", "close", "cumadj", "amt")}
        if not all(np.isfinite(value) for value in prices.values()):
            return reject("missing_bar_or_factor")
        if prices["amt"] <= 0:
            return reject("zero_amount")
        if side == "buy" and (position == 0 or self.data.history_counts[position - 1, self.data.symbol_index[symbol]] < self.config.min_history):
            return reject("insufficient_history")
        upper, lower = self._limits(position, symbol, state)
        if upper is not None and (not np.isfinite(upper) or not np.isfinite(lower)):
            return reject("missing_limit_reference")
        if upper is not None and side == "buy" and prices["open"] >= upper - 1e-8:
            return reject("open_at_limit_up")
        if lower is not None and side == "sell" and prices["open"] <= lower + 1e-8:
            return reject("open_at_limit_down")
        price = slipped_price(prices["open"], side, self.config.slippage_bps)
        if price <= 0 or price > prices["high"] + 1e-8 or price < prices["low"] - 1e-8:
            return reject("slippage_outside_daily_range")
        if upper is not None and (price > upper + 1e-8 or price < lower - 1e-8):
            return reject("slippage_outside_price_limits")
        previous_amount = self.data.value("amt", position - 1, symbol) * self.config.amount_multiplier if position > 0 else math.nan
        if not np.isfinite(previous_amount) or previous_amount <= 0:
            return reject("missing_previous_liquidity")
        remaining_capacity = max(0.0, previous_amount * self.config.participation_rate - self.daily_notional[symbol])
        capacity_quantity = math.floor(remaining_capacity / price + 1e-8)
        available = self._available(date, position, symbol)
        allowed = min(desired, capacity_quantity, max_order_quantity(symbol))
        if side == "sell":
            allowed = min(allowed, math.floor(available + 1e-8))
            if available < 1:
                return reject("t_plus_one_or_fractional_residual")
        if side == "buy" and state.get("is_st"):
            allowed = min(allowed, 500_000 - self.daily_buys[symbol])
        quantity = order_quantity(symbol, allowed, side, available)
        if side == "buy":
            cash_budget = min(self.cash, budget if budget is not None else self.cash)
            lower_quantity, upper_quantity = 0, quantity
            while lower_quantity < upper_quantity:
                candidate = (lower_quantity + upper_quantity + 1) // 2
                cost = candidate * price + trading_fees(date, side, candidate * price, self.config)["total"]
                if cost <= cash_budget + 1e-8:
                    lower_quantity = candidate
                else:
                    upper_quantity = candidate - 1
            quantity = order_quantity(symbol, lower_quantity, side)
        if quantity <= 0:
            return reject("cash_liquidity_or_lot_size")
        gross = tick_round(quantity * price)
        fees = trading_fees(date, side, gross, self.config)
        scale = self._scale(position, symbol)
        units = quantity / scale
        if side == "buy":
            self.cash -= gross + fees["total"]
            next_day = self.data.dates[position + 1] if position + 1 < len(self.data.dates) else date + pd.Timedelta(days=1)
            self.lots.setdefault(symbol, []).append(Lot(units, next_day))
            self.daily_buys[symbol] += quantity
            self.last_factors[symbol] = prices["cumadj"]
            self.last_marks[symbol] = prices["open"] * scale
        else:
            if self.cash + gross < fees["total"]:
                return reject("insufficient_cash_for_fees")
            remaining = units
            for lot in self.lots.get(symbol, []):
                if lot.available_date <= date:
                    deducted = min(lot.units, remaining)
                    lot.units -= deducted
                    remaining -= deducted
            self.lots[symbol] = [lot for lot in self.lots[symbol] if lot.units > 1e-10]
            self.cash += gross - fees["total"]
        if self.cash < -1e-6:
            raise RuntimeError("账户现金为负，资金约束失败")
        self.cash = max(0.0, self.cash)
        self.daily_notional[symbol] += gross
        record["filled_quantity"] = quantity
        record["status"] = "filled" if quantity == desired else "partial"
        record["reason"] = "" if quantity == desired else "cash_liquidity_t1_or_lot_cap"
        self.order_records.append(record)
        self.trade_records.append({"date": str(date.date()), "signal_date": record["signal_date"],
                                   "symbol": symbol, "side": side, "quantity": quantity,
                                   "price": price, "gross": gross, "commission": fees["commission"],
                                   "stamp_tax": fees["stamp_tax"], "transfer_fee": fees["transfer_fee"],
                                   "total_fee": fees["total"], "cash_after": self.cash})
        return quantity

    def _submit_order(self, position, symbol, side, desired, signal_date, budget=None):
        if not self.config.split_orders:
            return self._order(position, symbol, side, desired, signal_date, budget)
        remaining = max(0, math.floor(desired + 1e-8))
        total = 0
        cash_before = self.cash
        while remaining > 0:
            child = min(remaining, max_order_quantity(symbol))
            child_budget = max(0, budget - (cash_before - self.cash)) if budget is not None else None
            filled = self._order(position, symbol, side, child, signal_date, child_budget)
            total += filled
            remaining -= filled
            if filled < child:
                break
        return total

    def _rebalance(self, position, targets, signal_date):
        equity, values = self._mark(position, "open")
        if equity <= 0:
            raise ValueError("账户权益非正，无法继续调仓")
        for symbol in sorted(self.lots):
            if self._units(symbol) <= 1e-10:
                continue
            price = self.data.value("open", position, symbol)
            scale = self._scale(position, symbol)
            if not np.isfinite(price) or not np.isfinite(scale):
                self._submit_order(position, symbol, "sell", 1, signal_date)
                continue
            desired_value = equity * targets.get(symbol, 0)
            excess = self._units(symbol) * price * scale - desired_value
            if excess >= price - 1e-8:
                self._submit_order(position, symbol, "sell", excess / price, signal_date)
        for symbol, weight in sorted(targets.items(), key=lambda item: (-item[1], item[0])):
            price = self.data.value("open", position, symbol)
            scale = self._scale(position, symbol)
            if not np.isfinite(price) or not np.isfinite(scale):
                self._submit_order(position, symbol, "buy", 1, signal_date)
                continue
            current_value = self._units(symbol) * price * scale
            budget = equity * weight - current_value
            if budget >= price - 1e-8:
                slipped = slipped_price(price, "buy", self.config.slippage_bps)
                self._submit_order(position, symbol, "buy", budget / slipped, signal_date, budget)

    def run(self, strategy, progress=None):
        if self.has_run:
            raise RuntimeError("每次回测请创建新的 BacktestEngine 和策略实例")
        self.has_run = True
        config = self.config
        indices = np.flatnonzero((self.data.dates >= pd.Timestamp(config.start)) & (self.data.dates <= pd.Timestamp(config.end)))
        if len(indices) < 2:
            raise ValueError("回测区间至少需要两个有效观测日期")
        if config.liquidate_on_end and config.liquidation_window >= len(indices):
            raise ValueError("退出窗口必须短于回测区间，避免首日无法建仓")
        pending = None
        signal_date = None
        if config.seed_previous_close:
            previous = int(indices[0]) - 1
            if previous < 0:
                raise DataCoverageError("首日建仓需要开始日前一交易日行情，用于生成收盘信号")
            context = Context(self.data, previous, config, self.cash, self.cash, {})
            pending = validate_targets(strategy.on_close(context), self.data.symbols, config.max_weight)
            signal_date = self.data.dates[previous]
            self.event_records.append({"date": str(signal_date.date()), "symbol": "",
                                       "event": "initial_signal_from_previous_close", "value": len(pending or {})})
        equity_records, position_records = [], []
        for sequence, position in enumerate(indices):
            date = self.data.dates[position]
            self.daily_notional.clear()
            self.daily_buys.clear()
            self._corporate_actions(position)
            trade_start = len(self.trade_records)
            exiting = config.liquidate_on_end and sequence >= len(indices) - config.liquidation_window
            if exiting:
                self._rebalance(position, {}, self.data.dates[position - 1] if position else date)
                if sequence == len(indices) - 1 and config.accounting == "research" and config.write_off_research_dust:
                    for symbol in list(self.lots):
                        scale = self._scale(position, symbol)
                        if not np.isfinite(scale):
                            scale = self.last_factors.get(symbol, math.nan)
                        shares = self._units(symbol) * scale
                        if 0 < shares < 1 - 1e-8:
                            price = self.data.value("close", position, symbol)
                            value = shares * price if np.isfinite(price) else self._units(symbol) * self.last_marks.get(symbol, math.nan)
                            if np.isfinite(value):
                                self.event_records.append({"date": str(date.date()), "symbol": symbol,
                                                           "event": "research_fractional_dust_writeoff", "value": -value,
                                                           "shares_before": shares, "shares_after": 0})
                                self.lots[symbol] = []
                self.event_records.append({"date": str(date.date()), "symbol": "",
                                           "event": "scheduled_terminal_liquidation" if sequence == len(indices) - 1 else "scheduled_exit_window",
                                           "value": sum(self._units(symbol) > 1e-10 for symbol in self.lots)})
            elif pending is not None:
                self._rebalance(position, pending, signal_date)
            equity, values = self._mark(position, "close", end_of_day=True)
            if not np.isfinite(equity) or equity <= 0:
                raise ValueError(f"{date.date()} 账户权益非正或非有限")
            gross = sum(trade["gross"] for trade in self.trade_records[trade_start:])
            fees = sum(trade["total_fee"] for trade in self.trade_records[trade_start:])
            equity_records.append({"date": date, "equity": equity, "cash": self.cash,
                                   "receivables": self._cash_claims(), "market_value": sum(values.values()),
                                   "exposure": sum(values.values()) / equity,
                                   "traded_notional": gross, "fees": fees,
                                   "stale_positions": sum(self.stale_days[symbol] > 0 for symbol in values)})
            quantities = {}
            for symbol, value in values.items():
                factor = self._scale(position, symbol)
                if not np.isfinite(factor):
                    factor = self.last_factors.get(symbol, math.nan)
                quantities[symbol] = self._units(symbol) * factor
                position_records.append({"date": str(date.date()), "symbol": symbol,
                                         "units": self._units(symbol), "equivalent_shares": quantities[symbol],
                                         "sellable_shares": self._available(date, position, symbol),
                                         "market_value": value, "weight": value / equity,
                                         "stale_days": self.stale_days[symbol]})
            context = Context(self.data, position, config, self.cash, equity, quantities)
            pending = validate_targets(strategy.on_close(context), self.data.symbols, config.max_weight)
            signal_date = date
            if sequence == len(indices) - 1 and pending is not None:
                self.event_records.append({"date": str(date.date()), "symbol": "",
                                           "event": "last_signal_not_executed", "value": len(pending)})
            if progress is not None and (sequence % 10 == 0 or sequence == len(indices) - 1):
                progress((sequence + 1) / len(indices), str(date.date()))
        curve = pd.DataFrame(equity_records).set_index("date")
        curve["net_value"] = curve.equity / config.initial_cash
        curve["daily_return"] = curve.equity.pct_change(fill_method=None).fillna(0.0)
        curve.loc[curve.index[0], "daily_return"] = curve.net_value.iloc[0] - 1
        curve["drawdown"] = curve.net_value / curve.net_value.cummax().clip(lower=1.0) - 1
        if config.benchmark:
            if config.benchmark not in self.data.symbol_index:
                raise ValueError("基准代码不在载入股票池中")
            prices = self.data._panels["close"][config.benchmark] * self.data._panels["cumadj"][config.benchmark]
            prices = prices.reindex(curve.index)
            if prices.isna().any():
                raise DataCoverageError("基准区间存在缺失，未自动填补")
            curve["benchmark_net_value"] = prices / prices.iloc[0]
        trades = pd.DataFrame(self.trade_records, columns=TRADE_COLUMNS)
        orders = pd.DataFrame(self.order_records, columns=ORDER_COLUMNS)
        positions = pd.DataFrame(position_records, columns=POSITION_COLUMNS)
        assumptions = self._assumptions()
        from .report import performance_metrics
        metrics = performance_metrics(curve, trades, orders, config)
        metrics["terminal_position_count"] = sum(self._units(symbol) > 1e-10 for symbol in self.lots)
        metrics["terminal_market_value"] = float(curve.market_value.iloc[-1])
        metrics["liquidation_complete"] = metrics["terminal_position_count"] == 0 and self._cash_claims() == 0
        metrics["research_dust_writeoff"] = -sum(event.get("value", 0) for event in self.event_records
                                                if event["event"] == "research_fractional_dust_writeoff")
        return BacktestResult(curve, trades, orders, positions, pd.DataFrame(self.event_records, columns=EVENT_COLUMNS),
                              config.to_dict(), self.data.summary(), assumptions,
                              getattr(strategy, "name", type(strategy).__name__), metrics)

    def _assumptions(self):
        notes = ["收盘后形成目标权重，下一观测日开盘尝试成交；首日为空仓，末日信号不执行。",
                 "未成交及剩余订单当日失效；None 表示不下新单，空字典表示尝试清仓。",
                 "开盘即涨停不买、开盘即跌停不卖；日线不能重现排队成交和盘中路径。",
                 "成交规模上限采用上一观测日成交额，当前全天成交额仅用于排除零成交记录。",
                 "佣金为可配置的假定全佣，过户费另计；应按实际券商口径调整，避免重复收费。",
                 "成交额按 amount_multiplier 换算为元；原文件金额单位仍需老师确认。",
                 "缺失行情按最后有效价格估值并保留持仓，不自动卖出或把退市损失忽略为已实现盈利。",
                 "不含融资融券、日内交易、竞价排队和个人投资者适当性审查；不强制期末清仓。"]
        if self.config.accounting == "research":
            notes.append("复权研究模式：持仓按 P×cumadj 的权益份额计价，隐含复权再投资近似；不是实际股数、现金分红和税款账户。买卖申报按整数股及板块数量规则约束，折算不足一股的残余权益留在持仓。")
        else:
            notes.append("严格账户模式：实际股数和现金核算；持仓因子变化缺少公司行为时停止。支持普通现金分红、送转/缩股及到账日期；股息税按配置固定税率，不模拟持有期补税，配股等需另行扩展。")
        if not self.config.strict_rules:
            notes.append("状态覆盖为研究近似：缺失每日状态时按普通股及板块规则推算涨跌停。没有历史 ST 标记时无法保证 ST 排除；首次观测日不被当作真实上市日。")
        else:
            notes.append("严格交易状态模式：每次成交检查需要逐日 ST、停牌、是否无涨跌幅及精确涨跌停价；覆盖不足立即停止。")
        if not self.config.benchmark:
            notes.append("未指定外部基准；净值图中的 1.0 为初始资金基线，不是沪深300或全市场指数。")
        if self.config.seed_previous_close:
            notes[0] = "使用开始日前一交易日收盘信号，在考核首日开盘尝试建仓；首日损益和费用计入收益。"
        if self.config.liquidate_on_end:
            notes[7] = "末日开盘执行预先安排的清仓，不再开新仓；仍检查 T+1、跌停、停牌和成交规模。无法卖出的持仓保留并报告清仓未完成，不假造成交。"
            if self.config.liquidation_window > 1:
                notes[7] = f"结束前 {self.config.liquidation_window} 个交易观测日进入只卖不买退出期，每日开盘重试清仓。所有 T+1、跌停、停牌及容量约束仍生效，受阻持仓保留并报告。"
        if self.config.require_st_history:
            notes.append("非 ST 硬约束：选股时和买入时均须有当日明确的非 ST 状态；未知状态不买，已持仓变 ST 允许卖出。")
        if self.config.stamp_tax_rate is not None or self.config.transfer_fee_rate is not None:
            notes.append(f"费用覆盖：印花税单边卖出 {self.config.stamp_tax_rate if self.config.stamp_tax_rate is not None else '按历史'}；过户费 {self.config.transfer_fee_rate if self.config.transfer_fee_rate is not None else '按历史'}。")
        if self.config.split_orders:
            notes.append("大额调仓按板块单笔数量上限拆成子订单；子订单共享当日流动性额度，逐笔计费，不假设无限容量。")
        if self.config.write_off_research_dust:
            notes.append("仅研究模式：末日不足一股的复权模型残余权益按零值核销、计入损失，不产生卖出成交或现金收入；完整股份受阻仍保留，不以核销伪装清仓。")
        for reason, count in sorted(self.warnings.items()):
            notes.append(f"数据/模型事件 {reason}：{count} 次。")
        stale = {symbol: count for symbol, count in self.stale_days.items() if count >= self.config.stale_warning_days and self._units(symbol) > 1e-10}
        if stale:
            notes.append(f"期末存在长期陈旧估值持仓，需核实退市/停牌/数据缺失并进行损失情景分析：{stale}")
        return notes
