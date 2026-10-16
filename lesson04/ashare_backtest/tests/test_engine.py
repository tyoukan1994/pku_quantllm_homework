from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ashare import BacktestConfig, BacktestEngine, DataBundle, Strategy
from ashare.engine import DataCoverageError
from ashare.report import export_result
from ashare.rules import limit_rate, order_quantity, tick_round, trading_fees
from ashare.strategy import Context, load_strategy, validate_targets


SYMBOL = "000001.SZ"


def dataset(prices=None, opens=None, factors=None, amounts=None, symbols=None, states=None, actions=None):
    prices = np.asarray(prices if prices is not None else [10, 10, 10, 10], dtype=float)
    if prices.ndim == 1:
        prices = prices[:, None]
    symbols = symbols or [SYMBOL]
    dates = pd.bdate_range("2024-01-02", periods=len(prices))
    opens = prices.copy() if opens is None else np.asarray(opens, dtype=float).reshape(prices.shape)
    factors = np.ones_like(prices) if factors is None else np.asarray(factors, dtype=float).reshape(prices.shape)
    amounts = np.full_like(prices, 1e9) if amounts is None else np.asarray(amounts, dtype=float).reshape(prices.shape)
    panels = {"open": opens, "close": prices, "high": np.maximum(opens, prices) * 1.01,
              "low": np.minimum(opens, prices) * 0.99, "cumadj": factors, "amt": amounts}
    panels = {field: pd.DataFrame(values, index=dates, columns=symbols) for field, values in panels.items()}
    return DataBundle(panels, states=states, actions=actions)


def configuration(**settings):
    return BacktestConfig(start="2024-01-02", end="2024-02-01", initial_cash=100_000,
                          commission_rate=0, minimum_commission=0, slippage_bps=0,
                          participation_rate=1, min_history=1, **settings)


class SequenceStrategy(Strategy):
    def __init__(self, instructions):
        self.instructions = iter(instructions)

    def on_close(self, context):
        return next(self.instructions, None)


class EngineTests(unittest.TestCase):
    def test_close_signal_executes_next_open_and_excludes_prior_gap(self):
        data = dataset([10, 10.5, 10.5], opens=[10, 10.5, 10.5])
        result = BacktestEngine(data, configuration()).run(SequenceStrategy([{SYMBOL: 0.5}]))
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades.iloc[0].date, "2024-01-03")
        self.assertEqual(result.trades.iloc[0].price, 10.5)
        self.assertLess(abs(result.metrics["total_return"]), 0.0001)

    def test_none_preserves_holdings_instead_of_daily_reweighting(self):
        data = dataset([10, 10, 10.5, 10.8])
        result = BacktestEngine(data, configuration()).run(SequenceStrategy([{SYMBOL: 0.5}, None, None]))
        self.assertEqual(len(result.trades), 1)
        self.assertGreater(result.equity.exposure.iloc[-1], 0.5)

    def test_empty_targets_sells_on_following_day(self):
        result = BacktestEngine(dataset(), configuration()).run(SequenceStrategy([{SYMBOL: 0.5}, {}]))
        self.assertEqual(list(result.trades.side), ["buy", "sell"])
        self.assertEqual(result.trades.iloc[-1].date, "2024-01-04")

    def test_t_plus_one_blocks_same_day_sale(self):
        data = dataset()
        engine = BacktestEngine(data, configuration())
        engine._order(1, SYMBOL, "buy", 100, data.dates[0])
        self.assertEqual(engine._order(1, SYMBOL, "sell", 100, data.dates[0]), 0)
        self.assertEqual(engine._order(2, SYMBOL, "sell", 100, data.dates[1]), 100)

    def test_open_limit_up_blocks_buy(self):
        result = BacktestEngine(dataset([10, 11, 11]), configuration()).run(SequenceStrategy([{SYMBOL: 0.5}]))
        self.assertTrue(result.trades.empty)
        self.assertIn("open_at_limit_up", result.orders.reason.tolist())

    def test_open_limit_down_preserves_unsold_position(self):
        result = BacktestEngine(dataset([10, 10, 9]), configuration()).run(SequenceStrategy([{SYMBOL: 0.5}, {}]))
        self.assertEqual(result.trades.side.tolist(), ["buy"])
        self.assertIn("open_at_limit_down", result.orders.reason.tolist())
        self.assertGreater(result.equity.market_value.iloc[-1], 0)

    def test_ex_dividend_limit_reference_uses_current_factor(self):
        data = dataset([10, 9.9, 9.9], factors=[1, 10 / 9, 10 / 9])
        result = BacktestEngine(data, configuration()).run(SequenceStrategy([{SYMBOL: 0.5}]))
        self.assertTrue(result.trades.empty)
        self.assertIn("open_at_limit_up", result.orders.reason.tolist())

    def test_research_split_continuity_without_double_adjustment(self):
        data = dataset([10, 10, 5, 5], factors=[1, 1, 2, 2])
        result = BacktestEngine(data, configuration()).run(SequenceStrategy([{SYMBOL: 0.5}]))
        self.assertAlmostEqual(result.equity.equity.iloc[1], result.equity.equity.iloc[2])
        self.assertEqual(result.positions.iloc[-1].equivalent_shares, 2 * result.trades.quantity.iloc[0])

    def test_strict_account_requires_actions_for_held_factor_jump(self):
        data = dataset([10, 10, 5], factors=[1, 1, 2])
        with self.assertRaises(DataCoverageError):
            BacktestEngine(data, configuration(accounting="strict")).run(SequenceStrategy([{SYMBOL: 0.5}]))

    def test_strict_dividend_is_receivable_until_pay_date(self):
        actions = pd.DataFrame([{"date": "2024-01-04", "symbol": SYMBOL, "cash_per_share": 1,
                                 "share_multiplier": 1, "pay_date": "2024-01-05",
                                 "share_available_date": "2024-01-04"}])
        data = dataset([10, 10, 9, 9], factors=[1, 1, 10 / 9, 10 / 9], actions=actions)
        result = BacktestEngine(data, configuration(accounting="strict")).run(SequenceStrategy([{SYMBOL: 0.5}]))
        dividend = result.trades.quantity.iloc[0]
        self.assertEqual(result.equity.receivables.iloc[2], dividend)
        self.assertEqual(result.equity.receivables.iloc[3], 0)
        self.assertAlmostEqual(result.equity.equity.iloc[1], result.equity.equity.iloc[2])
        self.assertAlmostEqual(result.equity.cash.iloc[3] - result.equity.cash.iloc[2], dividend)

    def test_strict_bonus_shares_respect_available_date(self):
        actions = pd.DataFrame([{"date": "2024-01-04", "symbol": SYMBOL, "cash_per_share": 0,
                                 "share_multiplier": 2, "pay_date": "2024-01-04",
                                 "share_available_date": "2024-01-05"}])
        data = dataset([10, 10, 5, 5], factors=[1, 1, 2, 2], actions=actions)
        result = BacktestEngine(data, configuration(accounting="strict")).run(SequenceStrategy([{SYMBOL: 0.5}, {}, {}]))
        self.assertEqual(result.trades.quantity.tolist(), [result.trades.quantity.iloc[0]] * 3)
        self.assertEqual(result.equity.market_value.iloc[-1], 0)

    def test_missing_held_price_is_marked_stale_and_cannot_sell(self):
        data = dataset([10, 10, np.nan, 10])
        result = BacktestEngine(data, configuration()).run(SequenceStrategy([{SYMBOL: 0.5}, {}]))
        self.assertEqual(result.equity.stale_positions.iloc[2], 1)
        self.assertEqual(result.trades.side.tolist(), ["buy"])
        self.assertAlmostEqual(result.equity.equity.iloc[1], result.equity.equity.iloc[2])

    def test_zero_amount_blocks_execution(self):
        data = dataset(amounts=[1e9, 0, 1e9, 1e9])
        result = BacktestEngine(data, configuration()).run(SequenceStrategy([{SYMBOL: 0.5}]))
        self.assertTrue(result.trades.empty)
        self.assertIn("zero_amount", result.orders.reason.tolist())

    def test_liquidity_uses_previous_day_not_future_amount(self):
        data = dataset(amounts=[1000, 1e9, 1e9, 1e9])
        result = BacktestEngine(data, configuration()).run(SequenceStrategy([{SYMBOL: 0.5}]))
        self.assertEqual(result.trades.quantity.iloc[0], 100)

    def test_all_cash_target_does_not_overdraw_after_fees(self):
        config = replace(configuration(), commission_rate=0.0003, minimum_commission=5)
        result = BacktestEngine(dataset(), config).run(SequenceStrategy([{SYMBOL: 1.0}]))
        self.assertGreaterEqual(result.equity.cash.min(), 0)
        self.assertEqual(result.trades.quantity.iloc[0] % 100, 0)

    def test_strict_state_coverage_fails_closed(self):
        with self.assertRaises(DataCoverageError):
            BacktestEngine(dataset(), configuration(strict_rules=True)).run(SequenceStrategy([{SYMBOL: 0.5}]))

    def test_explicit_no_limit_overrides_normal_limits(self):
        states = pd.DataFrame([{"date": "2024-01-03", "symbol": SYMBOL, "is_st": False,
                                "suspended": False, "no_limit": True}])
        data = dataset([10, 15, 15], states=states)
        result = BacktestEngine(data, configuration(strict_rules=True)).run(SequenceStrategy([{SYMBOL: 0.5}]))
        self.assertEqual(len(result.trades), 1)

    def test_explicit_st_exclusion(self):
        states = pd.DataFrame([{"date": "2024-01-03", "symbol": SYMBOL, "is_st": True}])
        result = BacktestEngine(dataset(states=states), configuration()).run(SequenceStrategy([{SYMBOL: 0.5}]))
        self.assertIn("st_excluded", result.orders.reason.tolist())

    def test_missing_factor_blocks_new_buy(self):
        data = dataset(factors=[1, np.nan, 1, 1])
        result = BacktestEngine(data, configuration()).run(SequenceStrategy([{SYMBOL: 0.5}]))
        self.assertIn("missing_bar_or_factor", result.orders.reason.tolist())

    def test_final_signal_is_not_executed(self):
        result = BacktestEngine(dataset([10, 10]), configuration()).run(SequenceStrategy([None, {SYMBOL: 0.5}]))
        self.assertTrue(result.trades.empty)
        self.assertIn("last_signal_not_executed", result.events.event.tolist())

    def test_history_is_cutoff_and_copy(self):
        data = dataset([10, 10, 100, 100])
        context = Context(data, 1, configuration(), 1, 1, {})
        history = context.history("close", 100)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[SYMBOL].max(), 10)
        history.iloc[0, 0] = 999
        self.assertEqual(data.value("close", 0, SYMBOL), 10)

    def test_data_axis_mismatch_rejected(self):
        data = dataset()
        panels = {field: frame.copy() for field, frame in data._panels.items()}
        panels["amt"] = panels["amt"].iloc[::-1]
        with self.assertRaises(ValueError):
            DataBundle(panels)

    def test_new_listing_not_backfilled_into_universe(self):
        data = dataset([np.nan, np.nan, 10, 10])
        self.assertEqual(Context(data, 1, configuration(), 0, 0, {}).universe(), [])
        self.assertEqual(Context(data, 2, configuration(), 0, 0, {}).universe(), [SYMBOL])

    def test_export_has_metrics_trades_and_report(self):
        result = BacktestEngine(dataset(), configuration()).run(SequenceStrategy([{SYMBOL: 0.5}]))
        with tempfile.TemporaryDirectory() as directory:
            export_result(result, directory, "source snapshot")
            self.assertIn("沪深日频回测报告", (Path(directory) / "report.html").read_text())
            self.assertTrue((Path(directory) / "summary.json").exists())
            self.assertTrue((Path(directory) / "strategy.py").exists())

    def test_builtin_moving_average_and_custom_template_can_trade(self):
        root = Path(__file__).resolve().parents[1]
        data = dataset([10, 10.1, 10.2, 10.3, 10.4, 10.5])
        for filename, parameters in [("moving_average.py", {"fast": 2, "slow": 3, "rebalance_days": 1}),
                                     ("custom_template.py", {"lookback": 2, "top_n": 1})]:
            strategy = load_strategy(root / "strategies" / filename, parameters)
            result = BacktestEngine(data, configuration()).run(strategy)
            self.assertFalse(result.trades.empty)

    def test_partial_sale_keeps_nonzero_target_holdings(self):
        result = BacktestEngine(dataset(), configuration()).run(SequenceStrategy([{SYMBOL: 0.8}, {SYMBOL: 0.4}]))
        self.assertEqual(result.trades.side.tolist(), ["buy", "sell"])
        self.assertGreater(result.equity.market_value.iloc[-1], 0)

    def test_suspended_sale_does_not_fund_new_purchase(self):
        other = "600000.SH"
        states = pd.DataFrame([{"date": "2024-01-04", "symbol": SYMBOL, "suspended": True}])
        data = dataset([[10, 10]] * 4, symbols=[SYMBOL, other], states=states)
        result = BacktestEngine(data, configuration()).run(SequenceStrategy([{SYMBOL: 0.95}, {other: 0.95}]))
        purchases = result.trades[result.trades.symbol == other]
        self.assertLess(purchases.gross.sum(), 6000)
        self.assertIn("suspended", result.orders.reason.tolist())


class RuleTests(unittest.TestCase):
    def test_board_quantity_and_odd_lot_rules(self):
        self.assertEqual(order_quantity(SYMBOL, 199, "buy"), 100)
        self.assertEqual(order_quantity("688001.SH", 199, "buy"), 0)
        self.assertEqual(order_quantity("688001.SH", 201, "buy"), 201)
        self.assertEqual(order_quantity(SYMBOL, 57, "sell", 57), 57)
        self.assertEqual(order_quantity("688001.SH", 57, "sell", 57), 57)

    def test_historical_limit_regimes(self):
        self.assertEqual(limit_rate("300750.SZ", "2020-08-21"), 0.1)
        self.assertEqual(limit_rate("300750.SZ", "2020-08-24"), 0.2)
        self.assertEqual(limit_rate(SYMBOL, "2026-07-03", True), 0.05)
        self.assertEqual(limit_rate(SYMBOL, "2026-07-06", True), 0.1)
        self.assertEqual(limit_rate("688001.SH", "2024-01-01", True), 0.2)

    def test_fee_regime_dates_and_sell_only_stamp(self):
        config = configuration()
        self.assertEqual(trading_fees("2023-08-25", "sell", 100_000, config)["stamp_tax"], 100)
        self.assertEqual(trading_fees("2023-08-28", "sell", 100_000, config)["stamp_tax"], 50)
        self.assertEqual(trading_fees("2023-08-28", "buy", 100_000, config)["stamp_tax"], 0)
        self.assertEqual(trading_fees("2022-04-28", "buy", 100_000, config)["transfer_fee"], 2)
        self.assertEqual(trading_fees("2022-04-29", "buy", 100_000, config)["transfer_fee"], 1)

    def test_tick_half_up_rounding(self):
        self.assertEqual(tick_round(10.005), 10.01)

    def test_invalid_weights_and_leverage(self):
        for weights in ({SYMBOL: -0.1}, {SYMBOL: np.nan}, {SYMBOL: 1.1}, {"999999.SH": 0.1}):
            with self.assertRaises(ValueError):
                validate_targets(weights, [SYMBOL], 1)

    def test_strategy_module_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.py"
            path.write_text("class UserStrategy: pass")
            with self.assertRaises(ValueError):
                load_strategy(path)


if __name__ == "__main__":
    unittest.main()
