from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ashare import BacktestEngine
from ashare.factors import factor_scores
from ashare.research import experiment_config, training_ic
from ashare.rules import max_order_quantity, trading_fees
from ashare.strategy import Context, load_strategy
from test_engine import SYMBOL, SequenceStrategy, configuration, dataset


class ResearchTests(unittest.TestCase):
    def test_requested_fees_override_history(self):
        config = experiment_config()
        for date in ("2020-01-02", "2025-01-02"):
            self.assertEqual(trading_fees(date, "buy", 100_000, config)["total"], 10)
            self.assertEqual(trading_fees(date, "sell", 100_000, config)["total"], 60)

    def test_unknown_st_is_not_treated_as_ordinary(self):
        data = dataset()
        config = configuration(require_st_history=True)
        self.assertEqual(Context(data, 1, config, 1, 1, {}).universe(), [])
        result = BacktestEngine(data, config).run(SequenceStrategy([{SYMBOL: 0.5}]))
        self.assertTrue(result.trades.empty)
        self.assertIn("unknown_st_status", result.orders.reason.tolist())

    def test_dense_metadata_known_false_and_missing_remain_distinct(self):
        data = dataset()
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame({"date": data.dates[:2], SYMBOL: [False, True]}).to_feather(Path(directory) / "is_st.feather")
            data.add_metadata(directory)
        config = configuration(require_st_history=True)
        self.assertEqual(Context(data, 0, config, 1, 1, {}).universe(), [SYMBOL])
        self.assertEqual(Context(data, 1, config, 1, 1, {}).universe(), [])
        self.assertEqual(Context(data, 2, config, 1, 1, {}).universe(), [])

    def test_st_switch_on_execution_day_blocks_pending_buy(self):
        data = dataset(states=pd.DataFrame([{"date": "2024-01-02", "symbol": SYMBOL, "is_st": False},
                                             {"date": "2024-01-03", "symbol": SYMBOL, "is_st": True}]))
        result = BacktestEngine(data, configuration(require_st_history=True)).run(SequenceStrategy([{SYMBOL: 0.5}]))
        self.assertIn("st_excluded", result.orders.reason.tolist())

    def test_previous_close_signal_builds_on_first_evaluation_day(self):
        config = replace(configuration(seed_previous_close=True), start="2024-01-03")
        result = BacktestEngine(dataset([10, 10.5, 10.5], opens=[10, 10, 10.5]), config).run(SequenceStrategy([{SYMBOL: 0.5}]))
        self.assertEqual(result.trades.iloc[0].date, "2024-01-03")
        self.assertEqual(result.trades.iloc[0].signal_date, "2024-01-02")
        self.assertGreater(result.equity.daily_return.iloc[0], 0)
        self.assertEqual(result.metrics["return_periods"], 2)
        self.assertAlmostEqual(np.prod(1 + result.equity.daily_return), 1 + result.metrics["total_return"])

    def test_terminal_exit_does_not_execute_new_buys(self):
        result = BacktestEngine(dataset(), configuration(liquidate_on_end=True)).run(SequenceStrategy([{SYMBOL: 0.5}, None, {SYMBOL: 1}]))
        self.assertEqual(result.trades.side.tolist(), ["buy", "sell"])
        self.assertEqual(result.trades.iloc[-1].date, "2024-01-05")
        self.assertTrue(result.metrics["liquidation_complete"])

    def test_terminal_exit_cannot_override_limit_down(self):
        result = BacktestEngine(dataset([10, 10, 10, 9]), configuration(liquidate_on_end=True)).run(SequenceStrategy([{SYMBOL: 0.5}]))
        self.assertFalse(result.metrics["liquidation_complete"])
        self.assertEqual(result.metrics["terminal_position_count"], 1)
        self.assertIn("open_at_limit_down", result.orders.reason.tolist())

    def test_prearranged_exit_window_retries_without_new_buys(self):
        data = dataset([10] * 8, amounts=[1e9, 1e9, 1e9, 1e9, 10_000, 10_000, 10_000, 10_000])
        config = configuration(liquidate_on_end=True, liquidation_window=3)
        result = BacktestEngine(data, config).run(SequenceStrategy([{SYMBOL: 0.5}] * 8))
        exits = result.trades[result.trades.date >= "2024-01-09"]
        self.assertEqual(set(exits.side), {"sell"})
        self.assertGreater(len(exits), 1)
        self.assertIn("scheduled_exit_window", result.events.event.tolist())

    def test_fractional_writeoff_is_loss_not_fake_sell_or_cash(self):
        data = dataset([10, 10, 10, 10], factors=[1, 1, 1.00013, 1.00013])
        config = replace(configuration(liquidate_on_end=True, write_off_research_dust=True), transfer_fee_rate=0)
        result = BacktestEngine(data, config).run(SequenceStrategy([{SYMBOL: 0.5}]))
        self.assertTrue(result.metrics["liquidation_complete"])
        self.assertGreater(result.metrics["research_dust_writeoff"], 0)
        self.assertEqual(result.trades.side.tolist(), ["buy", "sell"])
        self.assertTrue((result.trades.quantity % 1 == 0).all())
        self.assertAlmostEqual(result.metrics["final_equity"], result.metrics["final_cash"])

    def test_split_orders_share_liquidity_cap(self):
        config = replace(configuration(split_orders=True), initial_cash=1e9)
        data = dataset(amounts=[1e9, 50e6, 1e9, 1e9])
        engine = BacktestEngine(data, config)
        engine._submit_order(1, SYMBOL, "buy", 2_500_000, data.dates[0])
        self.assertEqual([record["quantity"] for record in engine.trade_records], [1_000_000, 1_000_000, 500_000])
        self.assertTrue(all(record["quantity"] <= max_order_quantity(SYMBOL) for record in engine.trade_records))
        engine._submit_order(2, SYMBOL, "buy", 10_000_000, data.dates[1])
        today = [record for record in engine.trade_records if record["date"] == "2024-01-04"]
        self.assertLessEqual(sum(record["gross"] for record in today), 50e6)

    def test_factor_definition_uses_adjusted_close_return_and_amount(self):
        dates = pd.bdate_range("2023-01-02", periods=61)
        close = pd.DataFrame({SYMBOL: np.arange(100, 161)}, index=dates)
        amount = close * 0 + 1e8
        scores = factor_scores(close, amount.tail(20), close.tail(5) + 1, close.tail(5) - 1)
        self.assertAlmostEqual(scores.loc[SYMBOL, "reversal5"], -(160 / 155 - 1))
        self.assertAlmostEqual(scores.loc[SYMBOL, "amount_reversal5"], scores.loc[SYMBOL, "reversal5"])
        self.assertAlmostEqual(scores.loc[SYMBOL, "close_pressure5"], 0)

    def test_future_market_changes_do_not_change_past_signals(self):
        root = Path(__file__).resolve().parents[1]
        prices = np.arange(100, 180, dtype=float)
        data = dataset(prices)
        changed = dataset(np.r_[prices[:66], np.full(14, 9999)])
        config = configuration()
        first = load_strategy(root / "strategies/price_amount.py", {"top_n": 1})
        second = load_strategy(root / "strategies/price_amount.py", {"top_n": 1})
        for position in range(60, 66):
            self.assertEqual(first.on_close(Context(data, position, config, 1e9, 1e9, {})),
                             second.on_close(Context(changed, position, config, 1e9, 1e9, {})))

    def test_training_labels_never_extend_past_training_end(self):
        data = dataset(np.tile(np.linspace(100, 120, 85)[:, None], (1, 40)) * np.linspace(1, 1.1, 40),
                       symbols=[f"{position:06d}.SZ" for position in range(40)])
        observations, summary = training_ic(data, configuration())
        self.assertFalse(observations.empty)
        self.assertLessEqual(pd.Timestamp(observations.label_end.max()), data.dates[-1])
        self.assertTrue((pd.to_datetime(observations.label_start) > pd.to_datetime(observations.date)).all())


if __name__ == "__main__":
    unittest.main()
