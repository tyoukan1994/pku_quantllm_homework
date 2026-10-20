import unittest

import numpy as np
import pandas as pd

from ashare.config import BacktestConfig
from ashare.engine import BacktestEngine
from ashare.literature import FACTORS, LiteratureStrategy, literature_scores
from ashare.strategy import Context
from test_technical import sample_data


class LiteratureTests(unittest.TestCase):
    def test_declared_literature_candidates_are_unique_and_constructible(self):
        from ashare.literature_discovery import candidate_list
        candidates = candidate_list()
        self.assertEqual(len(candidates), 12)
        self.assertEqual(len({row["id"] for row in candidates}), 12)
        for row in candidates:
            LiteratureStrategy(**row["parameters"])

    def test_alpha33_and_alpha101_price_formula(self):
        data = sample_data()
        panels = data._panels
        row = literature_scores(*(panels[name] for name in ("close", "open", "high", "low", "amt")))
        expected33 = panels["open"].iloc[-1] / panels["close"].iloc[-1] - 1
        expected101 = (panels["close"].iloc[-1] - panels["open"].iloc[-1]) / (panels["high"].iloc[-1] - panels["low"].iloc[-1] + 0.001)
        np.testing.assert_allclose(row.alpha33, expected33)
        np.testing.assert_allclose(row.alpha101, expected101)

    def test_52_week_high_requires_full_window(self):
        data = sample_data()
        arguments = [data._panels[name].iloc[:251] for name in ("close", "open", "high", "low", "amt")]
        self.assertTrue(literature_scores(*arguments).high252.isna().all())

    def test_amihud_uses_amount_not_number_of_trades(self):
        data = sample_data()
        panels = data._panels
        row = literature_scores(*(panels[name] for name in ("close", "open", "high", "low", "amt")))
        expected = (panels["close"].pct_change(fill_method=None).abs() / panels["amt"]).tail(20).mean()
        np.testing.assert_allclose(row.amihud_high, expected)
        np.testing.assert_allclose(row.amihud_low, -expected)

    def test_each_factor_produces_valid_long_only_targets(self):
        data = sample_data()
        context = Context(data, 279, BacktestConfig(), 1e9, 1e9, {})
        for factor in sorted(FACTORS):
            targets = LiteratureStrategy(factor=factor, top_n=3, liquid_pool=3).on_close(context)
            self.assertLessEqual(sum(targets.values()), 0.95 + 1e-12)
            self.assertTrue(all(value > 0 for value in targets.values()))

    def test_future_prices_cannot_change_current_targets(self):
        data = sample_data()
        changed = sample_data()
        for field in ("close", "open", "high", "low"):
            changed._panels[field].iloc[260:] *= 100
        for factor in sorted(FACTORS):
            first = LiteratureStrategy(factor=factor, top_n=2).on_close(Context(data, 259, BacktestConfig(), 1e9, 1e9, {}))
            second = LiteratureStrategy(factor=factor, top_n=2).on_close(Context(changed, 259, BacktestConfig(), 1e9, 1e9, {}))
            self.assertEqual(first, second)

    def test_account_uses_next_day_execution(self):
        data = sample_data()
        config = BacktestConfig(start=str(data.dates[210].date()), end=str(data.dates[220].date()), min_history=120)
        result = BacktestEngine(data, config).run(LiteratureStrategy(factor="amihud_low", top_n=3, liquid_pool=3))
        self.assertFalse(result.trades.empty)
        self.assertTrue((pd.to_datetime(result.trades.signal_date) < pd.to_datetime(result.trades.date)).all())

    def test_invalid_parameters(self):
        for parameters in ({"factor": "vwap_fake"}, {"top_n": True}, {"rebalance_days": 1.5},
                           {"investment": float("inf")}, {"minimum_amount": -1}):
            with self.assertRaises(ValueError):
                LiteratureStrategy(**parameters)


if __name__ == "__main__":
    unittest.main()
