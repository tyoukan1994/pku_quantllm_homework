from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from ashare.config import BacktestConfig
from ashare.data import DataBundle
from ashare.engine import BacktestEngine
from ashare.strategy import Context
from ashare.technical import (OnlineFeatures, PreparedFeatures, TechnicalStrategy,
                              prepare_features, wilder_rsi)


def sample_data():
    dates = pd.bdate_range("2020-01-02", periods=280)
    generator = np.random.default_rng(81)
    columns = ["000001.SZ", "600000.SH", "300750.SZ"]
    close = pd.DataFrame(20 * np.exp(np.cumsum(generator.normal(0, 0.025, (280, 3)), axis=0)), index=dates, columns=columns)
    close.iloc[230:232, 1] = np.nan
    panels = {"close": close, "open": close * 0.999, "high": close * 1.02, "low": close * 0.98,
              "cumadj": close * 0 + 1, "amt": close * 0 + 2_000_000_000}
    return DataBundle(panels)


class TechnicalTests(unittest.TestCase):
    def test_wilder_seed_and_smoothing(self):
        close = pd.DataFrame({"stock": [10, 11, 12, 11, 13, 12]})
        result = wilder_rsi(close, 3).stock
        self.assertTrue(result.iloc[:3].isna().all())
        self.assertAlmostEqual(result.iloc[3], 100 * 2 / 3)
        self.assertAlmostEqual(result.iloc[4], 100 * 10 / 12)
        self.assertAlmostEqual(result.iloc[5], 100 * 20 / 33)

    def test_extremes_flat_and_missing_restart(self):
        close = pd.DataFrame({"rise": np.arange(25), "fall": -np.arange(25), "flat": np.ones(25)})
        result = wilder_rsi(close, 14)
        np.testing.assert_array_equal(result.iloc[-1], [100, 0, 50])
        close.iloc[19, 0] = np.nan
        result = wilder_rsi(close, 2)
        self.assertTrue(result.rise.iloc[19:22].isna().all())
        self.assertEqual(result.rise.iloc[22], 100)

    def test_batch_features_do_not_depend_on_future(self):
        data = sample_data()
        arguments = [data._panels[name] for name in ("close", "amt", "high", "low")]
        all_features = prepare_features(*arguments)
        prefix = prepare_features(*(frame.iloc[:220] for frame in arguments))
        for name in all_features:
            pd.testing.assert_frame_equal(all_features[name].iloc[:220], prefix[name])

    def test_online_and_prepared_features_agree(self):
        data = sample_data()
        prepared = PreparedFeatures(prepare_features(*(data._panels[name] for name in ("close", "amt", "high", "low"))))
        online = OnlineFeatures()
        config = BacktestConfig()
        for position in (210, 211, 215, 232, 250, 279):
            context = Context(data, position, config, 1_000_000, 1_000_000, {})
            np.testing.assert_allclose(online.at(context), prepared.at(context), atol=1e-6, rtol=1e-5, equal_nan=True)

    def test_prepared_and_online_account_equity_agree(self):
        data = sample_data()
        config = BacktestConfig(start=str(data.dates[210].date()), end=str(data.dates[240].date()), min_history=120)
        for mode in ("rsi_pullback", "lowvol_reversal", "breakout"):
            parameters = dict(mode=mode, top_n=3, liquid_pool=3, entry_threshold=40, rebalance_days=5)
            online = TechnicalStrategy(**parameters)
            prepared = TechnicalStrategy(**parameters)
            prepared.features = PreparedFeatures(prepare_features(*(data._panels[name] for name in ("close", "amt", "high", "low"))))
            first = BacktestEngine(data, config).run(online)
            second = BacktestEngine(data, config).run(prepared)
            pd.testing.assert_frame_equal(first.equity, second.equity)
            pd.testing.assert_frame_equal(first.trades, second.trades)

    def test_targets_leave_unused_slots_in_cash(self):
        data = sample_data()
        context = Context(data, 210, BacktestConfig(), 1e9, 1e9, {})
        strategy = TechnicalStrategy(mode="liquid_equal", top_n=100)
        targets = strategy.on_close(context)
        self.assertEqual(len(targets), 3)
        self.assertAlmostEqual(sum(targets.values()), 0.0285)

    def test_age_starts_from_actual_holding_not_unfilled_signal(self):
        data = sample_data()
        strategy = TechnicalStrategy(mode="rsi_pullback", top_n=3, entry_threshold=99, exit_threshold=100)
        config = BacktestConfig()
        for position in (210, 211, 212):
            strategy.on_close(Context(data, position, config, 1e9, 1e9, {}))
        self.assertEqual(strategy.ages, {})
        strategy.on_close(Context(data, 213, config, 1e9, 1e9, {"000001.SZ": 100}))
        self.assertEqual(strategy.ages["000001.SZ"], 0)

    def test_invalid_parameters(self):
        for parameters in ({"lookback": 0}, {"mode": "future"}, {"entry_threshold": 70},
                           {"mode": "breakout", "lookback": 5}, {"investment": float("nan")},
                           {"top_n": True}, {"top_n": 2.5}, {"trend_window": 5}):
            with self.assertRaises(ValueError):
                TechnicalStrategy(**parameters)


if __name__ == "__main__":
    unittest.main()
