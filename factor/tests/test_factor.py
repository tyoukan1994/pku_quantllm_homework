import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_source import CsvBarDataSource, JqDataSource, normalize_bars  # noqa: E402
from momentum import compute_volatility_adjusted_momentum  # noqa: E402
from research_policy import (  # noqa: E402
    SampleLeakageError,
    require_discovery_window,
    require_final_feature_window,
    require_final_return_window,
    require_final_signal_window,
)
from ff3_combo import generate_weight_grid, prepare_features  # noqa: E402


def sample_bars(count=40):
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=count, freq="D"),
            "symbol": ["rb.main"] * count,
            "open": [100 + index for index in range(count)],
            "high": [101 + index for index in range(count)],
            "low": [99 + index for index in range(count)],
            "close": [100 + index for index in range(count)],
            "volume": [1000 + index for index in range(count)],
        }
    )


class FactorTests(unittest.TestCase):
    def test_ff3_weight_grid_has_35_combinations(self):
        weights = generate_weight_grid()
        self.assertEqual(len(weights), 35)
        self.assertTrue(all(abs(sum(row.values()) - 1) < 1e-12 for row in weights))

    def test_ff3_feature_preparation_is_cross_sectionally_normalized(self):
        rows = []
        for date in pd.to_datetime(["2022-01-31", "2022-02-28"]):
            for index in range(20):
                rows.append(
                    {
                        "signal_date": date,
                        "return_date": date + pd.offsets.MonthEnd(),
                        "symbol": f"stock{index:02d}",
                        "market_cap": 100 + index,
                        "book_to_market_ratio_lf": 0.5 + index / 100,
                        "return_on_equity_ttm": 5 + index,
                        "RSI10": 30 + index,
                        "forward_return": index / 1000,
                    }
                )
        result = prepare_features(pd.DataFrame(rows))
        means = result.groupby("signal_date")[["value", "size", "quality", "technical"]].mean()
        self.assertTrue((means.abs() < 1e-10).all().all())

    def test_discovery_window_accepts_2015_through_2024(self):
        start, end = require_discovery_window("2015-01-01", "2024-12-31")
        self.assertEqual(start.isoformat(), "2015-01-01")
        self.assertEqual(end.isoformat(), "2024-12-31")

    def test_discovery_window_rejects_sealed_sample(self):
        with self.assertRaisesRegex(SampleLeakageError, "受保护区间"):
            require_discovery_window("2024-12-01", "2025-01-01")

    def test_final_holdout_windows_accept_only_frozen_ranges(self):
        self.assertEqual(
            require_final_signal_window("2024-12-01", "2025-11-30"),
            (pd.Timestamp("2024-12-01").date(), pd.Timestamp("2025-11-30").date()),
        )
        self.assertEqual(
            require_final_return_window("2025-01-01", "2025-12-31"),
            (pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-12-31").date()),
        )
        require_final_feature_window("2024-09-01", "2025-11-30")
        with self.assertRaises(SampleLeakageError):
            require_final_return_window("2025-01-01", "2026-01-01")

    def test_normalize_rejects_duplicate_datetimes(self):
        bars = sample_bars(3)
        bars.loc[2, "datetime"] = bars.loc[1, "datetime"]
        with self.assertRaisesRegex(ValueError, "重复"):
            normalize_bars(bars)

    def test_signal_is_shifted_before_strategy_return(self):
        result = compute_volatility_adjusted_momentum(sample_bars(), lookback=5)
        pd.testing.assert_series_equal(
            result["position"].iloc[1:].reset_index(drop=True),
            result["signal"].shift(1).iloc[1:].reset_index(drop=True),
            check_names=False,
        )

    def test_changing_last_close_cannot_change_prior_results(self):
        bars = sample_bars()
        original = compute_volatility_adjusted_momentum(bars, lookback=5)
        changed = bars.copy()
        changed.loc[len(changed) - 1, "close"] *= 10
        revised = compute_volatility_adjusted_momentum(changed, lookback=5)
        pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])

    def test_csv_source_filters_symbol(self):
        bars = sample_bars(5)
        extra = bars.copy()
        extra["symbol"] = "au.main"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bars.csv"
            pd.concat([bars, extra], ignore_index=True).to_csv(path, index=False)
            loaded = CsvBarDataSource(path).load_bars("RB.MAIN")
        self.assertEqual(len(loaded), 5)
        self.assertEqual(set(loaded["symbol"]), {"rb.main"})

    def test_jqdata_source_normalizes_history(self):
        class FakeJqData:
            def __init__(self):
                self.auth_calls = []
                self.price_kwargs = None

            def auth(self, username, password):
                self.auth_calls.append((username, password))

            def is_auth(self):
                return bool(self.auth_calls)

            def get_price(self, **kwargs):
                self.price_kwargs = kwargs
                bars = sample_bars(3).set_index("datetime")
                return bars[["open", "high", "low", "close", "volume"]]

        sdk = FakeJqData()
        source = JqDataSource("user", "secret", sdk=sdk)
        loaded = source.load_bars("000300.XSHG", "2026-01-01", "2026-01-03")

        self.assertEqual(sdk.auth_calls, [("user", "secret")])
        self.assertEqual(set(loaded["symbol"]), {"000300.XSHG"})
        self.assertEqual(len(loaded), 3)
        self.assertIsNone(sdk.price_kwargs["fq"])
        self.assertFalse(sdk.price_kwargs["panel"])

    def test_jqdata_password_is_hidden_from_repr(self):
        source = JqDataSource("user", "secret", sdk=object())
        self.assertNotIn("secret", repr(source))


if __name__ == "__main__":
    unittest.main()
