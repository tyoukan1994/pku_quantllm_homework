import unittest

import numpy as np
import pandas as pd

from alpha_ml import (
    ALPHA158_FEATURES,
    adjusted_vwap,
    calculate_monthly_features,
    cross_sectional_rank_label,
    expanding_folds,
    robust_scale,
)
from research_policy import SampleLeakageError, require_feature_context_window


class AlphaMlTest(unittest.TestCase):
    def make_daily(self) -> pd.DataFrame:
        dates = pd.bdate_range("2023-08-01", periods=90)
        frames = []
        for offset, symbol in enumerate(("A", "B"), start=1):
            base = np.arange(len(dates), dtype=float) + 10 + offset
            close = base + np.sin(np.arange(len(dates)) / 3)
            open_ = close * (1 + 0.001 * np.cos(np.arange(len(dates))))
            high = np.maximum(open_, close) * 1.01
            low = np.minimum(open_, close) * 0.99
            volume = 1000 + offset * 100 + np.arange(len(dates)) * (3 + offset)
            frames.append(
                pd.DataFrame(
                    {
                        "datetime": dates,
                        "symbol": symbol,
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "total_turnover": volume * close,
                        "vwap": close,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def test_generates_all_158_features(self):
        daily = self.make_daily()
        signal_date = daily["datetime"].max()
        members = pd.DataFrame(
            {"symbol": ["A", "B"], "signal_date": [signal_date, signal_date]}
        )
        result = calculate_monthly_features(daily, members)
        self.assertEqual(len(ALPHA158_FEATURES), 158)
        self.assertEqual(result.shape, (2, 164))
        self.assertTrue(set(ALPHA158_FEATURES).issubset(result.columns))
        self.assertTrue(np.isfinite(result["ma_60"]).all())

    def test_rank_label_is_cross_sectional(self):
        frame = pd.DataFrame(
            {
                "signal_date": ["2024-01-31"] * 3,
                "forward_return": [-0.1, 0.0, 0.1],
            }
        )
        label = cross_sectional_rank_label(frame)
        self.assertTrue(label.is_monotonic_increasing)
        self.assertAlmostEqual(float(label.iloc[1]), 3.46 * (2 / 3 - 0.5))

    def test_scaler_fits_train_only(self):
        train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        test = pd.DataFrame({"x": [1000.0]})
        train_scaled, test_scaled = robust_scale(train, test, columns=["x"])
        self.assertAlmostEqual(float(train_scaled["x"].median()), 0.0)
        self.assertEqual(float(test_scaled["x"].iloc[0]), 3.0)

    def test_expanding_folds_do_not_overlap(self):
        for fold in expanding_folds():
            self.assertLess(fold.train_end, fold.valid_year)
            self.assertLess(fold.valid_year, fold.test_year)

    def test_feature_context_still_seals_2025(self):
        require_feature_context_window("2014-09-01", "2024-12-31")
        with self.assertRaises(SampleLeakageError):
            require_feature_context_window("2024-09-01", "2025-01-02")

    def test_vwap_uses_raw_volume_before_price_adjustment(self):
        value = adjusted_vwap(
            pd.Series([5.0]),
            pd.Series([10.0]),
            pd.Series([100.0]),
            pd.Series([1_020.0]),
        )
        self.assertAlmostEqual(float(value.iloc[0]), 5.1)


if __name__ == "__main__":
    unittest.main()
