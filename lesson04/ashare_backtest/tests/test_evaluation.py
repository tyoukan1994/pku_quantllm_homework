from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ashare.evaluation import EvaluationConfig, collect_observations, dynamic_diagnostics, evaluate_observations, hac_mean
from ashare.platform.evaluations import EvaluationStore
from test_engine import dataset


def evaluation_config(**overrides):
    values = dict(start="2024-01-02", end="2025-12-31", factors=("custom",), horizons=(1, 5),
                  windows=(20, 60), fast_window=20, reference_window=60, block_window=40,
                  min_stocks=5, min_history=1, require_st_history=False)
    values.update(overrides)
    return EvaluationConfig(**values)


def observations_from_series(values, horizon=5):
    dates = pd.bdate_range("2024-01-02", periods=len(values) + horizon + 1)
    rows = []
    for position, value in enumerate(values):
        row = dict(date=dates[position], label_start=dates[position + 1], label_end=dates[position + horizon + 1],
                   factor="custom", horizon=horizon, rank_ic=value, pearson_ic=value,
                   stocks=100, eligible_stocks=100, coverage=1.0, spread=value / 100)
        row.update({f"q{group}": value * group / 100 for group in range(1, 6)})
        rows.append(row)
    return pd.DataFrame(rows), dates


class EvaluationTests(unittest.TestCase):
    def test_config_rejects_invalid_or_automatic_direction(self):
        for values in ({"direction": 0}, {"fast_window": 5}, {"reference_window": 20},
                       {"min_fraction": 0}, {"minimum_amount": float("nan")}, {"horizons": (0,)},
                       {"horizons": (5, 5)}, {"effect_floor": -0.1}, {"factors": ()}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                evaluation_config(**values)

    def test_future_labels_start_next_open_and_stop_at_boundary(self):
        symbols = [f"00000{number}.SZ" for number in range(1, 7)]
        prices = np.exp(np.arange(30)[:, None] * np.arange(1, 7)[None, :] / 100)
        data = dataset(prices, symbols=symbols)
        config = evaluation_config(end=str(data.dates[20].date()))
        provider = lambda context: pd.DataFrame({"custom": np.arange(6)}, index=context.universe())
        observations = collect_observations(data, config, provider)
        self.assertTrue((observations.label_end <= data.dates[20]).all())
        self.assertEqual(observations.iloc[0].label_start, data.dates[1])
        self.assertEqual(observations[observations.horizon == 5].iloc[0].label_end, data.dates[6])
        self.assertTrue(np.allclose(observations.rank_ic, 1))

    def test_unknown_st_excluded_without_looking_at_future_status(self):
        symbols = [f"00000{number}.SZ" for number in range(1, 7)]
        data = dataset(np.full((30, 6), 10.0), symbols=symbols)
        config = evaluation_config(require_st_history=True)
        provider = lambda context: pd.DataFrame({"custom": np.arange(6)}, index=symbols)
        observations = collect_observations(data, config, provider)
        self.assertTrue((observations.eligible_stocks == 0).all())
        self.assertTrue(observations.rank_ic.isna().all())

    def test_constant_cross_section_is_not_zero_ic(self):
        symbols = [f"00000{number}.SZ" for number in range(1, 7)]
        data = dataset(np.exp(np.arange(30)[:, None] * np.arange(1, 7)[None, :] / 100), symbols=symbols)
        observations = collect_observations(data, evaluation_config(),
            lambda context: pd.DataFrame({"custom": 1.0}, index=context.universe()))
        self.assertTrue(observations.rank_ic.isna().all())

    def test_overlapping_returns_use_hac_not_iid_standard_error(self):
        generator = np.random.default_rng(42)
        innovations = generator.normal(size=800)
        values = np.convolve(innovations, np.ones(5) / 5, mode="valid")
        self.assertGreater(hac_mean(values, 4)["se"], hac_mean(values, 0)["se"] * 1.5)

    def test_zero_variance_does_not_create_infinite_certainty(self):
        stats = hac_mean(np.ones(100) * 0.1, 5)
        self.assertTrue(np.isnan(stats["ir"]))
        self.assertTrue(np.isnan(stats["lower"]))

    def test_short_window_long_horizon_has_no_false_precision(self):
        stats = hac_mean(np.linspace(-0.1, 0.1, 21), 19)
        self.assertTrue(np.isfinite(stats["mean"]))
        self.assertTrue(np.isnan(stats["lower"]))

    def test_wrong_horizon_or_entry_day_is_rejected(self):
        observations, calendar = observations_from_series(np.linspace(0, 0.2, 50))
        observations.loc[0, "horizon"] = 1
        with self.assertRaisesRegex(ValueError, "严格匹配"):
            evaluate_observations(observations, calendar, evaluation_config())

    def test_maturity_alignment_does_not_backdate_ic(self):
        observations, calendar = observations_from_series(np.linspace(0.01, 0.2, 100))
        result = evaluate_observations(observations, calendar, evaluation_config(horizons=(5,)))
        dynamic = result["dynamic"]
        self.assertEqual(dynamic.iloc[24].state, "样本不足")
        self.assertEqual(dynamic.iloc[25]["count_20"], 20)
        expected = observations.iloc[:20].rank_ic.mean()
        self.assertAlmostEqual(dynamic.iloc[25]["mean_20"], expected)

    def test_future_changes_cannot_rewrite_past_diagnostics(self):
        generator = np.random.default_rng(7)
        values = generator.normal(0.1, 0.1, 180)
        dates = pd.bdate_range("2024-01-02", periods=len(values))
        first = dynamic_diagnostics(pd.Series(values, index=dates), 5, evaluation_config())
        changed = values.copy()
        changed[100:] = -0.8
        second = dynamic_diagnostics(pd.Series(changed, index=dates), 5, evaluation_config())
        pd.testing.assert_frame_equal(first.iloc[:100], second.iloc[:100])

    def test_missing_days_are_not_compressed_into_recent_window(self):
        values = np.r_[np.linspace(0.1, 0.2, 60), np.full(20, np.nan)]
        dates = pd.bdate_range("2024-01-02", periods=len(values))
        result = dynamic_diagnostics(pd.Series(values, index=dates), 5, evaluation_config())
        self.assertEqual(result.iloc[-1].state, "样本不足")
        self.assertEqual(result.iloc[-1]["count_20"], 0)

    def test_detects_reversal_and_respects_fixed_negative_direction(self):
        generator = np.random.default_rng(9)
        values = np.r_[generator.normal(0.2, 0.025, 120), generator.normal(-0.2, 0.025, 30)]
        dates = pd.bdate_range("2024-01-02", periods=len(values))
        result = dynamic_diagnostics(pd.Series(values, index=dates), 5, evaluation_config())
        self.assertEqual(result.iloc[-1].state, "近期反向")
        self.assertTrue(result.iloc[-1].persistent_alert)
        negative = dynamic_diagnostics(pd.Series(-values[:100], index=dates[:100]), 5, evaluation_config(direction=-1))
        self.assertEqual(negative.iloc[-1].state, "正向支持")

    def test_weakening_detected_against_purged_nonoverlapping_reference(self):
        generator = np.random.default_rng(11)
        values = np.r_[generator.normal(0.3, 0.02, 100), generator.normal(0.06, 0.02, 20)]
        dates = pd.bdate_range("2024-01-02", periods=len(values))
        result = dynamic_diagnostics(pd.Series(values, index=dates), 5, evaluation_config())
        self.assertEqual(result.iloc[-1].state, "近期衰减")
        self.assertAlmostEqual(result.iloc[-1].reference_mean, values[36:96].mean())

    def test_blocks_do_not_share_return_intervals_or_count_incomplete_tail(self):
        observations, calendar = observations_from_series(np.sin(np.arange(120)) * 0.1)
        result = evaluate_observations(observations, calendar, evaluation_config(horizons=(5,)))
        blocks = result["blocks"]
        self.assertFalse(blocks.iloc[-1].usable)
        self.assertEqual(blocks.iloc[0]["count"], 34)
        self.assertEqual(result["transitions"].iloc[0].adjacent_pairs, 2)

    def test_boundary_and_duplicate_observations_are_rejected(self):
        observations, calendar = observations_from_series(np.linspace(0, 0.2, 50))
        with self.assertRaises(ValueError):
            evaluate_observations(pd.concat([observations, observations.iloc[:1]]), calendar, evaluation_config())
        observations.loc[0, "label_end"] = calendar[-1] + pd.Timedelta(days=1)
        with self.assertRaises(ValueError):
            evaluate_observations(observations, calendar, evaluation_config())

    def test_store_has_unique_runs_integrity_checks_and_failed_records(self):
        symbols = [f"00000{number}.SZ" for number in range(1, 7)]
        generator = np.random.default_rng(8)
        data = dataset(10 * np.exp(np.cumsum(generator.normal(0, 0.01, (150, 6)), axis=0)), symbols=symbols)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data"
            source.mkdir()
            for name, panel in data._panels.items():
                panel.reset_index().to_feather(source / (name + ".feather"))
            store = EvaluationStore(Path(directory) / "evaluations")
            config = evaluation_config(factors=("reversal5",), minimum_amount=0)
            identifier = store.run(source, None, config)
            second = store.run(source, None, config)
            self.assertNotEqual(identifier, second)
            request, tables = store.load(identifier)
            self.assertEqual(request["status"], "completed")
            self.assertEqual(len(tables["static"]), 2)
            with (store.path(identifier) / "static.csv").open("a") as handle:
                handle.write("changed")
            with self.assertRaises(ValueError):
                store.load(identifier)
            with self.assertRaises(ValueError):
                store.run(source, None, replace(config, factors=("unknown",)))
            self.assertIn("failed", {record["status"] for record in store.list()})
            with self.assertRaises(ValueError):
                store.path("../outside")


if __name__ == "__main__":
    unittest.main()
