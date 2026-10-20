from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ashare import BacktestConfig
from ashare.cli import run_backtest
from ashare.platform import RunStore, StrategyStore, StudyStore
from ashare.platform.runs import read_json, write_json
from ashare.platform.studies import expand_grid


SOURCE = '''from ashare import Strategy

class UserStrategy(Strategy):
    def __init__(self, weight=0.5):
        self.weight = weight

    def on_close(self, context):
        if "000001.SZ" in context.universe():
            return {"000001.SZ": self.weight}
        return {}
'''


class PlatformTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "data"
        self.source.mkdir()
        dates = pd.bdate_range("2024-01-02", periods=80)
        prices = 10 + np.sin(np.arange(80) / 5) * 0.1
        for field, values in {"open": prices, "close": prices, "high": prices + 0.1,
                              "low": prices - 0.1, "amt": prices * 0 + 1e8,
                              "cumadj": prices * 0 + 1}.items():
            pd.DataFrame({"date": dates, "000001.SZ": values}).to_feather(self.source / f"{field}.feather")
        self.metadata = self.root / "metadata"
        self.metadata.mkdir()
        pd.DataFrame({"date": dates, "000001.SZ": False}).to_feather(self.metadata / "is_st.feather")
        self.strategies = StrategyStore(self.root / "strategies")
        self.strategy = self.strategies.save("测试策略", SOURCE, {"weight": 0.5})
        self.runs = RunStore(self.root / "runs")
        self.studies = StudyStore(self.root / "studies", self.runs)
        self.config = BacktestConfig(start="2024-01-03", end="2024-01-31", initial_cash=1e6,
                                      commission_rate=0.0001, minimum_commission=0, stamp_tax_rate=0.0005,
                                      transfer_fee_rate=0, slippage_bps=0, participation_rate=1,
                                      min_history=1, require_st_history=True, seed_previous_close=True)

    def run_once(self, weight=0.5, config=None):
        return self.runs.run(self.source, self.metadata, self.strategy, {"weight": weight}, config or self.config)

    def search(self):
        return self.studies.search(self.source, self.metadata, self.strategy, {}, {"weight": [0.3, 0.6]}, self.config)

    def validation(self):
        return replace(self.config, start="2024-02-01", end="2024-02-29")

    def test_library_has_multiple_templates_and_versioned_user_sources(self):
        self.assertGreaterEqual(sum(spec.builtin for spec in self.strategies.list()), 6)
        updated = self.strategies.save("修改后", SOURCE + "\n", {"weight": 0.2}, self.strategy.identifier)
        self.assertEqual(updated.revision, 2)
        self.assertEqual(updated.identifier, self.strategy.identifier)
        self.assertEqual(self.strategy.path.read_text(), SOURCE)
        self.assertEqual(updated.parameters, {"weight": 0.2})

    def test_saving_source_does_not_execute_it(self):
        marker = self.root / "must_not_exist"
        self.strategies.save("不执行", f"from pathlib import Path\nPath({str(marker)!r}).touch()\n" + SOURCE, {})
        self.assertFalse(marker.exists())

    def test_cannot_overwrite_builtin_or_save_invalid_contract(self):
        with self.assertRaises(ValueError):
            self.strategies.save("模板", SOURCE, {}, "builtin:momentum")
        for source in ("pass", "class Other: pass"):
            with self.assertRaises(ValueError):
                self.strategies.save("错误", source, {})
        with self.assertRaises(ValueError):
            self.strategies.save("错误", SOURCE, [])

    def test_run_persists_inputs_and_immutable_source(self):
        identifier, result = self.run_once()
        directory = self.runs.path(identifier)
        request = read_json(directory / "request.json")
        self.assertEqual(request["status"], "completed")
        self.assertEqual(request["parameters"], {"weight": 0.5})
        self.assertEqual(request["strategy_sha256"], hashlib.sha256(SOURCE.encode()).hexdigest())
        self.strategies.save("新版本", SOURCE + "\n", {}, self.strategy.identifier)
        self.assertEqual((directory / "strategy.py").read_text(), SOURCE)
        self.assertEqual(self.runs.load(identifier).metrics, result.metrics)

    def test_failed_run_remains_in_catalog(self):
        with self.assertRaises(ValueError):
            self.run_once(weight=1.5)
        records = self.runs.list()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "failed")
        self.assertTrue(records[0]["error"])

    def test_old_summaries_are_cataloged_without_special_paths(self):
        identifier, result = self.run_once()
        (self.runs.path(identifier) / "request.json").unlink()
        self.assertEqual(self.runs.list()[0]["kind"], "legacy")
        self.assertEqual(self.runs.load(identifier).metrics, result.metrics)

    def test_catalog_orders_different_timezones_by_actual_time(self):
        earlier, result = self.run_once()
        later, result = self.run_once()
        for identifier, created in ((earlier, "2026-10-20T10:00:00+08:00"),
                                    (later, "2026-10-20T03:00:00+00:00")):
            path = self.runs.path(identifier) / "request.json"
            request = read_json(path)
            request["created_at"] = created
            write_json(path, request)
        self.assertEqual([record["id"] for record in self.runs.list()], [later, earlier])

    def test_comparison_accepts_same_setup_and_flags_different_costs(self):
        first, result = self.run_once(0.3)
        second, result = self.run_once(0.6)
        table, curves, differences = self.runs.comparison([first, second])
        self.assertEqual(len(table), 2)
        self.assertEqual(set(curves.run), {first, second})
        self.assertEqual(differences, [])
        third, result = self.run_once(config=replace(self.config, commission_rate=0.0002))
        self.assertIn("commission_rate", self.runs.comparison([first, third])[2])

    def test_invalid_paths_and_duplicate_comparison_rejected(self):
        with self.assertRaises(ValueError):
            self.runs.path("../../outside")
        with self.assertRaises(ValueError):
            self.studies.path("../outside")
        identifier, result = self.run_once()
        with self.assertRaises(ValueError):
            self.runs.comparison([identifier, identifier])

    def test_parameter_grid_is_bounded_and_preserves_base_parameters(self):
        self.assertEqual(expand_grid({"fixed": 2}, {"weight": [0.3, 0.6]}),
                         [{"fixed": 2, "weight": 0.3}, {"fixed": 2, "weight": 0.6}])
        for grid in ({}, {"weight": []}, {"weight": 0.5}, {"weight": list(range(25))}):
            with self.assertRaises(ValueError):
                expand_grid({}, grid)

    def test_training_only_uses_shared_runner_and_training_cutoff(self):
        with patch("ashare.platform.runs.run_backtest", wraps=run_backtest) as runner:
            identifier = self.search()
            self.assertEqual(runner.call_count, 2)
            self.assertTrue(all(call.args[3].end == self.config.end for call in runner.call_args_list))
        study = read_json(self.studies.path(identifier) / "study.json")
        self.assertFalse((self.studies.path(identifier) / "frozen.json").exists())
        for candidate in study["candidates"]:
            summary = read_json(self.runs.path(candidate["run_id"]) / "summary.json")
            self.assertEqual(summary["data"]["end"], "2024-01-31")
        self.assertEqual(len(self.runs.list()), 2)

    def test_freeze_rejects_overlap_and_changed_execution_costs(self):
        identifier = self.search()
        candidate = self.studies.list()[0]["candidates"][0]["run_id"]
        with self.assertRaisesRegex(ValueError, "晚于"):
            self.studies.freeze_and_validate(identifier, candidate, replace(self.validation(), start="2024-01-31"))
        with self.assertRaisesRegex(ValueError, "成交配置"):
            self.studies.freeze_and_validate(identifier, candidate, replace(self.validation(), commission_rate=0.01))
        self.assertFalse((self.studies.path(identifier) / "frozen.json").exists())

    def test_freeze_precedes_validation_and_reuses_original_strategy_snapshot(self):
        identifier = self.search()
        candidate = self.studies.list()[0]["candidates"][0]["run_id"]
        self.strategies.save("修改后", SOURCE.replace('return {"000001.SZ": self.weight}', 'return {}'), {}, self.strategy.identifier)

        def check_frozen(*args, **kwargs):
            frozen = read_json(self.studies.path(identifier) / "frozen.json")
            self.assertEqual(frozen["status"], "frozen")
            self.assertEqual(frozen["parameters"], {"weight": 0.3})
            return run_backtest(*args, **kwargs)

        with patch("ashare.platform.runs.run_backtest", side_effect=check_frozen):
            run_id, result = self.studies.freeze_and_validate(identifier, candidate, self.validation())
        self.assertFalse(result.trades.empty)
        self.assertEqual(result.metrics["start"], "2024-02-01")
        self.assertEqual(read_json(self.studies.path(identifier) / "frozen.json")["run_id"], run_id)
        with self.assertRaisesRegex(ValueError, "已完成验证"):
            self.studies.freeze_and_validate(identifier, candidate, self.validation())

    def test_modified_training_snapshot_cannot_be_validated(self):
        identifier = self.search()
        candidate = self.studies.list()[0]["candidates"][0]["run_id"]
        (self.runs.path(candidate) / "strategy.py").write_text(SOURCE + "\n")
        with self.assertRaisesRegex(ValueError, "源码快照"):
            self.studies.freeze_and_validate(identifier, candidate, self.validation())

    def test_changed_input_version_cannot_be_validated(self):
        identifier = self.search()
        candidate = self.studies.list()[0]["candidates"][0]["run_id"]
        frame = pd.read_feather(self.source / "amt.feather")
        frame.iloc[0, 1] = 1
        frame.to_feather(self.source / "amt.feather")
        with self.assertRaisesRegex(ValueError, "输入版本不一致"):
            self.studies.freeze_and_validate(identifier, candidate, self.validation())

    def test_failed_validation_only_allows_retrying_same_frozen_choice(self):
        identifier = self.search()
        first, second = [record["run_id"] for record in self.studies.list()[0]["candidates"]]
        with patch("ashare.platform.runs.run_backtest", side_effect=OSError("临时失败")):
            with self.assertRaises(OSError):
                self.studies.freeze_and_validate(identifier, first, self.validation())
        with self.assertRaisesRegex(ValueError, "只能重试"):
            self.studies.freeze_and_validate(identifier, second, self.validation())
        run_id, result = self.studies.freeze_and_validate(identifier, first, self.validation())
        self.assertTrue((self.runs.path(run_id) / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
