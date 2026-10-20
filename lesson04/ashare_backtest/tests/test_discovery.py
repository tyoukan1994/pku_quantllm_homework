import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from ashare.cli import file_hash
from ashare.discovery import annual_statistics, candidates, changed_inputs, choose_shortlist, robustness_score, run_provenance


class DiscoveryTests(unittest.TestCase):
    def test_research_runs_keep_comparable_execution_provenance(self):
        result = run_provenance({"metadata_path": "/tmp/metadata",
            "input_sha256": {"/tmp/data/close.feather": "price", "/tmp/metadata/is_st.feather": "st"},
            "code_sha256": {"ashare/engine.py": "engine", "ashare/ui/common.py": "ui"}})
        self.assertEqual(result["input_files"], {"close.feather": {"sha256": "price"}})
        self.assertEqual(result["metadata_files"], {"is_st.feather": {"sha256": "st"}})
        self.assertEqual(result["engine_source_sha256"], {"engine.py": "engine"})

    def test_final_input_audit_accepts_json_string_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text("original", encoding="utf-8")
            fingerprints = {str(path): file_hash(path)}
            self.assertEqual(changed_inputs(fingerprints), [])
            path.write_text("changed", encoding="utf-8")
            self.assertEqual(changed_inputs(fingerprints), [str(path)])

    def test_candidate_scope_is_bounded_unique_and_constructible(self):
        from ashare.technical import TechnicalStrategy
        scope = candidates()
        self.assertEqual(len(scope), 24)
        self.assertEqual(len({row["id"] for row in scope}), 24)
        self.assertEqual(len({row["family"] for row in scope}), 12)
        for row in scope:
            strategy = TechnicalStrategy(**row["parameters"])
            self.assertEqual(strategy.top_n, 100)

    def test_shortlist_enforces_family_diversity_and_training_gate(self):
        rows = [{"id": "weak", "family": "RSI", "qualified": False, "robustness": 10},
                {"id": "first", "family": "低波", "qualified": True, "robustness": 1},
                {"id": "same-family", "family": "低波", "qualified": True, "robustness": 0.9},
                {"id": "second", "family": "反转", "qualified": True, "robustness": 0.8}]
        self.assertEqual([row["id"] for row in choose_shortlist(rows, 2)], ["first", "second"])

    def test_score_does_not_use_confirmation_performance(self):
        metrics = {"total_return": 0.2, "average_exposure": 0.7, "max_drawdown": -0.2}
        years = [{"year": year, "sharpe": 1, "return": 0.1} for year in (2021, 2022, 2023)]
        first = robustness_score(metrics, years)
        second = robustness_score(metrics, years + [{"year": 2024, "sharpe": -100, "return": -0.99}])
        self.assertEqual(first, second)
        self.assertTrue(first[1])

    def test_training_gate_rejects_low_exposure_and_bad_drawdown(self):
        years = [{"year": year, "sharpe": 1, "return": 0.1} for year in (2021, 2022, 2023)]
        for metrics in ({"total_return": 0.2, "average_exposure": 0.1, "max_drawdown": -0.2},
                        {"total_return": 0.2, "average_exposure": 0.7, "max_drawdown": -0.5}):
            self.assertFalse(robustness_score(metrics, years)[1])

    def test_yearly_returns_include_first_execution_and_year_boundary(self):
        curve = pd.DataFrame({"daily_return": [0.1, -0.1, 0.2], "exposure": [0.5, 0.6, 0.7]},
                             index=pd.to_datetime(["2022-12-29", "2022-12-30", "2023-01-03"]))
        rows = annual_statistics(curve)
        self.assertAlmostEqual(rows[0]["return"], -0.01)
        self.assertAlmostEqual(rows[1]["return"], 0.2)
        self.assertAlmostEqual(rows[0]["drawdown"], -0.1)


if __name__ == "__main__":
    unittest.main()
