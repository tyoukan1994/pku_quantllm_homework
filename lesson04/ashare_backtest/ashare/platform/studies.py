from dataclasses import replace
from datetime import datetime, timezone
import itertools
import math
from pathlib import Path

import pandas as pd

from ..cli import file_hash
from ..config import BacktestConfig
from .runs import new_identifier, read_json, write_json
from .strategies import StrategySpec


def expand_grid(parameters, grid, limit=24):
    if not isinstance(parameters, dict) or not isinstance(grid, dict) or not grid:
        raise ValueError("基础参数和非空参数网格必须为JSON对象")
    if any(not isinstance(values, list) or not values for values in grid.values()):
        raise ValueError("每个网格参数须提供非空列表")
    if math.prod(len(values) for values in grid.values()) > limit:
        raise ValueError(f"一次研究最多{limit}个候选组合")
    return [{**parameters, **dict(zip(grid, values))} for values in itertools.product(*grid.values())]


class StudyStore:
    def __init__(self, root, runs):
        self.root = Path(root)
        self.runs = runs

    def path(self, identifier):
        path = (self.root / identifier).resolve()
        if not path.is_relative_to(self.root.resolve()) or path == self.root.resolve():
            raise ValueError("无效研究路径")
        return path

    def list(self):
        return [read_json(path) for path in sorted(self.root.glob("*/study.json"), reverse=True)]

    def search(self, source, metadata, strategy, parameters, grid, config, symbols=None,
               label="", metric="sharpe", progress=None):
        if metric not in {"sharpe", "total_return", "max_drawdown"}:
            raise ValueError("不支持的排序指标")
        candidates = expand_grid(parameters, grid)
        identifier = new_identifier("study")
        directory = self.path(identifier)
        directory.mkdir(parents=True, exist_ok=False)
        snapshot = directory / "strategy.py"
        snapshot.write_text(strategy.path.read_text(encoding="utf-8"), encoding="utf-8")
        pinned = replace(strategy, path=snapshot)
        study = {"id": identifier, "label": label.strip() or strategy.name, "strategy_id": strategy.identifier,
                 "strategy_name": strategy.name, "train_config": config.to_dict(), "symbols": symbols,
                 "data_path": str(Path(source).expanduser().resolve()),
                 "metadata_path": str(Path(metadata).expanduser().resolve()) if metadata else None,
                 "grid": grid, "base_parameters": parameters, "metric": metric, "candidates": [], "status": "running",
                 "created_at": datetime.now(timezone.utc).isoformat()}
        write_json(directory / "study.json", study)
        for number, candidate in enumerate(candidates):
            try:
                run_id, result = self.runs.run(source, metadata, pinned, candidate, config, symbols,
                                               kind="training", parent_study=identifier,
                                               label=f"{study['label']} / {number + 1}")
                study["candidates"].append({"run_id": run_id, "parameters": candidate,
                                            "metrics": result.metrics, "status": "completed"})
            except Exception as error:
                study["candidates"].append({"parameters": candidate, "status": "failed", "error": str(error)})
            write_json(directory / "study.json", study)
            if progress:
                progress((number + 1) / len(candidates), f"{number + 1}/{len(candidates)}")
        study["status"] = "completed" if any(record["status"] == "completed" for record in study["candidates"]) else "failed"
        write_json(directory / "study.json", study)
        return identifier

    def freeze_and_validate(self, identifier, candidate_id, validation_config, previously_viewed=True, progress=None):
        directory = self.path(identifier)
        study = read_json(directory / "study.json")
        if pd.Timestamp(validation_config.start) <= pd.Timestamp(study["train_config"]["end"]):
            raise ValueError("验证开始日期必须晚于训练结束日期")
        candidate = next((record for record in study["candidates"] if record.get("run_id") == candidate_id), None)
        if not candidate or candidate["status"] != "completed":
            raise ValueError("只能冻结该研究中已完成的训练候选")
        differences = [key for key, value in study["train_config"].items()
                       if key not in {"start", "end", "seed_previous_close"} and validation_config.to_dict()[key] != value]
        if differences:
            raise ValueError("验证必须沿用训练的账户及成交配置：" + ", ".join(differences))
        request = read_json(self.runs.path(candidate_id) / "request.json")
        strategy_path = self.runs.path(candidate_id) / "strategy.py"
        if file_hash(strategy_path) != request["strategy_sha256"]:
            raise ValueError("训练源码快照已改变，不能继续验证")
        summary = read_json(self.runs.path(candidate_id) / "summary.json")
        for input_directory, field in ((study["data_path"], "input_files"), (study["metadata_path"], "metadata_files")):
            for filename, fingerprint in summary.get(field, {}).items():
                if not input_directory or file_hash(Path(input_directory) / filename) != fingerprint["sha256"]:
                    raise ValueError("训练与验证输入版本不一致：" + filename)
        frozen = {"candidate_id": candidate_id, "parameters": candidate["parameters"],
                  "strategy_sha256": request["strategy_sha256"], "validation_config": validation_config.to_dict(),
                  "previously_viewed": bool(previously_viewed), "status": "frozen"}
        frozen_path = directory / "frozen.json"
        if frozen_path.exists():
            existing = read_json(frozen_path)
            if existing.get("status") == "completed":
                raise ValueError("该研究已完成验证；不能更换候选反复验证")
            if any(existing.get(key) != value for key, value in frozen.items() if key != "status"):
                raise ValueError("冻结配置已存在，只能重试同一候选及同一验证配置")
            frozen["frozen_at"] = existing["frozen_at"]
        else:
            frozen["frozen_at"] = datetime.now(timezone.utc).isoformat()
        write_json(frozen_path, frozen)
        pinned = StrategySpec(study["strategy_id"], study["strategy_name"],
                              strategy_path, candidate["parameters"], {}, False)
        run_id, result = self.runs.run(study["data_path"], study["metadata_path"], pinned,
                                       candidate["parameters"], validation_config, study["symbols"],
                                       kind="validation", parent_study=identifier,
                                       label=study["label"] + " / 验证", progress=progress)
        frozen.update(status="completed", run_id=run_id)
        write_json(frozen_path, frozen)
        return run_id, result
