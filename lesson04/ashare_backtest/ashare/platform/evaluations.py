from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..cli import file_hash
from ..data import DataBundle
from ..evaluation import collect_observations, evaluate_observations
from .runs import new_identifier, read_json, write_json


TABLES = ("observations", "static", "dynamic", "blocks", "transitions", "monthly")


class EvaluationStore:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, identifier):
        destination = (self.root / identifier).resolve()
        if not destination.is_relative_to(self.root.resolve()) or destination == self.root.resolve():
            raise ValueError("无效因子评价路径")
        return destination

    def run(self, source, metadata, config, symbols=None, progress=None):
        identifier = new_identifier("factor")
        destination = self.path(identifier)
        destination.mkdir(parents=True, exist_ok=False)
        request = {"id": identifier, "status": "running", "config": config.to_dict(),
                   "symbols": list(symbols) if symbols else None,
                   "created_at": datetime.now(timezone.utc).isoformat(), "schema_version": 1,
                   "sources": {}, "code_sha256": file_hash(Path(__file__).parents[1] / "evaluation.py"),
                   "code_files": {name: file_hash(Path(__file__).parents[1] / name)
                                  for name in ("evaluation.py", "factors.py", "strategy.py", "data.py", "config.py")},
                   "label": "次日开盘至h日后开盘的复权收益；在退出日收盘后更新诊断",
                   "interval": "95%逐点HAC正态近似；不作顺序检验或多重比较校正"}
        write_json(destination / "request.json", request)
        try:
            for kind, directory in (("data", source), ("metadata", metadata)):
                if directory:
                    directory = Path(directory).expanduser()
                    request["sources"][kind] = {path.name: file_hash(path) for path in sorted(directory.glob("*"))
                                                if path.is_file() and path.suffix in {".feather", ".csv"}}
            write_json(destination / "request.json", request)
            data = DataBundle.from_feathers(source, symbols=symbols, metadata_dir=metadata, end=config.end)
            observations = collect_observations(data, config, progress=progress)
            results = evaluate_observations(observations, data.dates, config)
            for name, table in results.items():
                table.to_csv(destination / (name + ".csv"), index=False)
            request.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat(),
                           date_count=len(data.dates[(data.dates >= pd.Timestamp(config.start))]),
                           symbol_count=len(data.symbols), table_sha256={name: file_hash(destination / (name + ".csv"))
                                                                       for name in TABLES})
            write_json(destination / "request.json", request)
        except Exception as error:
            request.update(status="failed", error=str(error))
            write_json(destination / "request.json", request)
            raise
        return identifier

    def list(self):
        if not self.root.exists():
            return []
        return [read_json(path) for path in sorted(self.root.glob("*/request.json"), reverse=True)]

    def load(self, identifier):
        destination = self.path(identifier)
        request = read_json(destination / "request.json")
        if request["status"] != "completed":
            raise ValueError("此评价尚未完成")
        for name, digest in request["table_sha256"].items():
            if file_hash(destination / (name + ".csv")) != digest:
                raise ValueError("评价结果哈希不一致，请检查文件是否被修改")
        return request, {name: pd.read_csv(destination / (name + ".csv")) for name in TABLES}
