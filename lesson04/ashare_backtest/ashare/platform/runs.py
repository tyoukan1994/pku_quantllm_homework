from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import pandas as pd

from ..cli import run_backtest
from ..saved import load_saved_result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def new_identifier(prefix):
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


class RunStore:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, identifier):
        path = (self.root / identifier).resolve()
        if not path.is_relative_to(self.root.resolve()) or path == self.root.resolve():
            raise ValueError("无效运行路径")
        return path

    def run(self, source, metadata, strategy, parameters, config, symbols=None, label="",
            kind="backtest", parent_study=None, progress=None):
        if not isinstance(parameters, dict):
            raise ValueError("策略参数必须为JSON对象")
        parameters = json.loads(json.dumps(parameters, allow_nan=False))
        identifier = new_identifier(kind)
        destination = self.path(identifier)
        destination.mkdir(parents=True, exist_ok=False)
        snapshot = strategy.path.read_text(encoding="utf-8")
        strategy_path = destination / "strategy.py"
        strategy_path.write_text(snapshot, encoding="utf-8")
        request = {"id": identifier, "label": label.strip() or strategy.name,
                   "kind": kind, "parent_study": parent_study, "status": "running",
                   "strategy_id": strategy.identifier, "strategy_name": strategy.name,
                   "strategy_revision": strategy.revision, "parameters": parameters,
                   "strategy_sha256": hashlib.sha256(snapshot.encode()).hexdigest(),
                   "data_path": str(Path(source).expanduser().resolve()),
                   "metadata_path": str(Path(metadata).expanduser().resolve()) if metadata else None,
                   "symbols": list(symbols) if symbols else None, "config": config.to_dict(),
                   "created_at": datetime.now(timezone.utc).isoformat()}
        write_json(destination / "request.json", request)
        try:
            result = run_backtest(source, strategy_path, parameters, config, destination,
                                  symbols=symbols, progress=progress, metadata_dir=metadata or None)
        except Exception as error:
            request.update(status="failed", error=str(error))
            write_json(destination / "request.json", request)
            raise
        request.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
        write_json(destination / "request.json", request)
        return identifier, result

    def list(self):
        paths = set(self.root.rglob("summary.json")) | set(self.root.rglob("request.json"))
        result = []
        for directory in sorted({path.parent for path in paths}):
            summary_path = directory / "summary.json"
            request_path = directory / "request.json"
            if request_path.exists():
                request = read_json(request_path)
            else:
                request = {}
            summary = read_json(summary_path) if summary_path.exists() else {}
            if not request and not {"metrics", "config", "strategy"}.issubset(summary):
                continue
            identifier = directory.relative_to(self.root).as_posix()
            result.append({"id": identifier, "label": request.get("label", summary.get("strategy", directory.name)),
                           "strategy": request.get("strategy_name", summary.get("strategy", "")),
                           "kind": request.get("kind", "legacy"),
                           "status": request.get("status", "completed"),
                           "created_at": request.get("created_at", summary.get("created_at", "")),
                           "config": summary.get("config", request.get("config", {})),
                           "parameters": request.get("parameters", summary.get("strategy_parameters", {})),
                           "metrics": summary.get("metrics", {}), "error": request.get("error", "")})
        def ordering(record):
            try:
                timestamp = pd.Timestamp(record["created_at"])
                if pd.isna(timestamp):
                    raise ValueError("missing timestamp")
                if timestamp.tz is None:
                    timestamp = timestamp.tz_localize("Asia/Shanghai")
                instant = timestamp.tz_convert("UTC").value
            except (TypeError, ValueError):
                instant = 0
            return instant, record["id"]

        return sorted(result, key=ordering, reverse=True)

    def load(self, identifier):
        return load_saved_result(self.path(identifier))

    def comparison(self, identifiers):
        if len(identifiers) < 2 or len(set(identifiers)) != len(identifiers):
            raise ValueError("至少选择两个不同的已完成运行")
        results = [(identifier, self.load(identifier)) for identifier in identifiers]
        first = results[0][1]
        keys = ("initial_cash", "commission_rate", "minimum_commission", "stamp_tax_rate", "transfer_fee_rate",
                "slippage_bps", "participation_rate", "amount_multiplier", "accounting", "annual_days",
                "risk_free_rate", "min_history", "exclude_st", "require_st_history", "strict_rules",
                "seed_previous_close", "liquidate_on_end", "liquidation_window", "max_weight", "split_orders",
                "write_off_research_dust", "dividend_tax_rate", "factor_tolerance")
        differences = set()
        rows, curves = [], []
        requests = []
        for identifier, result in results:
            path = self.path(identifier) / "request.json"
            requests.append(read_json(path) if path.exists() else None)
            if not result.equity.index.equals(first.equity.index):
                differences.add("交易日期")
            differences.update(key for key in keys if result.config.get(key) != first.config.get(key))
            rows.append({"运行": identifier, "策略": result.strategy_name, "开始": result.metrics["start"],
                         "结束": result.metrics["end"], "收益(%)": result.metrics["total_return"] * 100,
                         "回撤(%)": result.metrics["max_drawdown"] * 100, "夏普": result.metrics["sharpe"],
                         "费用": result.metrics["total_fees"], "期末股票市值": result.metrics.get("terminal_market_value")})
            frame = result.equity[["net_value"]].reset_index()
            frame["run"] = identifier
            curves.append(frame)
        if any(request is None for request in requests):
            differences.add("历史记录缺少统一请求信息")
        else:
            for request in requests[1:]:
                if request["symbols"] != requests[0]["symbols"]:
                    differences.add("载入股票池")
        summaries = [read_json(self.path(identifier) / "summary.json") for identifier in identifiers]
        for field in ("input_files", "metadata_files"):
            fingerprints = [{name: value.get("sha256") for name, value in summary.get(field, {}).items()}
                            for summary in summaries]
            if (not all(field in summary for summary in summaries)
                    or (field == "input_files" and not all(fingerprints))
                    or any(value != fingerprints[0] for value in fingerprints[1:])):
                differences.add("行情来源/状态版本")
        engines = [summary.get("engine_source_sha256") for summary in summaries]
        if not all(engines) or any(value != engines[0] for value in engines[1:]):
            differences.add("引擎版本")
        return pd.DataFrame(rows), pd.concat(curves, ignore_index=True), sorted(differences)
