import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

from .config import BacktestConfig
from .data import DataBundle
from .engine import BacktestEngine
from .report import export_result
from .strategy import load_strategy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = os.environ.get("ASHARE_DATA_DIR", str(Path.home() / "Downloads" / "feathers"))


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_backtest(source, strategy_path, parameters, config, destination, symbols=None, progress=None, metadata_dir=None):
    input_files = {path.name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                              "sha256": file_hash(path)}
                   for path in Path(source).expanduser().glob("*")
                   if path.suffix in {".feather", ".csv"}}
    metadata_files = {path.name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                                  "sha256": file_hash(path)}
                      for path in Path(metadata_dir).expanduser().glob("*")
                      if path.suffix in {".feather", ".csv", ".json"}} if metadata_dir else {}
    data = DataBundle.from_feathers(source, symbols, metadata_dir, end=config.end)
    strategy = load_strategy(strategy_path, parameters)
    result = BacktestEngine(data, config).run(strategy, progress)
    for directory, fingerprints in ((source, input_files), (metadata_dir, metadata_files)):
        for filename, fingerprint in fingerprints.items():
            path = Path(directory).expanduser() / filename
            if path.stat().st_size != fingerprint["bytes"] or path.stat().st_mtime_ns != fingerprint["mtime_ns"]:
                raise RuntimeError(f"回测期间输入文件发生变化，请重新运行：{filename}")
    export_result(result, destination, Path(strategy_path).read_text(encoding="utf-8"))
    summary_path = Path(destination) / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["strategy_parameters"] = parameters
    payload["strategy_sha256"] = hashlib.sha256(Path(strategy_path).read_bytes()).hexdigest()
    payload["input_files"] = input_files
    payload["metadata_files"] = metadata_files
    payload["python_version"] = sys.version
    import numpy
    import pandas
    import pyarrow
    payload["package_versions"] = {"numpy": numpy.__version__, "pandas": pandas.__version__, "pyarrow": pyarrow.__version__}
    payload["engine_source_sha256"] = {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="沪深 A 股日频回测：收盘信号、次日开盘成交")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--strategy", type=Path, default=ROOT / "strategies/momentum.py")
    parser.add_argument("--params", default="{}", help="策略参数 JSON")
    parser.add_argument("--config", type=Path, help="BacktestConfig 参数 JSON")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--symbols", help="逗号分隔的股票代码；不填则加载全部")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata", type=Path, help="本机私有历史 ST、停牌和上市日期目录")
    parser.add_argument("--inspect", action="store_true")
    arguments = parser.parse_args()
    symbols = [symbol.strip() for symbol in arguments.symbols.split(",") if symbol.strip()] if arguments.symbols else None
    if arguments.inspect:
        print(json.dumps(DataBundle.from_feathers(arguments.data, symbols, arguments.metadata).summary(), ensure_ascii=False, indent=2))
        return
    settings = json.loads(arguments.config.read_text(encoding="utf-8")) if arguments.config else {}
    for field in ("start", "end"):
        if getattr(arguments, field):
            settings[field] = getattr(arguments, field)
    parameters = json.loads(arguments.params)
    if not isinstance(parameters, dict):
        raise ValueError("--params 必须为 JSON 对象")
    destination = arguments.output or ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    result = run_backtest(arguments.data, arguments.strategy, parameters, BacktestConfig(**settings), destination, symbols, metadata_dir=arguments.metadata)
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"报告：{destination.resolve() / 'report.html'}")


if __name__ == "__main__":
    main()
