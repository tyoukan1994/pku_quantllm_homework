import ast
from dataclasses import dataclass
import json
from pathlib import Path
import uuid

from .runs import write_json

@dataclass(frozen=True)
class StrategySpec:
    identifier: str
    name: str
    path: Path
    parameters: dict
    grid: dict
    builtin: bool
    revision: int = 0


BUILTINS = [
    ("moving_average", "均线趋势", {"fast": 20, "slow": 60, "top_n": 20, "rebalance_days": 5}, {"fast": [10, 20], "slow": [60, 120]}),
    ("momentum", "动量选股", {"lookback": 60, "skip": 5, "top_n": 20, "rebalance_days": 5}, {"lookback": [20, 60], "top_n": [10, 20]}),
    ("equal_weight", "等权持仓", {"symbols": ["000001.SZ", "600000.SH", "600519.SH", "300750.SZ"], "rebalance_days": 20}, {"rebalance_days": [5, 20]}),
    ("five_day_reversal", "五日反转", {"top_n": 200, "rebalance_days": 5}, {"top_n": [100, 200], "rebalance_days": [5, 10]}),
    ("price_amount", "量价因子选股", {"factor": "reversal5", "top_n": 200, "rebalance_days": 5}, {"factor": ["reversal5", "low_volatility20"], "top_n": [100, 200]}),
    ("technical_selection", "RSI与技术选股", {"mode": "rsi_rebound", "top_n": 100, "rebalance_days": 1}, {"entry_threshold": [25, 30], "trend_window": [0, 60]}),
    ("literature_selection", "文献量价选股", {"factor": "high252", "top_n": 100, "rebalance_days": 20}, {"factor": ["high252", "amihud_low", "alpha33", "intraday_reversal"]}),
    ("custom_template", "自定义模板", {"lookback": 20, "top_n": 5}, {"lookback": [10, 20]}),
]


class StrategyStore:
    def __init__(self, root, builtin_dir=None):
        self.root = Path(root)
        self.builtin_dir = Path(builtin_dir) if builtin_dir else Path(__file__).resolve().parents[2] / "strategies"

    def list(self):
        result = [StrategySpec(f"builtin:{identifier}", name, self.builtin_dir / f"{identifier}.py",
                               dict(parameters), dict(grid), True)
                  for identifier, name, parameters, grid in BUILTINS]
        for path in sorted(self.root.glob("*/metadata.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            result.append(StrategySpec(record["identifier"], record["name"], path.parent / record["source"],
                                       record["parameters"], record.get("grid", {}), False, record["revision"]))
        return result

    def get(self, identifier):
        for spec in self.list():
            if spec.identifier == identifier:
                return spec
        raise ValueError("策略不存在")

    def save(self, name, source, parameters, identifier=None):
        name = name.strip()
        if not name or len(name) > 100:
            raise ValueError("策略名称须为1—100个字符")
        if not isinstance(parameters, dict):
            raise ValueError("默认参数必须为JSON对象")
        if len(source.encode("utf-8")) > 1_000_000:
            raise ValueError("策略源码不能超过1MB")
        json.dumps(parameters, allow_nan=False)
        tree = ast.parse(source)
        if not any(isinstance(node, ast.ClassDef) and node.name == "UserStrategy" for node in tree.body):
            raise ValueError("策略需定义UserStrategy类")
        if identifier:
            previous = self.get(identifier)
            if previous.builtin:
                raise ValueError("内置模板不可覆盖，请另存为新策略")
            directory = previous.path.parent
            revision = previous.revision + 1
        else:
            identifier = "user:" + uuid.uuid4().hex
            directory = self.root / identifier.split(":", 1)[1]
            revision = 1
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"source-{revision:04d}.py"
        record = {"identifier": identifier, "name": name, "parameters": parameters,
                  "source": filename, "revision": revision}
        with (directory / filename).open("x", encoding="utf-8") as stream:
            stream.write(source)
        write_json(directory / "metadata.json", record)
        return self.get(identifier)
