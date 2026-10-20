import json
from pathlib import Path

import pandas as pd

from .engine import BacktestResult


def load_saved_result(directory):
    directory = Path(directory)
    payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    frames = {}
    for name in ("equity", "trades", "orders", "positions", "events"):
        frames[name] = pd.read_csv(directory / f"{name}.csv")
    frames["equity"]["date"] = pd.to_datetime(frames["equity"].date)
    frames["equity"] = frames["equity"].set_index("date")
    frames["orders"]["reason"] = frames["orders"].reason.fillna("")
    return BacktestResult(**frames, config=payload["config"], data_summary=payload["data"],
                          assumptions=payload["assumptions"], strategy_name=payload["strategy"],
                          metrics=payload["metrics"])
