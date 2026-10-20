from dataclasses import replace
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ashare.cli import file_hash
from ashare.discovery import annual_statistics, choose_shortlist, robustness_score, table_rows
from ashare.platform import RunStore, StrategyStore
from ashare.platform.runs import read_json, write_json
from ashare.platform.strategies import StrategySpec
from ashare.research import experiment_config


SOURCES = [
    {"title": "101 Formulaic Alphas：仅使用可复现价格公式", "url": "https://arxiv.org/abs/1601.00991"},
    {"title": "52周高点与动量", "url": "https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2004.00695.x"},
    {"title": "隔夜与日内收益的拉锯", "url": "https://personal.lse.ac.uk/polk/research/TugOfWar.pdf"},
    {"title": "Amihud流动性度量", "url": "https://vlab.stern.nyu.edu/docs/liquidity/ILLIQ-HIST"},
    {"title": "High-Volume Return Premium：本次仅测试成交额改编", "url": "https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00349"},
    {"title": "Lee与Swaminathan：价格动量与成交量（缺少换手率，不原样复现）", "url": "https://doi.org/10.1111/0022-1082.00280"},
    {"title": "RiceQuant：内置量价/技术/Alpha101因子接口", "url": "https://www.ricequant.com/doc/rqfactor/api/biult-in-factor"},
]


def candidate_list():
    settings = [
        ("high252", "接近52周高点", "高点动量", 20),
        ("high252_lowvol", "52周高点 + 低波动", "高点组合", 20),
        ("intraday_reversal", "五日日内收益反转", "收益拆分", 5),
        ("overnight_reversal", "五日隔夜收益反转", "收益拆分", 5),
        ("amihud_low", "Amihud低非流动性", "流动性", 5),
        ("amihud_high", "Amihud高非流动性", "流动性", 5),
        ("alpha33", "Alpha101 #33 / 日内反转", "公式Alpha", 5),
        ("alpha101", "Alpha101 #101 / K线实体", "公式Alpha", 5),
        ("alpha101_reverse", "K线实体反转 / 改编对照", "K线反转", 5),
        ("range_reversal", "ATR归一化五日反转", "风险归一化", 5),
        ("amount_expansion", "成交额异常放大 / 成交量文献改编", "量能改编", 20),
        ("price_amount_divergence", "收益与成交额变化背离", "量价背离", 5),
    ]
    return [{"id": factor, "name": name, "family": family,
             "parameters": {"factor": factor, "top_n": 100, "rebalance_days": interval}}
            for factor, name, family, interval in settings]


def run(source, metadata, campaign):
    campaign = Path(campaign).resolve()
    if not campaign.is_relative_to(ROOT / "runs"):
        raise ValueError("研究目录须位于平台runs内")
    campaign.mkdir(parents=True, exist_ok=True)
    source, metadata = Path(source).expanduser().resolve(), Path(metadata).expanduser().resolve()
    config = experiment_config(start="2020-01-02", end="2023-12-29", min_history=120,
                               seed_previous_close=False, liquidate_on_end=False, write_off_research_dust=False)
    strategy_path = ROOT / "strategies" / "literature_selection.py"
    protocol = {"candidates": candidate_list(), "training_config": config.to_dict(),
                "confirmation": ["2024-01-02", "2024-12-31"], "holdout_run": False,
                "references": SOURCES,
                "selection": "沿用首批研究的年度稳健评分与训练门槛，每族最多一个，前三名先冻结再确认2024，不读2025—2026。",
                "input_sha256": {str(path): file_hash(path) for directory in (source, metadata) for path in sorted(directory.glob("*")) if path.suffix in {".feather", ".csv", ".json"}},
                "code_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in sorted((ROOT / "ashare").rglob("*.py"))},
                "adaptations": "价格类信号可计算；成交额不是成交量/换手率，不复现需volume、vwap、行业和市值的全部Alpha101；原文多空/当日收盘执行均改为A股多头、次日开盘和容量约束。"}
    protocol_path = campaign / "protocol.json"
    if protocol_path.exists():
        raise ValueError("该目录已有研究记录，请换新目录，避免重复试验或覆盖冻结结果")
    else:
        write_json(protocol_path, {**protocol, "created_at": datetime.now(timezone.utc).isoformat()})
    record = {"id": campaign.name, "name": "网络文献量价探索 · 12候选", "status": "running",
              "calibration": [], "confirmation": [], "sensitivity": [], "baseline": [],
              "references": SOURCES, "holdout_run": False}
    runs = RunStore(ROOT / "runs")

    def execute(candidate, settings, stage):
        spec = StrategySpec("builtin:literature_selection", candidate["name"], strategy_path, candidate["parameters"], {}, True)
        started = time.monotonic()
        identifier, result = runs.run(source, metadata, spec, candidate["parameters"], settings,
                                       label=candidate["name"] + " / " + stage,
                                       kind="training" if stage == "calibration" else "validation", parent_study=campaign.name)
        (runs.path(identifier) / "literature_snapshot.py").write_text((ROOT / "ashare" / "literature.py").read_text(encoding="utf-8"), encoding="utf-8")
        print(f"{stage} {candidate['id']}: 收益={result.metrics['total_return']:.2%}, 夏普={result.metrics['sharpe']}, {time.monotonic()-started:.1f}秒", flush=True)
        return identifier, result

    for candidate in candidate_list():
        identifier, result = execute(candidate, config, "calibration")
        years = annual_statistics(result.equity)
        score, qualified, positive = robustness_score(result.metrics, years)
        record["calibration"].append({**candidate, "run_id": identifier, "metrics": result.metrics,
                                      "years": years, "robustness": score, "qualified": qualified, "positive_years": positive})
        write_json(campaign / "campaign.json", record)
        pd.DataFrame(table_rows(record["calibration"])).to_csv(campaign / "开发候选.csv", index=False, encoding="utf-8-sig")
        del result
    shortlist = choose_shortlist(record["calibration"], 3)
    write_json(campaign / "frozen_shortlist.json", {"candidate_ids": [row["id"] for row in shortlist],
                "candidates": shortlist, "protocol_sha256": file_hash(protocol_path),
                "frozen_at": datetime.now(timezone.utc).isoformat(), "selected_without_2024_or_2025_2026": True})
    confirmation = replace(config, start="2024-01-02", end="2024-12-31", seed_previous_close=True)
    for candidate in shortlist:
        identifier, result = execute(candidate, confirmation, "confirmation")
        record["confirmation"].append({**candidate, "run_id": identifier, "metrics": result.metrics, "years": annual_statistics(result.equity)})
        write_json(campaign / "campaign.json", record)
        del result
    for candidate in shortlist[:1]:
        for slippage in (5, 10):
            stress = {**candidate, "name": candidate["name"] + f" / 单边{slippage}bp"}
            identifier, result = execute(stress, replace(confirmation, slippage_bps=slippage), "sensitivity")
            record["sensitivity"].append({**stress, "run_id": identifier, "metrics": result.metrics})
            del result
    store = StrategyStore(ROOT / "workspace" / "strategies")
    record["saved_strategies"] = []
    for candidate in shortlist:
        spec = store.save(candidate["name"] + " · 文献候选", strategy_path.read_text(encoding="utf-8"), candidate["parameters"])
        record["saved_strategies"].append({"id": candidate["id"], "strategy_id": spec.identifier})
    record.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
    for filename, rows in (("确认候选.csv", record["confirmation"]), ("滑点压力.csv", record["sensitivity"])):
        pd.DataFrame(table_rows(rows)).to_csv(campaign / filename, index=False, encoding="utf-8-sig")
    write_json(campaign / "campaign.json", record)
    print("文献研究完成：" + str(campaign), flush=True)


def main():
    parser = argparse.ArgumentParser(description="12个文献量价候选；开发截至2023、确认2024，不运行2025—2026")
    parser.add_argument("--data", default=str(Path.home() / "Downloads" / "feathers"))
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.data, arguments.metadata, arguments.output)


if __name__ == "__main__":
    main()
