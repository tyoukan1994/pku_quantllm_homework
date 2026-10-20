import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cli import DEFAULT_DATA, ROOT, file_hash
from .config import BacktestConfig
from .data import DataBundle
from .engine import BacktestEngine, DataCoverageError
from .factors import FACTOR_NAMES, eligible_scores
from .report import export_result
from .strategy import Context, load_strategy


def experiment_config(**overrides):
    settings = dict(start="2025-01-01", end="2026-09-30", initial_cash=1_000_000_000,
                    commission_rate=0.0001, minimum_commission=0,
                    stamp_tax_rate=0.0005, transfer_fee_rate=0,
                    slippage_bps=0, participation_rate=0.01, min_history=60,
                    max_weight=0.02, exclude_st=True, require_st_history=True,
                    seed_previous_close=True, liquidate_on_end=True,
                    split_orders=True, write_off_research_dust=True)
    settings.update(overrides)
    return BacktestConfig(**settings)


def training_ic(data, config):
    rows = []
    adjusted_open = data.arrays["open"] * data.arrays["cumadj"]
    for position in range(60, len(data.dates) - 6, 5):
        context = Context(data, position, config, config.initial_cash, config.initial_cash, {})
        scores = eligible_scores(context)
        with np.errstate(invalid="ignore", divide="ignore"):
            forward = adjusted_open[position + 6] / adjusted_open[position + 1] - 1
        labels = pd.Series(forward, index=data.symbols)
        for factor in FACTOR_NAMES:
            sample = pd.concat([scores[factor].rename("score"), labels.rename("label")], axis=1).dropna()
            sample = sample[np.isfinite(sample.label)]
            if len(sample) < 30 or sample.score.nunique() < 2 or sample.label.nunique() < 2:
                continue
            ranked = sample.rank()
            ic = ranked.score.corr(ranked.label)
            top = sample.sort_index().nlargest(max(1, len(sample) // 5), "score").label.mean()
            bottom = sample.sort_index().nsmallest(max(1, len(sample) // 5), "score").label.mean()
            rows.append({"date": str(context.date.date()), "factor": factor,
                         "label_start": str(data.dates[position + 1].date()),
                         "label_end": str(data.dates[position + 6].date()),
                         "rank_ic": ic, "top_minus_bottom": top - bottom, "stocks": len(sample)})
    observations = pd.DataFrame(rows)
    summary = observations.groupby("factor").agg(mean_rank_ic=("rank_ic", "mean"),
                                                  ic_std=("rank_ic", "std"),
                                                  mean_five_day_spread=("top_minus_bottom", "mean"),
                                                  samples=("rank_ic", "size"),
                                                  label_end_max=("label_end", "max"))
    summary["ic_ir_unannualized"] = summary.mean_rank_ic / summary.ic_std
    return observations, summary.sort_values("mean_rank_ic", ascending=False, kind="stable")


def run_study(source, metadata, destination, progress=None):
    destination = Path(destination)
    if (destination / "frozen_selection.json").exists():
        raise ValueError("目标目录已有冻结实验，请换新目录，防止覆盖样本外记录")
    destination.mkdir(parents=True, exist_ok=True)

    def notify(message):
        print(message, flush=True)
        if progress:
            progress(message)

    base = experiment_config()
    strategy_path = ROOT / "strategies" / "price_amount.py"
    provenance = {"raw_inputs": {path.name: file_hash(path) for path in sorted(Path(source).glob("*.feather"))},
                  "code": {str(path.relative_to(ROOT)): file_hash(path)
                           for path in sorted(ROOT.rglob("*.py")) if "runs" not in path.parts},
                  "metadata": {path.name: file_hash(path) for path in sorted(Path(metadata).glob("*")) if path.is_file()}}
    protocol = {"calibration": ["2020-01-02", "2023-12-29"], "validation": ["2024-01-02", "2024-12-31"],
                "holdout": ["2025-01-01", "2026-09-30"], "factors": FACTOR_NAMES,
                "screen": "2020-2023 每隔5交易日计算未来5日开盘到开盘 RankIC，取均值最高3因子；标签结束不超过2023-12-29",
                "candidate_top_n": [100, 200], "rebalance_days": 5,
                "selection": "2024年费用后夏普最高；相同则累计收益最高、因子名及数量确定性排序。所有候选及负收益完整保留。",
                "liquidity": "过去20日平均成交额至少5000万元，取流动性前1500只；上一日成交额1%执行容量",
                "portfolio": "等权，95%目标仓位，最低60个有效历史观测",
                "costs": base.to_dict(), "provenance": provenance,
                "created_before_calibration": datetime.now(timezone.utc).isoformat()}
    (destination / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    notify("1/5 读取截至2023年的训练子集；样本外行情不载入研究接口")
    calibration = DataBundle.from_feathers(source, metadata_dir=metadata, end="2023-12-29")
    if "is_st" not in calibration.status_arrays:
        raise DataCoverageError("必须先补齐历史 ST 数据，不能将未知状态当作非 ST")
    observations, screen = training_ic(calibration, base)
    observations.to_csv(destination / "training_ic.csv", index=False, encoding="utf-8-sig")
    screen.to_csv(destination / "factor_screen.csv", encoding="utf-8-sig")
    shortlist = screen.head(3).index.tolist()
    notify("2/5 仅在2024年验证候选：" + ", ".join(shortlist))
    del calibration
    validation = DataBundle.from_feathers(source, metadata_dir=metadata, end="2024-12-31")
    trials = []
    for factor in shortlist:
        for top_n in (100, 200):
            parameters = {"factor": factor, "top_n": top_n, "rebalance_days": 5}
            config = replace(base, start="2024-01-02", end="2024-12-31")
            result = BacktestEngine(validation, config).run(load_strategy(strategy_path, parameters))
            export_result(result, destination / "validation" / f"{factor}-{top_n}")
            trials.append({"factor": factor, "top_n": top_n, **result.metrics})
            notify(f"  {factor} Top{top_n}: 2024收益={result.metrics['total_return']:.2%}, 夏普={result.metrics['sharpe']}")
    table = pd.DataFrame(trials).sort_values(["sharpe", "total_return", "factor", "top_n"],
                                             ascending=[False, False, True, True], na_position="last", kind="stable")
    table.to_csv(destination / "validation_candidates.csv", index=False, encoding="utf-8-sig")
    winner = table.iloc[0]
    parameters = {"factor": str(winner.factor), "top_n": int(winner.top_n), "rebalance_days": 5}
    frozen = {"parameters": parameters, "config": base.to_dict(), "train_end": "2024-12-31",
              "calibration_label_end_max": observations.label_end.max(),
              "selected_using": "2024 validation net Sharpe only, never 2025-2026",
              "frozen_at": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": file_hash(destination / "protocol.json"),
              "strategy_sha256": file_hash(strategy_path)}
    encoded = json.dumps(frozen, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    frozen["selection_sha256"] = hashlib.sha256(encoded).hexdigest()
    (destination / "frozen_selection.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    notify("3/5 参数已冻结；保存2020—2024年拟合内表现，不据此重新选参")
    training = BacktestEngine(validation, replace(base, start="2020-01-02", end="2024-12-31", seed_previous_close=False)).run(load_strategy(strategy_path, parameters))
    export_result(training, destination / "training", strategy_path.read_text(encoding="utf-8"))
    del validation
    notify("4/5 首次载入考核行情：2025首日建仓，2026年9月末预定清仓")
    holdout_data = DataBundle.from_feathers(source, metadata_dir=metadata, end=base.end)
    holdout = BacktestEngine(holdout_data, base).run(load_strategy(strategy_path, parameters))
    export_result(holdout, destination / "holdout", strategy_path.read_text(encoding="utf-8"))
    buys = holdout.trades[holdout.trades.side == "buy"]
    audit = {"first_execution": buys.date.min() if not buys.empty else None,
             "last_execution": holdout.trades.date.max() if not holdout.trades.empty else None,
             "st_buys": sum(holdout_data.state(pd.Timestamp(record.date), record.symbol).get("is_st") is not False
                            for record in buys.itertuples()),
             "first_signal_date": holdout.events.loc[holdout.events.event == "initial_signal_from_previous_close", "date"].iloc[0],
             "holdout_first_day": holdout.metrics["start"], "holdout_last_day": holdout.metrics["end"],
             "selection_sha256": frozen["selection_sha256"], "liquidation_complete": holdout.metrics["liquidation_complete"],
             "status_coverage": holdout_data.summary()["status_coverage"],
             "warnings": "公开研究近似：没有精确逐日涨跌停价和公司行为现金明细；日频不能识别队列和开盘容量。样本框为老师所给5451个代码，不承诺无幸存者偏差。"}
    (destination / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    notify("5/5 完成；考核结果已单独保存，不回流选参")
    return holdout


def run_exit_revision(source, metadata, study, window=20):
    study = Path(study)
    frozen = json.loads((study / "frozen_selection.json").read_text(encoding="utf-8"))
    protocol = json.loads((study / "protocol.json").read_text(encoding="utf-8"))
    strategy_path = ROOT / "strategies" / "price_amount.py"
    if file_hash(strategy_path) != frozen["strategy_sha256"]:
        raise ValueError("冻结策略源码已改变，不能标作只修改退出机制")
    for filename, digest in protocol["provenance"]["raw_inputs"].items():
        if file_hash(Path(source) / filename) != digest:
            raise ValueError(f"原始输入已改变：{filename}")
    for filename, digest in protocol["provenance"]["metadata"].items():
        if file_hash(Path(metadata) / filename) != digest:
            raise ValueError(f"历史状态已改变：{filename}")
    destination = study / f"exit_window_{window}"
    if destination.exists():
        raise ValueError("退出对照已存在，不覆盖原记录")
    destination.mkdir()
    config = replace(BacktestConfig(**frozen["config"]), liquidation_window=window)
    revision = {"type": "user_approved_execution_revision", "liquidation_window": window,
                "parameters_unchanged": frozen["parameters"], "original_selection_sha256": frozen["selection_sha256"],
                "original_holdout_preserved": "../holdout", "recorded_at": datetime.now(timezone.utc).isoformat(),
                "not_a_new_blind_holdout": True,
                "reason": "用户已查看末日容量约束，明确同意提前20个交易日只卖不买并保留原始考核。不是根据新收益重新选因子或调仓参数。",
                "additional_engine_fix": "研究模式末日不足1股权益可用最近有效估值核销为损失，包括已退市且末日行情缺失的模型残余；不产生现金或虚构卖出。",
                "engine_source_sha256": {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}}
    (destination / "execution_revision.json").write_text(json.dumps(revision, ensure_ascii=False, indent=2), encoding="utf-8")
    data = DataBundle.from_feathers(source, metadata_dir=metadata, end=config.end)
    result = BacktestEngine(data, config).run(load_strategy(strategy_path, frozen["parameters"]))
    export_result(result, destination, strategy_path.read_text(encoding="utf-8"))
    original = json.loads((study / "holdout" / "summary.json").read_text(encoding="utf-8"))
    pd.DataFrame([{"experiment": "原始：末日单次清仓", **original["metrics"]},
                  {"experiment": f"执行对照：提前{window}日分批退出", **result.metrics}]).to_csv(study / "execution_comparison.csv", index=False, encoding="utf-8-sig")
    buys = result.trades[result.trades.side == "buy"]
    audit = {"selection_sha256": frozen["selection_sha256"], "parameters_unchanged": True,
             "first_execution": buys.date.min() if not buys.empty else None,
             "first_signal_date": result.events.loc[result.events.event == "initial_signal_from_previous_close", "date"].iloc[0],
             "first_exit_day": str(result.equity.index[-window].date()),
             "buys_in_exit_window": int((buys.date >= str(result.equity.index[-window].date())).sum()),
             "st_or_unknown_buys": sum(data.state(pd.Timestamp(record.date), record.symbol).get("is_st") is not False
                                       for record in buys.itertuples()),
             "liquidation_complete": result.metrics["liquidation_complete"],
             "terminal_position_count": result.metrics["terminal_position_count"],
             "not_a_new_blind_holdout": True}
    (destination / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="训练内量价因子探索 + 一次冻结样本外考核")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exit-revision", type=int, help="仅在用户同意后，复核冻结模型的提前退出执行对照")
    arguments = parser.parse_args()
    result = (run_exit_revision(arguments.data, arguments.metadata, arguments.output, arguments.exit_revision)
              if arguments.exit_revision else run_study(arguments.data, arguments.metadata, arguments.output))
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
