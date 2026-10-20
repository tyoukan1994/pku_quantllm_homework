import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .cli import DEFAULT_DATA, ROOT, file_hash
from .data import DataBundle
from .engine import BacktestEngine, DataCoverageError
from .platform.runs import read_json, write_json
from .platform.strategies import StrategyStore
from .report import export_result
from .research import experiment_config
from .technical import PreparedFeatures, TechnicalStrategy, prepare_features


REFERENCES = [
    {"title": "TradingView：RSI与Wilder平滑", "url": "https://www.tradingview.com/support/solutions/43000502338-relative-strength-index-rsi/"},
    {"title": "Connors Research：短周期RSI", "url": "https://tradingmarkets.com/connorsrsi-download-the-code-now"},
    {"title": "John Bollinger：布林带使用规则", "url": "https://www.bollingerbands.com/bollinger-band-rules"},
    {"title": "QuantConnect：短期反转的实现与成本", "url": "https://www.quantconnect.com/research/15311/short-term-reversal-strategy-in-stocks/"},
    {"title": "AQR：Time Series Momentum", "url": "https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum"},
    {"title": "AQR：Betting Against Beta", "url": "https://www.aqr.com/insights/research/journal-article/betting-against-beta"},
]


def candidates():
    result = []

    def add(identifier, name, family, **parameters):
        result.append({"id": identifier, "name": name, "family": family,
                       "parameters": {"top_n": 100, "rebalance_days": 5, **parameters}})

    for trend in (0, 60):
        add(f"rsi14-{trend}", f"RSI14回升 / {'无趋势过滤' if trend == 0 else 'MA60过滤'}", "RSI14",
            mode="rsi_rebound", rebalance_days=1, trend_window=trend)
    for threshold in (5, 10):
        add(f"rsi2-{threshold}", f"RSI2回撤 / 阈值{threshold}", "RSI2",
            mode="rsi_pullback", entry_threshold=threshold, exit_threshold=70,
            rebalance_days=1, trend_window=200, max_hold=10)
    for trend in (0, 60):
        add(f"bollinger-{trend}", f"布林下轨回升 / {'无趋势过滤' if trend == 0 else 'MA60过滤'}", "布林带",
            mode="bollinger_rebound", trend_window=trend, rebalance_days=1)
    for lookback in (5, 20):
        add(f"reversal-{lookback}", f"{lookback}日反转", "反转", mode="reversal", lookback=lookback)
    for lookback in (60, 120):
        add(f"momentum-{lookback}", f"{lookback}日动量 / 跳过近期", "动量",
            mode="momentum", lookback=lookback, rebalance_days=20)
    for lookback in (20, 60):
        add(f"breakout-{lookback}", f"{lookback}日通道突破", "突破",
            mode="breakout", lookback=lookback, max_hold=60, rebalance_days=1)
    for lookback in (20, 60):
        add(f"ma-{lookback}", f"MA{lookback} / MA{60 if lookback == 20 else 120}", "均线趋势",
            mode="ma_trend", lookback=lookback, rebalance_days=10)
    for lookback in (20, 60):
        add(f"lowvol-{lookback}", f"{lookback}日低波动", "低波动", mode="low_volatility", lookback=lookback)
    for weight in (0.5, 0.75):
        add(f"lowvol-reversal-{weight}", f"低波动{weight:.0%} + 反转", "低波反转",
            mode="lowvol_reversal", blend_weight=weight)
    for weight in (0.5, 0.75):
        add(f"lowvol-momentum-{weight}", f"低波动{weight:.0%} + 动量", "低波动量",
            mode="lowvol_momentum", lookback=60, blend_weight=weight, rebalance_days=10)
    add("amount-reversal", "成交额放大反转", "量价", mode="amount_reversal")
    add("close-pressure", "收盘位置量价反转", "量价", mode="close_pressure")
    for weight in (0.5, 0.75):
        add(f"defensive-{weight}", f"市场宽度减仓 / 低波动{weight:.0%}", "宽度择时",
            mode="defensive_regime", blend_weight=weight)
    return result


def prepare(data):
    adjusted = {field: data._panels[field] * data._panels["cumadj"] for field in ("close", "high", "low")}
    return PreparedFeatures(prepare_features(adjusted["close"], data._panels["amt"], adjusted["high"], adjusted["low"]))


def annual_statistics(curve):
    rows = []
    for year, group in curve.groupby(curve.index.year):
        returns = group.daily_return
        net = (1 + returns).cumprod()
        deviation = returns.std(ddof=1)
        rows.append({"year": int(year), "return": float(net.iloc[-1] - 1),
                     "sharpe": float(returns.mean() / deviation * np.sqrt(252)) if deviation > 1e-12 else None,
                     "drawdown": float((net / net.cummax().clip(lower=1) - 1).min()),
                     "exposure": float(group.exposure.mean())})
    return rows


def robustness_score(metrics, years):
    evidence = [row for row in years if row["year"] in (2021, 2022, 2023)]
    sharpes = [row["sharpe"] if row["sharpe"] is not None else -10 for row in evidence]
    positive_years = sum(row["return"] > 0 for row in evidence)
    score = float(np.median(sharpes) + 0.25 * min(sharpes) - max(0, abs(metrics["max_drawdown"]) - 0.25))
    qualified = (len(evidence) == 3 and positive_years >= 2 and metrics["total_return"] > 0
                 and metrics["average_exposure"] >= 0.2 and metrics["max_drawdown"] >= -0.4)
    return score, qualified, positive_years


def choose_shortlist(rows, maximum=5):
    ordered = sorted(rows, key=lambda row: (-int(row["qualified"]), -row["robustness"], row["id"]))
    selected, families = [], set()
    for row in ordered:
        if row["family"] not in families:
            selected.append(row)
            families.add(row["family"])
        if len(selected) == maximum:
            break
    return selected


def changed_inputs(fingerprints):
    return [filename for filename, digest in fingerprints.items() if file_hash(Path(filename)) != digest]


def run_provenance(protocol):
    inputs, metadata = {}, {}
    metadata_directory = Path(protocol["metadata_path"]).resolve()
    for filename, digest in protocol["input_sha256"].items():
        path = Path(filename).resolve()
        destination = metadata if path.parent == metadata_directory else inputs
        destination[path.name] = {"sha256": digest}
    return {"input_files": inputs, "metadata_files": metadata,
            "engine_source_sha256": {Path(name).name: digest for name, digest in protocol["code_sha256"].items()
                                     if len(Path(name).parts) == 2}}


def run_candidate(data, features, config, candidate, campaign, stage):
    destination = campaign / stage / candidate["id"]
    request_path = destination / "request.json"
    if request_path.exists():
        request = read_json(request_path)
        if request["parameters"] != candidate["parameters"] or request["config"] != config.to_dict():
            raise ValueError("已保存运行的参数不同，不能覆盖")
        if request["status"] == "completed":
            from .saved import load_saved_result
            return load_saved_result(destination)
        raise ValueError("已有未完成记录，请保留记录并在新研究目录重跑")
    started = time.monotonic()
    request = {"id": destination.relative_to(ROOT / "runs").as_posix(),
               "label": candidate["name"] + " / " + stage, "kind": "training" if stage == "calibration" else "validation",
               "parent_study": campaign.name, "status": "running", "strategy_id": "builtin:technical_selection",
               "strategy_name": candidate["name"], "strategy_revision": 0,
               "parameters": candidate["parameters"], "config": config.to_dict(), "symbols": None,
               "data_path": data.source, "metadata_path": read_json(campaign / "protocol.json")["metadata_path"],
               "created_at": datetime.now(timezone.utc).isoformat(),
               "strategy_sha256": file_hash(ROOT / "strategies" / "technical_selection.py")}
    write_json(request_path, request)
    strategy = TechnicalStrategy(**candidate["parameters"])
    strategy.name = candidate["name"]
    strategy.features = features
    try:
        result = BacktestEngine(data, config).run(strategy)
        export_result(result, destination, (ROOT / "strategies" / "technical_selection.py").read_text(encoding="utf-8"))
        summary = read_json(destination / "summary.json")
        summary.update(run_provenance(read_json(campaign / "protocol.json")),
                       strategy_parameters=candidate["parameters"], strategy_sha256=request["strategy_sha256"])
        write_json(destination / "summary.json", summary)
        (destination / "technical_snapshot.py").write_text((ROOT / "ashare" / "technical.py").read_text(encoding="utf-8"), encoding="utf-8")
        buys = result.trades.loc[result.trades.side == "buy"]
        audit = {"st_or_unknown_buys": sum(data.state(row.date, row.symbol).get("is_st") is not False for row in buys.itertuples()),
                 "maximum_signal_date": str(result.equity.index[-1].date()),
                 "source_last_date": str(data.dates[-1].date()),
                 "integer_stale_positions_at_end": int(((result.positions.date == str(result.equity.index[-1].date()))
                     & (result.positions.equivalent_shares >= 1) & (result.positions.stale_days >= 20)).sum()),
                 "elapsed_seconds": round(time.monotonic() - started, 2)}
        write_json(destination / "audit.json", audit)
        if audit["st_or_unknown_buys"]:
            raise ValueError("出现ST或历史状态未知的买入，研究无效")
        request.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
        write_json(request_path, request)
        print(f"{stage} {candidate['id']}: 收益={result.metrics['total_return']:.2%}, 夏普={result.metrics['sharpe']}, 仓位={result.metrics['average_exposure']:.1%}, {audit['elapsed_seconds']}秒", flush=True)
        return result
    except Exception as error:
        request.update(status="failed", error=str(error))
        write_json(request_path, request)
        raise


def table_rows(rows):
    return [{"id": row["id"], "策略": row["name"], "类别": row["family"],
             "收益(%)": row["metrics"]["total_return"] * 100,
             "年化(%)": row["metrics"]["annual_return"] * 100,
             "回撤(%)": row["metrics"]["max_drawdown"] * 100,
             "夏普": row["metrics"]["sharpe"], "平均仓位(%)": row["metrics"]["average_exposure"] * 100,
             "费用(万元)": row["metrics"]["total_fees"] / 10_000,
             "双边换手": row["metrics"]["two_way_turnover"],
             "稳健评分": row.get("robustness"), "训练门槛通过": row.get("qualified"),
             **{f"{year['year']}收益(%)": year["return"] * 100 for year in row.get("years", [])}}
            for row in rows]


def write_report(campaign, record):
    def markdown_table(rows):
        lines = ["| 策略 | 收益 | 最大回撤 | 夏普 | 平均仓位 |", "|---|---:|---:|---:|---:|"]
        for row in rows:
            metrics = row["metrics"]
            sharpe = f"{metrics['sharpe']:.2f}" if metrics["sharpe"] is not None else "—"
            lines.append(f"| {row['name']} | {metrics['total_return']:.2%} | {metrics['max_drawdown']:.2%} | {sharpe} | {metrics['average_exposure']:.1%} |")
        return "\n".join(lines)

    calibration = sorted(record["calibration"], key=lambda row: (-int(row["qualified"]), -row["robustness"], row["id"]))
    text = "# 技术策略探索报告\n\n"
    text += "## 研究范围\n\n24个预先列明候选，另加流动性前100只等权参照。2020—2023年开发，2024年确认；未运行2025—2026年。所有结果均由平台原有撮合引擎产生，不使用简化的价格乘权重收益冒充账户回测。\n\n"
    text += "10亿元，买卖佣金各万一，卖出印花税万五；无最低佣金、额外过户费，基础滑点0。每股票每日成交额上限为上日成交额1%；沪深非ST且状态明确，过去120日价格完整，20日平均成交额≥1亿元，按流动性选前1500；最多100只、每只目标0.95%，不足保留现金。金额单位沿用平台研究假设为元。\n\n"
    text += "## 2020—2023年开发结果\n\n" + markdown_table(calibration) + "\n\n"
    text += "排序采用2021—2023年度夏普中位数+0.25×最差年度夏普−超过25%的全期回撤惩罚。训练门槛为三年中至少两年盈利、开发全期收益正、平均仓位≥20%、回撤不超过40%。按门槛及评分排序、每类最多一个，固定五个候选再读取2024年；无合格候选时仍保留诊断候选，但不推荐。2020年存在120/200日预热，单列而不参与年度稳健评分。\n\n"
    text += "## 2024年确认结果（按事前冻结顺序，不按2024收益重排）\n\n" + markdown_table(record["confirmation"]) + "\n\n"
    if record.get("baseline"):
        text += "## 同口径参照\n\n" + markdown_table(record["baseline"]) + "\n\n参照是流动性最高100只股票的每20日等权调仓账户，不是沪深300，也不是无成本全市场指数。\n\n"
    if record.get("sensitivity"):
        text += "## 固定前两名的2024滑点压力测试\n\n" + markdown_table(record["sensitivity"]) + "\n\n压力测试不用于重新选择赢家；超出日线高低价/涨跌停范围的滑点成交仍拒绝。\n\n"
    text += "## 策略定义\n\n- RSI14：昨日≤30、今日>30入场，RSI≥60或实际持仓满10日退出；对照增加MA60过滤。\n- RSI2：MA200上方、RSI2<5或<10买入；高于MA5、RSI2≥70或满10日卖出。借鉴Connors短周期RSI思想，不是三分量ConnorsRSI指标。\n- 布林带：20日均值、2倍总体标准差；从下轨外回到轨内买入，回到中轨或满10日退出，比较MA60过滤。触轨本身不等于买点。\n- 反转：5/20日复权收益最低者；动量：60日跳过最近5日，或120日跳过最近20日，只选正动量，每20日调仓。\n- 突破：收盘高于此前20/60日最高价买入，跌破此前10/30日最低价或持有60日退出。\n- 均线：MA20/60或MA60/120，按正的均线乖离排序，每10日调仓。\n- 低波：过去20/60日收益标准差低者，每5日调仓；不是低贝塔BAB多空策略的原样复现。\n- 组合：低波与5日反转、60日动量的横截面百分位排名加权；低波权重50%或75%。\n- 量价：5日反转×成交额5/20日比值（截断0.5—2），以及成交额加权的收盘区间位置反转。\n- 宽度：低波反转组合；可选流动性池中超过MA60的比例<50%时，目标总仓位减半，不引用未提供的市场指数。\n\n"
    text += "## 边界\n\n- 这是多重尝试后的探索性结果，未做多重检验校正，不宣称统计显著、稳定盈利或最佳策略。2024也曾在前次研究中使用，不能称为完全独立的盲测。2025—2026旧考核已被查看，本次未运行，不借该区间反向选参。\n- 本次开发/确认均不强制末日清仓，期末权益含真实剩余持仓市值；不是原作业的2026年9月离场考核。若之后考核，须另外冻结策略并沿用已批准的20日退出窗口。\n- 第一交易日只有信号满足且120/200日历史充足才能建仓，不能强行满仓。事件策略持有期从实际持仓开始计数，未成交信号不算持仓。\n- 复权为权益再投资近似；精确涨跌停价和公司行为现金明细仍不全。日频开盘成交与1%上日成交额不是实际开盘冲击模型。停牌、涨跌停受阻持仓不伪造成交。\n- 老师给定代码池未证明无幸存者偏差；无财务、行业、市值、指数数据，因此不声称实现基本面价值、行业中性、指数增强或准确BAB。\n- 全部候选CSV、年度结果、委托、成交、持仓、费用与历史ST审计均保留；极小复权残余与陈旧估值需分别审查。\n\n## 参考资料\n\n"
    text += "\n".join(f"- [{item['title']}]({item['url']})" for item in REFERENCES)
    (campaign / "研究报告.md").write_text(text + "\n", encoding="utf-8")


def run_discovery(source, metadata, campaign):
    campaign = Path(campaign).resolve()
    if not campaign.is_relative_to(ROOT / "runs"):
        raise ValueError("研究目录须位于平台runs内，便于实验记录发现")
    campaign.mkdir(parents=True, exist_ok=True)
    base = experiment_config(start="2020-01-02", end="2023-12-29", min_history=120,
                             seed_previous_close=False, liquidate_on_end=False,
                             write_off_research_dust=False)
    protocol_path = campaign / "protocol.json"
    fingerprints = {str(path): file_hash(path) for directory in (Path(source), Path(metadata))
                    for path in sorted(directory.glob("*")) if path.suffix in {".feather", ".csv", ".json"}}
    code = {str(path.relative_to(ROOT)): file_hash(path) for path in sorted((ROOT / "ashare").rglob("*.py"))}
    protocol = {"candidates": candidates(), "training_config": base.to_dict(),
                "data_path": str(Path(source).resolve()), "metadata_path": str(Path(metadata).resolve()),
                "confirmation": ["2024-01-02", "2024-12-31"], "holdout_access": "not_run",
                "selection_rule": "2021-2023年夏普中位数+0.25*最差年度夏普-超25%回撤惩罚；优先满足两年盈利/全期正收益/平均仓位20%/回撤40%，每族最多1名、共5名，先冻结后读取2024。",
                "input_sha256": fingerprints, "code_sha256": code, "references": REFERENCES}
    if protocol_path.exists():
        previous = read_json(protocol_path)
        if any(previous.get(key) != value for key, value in protocol.items()):
            raise ValueError("研究协议、输入或代码已变化，请建立新目录；不覆盖旧试验")
    else:
        write_json(protocol_path, {**protocol, "created_at": datetime.now(timezone.utc).isoformat()})
    record = {"id": campaign.name, "name": "技术策略探索 · 24候选", "status": "running",
              "calibration": [], "confirmation": [], "sensitivity": [], "baseline": [],
              "protocol": "protocol.json", "holdout_run": False}
    print("读取截至2023-12-29的开发数据，计算因果指标缓存", flush=True)
    data = DataBundle.from_feathers(source, metadata_dir=metadata, end=base.end)
    if "is_st" not in data.status_arrays:
        raise DataCoverageError("研究需要明确的历史ST状态")
    features = prepare(data)
    for candidate in candidates():
        result = run_candidate(data, features, base, candidate, campaign, "calibration")
        years = annual_statistics(result.equity)
        score, qualified, positive = robustness_score(result.metrics, years)
        record["calibration"].append({**candidate, "metrics": result.metrics, "years": years,
                                      "robustness": score, "qualified": qualified, "positive_years": positive})
        write_json(campaign / "campaign.json", record)
        pd.DataFrame(table_rows(record["calibration"])).to_csv(campaign / "开发候选.csv", index=False, encoding="utf-8-sig")
        del result
        gc.collect()
    baseline = {"id": "liquid-equal", "name": "流动性前100等权", "family": "参照",
                "parameters": {"mode": "liquid_equal", "top_n": 100, "rebalance_days": 20}}
    result = run_candidate(data, features, base, baseline, campaign, "calibration")
    record["baseline"].append({**baseline, "name": baseline["name"] + " / 2020—2023", "metrics": result.metrics})
    shortlist = choose_shortlist(record["calibration"])
    freeze_path = campaign / "frozen_shortlist.json"
    frozen = {"candidate_ids": [row["id"] for row in shortlist], "candidates": shortlist,
              "protocol_sha256": file_hash(protocol_path), "selected_without_2024_or_2025_2026": True}
    if freeze_path.exists():
        existing = read_json(freeze_path)
        if any(existing.get(key) != value for key, value in frozen.items()):
            raise ValueError("冻结候选不一致")
    else:
        write_json(freeze_path, {**frozen, "frozen_at": datetime.now(timezone.utc).isoformat()})
    del data, features, result
    gc.collect()
    print("候选已冻结：" + ", ".join(frozen["candidate_ids"]) + "；现在读取截至2024年底的数据", flush=True)
    config = replace(base, start="2024-01-02", end="2024-12-31", seed_previous_close=True)
    data = DataBundle.from_feathers(source, metadata_dir=metadata, end=config.end)
    features = prepare(data)
    for candidate in shortlist:
        result = run_candidate(data, features, config, candidate, campaign, "confirmation")
        record["confirmation"].append({**candidate, "metrics": result.metrics, "years": annual_statistics(result.equity)})
        write_json(campaign / "campaign.json", record)
        del result
    result = run_candidate(data, features, config, baseline, campaign, "confirmation")
    record["baseline"].append({**baseline, "name": baseline["name"] + " / 2024", "metrics": result.metrics})
    del result
    for candidate in shortlist[:2]:
        for slippage in (5, 10):
            stress = {**candidate, "id": candidate["id"] + f"-slippage{slippage}",
                      "name": candidate["name"] + f" / 单边{slippage}bp"}
            result = run_candidate(data, features, replace(config, slippage_bps=slippage), stress, campaign, "sensitivity")
            record["sensitivity"].append({**stress, "metrics": result.metrics})
            del result
    store = StrategyStore(ROOT / "workspace" / "strategies")
    source_text = (ROOT / "strategies" / "technical_selection.py").read_text(encoding="utf-8")
    saved = []
    for candidate in shortlist:
        spec = store.save(candidate["name"] + " · 探索候选", source_text, candidate["parameters"])
        saved.append({"id": candidate["id"], "strategy_id": spec.identifier})
    record.update(status="completed", saved_strategies=saved, finished_at=datetime.now(timezone.utc).isoformat())
    for filename, rows in (("确认候选.csv", record["confirmation"]), ("滑点压力.csv", record["sensitivity"])):
        pd.DataFrame(table_rows(rows)).to_csv(campaign / filename, index=False, encoding="utf-8-sig")
    yearly = [{"id": row["id"], "策略": row["name"], **year} for row in record["calibration"] for year in row["years"]]
    pd.DataFrame(yearly).to_csv(campaign / "年度表现.csv", index=False, encoding="utf-8-sig")
    changed = changed_inputs(fingerprints)
    if changed:
        raise RuntimeError("研究期间输入文件发生变化：" + ", ".join(changed))
    write_json(campaign / "campaign.json", record)
    write_report(campaign, record)
    print("研究完成：" + str(campaign / "研究报告.md"), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description="同一回测引擎上的24候选技术策略探索；不运行2025—2026")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_discovery(args.data, args.metadata, args.output)


if __name__ == "__main__":
    main()
