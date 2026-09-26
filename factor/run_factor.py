"""Run the candidate factor on a CSV export; Choice can replace the source later."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_source import CsvBarDataSource
from momentum import compute_volatility_adjusted_momentum, summarize_daily_backtest


def main() -> int:
    parser = argparse.ArgumentParser(description="候选波动率调整动量因子（日线）")
    parser.add_argument("--csv", required=True, type=Path, help="标准 OHLCV CSV")
    parser.add_argument("--symbol", required=True, help="CSV 中的合约或连续合约代码")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--cost-bps", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("factor/output"))
    args = parser.parse_args()

    source = CsvBarDataSource(args.csv)
    bars = source.load_bars(args.symbol, args.start, args.end)
    result = compute_volatility_adjusted_momentum(bars, args.lookback, args.cost_bps)
    metrics = summarize_daily_backtest(result)

    args.output.mkdir(parents=True, exist_ok=True)
    slug = "".join(character if character.isalnum() else "_" for character in args.symbol)
    result_path = args.output / f"{slug}_factor.csv"
    metrics_path = args.output / f"{slug}_metrics.json"
    result.to_csv(result_path, index=False)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"因子明细：{result_path}")
    print(f"指标摘要：{metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("注意：这是候选因子的历史回测，不代表未来收益；提交前必须补充数据来源、换月和样本外验证。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
