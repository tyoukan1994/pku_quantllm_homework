"""Securely download broad-index history from JQData to a gitignored CSV."""

from __future__ import annotations

import argparse
from datetime import date
from getpass import getpass
import os
from pathlib import Path

import pandas as pd

from data_source import JqDataSource


DEFAULT_SYMBOLS = (
    "000016.XSHG",  # SSE 50
    "000300.XSHG",  # CSI 300
    "000905.XSHG",  # CSI 500
    "000852.XSHG",  # CSI 1000
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从聚宽下载宽基指数历史行情")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--frequency", default="daily")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/private/jqdata/broad_indices_daily.csv"),
    )
    return parser.parse_args()


def prompt_credentials() -> tuple[str, str]:
    username = os.environ.get("JQDATA_USERNAME") or input("聚宽账号：").strip()
    password = os.environ.get("JQDATA_PASSWORD") or getpass("聚宽密码（输入不可见）：")
    if not username or not password:
        raise ValueError("聚宽账号和密码不能为空。")
    return username, password


def main() -> int:
    args = parse_args()
    username, password = prompt_credentials()

    try:
        import jqdatasdk as jq

        jq.set_params(enable_auth_prompt=False)
        jq.auth(username, password)
        quota = jq.get_query_count()
        print(
            "聚宽登录成功；今日总额度："
            f"{quota.get('total', '未知')}，剩余额度：{quota.get('spare', '未知')}"
        )

        source = JqDataSource(frequency=args.frequency, sdk=jq)
        frames: list[pd.DataFrame] = []
        for symbol in args.symbols:
            bars = source.load_bars(symbol, args.start, args.end)
            frames.append(bars)
            print(f"{symbol}: {len(bars)} 条")

        combined = pd.concat(frames, ignore_index=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.output, index=False)
        print(f"已保存：{args.output.resolve()}")
        return 0
    except Exception as exc:
        print(f"聚宽下载失败：{exc}")
        return 1
    finally:
        try:
            if "jq" in locals():
                jq.logout()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
