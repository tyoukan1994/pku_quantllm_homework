"""Safely verify RQData credentials and historical-data permission."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import re
from string import Template
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="验证 RQData 登录及一小段历史行情权限，不保存凭据。"
    )
    parser.add_argument("--symbol", default="000300.XSHG", help="RQData 标的代码")
    parser.add_argument("--start", default="2024-01-02", help="开始日期")
    parser.add_argument("--end", default="2024-01-05", help="结束日期")
    parser.add_argument(
        "--license-script",
        type=Path,
        help="米筐提供的 make.sh；仅解析内嵌 License，不执行脚本",
    )
    return parser.parse_args()


def load_license_uri(path: Path) -> str:
    """Parse the variables in RQData's make.sh without executing the script."""
    values: dict[str, str] = {}
    for raw_line in path.read_text(errors="strict").splitlines():
        line = raw_line.strip()
        match = re.fullmatch(r"(name|password|host|port|url)=(.*)", line)
        if not match:
            continue

        key, value = match.groups()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = Template(value).safe_substitute(values)

    required = {"name", "password", "host", "port", "url"}
    if required - values.keys() or "$" in values.get("url", ""):
        raise ValueError("无法从 make.sh 安全解析 RQData License")
    return values["url"]


def main() -> int:
    args = parse_args()

    try:
        import rqdatac
    except ImportError:
        print("尚未安装 rqdatac，请先在项目 .venv 中安装。", file=sys.stderr)
        return 2

    temporary_license: str | None = None
    if args.license_script:
        try:
            temporary_license = load_license_uri(args.license_script)
            os.environ["RQDATAC2_CONF"] = temporary_license
        except (OSError, UnicodeError, ValueError) as exc:
            print(f"读取 License 脚本失败：{exc}", file=sys.stderr)
            return 2

    using_environment = bool(
        os.environ.get("RQDATAC2_CONF") or os.environ.get("RQDATAC_CONF")
    )

    try:
        if using_environment:
            print("检测到 RQData License，将直接使用（不会打印配置内容）。")
            rqdatac.init()
        else:
            username = input("RQData 用户名或 License：").strip()
            password = getpass.getpass("RQData 密码或 Token（输入不可见）：")
            if not username or not password:
                print("用户名和密码不能为空。", file=sys.stderr)
                return 2
            rqdatac.init(username, password)

        bars = rqdatac.get_price(
            args.symbol,
            start_date=args.start,
            end_date=args.end,
            frequency="1d",
            fields=["open", "high", "low", "close", "volume"],
            adjust_type="none",
        )
        if bars is None or bars.empty:
            print(
                f"登录成功，但 {args.symbol} 在 {args.start} 至 {args.end} 没有返回行情。"
            )
            return 1

        first_date = bars.index[0]
        last_date = bars.index[-1]
        print("RQData 登录及历史行情权限验证成功。")
        print(f"标的：{args.symbol}")
        print(f"返回：{len(bars)} 行，{first_date} 至 {last_date}")
        return 0
    except Exception as exc:
        message = str(exc)
        if temporary_license:
            message = message.replace(temporary_license, "<已隐藏License>")
        print(f"RQData 验证失败：{type(exc).__name__}: {message}", file=sys.stderr)
        return 1
    finally:
        try:
            rqdatac.reset()
        except Exception:
            pass
        if temporary_license:
            os.environ.pop("RQDATAC2_CONF", None)


if __name__ == "__main__":
    raise SystemExit(main())
