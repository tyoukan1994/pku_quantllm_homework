"""Safely diagnose a local CTP connection without printing credentials."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vnpy.event import Event, EventEngine
from vnpy.trader.event import EVENT_LOG
from vnpy_ctp import CtpGateway


ROOT = Path(__file__).resolve().parents[1]
SETTING_PATH = ROOT / ".vntrader" / "connect_ctp.json"


def load_setting() -> dict[str, str]:
    """Load the ignored local setting and validate it without exposing secrets."""
    if not SETTING_PATH.exists():
        raise SystemExit("未找到本机 CTP 配置，请先在图形界面填写一次连接信息。")

    setting = json.loads(SETTING_PATH.read_text(encoding="utf-8"))
    required = (
        "用户名",
        "密码",
        "经纪商代码",
        "交易服务器",
        "行情服务器",
        "产品名称",
        "授权编码",
    )
    missing = [key for key in required if not setting.get(key)]
    if missing:
        raise SystemExit(f"配置缺少字段：{', '.join(missing)}")

    if setting["授权编码"] != "0" * 16:
        raise SystemExit("授权编码格式错误：应当精确填写 16 个 0。")

    return setting


def main() -> int:
    parser = argparse.ArgumentParser(description="安全检查 SimNow/CTP 连接日志")
    parser.add_argument("--seconds", type=int, default=20, help="诊断等待秒数")
    args = parser.parse_args()

    setting = load_setting()
    secrets = tuple(str(setting[key]) for key in ("用户名", "密码"))
    messages: list[str] = []

    def redact(message: str) -> str:
        for secret in secrets:
            if secret:
                message = message.replace(secret, "***")
        return message

    def on_log(event: Event) -> None:
        message = redact(str(event.data.msg))
        messages.append(message)
        print(f"[CTP] {message}", flush=True)

    print("开始连接；用户名和密码不会输出。", flush=True)
    print(f"交易服务器：{setting['交易服务器']}", flush=True)
    print(f"行情服务器：{setting['行情服务器']}", flush=True)

    event_engine = EventEngine()
    event_engine.register(EVENT_LOG, on_log)
    event_engine.start()
    gateway = CtpGateway(event_engine, "CTP")

    try:
        gateway.connect(setting)
        deadline = time.monotonic() + max(args.seconds, 1)
        while time.monotonic() < deadline:
            if any("交易服务器登录成功" in message for message in messages):
                break
            time.sleep(0.25)
    finally:
        gateway.close()
        event_engine.stop()

    if any("交易服务器登录成功" in message for message in messages):
        print("诊断结果：交易柜台登录成功。", flush=True)
        return 0

    print("诊断结果：在限定时间内未登录成功。", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
