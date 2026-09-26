"""Prompt locally for an authorized SimNow account and test login without saving it."""

from __future__ import annotations

import getpass
import time

from vnpy.event import Event, EventEngine
from vnpy.trader.event import EVENT_LOG
from vnpy_ctp import CtpGateway


ENVIRONMENTS = {
    "1": {
        "name": "7x24 API 测试环境",
        "交易服务器": "182.254.243.31:40001",
        "行情服务器": "182.254.243.31:40011",
    },
    "2": {
        "name": "第一套生产时段仿真环境（第一组）",
        "交易服务器": "182.254.243.31:30001",
        "行情服务器": "182.254.243.31:30011",
    },
    "3": {
        "name": "第一套生产时段仿真环境（第二组）",
        "交易服务器": "182.254.243.31:30002",
        "行情服务器": "182.254.243.31:30012",
    },
}


def prompt_required(prompt: str, *, secret: bool = False) -> str:
    """Read a non-empty value, hiding it when it is a secret."""
    reader = getpass.getpass if secret else input
    while True:
        value = reader(prompt).strip()
        if value:
            return value
        print("输入不能为空，请重试。")


def main() -> int:
    print("SimNow 安全登录诊断（不会保存账号或密码，也不会下单）")
    print("1. 7x24 API 测试环境（当前推荐）")
    print("2. 第一套生产时段仿真环境（第一组）")
    print("3. 第一套生产时段仿真环境（第二组）")
    environment_key = input("请选择环境 [1]：").strip() or "1"
    if environment_key not in ENVIRONMENTS:
        print("环境选择无效。")
        return 2

    userid = prompt_required("请输入已授权的 UserID：")
    password = prompt_required("请输入交易密码（输入时不显示）：", secret=True)
    environment = ENVIRONMENTS[environment_key]
    setting = {
        "用户名": userid,
        "密码": password,
        "经纪商代码": "9999",
        "交易服务器": environment["交易服务器"],
        "行情服务器": environment["行情服务器"],
        "产品名称": "simnow_client_test",
        "授权编码": "0" * 16,
    }

    messages: list[str] = []

    def redact(message: str) -> str:
        for secret in (userid, password):
            if secret:
                message = message.replace(secret, "***")
        return message

    def on_log(event: Event) -> None:
        message = redact(str(event.data.msg))
        messages.append(message)
        print(f"[CTP] {message}", flush=True)

    print(f"\n正在连接：{environment['name']}")
    print(f"交易前置：{environment['交易服务器']}")
    print(f"行情前置：{environment['行情服务器']}")
    print("本次只验证登录，不订阅行情、不发送委托。\n")

    event_engine = EventEngine()
    event_engine.register(EVENT_LOG, on_log)
    event_engine.start()
    gateway = CtpGateway(event_engine, "CTP")

    try:
        gateway.connect(setting)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if any("交易服务器登录成功" in message for message in messages):
                break
            time.sleep(0.25)
    finally:
        gateway.close()
        event_engine.stop()
        password = ""
        setting.clear()

    if any("交易服务器登录成功" in message for message in messages):
        print("\n诊断结果：交易柜台登录成功。")
        return 0

    print("\n诊断结果：限定时间内未登录成功。请把最后几行错误提示发给我。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
