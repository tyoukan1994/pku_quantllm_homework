"""Read-only SimNow market-data demo with local secret entry.

The script logs in, waits for the contract catalogue, subscribes to one
contract and prints ticks.  It deliberately imports no order request types.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from threading import Event as ThreadEvent

from vnpy.event import Event, EventEngine
from vnpy.trader.constant import Product
from vnpy.trader.event import EVENT_CONTRACT, EVENT_LOG, EVENT_TICK
from vnpy.trader.object import ContractData, SubscribeRequest, TickData
from vnpy_ctp import CtpGateway

from secure_login import ENVIRONMENTS, prompt_required
from signals import TickSignal


ROOT = Path(__file__).resolve().parents[1]
TICK_DIR = ROOT / ".vntrader" / "ticks"


def matching_contracts(
    contracts: dict[str, ContractData], query: str
) -> list[ContractData]:
    """Return futures contracts matching a symbol, vt_symbol or product prefix."""
    normalized = query.strip().casefold()
    futures = [contract for contract in contracts.values() if contract.product == Product.FUTURES]

    exact = [
        contract
        for contract in futures
        if contract.symbol.casefold() == normalized
        or contract.vt_symbol.casefold() == normalized
    ]
    if exact:
        return sorted(exact, key=lambda contract: contract.vt_symbol)

    return sorted(
        [contract for contract in futures if contract.symbol.casefold().startswith(normalized)],
        key=lambda contract: contract.vt_symbol,
    )


def choose_contract(contracts: dict[str, ContractData]) -> ContractData | None:
    """Interactively select one known futures contract."""
    query = input("请输入合约代码或品种前缀 [rb]：").strip() or "rb"
    matches = matching_contracts(contracts, query)
    if not matches:
        print(f"没有找到与 {query!r} 匹配的期货合约。")
        return None

    shown = matches[:20]
    print("\n可选合约：")
    for index, contract in enumerate(shown, start=1):
        print(
            f"{index:>2}. {contract.vt_symbol:<18} {contract.name:<12} "
            f"乘数={contract.size:g} 最小变动={contract.pricetick:g}"
        )

    raw_index = input("请选择序号 [1]：").strip() or "1"
    try:
        selected = int(raw_index)
    except ValueError:
        print("序号必须是整数。")
        return None

    if selected < 1 or selected > len(shown):
        print("序号超出范围。")
        return None
    return shown[selected - 1]


def tick_row(tick: TickData) -> list[str | float]:
    """Convert a tick into the stable CSV schema used by this demo."""
    return [
        tick.datetime.isoformat(),
        tick.vt_symbol,
        tick.last_price,
        tick.bid_price_1,
        tick.bid_volume_1,
        tick.ask_price_1,
        tick.ask_volume_1,
        tick.volume,
        tick.open_interest,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="SimNow 只读行情订阅 Demo")
    parser.add_argument("--ticks", type=int, default=10, help="收到多少条 Tick 后退出")
    parser.add_argument("--timeout", type=int, default=90, help="订阅后最多等待多少秒")
    args = parser.parse_args()
    if args.ticks < 1 or args.timeout < 1:
        parser.error("--ticks 和 --timeout 必须大于 0")

    print("SimNow 只读行情 Demo（不保存凭据，不包含下单功能）")
    print("1. 第二套 7x24 API 测试环境（推荐）")
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

    contracts: dict[str, ContractData] = {}
    messages: list[str] = []
    catalogue_ready = ThreadEvent()
    login_failed = ThreadEvent()
    tick_done = ThreadEvent()
    target_vt_symbol = ""
    tick_count = 0
    csv_file = None
    writer = None
    signal = TickSignal(window=10)

    def redact(message: str) -> str:
        for secret in (userid, password):
            if secret:
                message = message.replace(secret, "***")
        return message

    def on_log(event: Event) -> None:
        message = redact(str(event.data.msg))
        messages.append(message)
        print(f"[CTP] {message}", flush=True)
        if "登录失败" in message or "授权验证失败" in message:
            login_failed.set()
        if "合约信息查询成功" in message:
            catalogue_ready.set()

    def on_contract(event: Event) -> None:
        contract: ContractData = event.data
        contracts[contract.vt_symbol] = contract

    def on_tick(event: Event) -> None:
        nonlocal tick_count
        tick: TickData = event.data
        if tick.vt_symbol != target_vt_symbol or writer is None or csv_file is None:
            return

        tick_count += 1
        writer.writerow(tick_row(tick))
        csv_file.flush()
        print(
            f"[行情 {tick_count}/{args.ticks}] {tick.datetime.isoformat()} "
            f"{tick.vt_symbol} 最新={tick.last_price:g} "
            f"买一={tick.bid_price_1:g} 卖一={tick.ask_price_1:g} "
            f"成交量={tick.volume:g}",
            flush=True,
        )
        message = signal.update(tick.vt_symbol, tick.last_price)
        if message:
            print(f"[演示信号] {tick.vt_symbol} {message}，未自动下单", flush=True)
        if tick_count >= args.ticks:
            tick_done.set()

    event_engine = EventEngine()
    event_engine.register(EVENT_LOG, on_log)
    event_engine.register(EVENT_CONTRACT, on_contract)
    event_engine.register(EVENT_TICK, on_tick)
    event_engine.start()
    gateway = CtpGateway(event_engine, "CTP")

    try:
        print(f"\n正在连接：{environment['name']}……")
        gateway.connect(setting)

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not catalogue_ready.is_set():
            if login_failed.is_set():
                print("登录失败，停止行情测试。")
                return 1
            time.sleep(0.25)

        if not catalogue_ready.is_set():
            print("等待合约表超时，未发起行情订阅。")
            return 1

        print(f"\n已收到 {len(contracts)} 个合约定义。")
        contract = choose_contract(contracts)
        if contract is None:
            return 2

        target_vt_symbol = contract.vt_symbol
        TICK_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        csv_path = TICK_DIR / f"{contract.vt_symbol}-{timestamp}.csv"
        csv_file = csv_path.open("w", encoding="utf-8", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "datetime",
                "vt_symbol",
                "last_price",
                "bid_price_1",
                "bid_volume_1",
                "ask_price_1",
                "ask_volume_1",
                "volume",
                "open_interest",
            ]
        )

        gateway.subscribe(SubscribeRequest(contract.symbol, contract.exchange))
        print(f"\n已订阅 {contract.vt_symbol}，等待最多 {args.timeout} 秒……")
        tick_done.wait(args.timeout)

        if tick_count:
            print(f"\n共收到 {tick_count} 条 Tick，已保存到：{csv_path}")
            if tick_count < args.ticks:
                print("等待超时，但已验证行情推送链路。")
            return 0

        print("\n没有收到 Tick。登录和合约查询成功，但当前时段可能没有行情推送。")
        return 3
    except KeyboardInterrupt:
        print("\n用户停止行情测试。")
        return 130
    finally:
        if csv_file is not None:
            csv_file.close()
        gateway.close()
        event_engine.stop()
        password = ""
        setting.clear()


if __name__ == "__main__":
    raise SystemExit(main())
