"""Minimal vnpy_ctp desktop demo. Orders are manual in VeighNa Trader."""

from signals import TickSignal


def main():
    try:
        from vnpy.event import EventEngine
        from vnpy.trader.engine import MainEngine
        from vnpy.trader.event import EVENT_LOG, EVENT_ORDER, EVENT_TICK, EVENT_TRADE
        from vnpy.trader.ui import MainWindow, create_qapp
        from vnpy_ctp import CtpGateway
    except ImportError as exc:
        raise SystemExit(
            f"依赖导入失败：{exc}。请先按 README 的 Mac 安装指南配置 Python 3.10+。"
        ) from exc

    qapp = create_qapp()
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.add_gateway(CtpGateway)
    signal = TickSignal(window=10)

    def on_tick(event):
        tick = event.data
        print(
            f"[行情] {tick.datetime} {tick.vt_symbol} "
            f"最新={tick.last_price} 成交量={tick.volume}",
            flush=True,
        )
        message = signal.update(tick.vt_symbol, tick.last_price)
        if message:
            print(f"[演示信号] {tick.vt_symbol} {message}，未自动下单", flush=True)

    def on_log(event):
        print(f"[日志] {event.data.msg}", flush=True)

    def on_order(event):
        order = event.data
        print(f"[委托] {order.vt_orderid} {order.status.value}", flush=True)

    def on_trade(event):
        trade = event.data
        print(f"[成交] {trade.vt_tradeid} {trade.vt_symbol} {trade.price} x {trade.volume}", flush=True)

    event_engine.register(EVENT_TICK, on_tick)
    event_engine.register(EVENT_LOG, on_log)
    event_engine.register(EVENT_ORDER, on_order)
    event_engine.register(EVENT_TRADE, on_trade)

    window = MainWindow(main_engine, event_engine)
    window.showMaximized()
    qapp.exec()


if __name__ == "__main__":
    main()
