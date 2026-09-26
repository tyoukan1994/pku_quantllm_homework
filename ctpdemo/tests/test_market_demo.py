import sys
import unittest
from datetime import datetime
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_demo import matching_contracts, tick_row  # noqa: E402
from vnpy.trader.constant import Exchange, Product  # noqa: E402
from vnpy.trader.object import ContractData, TickData  # noqa: E402


def contract(symbol, exchange=Exchange.SHFE, product=Product.FUTURES):
    return ContractData(
        gateway_name="CTP",
        symbol=symbol,
        exchange=exchange,
        name=symbol,
        product=product,
        size=10,
        pricetick=1,
    )


class MarketDemoTests(unittest.TestCase):
    def setUp(self):
        items = [
            contract("rb2701"),
            contract("rb2705"),
            contract("au2612"),
            contract("rb2701C3500", product=Product.OPTION),
        ]
        self.contracts = {item.vt_symbol: item for item in items}

    def test_exact_vt_symbol_is_case_insensitive(self):
        matches = matching_contracts(self.contracts, "RB2701.shfe")
        self.assertEqual([item.vt_symbol for item in matches], ["rb2701.SHFE"])

    def test_prefix_only_returns_futures(self):
        matches = matching_contracts(self.contracts, "rb")
        self.assertEqual(
            [item.vt_symbol for item in matches],
            ["rb2701.SHFE", "rb2705.SHFE"],
        )

    def test_tick_row_has_stable_schema(self):
        tick = TickData(
            gateway_name="CTP",
            symbol="rb2701",
            exchange=Exchange.SHFE,
            datetime=datetime(2026, 9, 20, 9, 0),
            last_price=3500,
            bid_price_1=3499,
            ask_price_1=3501,
        )
        row = tick_row(tick)
        self.assertEqual(len(row), 9)
        self.assertEqual(row[1:4], ["rb2701.SHFE", 3500, 3499])


if __name__ == "__main__":
    unittest.main()
