import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signals import TickSignal  # noqa: E402


class TickSignalTests(unittest.TestCase):
    def test_crossing_uses_prior_window_and_no_orders(self):
        signal = TickSignal(window=3)
        result = [signal.update("rb.SHFE", value) for value in (10, 10, 10, 11, 9, 12)]
        self.assertEqual(result, [None, None, None, None, "下穿", "上穿"])

    def test_symbols_are_independent(self):
        signal = TickSignal(window=2)
        for price in (10, 10, 11):
            signal.update("rb.SHFE", price)
        self.assertIsNone(signal.update("au.SHFE", 200))

    def test_rejects_bad_window(self):
        with self.assertRaises(ValueError):
            TickSignal(window=1)


if __name__ == "__main__":
    unittest.main()
