import unittest
from unittest.mock import patch
from engine import HistoricalBook, Simulator, cents, stamp_to_ms, ms_to_stamp


def add(seq, oid, side, price, qty, time=93000000):
    return (seq, 'O', dict(BizIndex=seq, Channel=1002, Symbol=601688, OrderKind=65,
                          FunctionCode=side, OrderOriNo=oid, Price=price, Volume=qty, Time=time))


def cancel(seq, oid, side, price, qty, time):
    e = add(seq, oid, side, price, qty, time); e[2]['OrderKind'] = 68
    return e


def trade(seq, bid, ask, price, qty, flag, time):
    return (seq, 'T', dict(BizIndex=seq, Channel=1002, Symbol=601688,
                          BidOrder=bid, AskOrder=ask, Price=price, Volume=qty, BSFlag=flag, Time=time))


def sim(events=None, cash=100000, shares=1000, symbol=601688):
    if events is None: events = [add(1, 10, 66, 10, 100), add(2, 20, 83, 10.02, 100)]
    with patch('engine.load_stock', return_value=(events, [], 1000, 0)):
        s = Simulator(symbol=symbol, cash=cash, initial_shares=shares)
    s.advance(93000000)
    return s


class MarketOrderTests(unittest.TestCase):
    def test_five_price_levels_not_five_orders(self):
        events = [add(1, 10, 66, 10, 100), add(2, 20, 83, 10.01, 100),
                  add(3, 21, 83, 10.01, 100)]
        events += [add(i + 3, i + 21, 83, 10.01 + i / 100, 100) for i in range(1, 6)]
        s = sim(events)
        s.submit_market('B', 800, 'FIVE_CANCEL', 11)
        o = s.orders[0]
        self.assertEqual((o.status, o.remaining), ('CANCELLED', 200))
        self.assertEqual([f['price'] for f in s.fills], [10.01, 10.01, 10.02, 10.03, 10.04, 10.05])
        self.assertEqual(s.state()['frozen_cash'], 0)
        self.assertEqual(s.used_liquidity[26], 0)

    def test_five_limit_uses_this_orders_latest_fill_and_reserves_residual(self):
        s = sim([add(1, 10, 66, 10, 100), add(2, 20, 83, 10.01, 100),
                 add(3, 21, 83, 10.02, 100)])
        s.book.last_price = 999
        s.submit_market('B', 300, 'FIVE_LIMIT', 10.50)
        o = s.active
        self.assertTrue(o.converted_to_limit)
        self.assertEqual((o.price, o.remaining), (1002, 100))
        self.assertEqual(s.state()['frozen_cash'], 1002)
        self.assertEqual(s.state()['cash'], 97997)

    def test_five_limit_no_fill_falls_back_to_own_best(self):
        s = sim()
        s.submit_market('B', 100, 'FIVE_LIMIT', 10.01)
        self.assertFalse(s.fills)
        self.assertEqual(s.active.price, 1000)
        self.assertEqual(s.state()['frozen_cash'], 1000)
        self.assertEqual(s.active.ahead, {10})

    def test_no_reference_quote_cancels(self):
        for mode, events in [('SAME_BEST', [add(1, 20, 83, 10.02, 100)]),
                             ('OPPOSITE_BEST', [add(1, 10, 66, 10, 100)]),
                             ('FIVE_LIMIT', []), ('FIVE_CANCEL', [])]:
            with self.subTest(mode=mode):
                s = sim(events)
                s.submit_market('B', 100, mode, 11)
                self.assertEqual(s.orders[0].status, 'CANCELLED')
                self.assertFalse(s.fills)
                self.assertEqual(s.state()['available_cash'], 100000)

    def test_opposite_best_locks_one_price_and_queues_residual(self):
        s = sim([add(1, 10, 66, 10, 100), add(2, 20, 83, 10.01, 100),
                 add(3, 21, 83, 10.02, 100)])
        s.submit_market('B', 200, 'OPPOSITE_BEST', 11)
        self.assertEqual([(f['price'], f['quantity']) for f in s.fills], [(10.01, 100)])
        self.assertEqual((s.active.price, s.active.remaining), (1001, 100))
        self.assertEqual(s.used_liquidity[21], 0)

    def test_same_best_queues_at_entry_quote(self):
        s = sim()
        s.submit_market('B', 100, 'SAME_BEST', 11)
        self.assertFalse(s.fills)
        self.assertEqual(s.active.price, 1000)
        self.assertEqual(s.active.ahead, {10})

    def test_protection_caps_buy_and_floors_sell(self):
        events = [add(1, 10, 66, 10, 100), add(2, 11, 66, 9.99, 100),
                  add(3, 20, 83, 10.01, 100), add(4, 21, 83, 10.02, 100)]
        for side, cap, expected in [('B', 10.01, 10.01), ('S', 10, 10)]:
            with self.subTest(side=side):
                s = sim(events); s.submit_market(side, 200, 'FIVE_CANCEL', cap)
                self.assertEqual([(f['price'], f['quantity']) for f in s.fills], [(expected, 100)])
                self.assertEqual(s.orders[0].status, 'CANCELLED')
                self.assertEqual(s.state()['frozen_cash'], 0)
                self.assertEqual(s.state()['sellable_shares'], s.settled)

    def test_reference_and_conversion_outside_protection_cancel(self):
        for mode, cap in [('OPPOSITE_BEST', 10.01), ('SAME_BEST', 9.99), ('FIVE_LIMIT', 9.99)]:
            with self.subTest(mode=mode):
                s = sim(); s.submit_market('B', 100, mode, cap)
                self.assertFalse(s.fills)
                self.assertEqual(s.orders[0].status, 'CANCELLED')
                self.assertIn('原型', s.orders[0].reason)

    def test_market_protection_not_subject_to_limit_price_cage(self):
        s = sim()
        with self.assertRaisesRegex(ValueError, '有效申报'): s.submit('B', 10.50, 100)
        s.submit_market('B', 100, 'FIVE_CANCEL', 10.50)
        self.assertEqual(s.orders[0].status, 'FILLED')
        self.assertIsNone(s.state()['orders'][0]['price'])
        self.assertEqual(s.state()['orders'][0]['protection_price'], 10.50)

    def test_invalid_market_requests_leave_no_order(self):
        for mode, cap, qty in [('FIVE_CANCEL', None, 100), ('FIVE_CANCEL', 10.001, 100),
                               ('UNKNOWN', 11, 100), ('FIVE_CANCEL', 12, 100),
                               ('FIVE_CANCEL', 11, 101)]:
            with self.subTest(mode=mode, cap=cap, qty=qty):
                s = sim()
                with self.assertRaises(ValueError): s.submit_market('B', qty, mode, cap)
                self.assertEqual(s.orders, [])
        s = sim(cash=1000)
        with self.assertRaisesRegex(ValueError, '资金不足'): s.submit_market('B', 100, 'FIVE_CANCEL', 11)
        s = sim(shares=0)
        with self.assertRaisesRegex(ValueError, 'T\\+1'): s.submit_market('S', 100, 'FIVE_CANCEL', 10)
        s.advance(120000000)
        with self.assertRaisesRegex(ValueError, '连续竞价'): s.submit_market('B', 100, 'FIVE_CANCEL', 11)

    def test_star_market_quantity_limit_and_one_share_increments(self):
        s = sim(cash=2000000, symbol=688981)
        for qty in [100, 50001]:
            with self.assertRaises(ValueError): s.submit_market('B', qty, 'FIVE_CANCEL', 11)
        s.submit_market('B', 201, 'FIVE_CANCEL', 11)
        self.assertEqual(s.orders[0].remaining, 101)
        s.submit('B', 10, 100000)
        self.assertEqual(s.active.quantity, 100000)
        s = sim(symbol=688981, shares=150)
        s.submit_market('S', 150, 'FIVE_CANCEL', 10)
        self.assertEqual(s.settled, 50)

    def test_no_liquidity_reuse_between_limit_and_market(self):
        s = sim(); s.submit('B', 10.02, 100)
        s.submit_market('B', 100, 'FIVE_CANCEL', 11)
        self.assertEqual(s.orders[-1].status, 'CANCELLED')
        self.assertEqual(len(s.fills), 1)

    def test_used_level_does_not_promote_sixth_into_five_levels(self):
        s = sim([add(i, i + 20, 83, 10 + i / 100, 100) for i in range(1, 7)])
        s.submit('B', 10.01, 100)
        s.submit_market('B', 600, 'FIVE_CANCEL', 11)
        self.assertEqual(s.orders[-1].remaining, 200)
        self.assertEqual(s.used_liquidity[26], 0)

    def test_market_residual_can_fill_later_and_releases_reserve(self):
        events = [add(1, 10, 66, 10, 100), add(2, 20, 83, 10.01, 100),
                  add(3, 30, 83, 10.01, 100, 93001000)]
        s = sim(events); s.submit_market('B', 200, 'FIVE_LIMIT', 11)
        self.assertEqual(s.state()['frozen_cash'], 1001)
        s.advance(93001000)
        self.assertEqual(s.orders[0].status, 'FILLED')
        self.assertEqual(s.state()['frozen_cash'], 0)
        self.assertEqual(s.bought_today, 200)

    def test_recent_market_tape_is_bounded_and_separate_from_virtual_fills(self):
        events = [add(1, 10, 83, 10.02, 3000)]
        events += [trade(i + 2, i + 100, 10, 10.02, 100, 66, 93000010 + i * 10) for i in range(25)]
        s = sim(events); s.advance(93001000)
        st = s.state()
        self.assertEqual(len(st['market_trades']), 20)
        self.assertEqual(st['market_trades'][0]['seq'], 26)
        self.assertEqual(st['market_volume'], 2500)
        self.assertFalse(st['fills'])


class ReplayTests(unittest.TestCase):
    def test_residual_add_after_active_execution(self):
        b = HistoricalBook()
        for _, typ, x in [add(1, 10, 83, 10, 100), trade(2, 20, 10, 10, 100, 66, 93000010),
                          add(3, 20, 66, 10, 200, 93000010)]: b.apply(typ, x)
        self.assertEqual(b.orders[20][2], 200)
        self.assertEqual(b.stats['missing_active_trade'], 1)
        self.assertEqual(b.stats['missing_trade'], 0)

    def test_book_rejects_overdraw(self):
        b = HistoricalBook(); b.apply('O', add(1, 10, 66, 10, 100)[2])
        with self.assertRaises(ValueError): b.reduce(10, 101, 'cancel')

    def test_price_priority_multi_level_and_residual_cash(self):
        s = sim([add(1, 20, 83, 10.02, 100), add(2, 21, 83, 10.03, 100)])
        s.submit('B', 10.04, 300)
        self.assertEqual([(f['price'], f['quantity']) for f in s.fills], [(10.02, 100), (10.03, 100)])
        self.assertEqual(s.active.remaining, 100)
        self.assertEqual(s.state()['frozen_cash'], 1004)
        self.assertEqual(s.bought_today, 200)
        self.assertEqual(s.book.orders[20][2], 100)  # immutable historical market

    def test_no_repeat_sweep_of_same_historical_liquidity(self):
        s = sim(); s.submit('B', 10.02, 100); s.submit('B', 10.02, 100)
        self.assertEqual(len(s.fills), 1)
        self.assertEqual(s.active.remaining, 100)

    def test_fifo_historical_queue_and_later_order(self):
        events = [add(1, 10, 66, 10, 100), add(2, 20, 83, 10.02, 500),
                  add(3, 30, 66, 10, 100, 93001000),
                  trade(4, 10, 90, 10, 100, 83, 93002000),
                  trade(5, 30, 91, 10, 50, 83, 93003000)]
        s = sim(events); s.submit('B', 10, 100); s.advance(93002000)
        self.assertFalse(s.fills)
        s.advance(93003000)
        self.assertEqual(s.active.remaining, 50)
        self.assertEqual(s.fills[0]['quantity'], 50)

    def test_cancellation_advances_queue_without_filling(self):
        s = sim([add(1, 10, 66, 10, 100), cancel(2, 10, 66, 10, 100, 93001000)])
        s.submit('B', 10, 100); s.advance(93001000)
        self.assertFalse(s.fills)
        self.assertEqual(s.state()['orders'][0]['queue_ahead_quantity'], 0)
        s.cancel(); self.assertEqual(s.state()['frozen_cash'], 0)

    def test_new_opposite_residual_crosses_shadow_limit(self):
        s = sim([add(1, 10, 66, 9.99, 100), add(2, 20, 83, 10.02, 100),
                 add(3, 30, 83, 10, 100, 93001000)])
        s.submit('B', 10, 100); s.advance(93001000)
        self.assertEqual(s.orders[0].status, 'FILLED')
        self.assertEqual(s.fills[0]['price'], 10)
        self.assertEqual(s.used_liquidity[30], 100)

    def test_t_plus_one(self):
        s = sim(shares=0); s.submit('B', 10.02, 100)
        with self.assertRaisesRegex(ValueError, 'T\+1'): s.submit('S', 10, 100)

    def test_no_auction_execution_and_no_reverse_time(self):
        s = sim([add(1, 10, 66, 10, 100), add(2, 20, 83, 10.02, 100),
                 trade(3, 10, 20, 10, 100, 0, 150000000)])
        s.submit('B', 10, 100); s.advance(150000000)
        self.assertFalse(s.fills)
        self.assertEqual(s.orders[0].status, 'EXPIRED_AT_AUCTION_BOUNDARY')
        with self.assertRaises(ValueError): s.submit('B', 10, 100)
        with self.assertRaises(ValueError): s.advance(93000000)

    def test_quantities_cages_and_tick(self):
        s = sim()
        for p, q in [(10.001, 100), (10.03, 101), (10.50, 100), (12, 100)]:
            with self.assertRaises(ValueError): s.submit('B', p, q)
        star = sim(symbol=688981)
        with self.assertRaises(ValueError): star.submit('B', 10, 100)
        star.submit('B', 10, 201)
        self.assertEqual(star.active.quantity, 201)

    def test_time_arithmetic_and_half_up(self):
        self.assertEqual(ms_to_stamp(stamp_to_ms(95959000) + 1000), 100000000)
        self.assertEqual(cents('1.005'), 101)


if __name__ == '__main__': unittest.main()
