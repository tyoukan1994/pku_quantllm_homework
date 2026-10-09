"""Single-day SSE historical book replay and zero-impact shadow execution.

Feed mapping is inferred and audited, not a vendor-certified schema.
Continuous-auction limit orders and the four SSE market instructions.
"""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, asdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
import os
from typing import Optional
import pyarrow.parquet as pq

DEFAULT_DATA = Path(os.environ.get('SSE_L2_DATA_DIR', str(Path(__file__).resolve().parent / 'data' / 'sse50_20260923'))).expanduser()
DATE = '20260923'
MARKET_TYPES = {
    'FIVE_CANCEL': '最优五档即时成交剩余撤销',
    'FIVE_LIMIT': '最优五档即时成交剩余转限价',
    'SAME_BEST': '本方最优价格',
    'OPPOSITE_BEST': '对手方最优价格',
}


def cents(value):
    return int((Decimal(str(value)) * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def stamp_to_ms(stamp):
    stamp = int(stamp)
    h, rest = divmod(stamp, 10000000)
    m, rest = divmod(rest, 100000)
    s, ms = divmod(rest, 1000)
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
        raise ValueError('无效时间')
    return ((h * 60 + m) * 60 + s) * 1000 + ms


def ms_to_stamp(ms):
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return h * 10000000 + m * 100000 + s * 1000 + ms


def clock_text(stamp):
    s = f'{int(stamp):09d}'
    return f'{s[:2]}:{s[2:4]}:{s[4:6]}.{s[6:]}'


def continuous(t):
    return 93000000 <= t <= 113000000 or 130000000 <= t < 145700000


def phase(t):
    if t < 91500000: return '盘前'
    if t < 92500000: return '开盘集合竞价'
    if t < 93000000: return '开盘撮合后等待'
    if continuous(t): return '连续竞价'
    if t < 130000000: return '午间休市'
    if t <= 150000000: return '收盘集合竞价'
    return '竞价交易结束'


def load_stock(symbol, data_dir=DEFAULT_DATA):
    """Order by business sequence inside one symbol/channel, never timestamp alone."""
    data_dir = Path(data_dir)
    orders = pq.read_table(data_dir / f'ord_{DATE}.parquet', filters=[('Symbol', '=', symbol)]).to_pylist()
    trades = pq.read_table(data_dir / f'exe_{DATE}.parquet', filters=[('Symbol', '=', symbol)]).to_pylist()
    snapshots = pq.read_table(data_dir / f'snp_{DATE}.parquet', filters=[('Symbol', '=', symbol)]).to_pylist()
    events = [(x['BizIndex'], 'O', x) for x in orders]
    events += [(x['BizIndex'], 'T', x) for x in trades if x['Time'] <= 150000000]
    channels = {x['Channel'] for _, _, x in events}
    if len(channels) != 1:
        raise ValueError('原型要求一只股票的竞价事件属于同一频道')
    events.sort(key=lambda x: x[0])
    seqs = [s for s, _, _ in events]
    times = [x['Time'] for _, _, x in events]
    if len(seqs) != len(set(seqs)):
        raise ValueError('竞价事件业务序号重复')
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError('按业务序号排列后的时间倒退，需核对字段')
    preclose = next((cents(x['PreClose']) for x in snapshots if x['PreClose'] > 0), None)
    if not preclose: raise ValueError('缺少前收盘价')
    return events, snapshots, preclose, len(trades) - sum(t == 'T' for _, t, _ in events)


class HistoricalBook:
    def __init__(self):
        self.orders = {}  # original order id -> [side, price in cents, residual qty]
        self.levels = {66: defaultdict(int), 83: defaultdict(int)}
        self.stats = Counter()
        self.last_price = None
        self.trade_volume = 0
        self.recent_trades = deque(maxlen=20)

    def reduce(self, oid, qty, reason, missing_ok=False):
        order = self.orders.get(oid)
        if order is None:
            self.stats['missing_active_trade' if missing_ok else 'missing_' + reason] += 1
            return
        side, price, remaining = order
        if qty > remaining:
            self.stats[reason + '_overdraw'] += 1
            raise ValueError(f'订单 {oid} 的 {reason} 数量超过剩余量')
        order[2] -= qty
        self.levels[side][price] -= qty
        if self.levels[side][price] == 0: del self.levels[side][price]
        if order[2] == 0: del self.orders[oid]

    def apply(self, typ, x):
        if typ == 'O':
            kind = x['OrderKind']
            if kind == 65:  # ASCII A: quantity is the residual AFTER immediate matching
                oid, side, price, qty = x['OrderOriNo'], x['FunctionCode'], cents(x['Price']), x['Volume']
                if oid in self.orders: raise ValueError('新增订单号重复')
                if side not in (66, 83) or price <= 0 or qty <= 0: raise ValueError('新增委托字段不合法')
                self.orders[oid] = [side, price, qty]
                self.levels[side][price] += qty
                self.stats['adds'] += 1
            elif kind == 68:
                self.reduce(x['OrderOriNo'], x['Volume'], 'cancel')
                self.stats['cancels'] += 1
            elif kind == 83:
                self.stats['status_events'] += 1
            else:
                raise ValueError('未知订单事件编码')
        else:
            self.stats['trades'] += 1
            self.last_price = cents(x['Price'])
            self.trade_volume += x['Volume']
            self.recent_trades.append(dict(time=clock_text(x['Time']), price=self.last_price / 100,
                                           quantity=x['Volume'], side=x['BSFlag'], seq=x['BizIndex']))
            for side, key in [(66, 'BidOrder'), (83, 'AskOrder')]:
                # An immediately executed aggressor normally has no A event yet.
                self.reduce(x[key], x['Volume'], 'trade', missing_ok=(x['BSFlag'] == side))

    def top(self, side, depth=10):
        return sorted(self.levels[side].items(), reverse=(side == 66))[:depth]


@dataclass
class VirtualOrder:
    id: int
    side: str
    price: Optional[int]
    quantity: int
    remaining: int
    created_at: int
    ahead: set = field(default_factory=set)
    status: str = 'OPEN'
    order_type: str = 'LIMIT'
    protection_price: Optional[int] = None
    converted_to_limit: bool = False
    reason: str = ''


class Simulator:
    """One outstanding shadow order per stock; historical events stay immutable.

    Passive fills require a historical trade against a later order at the same
    price, or a trade through our limit. Cancellations only advance the queue.
    This is a zero-market-impact assumption, not an exchange counterfactual.
    """
    def __init__(self, symbol=601688, cash=1000000, initial_shares=0, data_dir=DEFAULT_DATA):
        if isinstance(initial_shares, bool) or not isinstance(initial_shares, int) or initial_shares < 0:
            raise ValueError('期初持仓须为非负整数')
        self.symbol = symbol
        self.events, self.snapshots, self.preclose, self.excluded_trades = load_stock(symbol, data_dir)
        self.book = HistoricalBook()
        self.cursor = 0
        self.time = 91500000
        self.cash = cents(cash)
        if self.cash < 0: raise ValueError('初始资金不能为负')
        self.settled = initial_shares
        self.bought_today = 0
        self.orders = []
        self.fills = []
        self.used_liquidity = defaultdict(int)

    @property
    def active(self):
        return next((o for o in self.orders if o.status == 'OPEN'), None)

    def _fill(self, order, price, qty, model, seq=None):
        if not 0 < qty <= order.remaining: raise ValueError('成交数量异常')
        if order.protection_price is not None and ((order.side == 'B' and price > order.protection_price) or
                                                   (order.side == 'S' and price < order.protection_price)):
            raise ValueError('成交价格越过保护限价')
        amount = price * qty
        if order.side == 'B':
            if amount > self.cash: raise ValueError('资金不足')
            self.cash -= amount
            self.bought_today += qty
        else:
            if qty > self.settled: raise ValueError('可卖持仓不足')
            self.cash += amount
            self.settled -= qty
        order.remaining -= qty
        if not order.remaining: order.status = 'FILLED'
        self.fills.append(dict(order_id=order.id, time=clock_text(self.time), price=price / 100,
                               quantity=qty, side=order.side, model=model, historical_seq=seq,
                               amount=amount / 100, fees=0))

    def advance(self, target):
        stamp_to_ms(target)
        if target < self.time: raise ValueError('回放不能倒退，请重置')
        if target > 150000000: raise ValueError('原型只回放到15:00；盘后数据另行处理')
        while self.cursor < len(self.events) and self.events[self.cursor][2]['Time'] <= target:
            seq, typ, x = self.events[self.cursor]
            self.time = x['Time']
            pending = self.active
            if pending and not continuous(self.time) and self.time >= 145700000:
                # Never pretend to match a continuous-only virtual order in the closing auction.
                pending.status = 'EXPIRED_AT_AUCTION_BOUNDARY'
                pending = None
            if pending and typ == 'T' and continuous(self.time):
                passive_side = 'S' if x['BSFlag'] == 66 else 'B' if x['BSFlag'] == 83 else None
                if passive_side == pending.side:
                    p = cents(x['Price'])
                    passive_id = x['AskOrder'] if passive_side == 'S' else x['BidOrder']
                    through = p > pending.price if pending.side == 'S' else p < pending.price
                    ahead_live = any(k in self.book.orders for k in pending.ahead)
                    same_and_behind = p == pending.price and passive_id not in pending.ahead and not ahead_live
                    if through or same_and_behind:
                        qty = min(pending.remaining, x['Volume'])
                        self._fill(pending, pending.price, qty, 'zero_impact_fifo', seq)
            if pending and pending.status == 'OPEN' and typ == 'O' and x['OrderKind'] == 65 and continuous(self.time):
                # A newly resting opposite order can cross a shadow quote that
                # did not exist in the historical market. Its limit is observed
                # only NOW, after any immediate historical executions.
                p = cents(x['Price'])
                opposite = x['FunctionCode'] == (83 if pending.side == 'B' else 66)
                crosses = p <= pending.price if pending.side == 'B' else p >= pending.price
                if opposite and crosses and not any(k in self.book.orders for k in pending.ahead):
                    qty = min(pending.remaining, x['Volume'])
                    self._fill(pending, pending.price, qty, 'zero_impact_fifo', seq)
                    self.used_liquidity[x['OrderOriNo']] += qty
            self.book.apply(typ, x)
            self.cursor += 1
            # Fully depleted original ids cannot be reused in this day's auction stream.
            if typ == 'T':
                for oid in (x['BidOrder'], x['AskOrder']):
                    if oid not in self.book.orders: self.used_liquidity.pop(oid, None)
            elif x['OrderKind'] == 68 and x['OrderOriNo'] not in self.book.orders:
                self.used_liquidity.pop(x['OrderOriNo'], None)
        self.time = target
        if target >= 145700000 and self.active:
            self.active.status = 'EXPIRED_AT_AUCTION_BOUNDARY'
        return self.state()

    def _validate(self, side, price, qty, order_type='LIMIT'):
        if not continuous(self.time): raise ValueError('此版本只接受连续竞价阶段的模拟委托')
        if self.active: raise ValueError('此版本每只股票同时只允许一笔未完成委托，请先撤单')
        if side not in ('B', 'S'): raise ValueError('方向须为 B 或 S')
        if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0: raise ValueError('数量须为正整数')
        star = str(self.symbol).startswith('688')
        maximum = (100000 if order_type == 'LIMIT' else 50000) if star else 1000000
        if qty > maximum: raise ValueError(f'超过单笔申报数量上限：{maximum}股')
        if star:
            if qty < 200 and not (side == 'S' and qty == self.settled): raise ValueError('科创板委托最少200股；不足200股的可卖余额须一次卖出')
        elif side == 'B' and qty % 100:
            raise ValueError('主板买入数量须为100股的整数倍')
        elif side == 'S' and qty % 100 and qty != self.settled:
            raise ValueError('此版本零股卖出须一次卖出全部可卖余额')
        ratio = Decimal('0.20') if star else Decimal('0.10')
        lower = int((Decimal(self.preclose) * (1 - ratio)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        upper = int((Decimal(self.preclose) * (1 + ratio)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        if not lower <= price <= upper: raise ValueError('价格超出普通已上市股票的当日涨跌幅范围')
        same_side = 66 if side == 'B' else 83
        opp_side = 83 if side == 'B' else 66
        opposite = self.book.top(opp_side, 1)
        same = self.book.top(same_side, 1)
        base = opposite[0][0] if opposite else same[0][0] if same else self.book.last_price or self.preclose
        factor = Decimal('1.02') if side == 'B' else Decimal('0.98')
        bound = int((Decimal(base) * factor).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        if not star: bound = max(bound, base + 10) if side == 'B' else min(bound, base - 10)
        # The 2% price cage applies to LIMIT instructions, not the market
        # instruction's protection price. All actual fills still obey protection.
        if order_type == 'LIMIT' and ((side == 'B' and price > bound) or (side == 'S' and price < bound)):
            raise ValueError('价格超出连续竞价有效申报价格范围')
        if side == 'B' and price * qty > self.cash: raise ValueError('可用资金不足（按委托限价冻结）')
        if side == 'S' and qty > self.settled: raise ValueError('可卖持仓不足；当日买入股票不能当日卖出（T+1）')

    @staticmethod
    def _price(value):
        if value is None or value == '': raise ValueError('请填写价格；市价单必须填写保护限价')
        try:
            parsed = Decimal(str(value))
        except InvalidOperation:
            raise ValueError('价格必须为正且精确到0.01元') from None
        if not parsed.is_finite() or parsed <= 0 or parsed * 100 != (parsed * 100).to_integral_value():
            raise ValueError('价格必须为正且精确到0.01元')
        return cents(value)

    def _queue(self, order):
        s = 66 if order.side == 'B' else 83
        order.ahead = {oid for oid, r in self.book.orders.items() if r[0] == s and r[1] == order.price}

    def _sweep(self, order, cap, depth=None):
        side = order.side
        opp = 83 if side == 'B' else 66
        levels = None if depth is None else {p for p, _ in self.book.top(opp, depth)}
        candidates = [(oid, r) for oid, r in self.book.orders.items() if r[0] == opp and
                      (r[1] <= cap if side == 'B' else r[1] >= cap) and
                      (levels is None or r[1] in levels)]
        candidates.sort(key=lambda it: (it[1][1] if side == 'B' else -it[1][1], it[0]))
        last_fill_price = None
        for oid, r in candidates:
            available = max(0, r[2] - self.used_liquidity[oid])
            qty = min(available, order.remaining)
            if qty:
                self._fill(order, r[1], qty, 'zero_impact_book_sweep')
                self.used_liquidity[oid] += qty
                last_fill_price = r[1]
            if not order.remaining: break
        return last_fill_price

    def submit(self, side, price, quantity, order_type='LIMIT', protection_price=None):
        if order_type != 'LIMIT':
            return self.submit_market(side, quantity, order_type, protection_price)
        p = self._price(price)
        self._validate(side, p, quantity)
        order = VirtualOrder(len(self.orders) + 1, side, p, quantity, quantity, self.time)
        self.orders.append(order)
        self._sweep(order, p)
        if order.remaining: self._queue(order)
        return self.state()

    def submit_market(self, side, quantity, order_type, protection_price):
        if order_type not in MARKET_TYPES: raise ValueError('未知市价指令类型')
        protection = self._price(protection_price)
        self._validate(side, protection, quantity, order_type)
        order = VirtualOrder(len(self.orders) + 1, side, None, quantity, quantity, self.time,
                             order_type=order_type, protection_price=protection)
        self.orders.append(order)
        own_side, opp_side = (66, 83) if side == 'B' else (83, 66)
        own = self.book.top(own_side, 1)
        opposite = self.book.top(opp_side, 1)

        def protected(p):
            return p <= protection if side == 'B' else p >= protection

        def finish_cancel(reason):
            order.status = 'CANCELLED'
            order.reason = reason

        if order_type in ('SAME_BEST', 'OPPOSITE_BEST'):
            reference = own if order_type == 'SAME_BEST' else opposite
            if not reference:
                finish_cancel('本方无报价，自动撤销' if order_type == 'SAME_BEST' else '对手方无报价，自动撤销')
            elif not protected(reference[0][0]):
                # Conservative handling when the reference quote is outside
                # protection: never invent a clipped reference price or fill.
                finish_cancel('最优报价超出保护限价，原型自动撤销')
            else:
                order.price = reference[0][0]
                self._sweep(order, order.price)
                if order.remaining:
                    self._queue(order)
                    order.reason = '以进入时的最优报价定价，剩余量排队'
        else:
            last_fill = self._sweep(order, protection, depth=5)
            if order.remaining:
                if order_type == 'FIVE_CANCEL':
                    finish_cancel('五档范围及保护限价内撮合结束，剩余量自动撤销')
                else:
                    limit = last_fill if last_fill is not None else own[0][0] if own else None
                    if limit is None:
                        finish_cancel('无成交且本方无报价，自动撤销')
                    elif not protected(limit):
                        finish_cancel('转限价价格超出保护限价，原型自动撤销')
                    else:
                        order.price = limit
                        order.converted_to_limit = True
                        order.reason = '剩余量按本单最新成交价转限价' if last_fill is not None else '无成交，按本方最优价转限价'
                        self._queue(order)
        return self.state()

    def cancel(self):
        if not continuous(self.time): raise ValueError('此版本仅在连续竞价阶段接受模拟撤单')
        if not self.active: raise ValueError('没有可撤销的模拟委托')
        self.active.status = 'CANCELLED'
        return self.state()

    def state(self):
        active = self.active
        frozen_cash = active.price * active.remaining if active and active.side == 'B' else 0
        frozen_shares = active.remaining if active and active.side == 'S' else 0
        last = self.book.last_price or self.preclose
        orders = []
        for o in self.orders:
            d = asdict(o)
            d.pop('ahead')
            d['price'] = o.price / 100 if o.price is not None else None
            d['protection_price'] = o.protection_price / 100 if o.protection_price is not None else None
            d['order_type_name'] = MARKET_TYPES.get(o.order_type, '限价')
            d['created_at'] = clock_text(o.created_at)
            d['queue_ahead_quantity'] = sum(self.book.orders[k][2] for k in o.ahead if k in self.book.orders)
            orders.append(d)
        return dict(symbol=self.symbol, date=DATE, time=clock_text(self.time), time_stamp=self.time,
                    phase=phase(self.time), events_processed=self.cursor, events_total=len(self.events),
                    preclose=self.preclose / 100, last_price=last / 100,
                    bids=[dict(price=p / 100, quantity=q) for p, q in self.book.top(66)],
                    asks=[dict(price=p / 100, quantity=q) for p, q in self.book.top(83)],
                    cash=self.cash / 100, available_cash=(self.cash - frozen_cash) / 100,
                    frozen_cash=frozen_cash / 100, settled_shares=self.settled,
                    sellable_shares=self.settled - frozen_shares, bought_today=self.bought_today,
                    equity=(self.cash + (self.settled + self.bought_today) * last) / 100,
                    orders=orders, fills=self.fills, audit=dict(self.book.stats),
                    market_trades=list(reversed(self.book.recent_trades)),
                    market_volume=self.book.trade_volume,
                    excluded_after_hours_trades=self.excluded_trades,
                    model='零市场冲击假设；连续竞价限价／市价委托；单笔在途；不含费用；零延迟')
