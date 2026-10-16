from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import math

import pandas as pd


def tick_round(price, rounding=ROUND_HALF_UP):
    return float(Decimal(str(price)).quantize(Decimal("0.01"), rounding=rounding))


def board(symbol):
    if symbol.startswith(("688", "689")) and symbol.endswith(".SH"):
        return "star"
    if symbol.startswith("30") and symbol.endswith(".SZ"):
        return "chinext"
    return "main"


def limit_rate(symbol, date, is_st=False):
    date = pd.Timestamp(date)
    market = board(symbol)
    if market == "star":
        return 0.20
    if market == "chinext":
        if date >= pd.Timestamp("2020-08-24"):
            return 0.20
        return 0.05 if is_st else 0.10
    if is_st and date < pd.Timestamp("2026-07-06"):
        return 0.05
    return 0.10


def order_quantity(symbol, desired, side, available=0):
    desired = max(0, int(math.floor(desired + 1e-8)))
    available = max(0, int(math.floor(available + 1e-8)))
    minimum = 200 if board(symbol) == "star" else 100
    if side == "sell" and desired >= available:
        return available
    if desired < minimum:
        return 0
    return desired if board(symbol) == "star" else desired // 100 * 100


def max_order_quantity(symbol):
    market = board(symbol)
    if market == "star":
        return 100_000
    if market == "chinext":
        return 300_000
    return 1_000_000


def trading_fees(date, side, gross, config):
    if gross <= 0:
        return {"commission": 0.0, "stamp_tax": 0.0, "transfer_fee": 0.0, "total": 0.0}
    date = pd.Timestamp(date)
    commission = tick_round(max(config.minimum_commission, gross * config.commission_rate))
    stamp_rate = 0.0005 if date >= pd.Timestamp("2023-08-28") else 0.001
    transfer_rate = 0.00001 if date >= pd.Timestamp("2022-04-29") else 0.00002
    stamp = tick_round(gross * stamp_rate) if side == "sell" else 0.0
    transfer = tick_round(gross * transfer_rate)
    return {"commission": commission, "stamp_tax": stamp, "transfer_fee": transfer,
            "total": round(commission + stamp + transfer, 2)}


def slipped_price(open_price, side, basis_points):
    direction = 1 if side == "buy" else -1
    rounding = ROUND_CEILING if side == "buy" else ROUND_FLOOR
    return tick_round(open_price * (1 + direction * basis_points / 10000), rounding)
