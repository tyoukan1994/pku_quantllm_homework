"""Audit all stocks. Future alignment is diagnostic ONLY, never simulation input."""
import csv
import json
from pathlib import Path
from statistics import median
import pyarrow.parquet as pq
import pyarrow.compute as pc
from engine import HistoricalBook, load_stock, cents, stamp_to_ms, continuous, DEFAULT_DATA, DATE


def audit(symbol):
    events, snapshots, preclose, excluded = load_stock(symbol)
    # Deterministic sample: every 100th continuous snapshot, avoiding session endpoints.
    selected = [s for s in snapshots if continuous(s['Time']) and s['Time'] not in (93000000, 113000000, 130000000)][::100]
    b = HistoricalBook()
    i = 0
    strict = aligned = nearby = volume_strict = 0
    shifts = []
    nearby_shifts = []
    for s in selected:
        target = stamp_to_ms(s['Time'])
        expected = {side: [(cents(s[f'{prefix}Price{n}']), int(s[f'{prefix}Volume{n}']))
                           for n in range(1, 11) if s[f'{prefix}Price{n}'] > 0]
                    for side, prefix in [(66, 'Bid'), (83, 'Ask')]}
        def match():
            return b.trade_volume == s['AccVolume'] and all(b.top(side) == expected[side] for side in (66, 83))
        before_shift = None
        while i < len(events) and stamp_to_ms(events[i][2]['Time']) <= target:
            _, typ, x = events[i]; b.apply(typ, x); i += 1
            delta = stamp_to_ms(x['Time']) - target
            if delta >= -2000 and match(): before_shift = delta
        strict += int(match())
        volume_strict += int(b.trade_volume == s['AccVolume'])
        found = match()
        if found:
            shifts.append(0)
            before_shift = 0
        # Primary audit: search this labeled second. Extra diagnostic: also
        # record exact matches in the preceding two seconds. A snapshot's
        # timestamp is not certified as the book's exchange-generation time.
        # Neither search may be used to give a strategy future observations.
        after_shift = None
        while not found and i < len(events) and stamp_to_ms(events[i][2]['Time']) <= target + 999:
            _, typ, x = events[i]; b.apply(typ, x); i += 1
            if match():
                found = True
                after_shift = stamp_to_ms(x['Time']) - target
                shifts.append(after_shift)
        aligned += int(found)
        choices = [d for d in (before_shift, after_shift) if d is not None]
        if choices:
            nearby += 1
            nearby_shifts.append(min(choices, key=abs))
    for _, typ, x in events[i:]: b.apply(typ, x)
    final_snapshots = [s for s in snapshots if 150000000 <= s['Time'] < 150500000]
    final_volume = final_snapshots[-1]['AccVolume'] if final_snapshots else None
    anomalies = sum(b.stats[k] for k in ('missing_cancel', 'missing_trade', 'cancel_overdraw', 'trade_overdraw'))
    return dict(symbol=symbol, channel=events[0][2]['Channel'], events=len(events), adds=b.stats['adds'],
                cancels=b.stats['cancels'], trades=b.stats['trades'], status_events=b.stats['status_events'],
                missing_active_legs=b.stats['missing_active_trade'], missing_passive_or_auction_legs=b.stats['missing_trade'],
                missing_cancels=b.stats['missing_cancel'], anomalies=anomalies,
                replay_auction_volume=b.trade_volume, snapshot_auction_volume=final_volume,
                final_volume_match=b.trade_volume == final_volume, snapshots_sampled=len(selected),
                strict_top10_and_volume_matches=strict, strict_volume_matches=volume_strict,
                aligned_top10_and_volume_matches=aligned,
                nearby_top10_and_volume_matches=nearby,
                min_nearby_alignment_ms=min(nearby_shifts) if nearby_shifts else None,
                median_alignment_ms=median(shifts) if shifts else None,
                max_alignment_ms=max(shifts) if shifts else None,
                excluded_after_hours_trades=excluded)


if __name__ == '__main__':
    symbols = sorted(pc.unique(pq.read_table(DEFAULT_DATA / f'ord_{DATE}.parquet', columns=['Symbol'])['Symbol']).to_pylist())
    rows = []
    for s in symbols:
        row = audit(s); rows.append(row)
        print(f"{s}: anomalies={row['anomalies']}, aligned={row['aligned_top10_and_volume_matches']}/{row['snapshots_sampled']}, final_volume={row['final_volume_match']}", flush=True)
    dest = Path(__file__).parent
    with (dest / 'audit_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary = dict(symbols=len(rows), events=sum(r['events'] for r in rows), anomalies=sum(r['anomalies'] for r in rows),
                   final_volume_matches=sum(r['final_volume_match'] for r in rows),
                   snapshots_sampled=sum(r['snapshots_sampled'] for r in rows),
                   strict_top10_and_volume_matches=sum(r['strict_top10_and_volume_matches'] for r in rows),
                   aligned_top10_and_volume_matches=sum(r['aligned_top10_and_volume_matches'] for r in rows),
                   nearby_top10_and_volume_matches=sum(r['nearby_top10_and_volume_matches'] for r in rows),
                   excluded_after_hours_trades=sum(r['excluded_after_hours_trades'] for r in rows), stocks=rows)
    (dest / 'audit_results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'stocks'}, ensure_ascii=False))
