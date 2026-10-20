from datetime import datetime
import html
import json
from pathlib import Path

import numpy as np


def performance_metrics(curve, trades, orders, config):
    returns = curve.daily_return if config.seed_previous_close else curve.daily_return.iloc[1:]
    periods = len(returns)
    total = float(curve.net_value.iloc[-1] - 1)
    annual_return = float((1 + total) ** (config.annual_days / periods) - 1)
    volatility = float(returns.std(ddof=1) * np.sqrt(config.annual_days)) if periods > 1 else 0.0
    daily_risk_free = (1 + config.risk_free_rate) ** (1 / config.annual_days) - 1 if config.risk_free_rate > -1 else 0.0
    deviation = float(returns.std(ddof=1)) if periods > 1 else 0.0
    sharpe = float((returns.mean() - daily_risk_free) / deviation * np.sqrt(config.annual_days)) if deviation > 1e-12 else None
    drawdown = float(curve.drawdown.min())
    previous_equity = curve.equity.shift(1).fillna(config.initial_cash)
    turnover = float((curve.traded_notional / previous_equity).sum())
    result = {"total_return": total, "annual_return": annual_return,
              "annual_volatility": volatility, "sharpe": sharpe,
              "max_drawdown": drawdown,
              "calmar": annual_return / abs(drawdown) if drawdown < -1e-12 else None,
              "total_fees": float(trades.total_fee.sum()), "fills": len(trades),
              "rejected_orders": int((orders.status == "rejected").sum()),
              "partial_orders": int((orders.status == "partial").sum()),
              "two_way_turnover": turnover, "average_exposure": float(curve.exposure.mean()),
              "final_equity": float(curve.equity.iloc[-1]),
              "final_cash": float(curve.cash.iloc[-1]),
              "stale_valuation_days": int((curve.stale_positions > 0).sum()),
              "start": str(curve.index[0].date()), "end": str(curve.index[-1].date()),
              "observations": len(curve), "return_periods": periods}
    if "benchmark_net_value" in curve:
        benchmark_return = float(curve.benchmark_net_value.iloc[-1] - 1)
        result["benchmark_return"] = benchmark_return
        result["return_minus_benchmark"] = total - benchmark_return
    return result


def chart_svg(curve):
    series = [("策略净值", curve.net_value.to_numpy(), "#2563eb")]
    if "benchmark_net_value" in curve:
        series.append(("指定基准", curve.benchmark_net_value.to_numpy(), "#d97706"))
    minimum = min(1.0, *(float(values.min()) for _, values, _ in series))
    maximum = max(1.0, *(float(values.max()) for _, values, _ in series))
    spread = max(maximum - minimum, 0.05)
    minimum -= spread * 0.08
    maximum += spread * 0.08
    parts = ['<svg viewBox="0 0 1000 340" role="img" aria-label="回测净值曲线">']
    for level in np.linspace(minimum, maximum, 5):
        vertical = 290 - (level - minimum) / (maximum - minimum) * 260
        parts.append(f'<line x1="65" x2="975" y1="{vertical:.2f}" y2="{vertical:.2f}" stroke="#e2e8f0"/><text x="8" y="{vertical + 4:.2f}" fill="#64748b" font-size="13">{level:.3f}</text>')
    for name, values, color in series:
        points = " ".join(f"{65 + position / max(1, len(values) - 1) * 910:.2f},{290 - (value - minimum) / (maximum - minimum) * 260:.2f}"
                          for position, value in enumerate(values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    parts.append(f'<text x="65" y="325" fill="#64748b">{curve.index[0].date()}</text><text x="880" y="325" fill="#64748b">{curve.index[-1].date()}</text></svg>')
    return "".join(parts)


def report_html(result):
    metrics = result.metrics
    cards = [("累计收益", f'{metrics["total_return"]:.2%}'), ("年化收益", f'{metrics["annual_return"]:.2%}'),
             ("最大回撤", f'{metrics["max_drawdown"]:.2%}'),
             ("夏普比率", f'{metrics["sharpe"]:.3f}' if metrics["sharpe"] is not None else "—"),
             ("成交笔数", str(metrics["fills"])), ("费用合计", f'{metrics["total_fees"]:,.2f}')]
    card_html = "".join(f"<div class='card'><span>{label}</span><strong>{value}</strong></div>" for label, value in cards)
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in result.assumptions)
    final_date = str(result.equity.index[-1].date())
    final_positions = result.positions[result.positions.date == final_date]
    orders = result.orders[result.orders.status != "filled"].tail(50)
    mode = "复权研究模式（权益份额近似）" if result.config["accounting"] == "research" else "严格账户模式"
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>沪深日频回测报告</title>
<style>body{{font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;background:#f4f7fb;color:#17233a;max-width:1160px;margin:40px auto;padding:0 24px}}h1{{font-size:32px;margin-bottom:6px}}h2{{margin-top:32px}}.muted{{color:#64748b}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card,section{{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:20px}}.card span{{display:block;color:#64748b}}.card strong{{font-size:27px}}section{{margin-top:20px;overflow:auto}}svg{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border-bottom:1px solid #e2e8f0;padding:8px;text-align:right}}pre{{white-space:pre-wrap}}.badge{{display:inline-block;background:#e0e7ff;color:#3730a3;padding:5px 12px;border-radius:20px}}@media(max-width:650px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}</style>
<h1>沪深日频回测报告</h1><p class="muted">{html.escape(result.strategy_name)} · {metrics['start']} — {metrics['end']} · {metrics['observations']} 个观测日</p><p class="badge">{mode}</p>
<div class="cards">{card_html}</div><section><h2>净值曲线</h2>{chart_svg(result.equity)}<p class="muted">费用后净值；期末持仓按市值计入。年化采用 {result.config['annual_days']} 个交易日，夏普无风险年利率 {result.config['risk_free_rate']:.2%}。无基准时不计算超额收益。</p></section>
<section><h2>模型口径与数据覆盖</h2><ul>{notes}</ul></section><section><h2>期末持仓</h2>{final_positions.to_html(index=False, escape=True)}</section>
<section><h2>最近 50 笔未完全成交订单</h2>{orders.to_html(index=False, escape=True)}</section>
<section><h2>运行配置</h2><pre>{html.escape(json.dumps(result.config, ensure_ascii=False, indent=2))}</pre><h2>数据范围</h2><pre>{html.escape(json.dumps(result.data_summary, ensure_ascii=False, indent=2))}</pre></section></html>'''


def export_result(result, destination, strategy_source=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("equity", "trades", "orders", "positions", "events"):
        frame = getattr(result, name)
        frame.to_csv(destination / f"{name}.csv", index=name == "equity", encoding="utf-8-sig")
    payload = {"metrics": result.metrics, "config": result.config, "data": result.data_summary,
               "strategy": result.strategy_name, "assumptions": result.assumptions,
               "created_at": datetime.now().astimezone().isoformat()}
    (destination / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (destination / "report.html").write_text(report_html(result), encoding="utf-8")
    if strategy_source is not None:
        (destination / "strategy.py").write_text(strategy_source, encoding="utf-8")
    return destination
