"""Local-only interactive demo. No broker APIs or external communication."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import argparse
import json
import pyarrow.parquet as pq
import pyarrow.compute as pc
from engine import Simulator, DEFAULT_DATA, DATE, stamp_to_ms, ms_to_stamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA, help='含 ord/exe/snp 三个 Parquet 文件的目录；也可设置 SSE_L2_DATA_DIR')
    args = parser.parse_args()
    args.data_dir = args.data_dir.expanduser()
    for kind in ('ord', 'exe', 'snp'):
        file = args.data_dir / f'{kind}_{DATE}.parquet'
        if not file.is_file():
            parser.error(f'行情文件不存在：{file}')
    symbols = sorted(pc.unique(pq.read_table(args.data_dir / f'ord_{DATE}.parquet', columns=['Symbol'])['Symbol']).to_pylist())
    sim = Simulator(data_dir=args.data_dir)
    sim.advance(93000000)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def send_json(self, data, code=200):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path == '/api/state': return self.send_json(dict(state=sim.state(), symbols=symbols))
            research_files = {
                '/research': ('report.html', 'text/html; charset=utf-8'),
                '/research/': ('report.html', 'text/html; charset=utf-8'),
                '/research/report.html': ('report.html', 'text/html; charset=utf-8'),
                '/research/overview.png': ('overview.png', 'image/png'),
                '/research/overview.pdf': ('overview.pdf', 'application/pdf'),
                '/research/factor_metrics.csv': ('factor_metrics.csv', 'text/csv; charset=utf-8'),
                '/research/factor_samples.parquet': ('factor_samples.parquet', 'application/octet-stream'),
            }
            if self.path in research_files:
                name, content_type = research_files[self.path]
                file = Path(__file__).with_name('research') / name
                if not file.is_file(): return self.send_error(404, 'Research report not generated')
                self.send_response(200); self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(file.stat().st_size))
                self.send_header('Cache-Control', 'no-store'); self.end_headers()
                with file.open('rb') as source:
                    while block := source.read(1024 * 1024): self.wfile.write(block)
                return
            if self.path in ('/', '/index.html'):
                body = Path(__file__).with_name('index.html').read_bytes()
                self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers(); self.wfile.write(body)
            else: self.send_error(404)

        def do_POST(self):
            nonlocal sim
            # Accept only browser requests from this local origin. No CORS.
            origin = self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{args.port}', f'http://localhost:{args.port}'):
                return self.send_json({'error': '仅接受本地页面操作'}, 403)
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 <= length <= 8192: raise ValueError('请求过大')
                p = json.loads(self.rfile.read(length) or b'{}')
                if self.path == '/api/reset':
                    symbol = int(p.get('symbol', 601688))
                    if symbol not in symbols: raise ValueError('代码不在数据中')
                    shares = p.get('initial_shares', 0)
                    if isinstance(shares, bool) or not isinstance(shares, int): raise ValueError('期初持仓须为整数')
                    replacement = Simulator(symbol, cash=p.get('cash', 1000000), initial_shares=shares, data_dir=args.data_dir)
                    replacement.advance(93000000)
                    sim = replacement
                elif self.path == '/api/advance':
                    target = p.get('target')
                    if target is None:
                        delta = p.get('milliseconds', p.get('seconds', 1) * 1000)
                        if isinstance(delta, bool) or not isinstance(delta, int) or not 0 < delta <= 3600000:
                            raise ValueError('单次推进1毫秒至3600秒')
                        target = min(150000000, ms_to_stamp(stamp_to_ms(sim.time) + delta))
                    sim.advance(int(target))
                elif self.path == '/api/order':
                    sim.submit(p['side'], p.get('price'), p['quantity'],
                               order_type=p.get('order_type', 'LIMIT'), protection_price=p.get('protection_price'))
                elif self.path == '/api/cancel': sim.cancel()
                else: return self.send_json({'error': '未知操作'}, 404)
                self.send_json(dict(state=sim.state(), symbols=symbols))
            except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
                self.send_json({'error': str(exc)}, 400)

    print(f'本地模拟交易：http://127.0.0.1:{args.port} （Ctrl+C停止）', flush=True)
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__': main()
