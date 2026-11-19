"""Project website: python serve.py, then open http://localhost:8000."""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from dbscan_project.brute_force import dbscan_brute_force
from dbscan_project.fast_queue import dbscan_spatial_queue


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / 'web'), **kwargs)

    def do_POST(self):
        if self.path != '/api/compare':
            self.send_error(404)
            return
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 1_000_000:
                raise ValueError('Invalid request size')
            data = json.loads(self.rfile.read(size))
            points, eps, minimum = data['points'], data['eps'], data['minPts']
            if not isinstance(points, list) or not 0 < len(points) <= 5000:
                raise ValueError('Use 1 to 5,000 points')
            if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
                raise ValueError('Radius must be positive and finite')
            if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
                raise ValueError('MinPts must be a positive integer')
            for p in points:
                if not isinstance(p, list) or len(p) != 2 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in p):
                    raise ValueError('Points must contain two finite coordinates')
            functions = [dbscan_brute_force, dbscan_spatial_queue]
            times, results = [[], []], [None, None]
            for trial in range(4):
                for index in ([0, 1] if trial % 2 == 0 else [1, 0]):
                    start = time.perf_counter()
                    labels, stats = functions[index](points, eps, minimum)
                    elapsed = (time.perf_counter() - start) * 1000
                    if trial:
                        times[index].append(elapsed)
                    results[index] = dict(labels=labels, checks=stats.distance_checks,
                                          clusters=len(set(labels) - {-1}))
            for index in range(2):
                results[index]['ms'] = statistics.median(times[index])
            self.reply(200, results)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            self.reply(400, {'error': str(error)})

    def reply(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    server = HTTPServer(('127.0.0.1', args.port), Handler)
    print(f'DBSCAN project website: http://localhost:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
