"""Bounded opt-in HTTP load-test harness for non-destructive health endpoints.

Examples: python load_test.py --url http://127.0.0.1:8767 --requests 500 --workers 20
Never targets writable or paid-API routes. Do not run against third-party services.
"""
import argparse
import concurrent.futures
import json
import statistics
import time
from urllib.request import urlopen


def run(base_url,requests=100,workers=10,timeout=5):
    if not (base_url.startswith('http://127.0.0.1:') or base_url.startswith('http://localhost:')):
        raise ValueError('Only locally operated localhost servers are supported.')
    if not 1<=requests<=20000 or not 1<=workers<=100:raise ValueError('Unsafe load-test parameters.')
    url=base_url.rstrip('/')+'/api/health'
    def one(_):
        start=time.perf_counter()
        try:
            with urlopen(url,timeout=timeout) as response:
                code=response.status;response.read(1024)
        except Exception:code=0
        return (code,time.perf_counter()-start)
    t=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes=list(pool.map(one,range(requests)))
    elapsed=max(time.perf_counter()-t,1e-6)
    durations=sorted(d for _,d in outcomes)
    def percentile(p):return round(durations[min(len(durations)-1,int(p*(len(durations)-1)))],4)
    good=sum(code==200 for code,_ in outcomes)
    return {'requests':requests,'successes':good,'failures':requests-good,
            'error_rate':round((requests-good)/requests,4),
            'requests_per_second':round(requests/elapsed,2),'duration_seconds':round(elapsed,2),
            'p50_seconds':percentile(.50),'p95_seconds':percentile(.95),'p99_seconds':percentile(.99)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',default='http://127.0.0.1:8767')
    p.add_argument('--requests',type=int,default=100)
    p.add_argument('--workers',type=int,default=10)
    args=p.parse_args()
    result=run(args.url,args.requests,args.workers)
    print(json.dumps(result,indent=2))
    if result['failures']:raise SystemExit(1)

if __name__=='__main__':main()
