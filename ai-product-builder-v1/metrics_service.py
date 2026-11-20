"""Bounded, low-cardinality request metrics; zero project labels or prompt content."""
import json
import shutil
import threading
import time
from pathlib import Path


class Metrics:
    def __init__(self):
        self.lock=threading.Lock()
        self.started=time.monotonic()
        self.rows={}
        self.inflight=0

    def enter(self):
        with self.lock:self.inflight+=1
        return time.monotonic()

    def leave(self,verb,route_class,status,started):
        elapsed=max(0.0,time.monotonic()-started)
        route_class=route_class if route_class in {'health','auth','projects','meetings','settings',
                         'workspace','releases','rag','website','static','other'} else 'other'
        status_group=str(int(status)//100)+'xx'
        key=(verb,route_class,status_group)
        with self.lock:
            row=self.rows.setdefault(key,{'count':0,'total_seconds':0.0,'over_1s':0,'over_5s':0})
            row['count']+=1
            row['total_seconds']+=elapsed
            row['over_1s']+=int(elapsed>1)
            row['over_5s']+=int(elapsed>5)
            self.inflight=max(0,self.inflight-1)

    def snapshot(self):
        with self.lock:
            return {'uptime_seconds':round(time.monotonic()-self.started,1),'inflight_requests':self.inflight,
                    'requests':[{'method':x[0],'route':x[1],'status_group':x[2],**v}
                                for x,v in sorted(self.rows.items())]}

    def prometheus(self):
        data=self.snapshot()
        lines=['# HELP builder_uptime_seconds Process uptime in seconds',
               '# TYPE builder_uptime_seconds gauge',f'builder_uptime_seconds {data["uptime_seconds"]}',
               '# TYPE builder_inflight_requests gauge',f'builder_inflight_requests {data["inflight_requests"]}',
               '# TYPE builder_http_requests_total counter',
               '# TYPE builder_http_request_duration_seconds_sum counter']
        for item in data['requests']:
            labels=f'method="{item["method"]}",route="{item["route"]}",status="{item["status_group"]}"'
            lines.append(f'builder_http_requests_total{{{labels}}} {item["count"]}')
            lines.append(f'builder_http_request_duration_seconds_sum{{{labels}}} {item["total_seconds"]:.6f}')
        return '\n'.join(lines)+'\n'


def capacity(store):
    disk=shutil.disk_usage(str(Path(store.filename).resolve().parent))
    with store.db() as db:
        active=db.execute("SELECT count(*) FROM messages WHERE status IN ('processing','queued')").fetchone()[0]
        running=db.execute("SELECT count(*) FROM workspace_runs WHERE status='running'").fetchone()[0]
    return {'disk_free_bytes':disk.free,'disk_total_bytes':disk.total,
            'queued_or_processing_agents':active,'running_workspace_checks':running}
