"""Bridge retained JS jobs and user events into the four-state stream contract."""
from __future__ import annotations

import json

from natlang.execution import CrispRequest
from natlang.streams import Poll, QueueSource


class JobEventSource:
    def __init__(self, engine, capacity: int = 32):
        self.engine = engine
        self.queue = QueueSource(capacity)
        self.jobs: dict[str, dict | None] = {}
        self.closing = False

    def put_user(self, event_id: str):
        self.queue.put({"kind": "user", "id": event_id})

    def watch(self, job_id: str):
        self.jobs[job_id] = None

    def close(self):
        self.closing = True

    def poll(self) -> Poll:
        queued = self.queue.poll()
        if queued.kind == "item":
            return queued
        for job_id in list(self.jobs):
            result = self.engine.run(CrispRequest(f"host.poll({json.dumps(job_id)})", {}, False,
                                                  "job-stream"), lambda *_: None)
            if result["status"] != "running":
                self.jobs[job_id] = result
                del self.jobs[job_id]
                return Poll("item", {"kind": "job_complete", "id": job_id})
        return Poll("closed") if self.closing and not self.jobs else Poll("empty")
