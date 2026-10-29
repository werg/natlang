"""Serialize mutation visibility behind durable replica state seals."""
from __future__ import annotations
import threading
import time
from pathlib import Path

class ReplicatedDispatch:
    def __init__(self, machine, executor, empty_payload):
        self.machine=machine;self.executor=executor;self.empty=Path(empty_payload)
        if self.empty.stat().st_size:raise ValueError('empty event payload is not empty')
        self.cond=threading.Condition();self.inflight=0

    def read(self, function):
        # The callback must capture a stable immutable result or open a file
        # descriptor before returning. Streaming can happen outside the queue.
        return self.executor.read(function)

    def replicate_file(self,name,path,sha256):
        from .replica_transport import transmit_file
        return transmit_file(self.executor.endpoint,self.executor.token,name,path,sha256,
            allow_http=self.executor.allow_http,gateway=self.executor.gateway,timeout=max(self.executor.timeout,3600))

    def call(self, kind, args, payload=None):
        with self.cond:
            self.inflight+=1
        source=Path(payload) if payload is not None else self.empty
        def event():return {'kind':kind,'args':args,'wall':time.time(),'monotonic':time.monotonic()}
        try:
            state=self.executor.submit(event,source,lambda e:self.machine.apply(e,source),seal=True,deferred=True)
            error=state['error']
            if error:
                if error['kind']=='conflict':
                    from .coordinator import Conflict
                    raise Conflict(error['message'],**error['detail'])
                raise ValueError(error['message'])
            return state['result']
        finally:
            with self.cond:
                self.inflight-=1;self.cond.notify_all()
