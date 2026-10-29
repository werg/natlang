"""Single ordered writer for replicated coordinator transactions.

The caller owns state-machine validation and deterministic replay; this class
must not be used to expose coordinator operations until those are integrated.
"""
from __future__ import annotations
import queue
import os
import tempfile
import threading
from .replica_transport import transmit


class ReplicatedExecutor:
    def __init__(self, journal, endpoint, token, *, allow_http=False, timeout=30, gateway=None):
        self.journal=journal;self.endpoint=endpoint;self.token=token;self.allow_http=allow_http;self.timeout=timeout;self.gateway=gateway
        self.queue=queue.Queue(maxsize=256);self.failure=None;self.rollover_callback=None;self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()

    def submit(self, event, payload, apply, *, seal=False, deferred=False):
        done=threading.Event();item={'event':event,'payload':payload,'apply':apply,'done':done,'seal':seal,'deferred':deferred};self.queue.put(item);done.wait()
        if 'error' in item:raise RuntimeError('replicated operation failed closed') from item['error']
        return item['result']

    def read(self, function):
        done=threading.Event();item={'read':function,'done':done};self.queue.put(item);done.wait()
        if 'error' in item:raise RuntimeError('replicated read failed closed') from item['error']
        return item['result']

    def _run(self):
        while True:
            item=self.queue.get()
            if item is None:return
            try:
                if self.failure:raise RuntimeError('replicated writer is fenced after an earlier failure')
                if 'read' in item:
                    item['result']=item['read']()
                    continue
                def over_budget():
                    return item['seal'] and (self.journal.sequence+2 > self.journal.max_transactions or
                        self.journal.payload_bytes+__import__('pathlib').Path(item['payload']).stat().st_size > self.journal.max_payload_bytes)
                if over_budget() and self.rollover_callback is not None:self.rollover_callback()
                if over_budget():raise ValueError('replica generation must rotate before this operation')
                event=item['event']() if item['deferred'] else item['event']
                sequence=self.journal.sequence+1;previous=self.journal.head
                local=self.journal.append(event,item['payload'],expected_sequence=sequence,expected_previous=previous)
                remote=transmit(self.endpoint,self.token,sequence=sequence,previous=previous,event=event,payload=item['payload'],allow_http=self.allow_http,timeout=self.timeout,gateway=self.gateway)
                if remote!=local:raise ValueError('replica acknowledgement chain differs from local durable record')
                result=item['apply'](event) if item['deferred'] else item['apply']()
                if item['seal']:
                    fd,name=tempfile.mkstemp(prefix='.seal-',dir=self.journal.root)
                    os.close(fd)
                    try:
                        seal={'kind':'seal','operation_sequence':sequence,'state':result}
                        next_sequence=self.journal.sequence+1;next_previous=self.journal.head
                        local=self.journal.append(seal,name,expected_sequence=next_sequence,expected_previous=next_previous)
                        remote=transmit(self.endpoint,self.token,sequence=next_sequence,previous=next_previous,event=seal,payload=name,allow_http=self.allow_http,timeout=self.timeout,gateway=self.gateway)
                        if remote!=local:raise ValueError('replica post-state acknowledgement differs')
                    finally:os.unlink(name)
                item['result']=result
            except BaseException as exc:
                if 'read' not in item:self.failure=exc
                item['error']=exc
            finally:item['done'].set()

    def close(self):
        self.queue.put(None);self.thread.join()
