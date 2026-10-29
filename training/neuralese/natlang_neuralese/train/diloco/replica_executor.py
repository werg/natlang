"""Single ordered writer for replicated coordinator transactions.

The caller owns state-machine validation and deterministic replay; this class
must not be used to expose coordinator operations until those are integrated.
"""
from __future__ import annotations
import http.client
import queue
import os
import sys
import tempfile
import threading
import time
import traceback
from .replica_transport import transmit

# A slow or dropped acknowledgement is retried with the IDENTICAL (sequence, previous, event, payload): the peer's
# journal acknowledges an identical record at its current sequence idempotently and rejects any divergent one
# (HTTP 400, never retried). Gateway 502/503/504 are upstream connection failures, not peer verdicts.
RETRY_BACKOFF_SECONDS = (1, 3, 9)
SEAL_TIMEOUT_SECONDS = 300
LARGE_PAYLOAD_BYTES_PER_SECOND = 4 << 20


def _retryable(exc):
    if isinstance(exc, (TimeoutError, ConnectionError, http.client.HTTPException)):return True
    if isinstance(exc, OSError) and not isinstance(exc, FileNotFoundError):return True
    return isinstance(exc, ValueError) and str(exc) in {'replica did not durably acknowledge transaction (HTTP %d)'%code for code in (502,503,504)}


class ReplicatedExecutor:
    def __init__(self, journal, endpoint, token, *, allow_http=False, timeout=30, gateway=None,
                 seal_timeout=SEAL_TIMEOUT_SECONDS, retry_backoff=RETRY_BACKOFF_SECONDS):
        self.journal=journal;self.endpoint=endpoint;self.token=token;self.allow_http=allow_http;self.timeout=timeout;self.gateway=gateway
        self.seal_timeout=seal_timeout;self.retry_backoff=tuple(retry_backoff);self.retries=[]
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
                remote=self._transmit(sequence,previous,event,item['payload'],seal=False)
                if remote!=local:raise ValueError('replica acknowledgement chain differs from local durable record')
                result=item['apply'](event) if item['deferred'] else item['apply']()
                if item['seal']:
                    fd,name=tempfile.mkstemp(prefix='.seal-',dir=self.journal.root)
                    os.close(fd)
                    try:
                        seal={'kind':'seal','operation_sequence':sequence,'state':result}
                        next_sequence=self.journal.sequence+1;next_previous=self.journal.head
                        local=self.journal.append(seal,name,expected_sequence=next_sequence,expected_previous=next_previous)
                        remote=self._transmit(next_sequence,next_previous,seal,name,seal=True)
                        if remote!=local:raise ValueError('replica post-state acknowledgement differs')
                    finally:os.unlink(name)
                item['result']=result
            except BaseException as exc:
                if 'read' not in item and not self.failure:
                    print('replicated writer fenced at sequence %d: %s' % (self.journal.sequence, ''.join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__))), file=sys.stderr, flush=True)
                if 'read' not in item:self.failure=exc
                item['error']=exc
            finally:item['done'].set()

    def _transmit(self, sequence, previous, event, payload, *, seal):
        """Send one already-durable local record; retry only an unknown outcome, never a peer verdict."""
        size=os.path.getsize(payload)
        timeout=max(self.timeout, self.seal_timeout if seal else 0, self.timeout+size/LARGE_PAYLOAD_BYTES_PER_SECOND)
        for attempt in range(len(self.retry_backoff)+1):
            try:
                return transmit(self.endpoint,self.token,sequence=sequence,previous=previous,event=event,payload=payload,
                                allow_http=self.allow_http,timeout=timeout,gateway=self.gateway)
            except Exception as exc:
                if not _retryable(exc) or attempt==len(self.retry_backoff):raise
                self.retries.append({'sequence':sequence,'attempt':attempt+1,'error':type(exc).__name__+': '+str(exc)[:200]})
                print('replica transmit retry %d for sequence %d after %s: %s' % (attempt+1, sequence, type(exc).__name__, exc),
                      file=sys.stderr, flush=True)
                time.sleep(self.retry_backoff[attempt])

    def close(self):
        self.queue.put(None);self.thread.join()
