"""The managed colocated RL actor, relaunched when it dies mid-collection (2026-11-10).

The SGLang engine of the production actor crashed under a live round; the controller only watched the actor process while
waiting for it to attest, so every worker then failed its episode on a refused connection for an hour. The collection now
asks ``ready()`` before each entry of the managed endpoint (as it already did for remote actors): a dead actor is started
again with the same command (same store directory, same policy, same actor.pid.json record) and the slot waits until it
attests the round's policy again. In-flight episodes of the dead actor are lost (unscored); nothing else is.
"""
from __future__ import annotations

import json
import time
from threading import Lock


class ManagedActor:
    """The ``Popen`` surface the controller uses (poll, terminate, wait, pid) over a process that ``ready`` replaces.

    ``launch()`` starts the actor and returns its process; ``attests()`` says whether the running actor serves the round's
    policy. A launch that has not attested within ``ready_seconds`` is replaced like a dead one; launches are at least
    ``backoff_seconds`` apart so a crash loop does not spin on model loads."""

    def __init__(self, process, launch, attests, *, ready_seconds, backoff_seconds=60., clock=None, log=print):
        self.process, self.launch, self.attests = process, launch, attests
        self.ready_seconds, self.backoff_seconds, self.clock, self.log = ready_seconds, backoff_seconds, clock or (lambda: time.monotonic()), log
        self.launched_at = self.clock()
        self.relaunches = 0
        self.lock = Lock()
        self.serving_at = None

    pid = property(lambda self: self.process.pid)

    def poll(self):
        return self.process.poll()

    def terminate(self):
        return self.process.terminate()

    def wait(self, *args, **kwargs):
        return self.process.wait(*args, **kwargs)

    def ready(self):
        """True while the actor process lives and attests the policy; relaunches a dead (or never-attesting) one."""
        with self.lock:
            now = self.clock()
            returncode = self.process.poll()
            if returncode is None:
                if self.serving_at is not None and now - self.serving_at < 10:
                    return True
                if self.attests():
                    self.serving_at = now
                    return True
                if now - self.launched_at < self.ready_seconds:
                    return False   # still loading
                reason = f'did not attest within {self.ready_seconds:g}s'
                self.process.terminate()
                try:
                    self.process.wait(timeout=120)   # it holds the actor's port until it is gone
                except Exception:
                    pass
            else:
                reason = f'exited {returncode}'
            if now - self.launched_at < self.backoff_seconds:
                return False
            self.relaunches += 1
            self.log(json.dumps({'event': 'relaunching the managed RL actor', 'reason': reason,
                                 'previous_pid': self.process.pid, 'relaunch': self.relaunches}), flush=True)
            self.process, self.launched_at, self.serving_at = self.launch(), now, None
            return False
