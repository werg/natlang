"""Deterministic replay of serialized coordinator operations.

Not a failover controller: callers must authenticate a baseline, contiguous
journal, post-state seals and an external primary fence before promotion.
"""
from __future__ import annotations
import hashlib
import json
import threading
import time as real_time
from pathlib import Path
from . import coordinator as module
from .protocol import Fragment

_CLOCK_LOCK=threading.Lock()

class _Clock:
    def __init__(self, wall, monotonic):self.wall=wall;self.mono=monotonic
    def time(self):return self.wall
    def monotonic(self):return self.mono
    def __getattr__(self, name):return getattr(real_time,name)


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        while block:=file.read(1<<20):h.update(block)
    return h.hexdigest()


class ReplayCoordinator:
    def __init__(self,root, *, quorum_since, clock_anchor=None):
        # Do not call Coordinator.__init__: it ticks pending quorums before the
        # acknowledged transaction log has been replayed.
        c=module.Coordinator.__new__(module.Coordinator);c.root=Path(root);c.manifest=module.read_manifest(c.root)
        c.fragments=tuple(Fragment.from_dict(x) for x in c.manifest['fragments'])
        c.progress=json.loads((c.root/'progress.json').read_text());c.lock=threading.RLock();c.progress_lock=threading.RLock()
        if not isinstance(quorum_since, dict) or any(type(k) is not int or not isinstance(v, (int, float)) for k, v in quorum_since.items()):
            raise ValueError('replay requires exact baseline quorum timers')
        c.published_versions=[];c.quorum_since=dict(quorum_since);c._sync_public();c._refresh_published();self.coordinator=c;self.last_clock=clock_anchor

    def apply(self,event,payload):
        if set(event)!={'kind','args','wall','monotonic'} or not isinstance(event['args'],dict):raise ValueError('invalid replay event')
        kind=event['kind'];args=dict(event['args']);c=self.coordinator
        if kind not in ('lease','heartbeat','identity_transition','completion','reopen','submit','tick','close','clock_rebase'):
            raise ValueError('unknown coordinator replay event')
        if not isinstance(event['wall'],(int,float)) or not isinstance(event['monotonic'],(int,float)):
            raise ValueError('invalid coordinator event clock')
        before=[c._version(f.index) for f in c.fragments]
        if kind=='submit':args['incoming']=Path(payload)
        with _CLOCK_LOCK:
            previous=module.time;module.time=_Clock(event['wall'],event['monotonic'])
            error=None
            try:
                try:
                    if kind=='clock_rebase':
                        if set(args)!={'old_wall','old_monotonic','old_timers'}:raise ValueError('invalid clock rebase custody')
                        if args['old_timers']!={str(k):v for k,v in sorted(c.quorum_since.items())}:raise ValueError('clock rebase timers differ')
                        old_wall=args['old_wall'];old_mono=args['old_monotonic']
                        if self.last_clock!={'wall':old_wall,'monotonic':old_mono}:raise ValueError('clock rebase does not follow sealed clock anchor')
                        if not all(isinstance(x,(int,float)) and __import__('math').isfinite(x) for x in (old_wall,old_mono,event['wall'],event['monotonic'])) or event['wall']<old_wall:
                            raise ValueError('clock rebase wall time is invalid or moved backwards')
                        elapsed=event['wall']-old_wall
                        if any(start>old_mono for start in c.quorum_since.values()):raise ValueError('clock rebase timer exceeds anchor')
                        c.quorum_since={key:event['monotonic']-(old_mono-start+elapsed) for key,start in c.quorum_since.items()}
                        result={'old_wall':old_wall,'old_monotonic':old_mono,'elapsed_wall':elapsed,'old_timers':args['old_timers'],
                                'new_wall':event['wall'],'new_monotonic':event['monotonic'],'new_timers':{str(k):v for k,v in sorted(c.quorum_since.items())}}
                    else:result=getattr(c,kind)(**args)
                except module.Conflict as exc:
                    result=None;error={'kind':'conflict','message':str(exc),'detail':exc.detail}
                except ValueError as exc:
                    result=None;error={'kind':'value','message':str(exc)}
            finally:module.time=previous
        self.last_clock={'wall':event['wall'],'monotonic':event['monotonic']}
        after=[c._version(f.index) for f in c.fragments]
        changed={str(i):_hash(c._path(i)) for i,(a,b) in enumerate(zip(before,after,strict=True)) if a!=b}
        indices = ([args['index']] if kind == 'submit' and type(args.get('index')) is int and 0 <= args['index'] < len(c.fragments)
                   else list(range(len(c.fragments))) if kind in ('tick','close','completion') else [])
        pending={str(i):{f.name:_hash(f) for f in sorted(c._pending_dir(i).glob('*.safetensors'))} for i in indices}
        timers={str(k):v for k,v in sorted(c.quorum_since.items())}
        return {'result':result,'error':error,'versions':after,'changed_fragments':changed,'pending':pending,
                'quorum_since':timers,'progress_sha256':_hash(c.root/'progress.json')}


def replay(journal, root, *, expected_sequence, expected_head, quorum_since, receiver_restart=False, clock_anchor=None):
    """Apply only complete sealed transactions from an authenticated baseline.

    The caller must establish baseline integrity and a supervisor fence before
    exposing the result. A valid shorter prefix is not a promotion proof.
    """
    journal.verify(expected_sequence=expected_sequence, expected_head=expected_head)
    if expected_sequence % 2 and not receiver_restart:
        raise ValueError('replica log ends with an unsealed coordinator operation')
    sealed_sequence=expected_sequence-(expected_sequence % 2)
    machine=ReplayCoordinator(root, quorum_since=quorum_since,clock_anchor=clock_anchor)
    for sequence in range(1,sealed_sequence+1,2):
        operation=journal.root/f'{sequence:020d}'
        seal=journal.root/f'{sequence+1:020d}'
        op=json.loads((operation/'record.json').read_text())
        proof=json.loads((seal/'record.json').read_text())['event']
        if set(proof)!={'kind','operation_sequence','state'} or proof['kind']!='seal' or proof['operation_sequence']!=sequence:
            raise ValueError('replica operation lacks its exact post-state seal')
        actual=machine.apply(op['event'],operation/'payload')
        if actual!=proof['state']:raise ValueError('replica replay differs from sealed coordinator state')
    return machine
