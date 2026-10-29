"""Lease-aware, crash-resumable retention of verified local replica generations."""
from __future__ import annotations
import contextlib
import fcntl
import json
import os
import re
import shutil
from pathlib import Path
from ...common.jsonio import write_json_durable


def _sync(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


class Retention:
    def __init__(self,root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'retention.json';self.lock=self.root/'retention.lock'
    @contextlib.contextmanager
    def _locked(self):
        with self.lock.open('a+b') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            yield
    def _read(self):
        if not self.path.exists():return {'schema':'natlang.diloco-replica-retention/1','generations':[],'leases':{},'retired':[]}
        state=json.loads(self.path.read_text())
        if state.get('schema')!='natlang.diloco-replica-retention/1':raise ValueError('unknown replica retention state')
        return state
    def register(self, *, generation, baseline_sha256, head, previous, replica_confirmation, active_pointer):
        if not re.fullmatch('[0-9a-f]{32}',generation):raise ValueError('invalid generation id')
        if not re.fullmatch('[0-9a-f]{64}',baseline_sha256) or not re.fullmatch('[0-9a-f]{64}',head):raise ValueError('invalid verified generation digest')
        if replica_confirmation!={'baseline_sha256':baseline_sha256,'new_head':head}:
            raise ValueError('physical replica generation confirmation differs')
        path=self.root/('generation-'+generation)
        if path.is_symlink() or not path.is_dir():raise ValueError('local verified generation is missing')
        from .replica_baseline import verify
        from .replica_journal import _identity
        verify(path/'baseline',baseline_sha256)
        if _identity(json.loads((path/'journal'/'identity.json').read_text()))!=head:
            raise ValueError('local generation journal identity differs')
        pointer=json.loads(Path(active_pointer).read_text())
        if (pointer.get('id')!=generation or pointer.get('baseline_sha256')!=baseline_sha256 or pointer.get('head')!=head):
            raise ValueError('local active generation pointer differs')
        with self._locked():
            state=self._read();entries=state['generations']
            existing=next((x for x in entries if x['id']==generation),None)
            if existing:
                if existing!={'id':generation,'baseline_sha256':baseline_sha256,'head':head,'previous':previous}:raise ValueError('generation registration differs')
                return
            if entries and previous!=entries[-1]['id']:raise ValueError('generation does not follow active generation')
            if not entries and previous is not None:raise ValueError('first generation has no registered predecessor')
            sealed=pointer.get('previous')
            if (not isinstance(sealed,dict) or set(sealed)!={'sequence','head'} or type(sealed['sequence']) is not int or
                    sealed['sequence']<0 or sealed['sequence']%2 or not re.fullmatch('[0-9a-f]{64}',str(sealed['head']))):
                raise ValueError('predecessor has no verified sealed head')
            if entries:entries[-1]['sealed']=sealed
            entries.append({'id':generation,'baseline_sha256':baseline_sha256,'head':head,'previous':previous})
            write_json_durable(self.path,state)
    def latest(self):
        with self._locked():
            entries=self._read()['generations']
            return entries[-1]['id'] if entries else None
    def acquire(self,generation,lease):
        if not re.fullmatch('[A-Za-z0-9_-]{1,128}',lease):raise ValueError('invalid generation lease')
        with self._locked():
            state=self._read()
            if generation not in {x['id'] for x in state['generations'] if not x.get('retiring')}:raise ValueError('cannot lease absent or retiring generation')
            if lease in state['leases']:raise ValueError('lease already exists')
            state['leases'][lease]=generation;write_json_durable(self.path,state)
    def release(self,lease,generation):
        with self._locked():
            state=self._read()
            if state['leases'].get(lease)!=generation:raise ValueError('generation lease differs')
            del state['leases'][lease];write_json_durable(self.path,state)
    @contextlib.contextmanager
    def leased_generation(self,generation,lease):
        """Pin a generation for the complete export or read operation."""
        self.acquire(generation,lease)
        try:yield self.root/('generation-'+generation)
        finally:self.release(lease,generation)
    def collect(self):
        """Durably mark before rename/delete; restart completes interrupted work."""
        with self._locked():
            state=self._read();entries=state['generations'];protected={x['id'] for x in entries[-2:]}|set(state['leases'].values())
            for entry in entries[:-2]:
                ident=entry['id']
                if ident in protected:continue
                source=self.root/('generation-'+ident);trash=self.root/('.retiring-'+ident)
                if source.is_symlink() or trash.is_symlink():raise ValueError('generation path is a symlink')
                if entry.get('retiring') is not True:
                    if not source.is_dir() or trash.exists():raise ValueError('generation disappeared before retirement')
                    entry['retiring']=True;write_json_durable(self.path,state)
                if source.exists():
                    if trash.exists():raise ValueError('ambiguous generation retirement')
                    os.replace(source,trash);_sync(self.root)
                if trash.exists():shutil.rmtree(trash);_sync(self.root)
                state['retired'].append({k:v for k,v in entry.items() if k!='retiring'})
                state['generations']=[x for x in state['generations'] if x['id']!=ident];write_json_durable(self.path,state)
            return {'remaining':[x['id'] for x in state['generations']],'retired':len(state['retired'])}
