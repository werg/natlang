"""Authenticate an append to an interrupted policy operation without replaying earned rows."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..common.hashing import sha256_file_hex_cached, canonical_json_sha256_hex
from ..common.jsonio import write_json_durable


class _Stream:
    def __init__(self, path):
        self.file = Path(path).open(encoding='utf-8')
        self.buf = ''
        self.at = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def more(self):
        if self.at:
            self.buf = self.buf[self.at:]
            self.at = 0
        chunk = self.file.read(1 << 20)
        self.eof = not chunk
        self.buf += chunk

    def ws(self):
        while True:
            while self.at < len(self.buf) and self.buf[self.at].isspace(): self.at += 1
            if self.at < len(self.buf) or self.eof: return
            self.more()

    def char(self, value):
        self.ws()
        if self.at >= len(self.buf) or self.buf[self.at] != value:
            raise ValueError('malformed policy objective JSON')
        self.at += 1

    def peek(self):
        self.ws()
        return self.buf[self.at] if self.at < len(self.buf) else ''

    def value(self):
        self.ws()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buf, self.at)
                self.at = end
                return value
            except json.JSONDecodeError:
                if self.eof: raise ValueError('malformed policy objective JSON')
                self.more()


def _object(stream, item):
    stream.char('{')
    if stream.peek() == '}':
        stream.char('}'); return
    while True:
        key = stream.value()
        if not isinstance(key, str): raise ValueError('policy objective key is not a string')
        stream.char(':')
        item(key, stream)
        if stream.peek() == '}':
            stream.char('}'); return
        stream.char(',')


def verify_objectives(old_path, new):
    """Check every old policy row and all non-row controls, retaining no second full objective in memory."""
    stream = _Stream(old_path)
    old_keys = set(); controls = {}
    def field(key, reader):
        if key != 'rows':
            if key in controls: raise ValueError('duplicate old policy objective field')
            controls[key] = reader.value(); return
        if old_keys: raise ValueError('duplicate old policy rows')
        def row(ident, source):
            if ident in old_keys: raise ValueError('duplicate old policy row')
            old_keys.add(ident)
            if canonical_json_sha256_hex(new.get('rows', {}).get(ident)) != canonical_json_sha256_hex(source.value()):
                raise ValueError('policy objective continuation changed or removed an earned row: ' + ident)
        _object(reader, row)
    try:
        _object(stream, field)
        if stream.peek(): raise ValueError('trailing policy objective data')
    finally:
        stream.file.close()
    if not old_keys or controls != {key:value for key,value in new.items() if key != 'rows'}:
        raise ValueError('policy objective continuation changed its policy, mode or global controls')
    return old_keys


def arrange_and_seal(out, graphs, objective):
    """Before replacing a rolling source, retain its exact files and prove an append-only successor."""
    out=Path(out); old_graph=out/'rl-native-graphs.jsonl'; old_objective=out/'rl-native-objective.json'
    slot=out/'checkpoint.pt'
    if not slot.exists(): return None
    if not old_graph.is_file() or not old_objective.is_file():
        raise ValueError('rolling policy checkpoint lacks its exact previous source and objective')
    existing=out/'policy-source-continuation.json'
    recovered=None
    if existing.is_file():
        candidate=json.loads(existing.read_text())
        if (candidate.get('schema')=='natlang.policy-source-continuation/1'
                and candidate.get('checkpoint_sha256')==sha256_file_hex_cached(slot)
                and (candidate['predecessor']['graphs']['sha256']!=candidate['successor']['graphs']['sha256']
                     or candidate['predecessor']['objective']['sha256']!=candidate['successor']['objective']['sha256'])
                and sha256_file_hex_cached(old_graph) in {candidate['predecessor']['graphs']['sha256'],candidate['successor']['graphs']['sha256']}
                and sha256_file_hex_cached(old_objective) in {candidate['predecessor']['objective']['sha256'],candidate['successor']['objective']['sha256']}):
            for item in candidate['predecessor'].values():
                if sha256_file_hex_cached(item['path'])!=item['sha256']:
                    raise ValueError('policy predecessor archive changed during retry')
            recovered=candidate
    prior_graph = Path(recovered['predecessor']['graphs']['path']) if recovered else old_graph
    prior_objective = Path(recovered['predecessor']['objective']['path']) if recovered else old_objective
    old_ids=verify_objectives(prior_objective, objective)
    by_id={row['id']:row for row in graphs}
    if len(by_id)!=len(graphs): raise ValueError('policy successor repeats a graph identity')
    ordered=[]
    with prior_graph.open(encoding='utf-8') as source:
        for line in source:
            old=json.loads(line); ident=old['id']
            if (ident not in old_ids or ident not in by_id or
                    canonical_json_sha256_hex(by_id.pop(ident))!=canonical_json_sha256_hex(old)):
                raise ValueError('policy continuation changed or removed an earned graph: '+ident)
            ordered.append(old)
    if set(row['id'] for row in ordered)!=old_ids:
        raise ValueError('old policy graph and objective inventories differ')
    graphs[:]=ordered+[row for row in graphs if row['id'] in by_id]
    if recovered is not None:
        if sorted(by_id)!=recovered['added_ids']:
            raise ValueError('policy continuation cannot change its appended rows before its first checkpoint')
        return recovered
    prior={}
    for label,path in [('graphs',old_graph),('objective',old_objective)]:
        digest=sha256_file_hex_cached(path)
        archive=out/(f'policy-source-predecessor-{label}-{digest}'+path.suffix)
        if not archive.exists(): os.link(path,archive)
        if sha256_file_hex_cached(archive)!=digest: raise ValueError('policy predecessor archive differs')
        prior[label]={'path':str(archive.resolve()),'sha256':digest}
    return {'schema':'natlang.policy-source-continuation/1','predecessor':prior,
            'checkpoint_sha256':sha256_file_hex_cached(slot), 'old_rows':len(old_ids),
            'added_rows':len(by_id), 'added_ids':sorted(by_id)}


def finish_seal(out, record, graph_path, objective_path, *, final_graph=None, final_objective=None):
    if record is None: return None
    successor={'graphs':{'path':str((final_graph or graph_path).resolve()),'sha256':sha256_file_hex_cached(graph_path)},
               'objective':{'path':str((final_objective or objective_path).resolve()),'sha256':sha256_file_hex_cached(objective_path)}}
    if record.get('successor') is not None and record['successor']!=successor:
        raise ValueError('retried policy successor differs from its sealed artifacts')
    record['successor']=successor
    path=Path(out)/'policy-source-continuation.json'
    write_json_durable(path,record)
    return path


def verify_seal(path, *, previous, current, checkpoint, train_sha):
    """Bind the authenticated append proof to the checkpoint and the exact loaded successor."""
    record=json.loads(Path(path).read_text())
    if record.get('schema')!='natlang.policy-source-continuation/1': raise ValueError('invalid policy source continuation')
    old=record['predecessor']; new=record['successor']
    for item in (*old.values(), *new.values()):
        if sha256_file_hex_cached(Path(item['path']))!=item['sha256']:
            raise ValueError('policy source continuation artifact changed')
    if (record.get('checkpoint_sha256')!=sha256_file_hex_cached(checkpoint)
            or previous.get('sha256')!=old['objective']['sha256']
            or current.get('sha256')!=new['objective']['sha256']
            or train_sha!=new['graphs']['sha256']):
        raise ValueError('policy source continuation differs from checkpoint or objective identities')
    return record
