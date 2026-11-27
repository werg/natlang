"""Edit layers for the training-data viewer: the native participant applies each retained `edit` here (edits.apply) and
records the outcome in the application journal (recordApplication). Live training files are never modified.

Layers (all under <data>/ops/data-edits/ unless noted; every write is atomic, every apply idempotent by operation id):
  log.jsonl          every edit that was applied or refused, with its outcome (the audit trail)
  state.json         current state per item: flagged, excluded, annotations, corrections (folded from log.jsonl)
  exclusions.jsonl   every item currently excluded, any kind (one line per dataset + id)
  corrections.jsonl  proposed corrections (field, content, reason); consumed by task re-qualification, never auto-applied
  ops/pool-errata/ops-dashboard.jsonl
                     RL task exclusions in the pool-errata format ({id, family, reason, source}); grpo and teacher_gen read
                     every pool-errata/*.jsonl at start / resume, so an exclusion reaches the next run without a code change
Held-out data is view-only: an edit naming a held task id or split group is refused (the domain refuses held kinds earlier).
Consumers that do not read these layers yet: SFT row / teacher demo / episode exclusions (sft_sources build), corrections
(no re-qualification consumer). See plans/neuralese/TRAINING_DATA_DISCOVERABILITY_20261120.md.
"""
import json
import os
import time

from data_service import handler, DASH
from data_items import OPS, is_held

LAYER = OPS + '/data-edits'
ERRATA_FILE = OPS + '/pool-errata/ops-dashboard.jsonl'
TASK_KINDS = ('task',)


def _atomic(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp-write'          # no .jsonl suffix: a reader globbing *.jsonl never sees a partial file
    with open(tmp, 'w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _log():
    out = []
    try:
        for line in open(LAYER + '/log.jsonl'):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def _fold(log):
    state = {}
    for e in log:
        if e.get('status') != 'applied':
            continue
        t = e['target']
        s = state.setdefault(f"{t['dataset']}|{t['id']}", {'target': t, 'flagged': None, 'excluded': None, 'annotations': [], 'corrections': []})
        brief = {'op': e['op'], 'text': e.get('text'), 'author': e.get('author'), 'ts': e.get('ts')}
        a = e['action']
        if a == 'flag':
            s['flagged'] = brief
        elif a == 'unflag':
            s['flagged'] = None
        elif a == 'exclude':
            s['excluded'] = brief
        elif a == 'include':
            s['excluded'] = None
        elif a == 'annotate':
            s['annotations'].append(brief)
        elif a == 'correct':
            s['corrections'].append({**brief, 'field': e.get('field'), 'format': e.get('correctionFormat'), 'basedOn': e.get('basedOn')})
        if e.get('supersedes'):
            for lst in (s['annotations'], s['corrections']):
                lst[:] = [x for x in lst if x['op'] != e['supersedes']]
    return state


def _write_layers(log):
    state = _fold(log)
    _atomic(LAYER + '/state.json', json.dumps(state, ensure_ascii=False, indent=1))
    excl = [{'dataset': s['target']['dataset'], 'kind': s['target']['kind'], 'id': s['target']['id'], 'reason': s['excluded']['text'],
             'op': s['excluded']['op'], 'author': s['excluded']['author'], 'ts': s['excluded']['ts']} for s in state.values() if s['excluded']]
    _atomic(LAYER + '/exclusions.jsonl', ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in excl))
    corr = [{'dataset': s['target']['dataset'], 'kind': s['target']['kind'], 'id': s['target']['id'], **c} for s in state.values() for c in s['corrections']]
    _atomic(LAYER + '/corrections.jsonl', ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in corr))
    full = {e['op']: e for e in log}
    errata = [{'id': x['id'], 'family': (full.get(x['op']) or {}).get('target', {}).get('label') or None,
               'reason': x['reason'], 'source': f"ops-dashboard edit {x['op']}", 'type': 'excluded-by-review'}
              for x in excl if x['kind'] in TASK_KINDS]
    _atomic(ERRATA_FILE, ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in errata))
    return len(excl), len(errata), len(corr)


@handler('edits.apply')
def edits_apply(edit):
    """Apply one edit (the retained payload plus author and operationId). Returns {status, detail, files}."""
    op, t, action = edit['operationId'], edit['target'], edit['action']
    log = _log()
    prior = next((e for e in log if e['op'] == op), None)
    if prior:
        return {'status': prior['status'], 'detail': prior.get('detail', 'already applied'), 'files': prior.get('files', '')}
    status, detail, files = 'applied', '', []
    if t['kind'].startswith('held') or is_held(t['id']):
        status, detail = 'refused', 'Held-out data is view-only: this id or its split group is in the held-v3/v2 exclusion set.'
    elif action in ('exclude', 'include') and t['kind'] not in TASK_KINDS:
        detail = (f"Recorded in ops/data-edits/exclusions.jsonl. No consumer reads {t['kind']} exclusions yet "
                  '(sft_sources build has no exclusion input); see the discoverability findings.')
    elif action in ('exclude', 'include'):
        detail = ('Task exclusion written to ops/pool-errata/ops-dashboard.jsonl: GRPO and teacher-gen exclude it at their next start or resume. '
                  'SFT builds do not read pool errata.') if action == 'exclude' else 'Removed from ops/pool-errata/ops-dashboard.jsonl at the next start or resume.'
    elif action == 'correct':
        detail = 'Recorded in ops/data-edits/corrections.jsonl for re-qualification; the source file is unchanged.'
    else:
        detail = f'{action} recorded in ops/data-edits/state.json.'
    files = ['ops/data-edits/log.jsonl']
    if status == 'applied':
        files.append('ops/data-edits/state.json')
        if action in ('exclude', 'include'):
            files += ['ops/data-edits/exclusions.jsonl'] + (['ops/pool-errata/ops-dashboard.jsonl'] if t['kind'] in TASK_KINDS else [])
        if action == 'correct':
            files.append('ops/data-edits/corrections.jsonl')
    rec = {'op': op, 'ts': int(time.time()), 'author': edit.get('author'), 'status': status, 'detail': detail, 'files': ', '.join(files),
           **{k: edit.get(k) for k in ('target', 'action', 'text', 'correction', 'correctionFormat', 'field', 'basedOn', 'supersedes')}}
    if isinstance(rec.get('correction'), list):
        rec['correction'] = ''.join(rec['correction'])
    os.makedirs(LAYER, exist_ok=True)
    with open(LAYER + '/log.jsonl', 'a') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    log.append(rec)
    if status == 'applied':
        _write_layers(log)
    return {'status': status, 'detail': detail, 'files': ', '.join(files)}


@handler('edits.layers')
def edits_layers(_):
    """The current layer state (for the agent and for checks)."""
    log = _log()
    return {'log': len(log), 'state': _fold(log), 'errata_file': ERRATA_FILE}
