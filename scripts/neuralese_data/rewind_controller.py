#!/usr/bin/env python3
"""Roll a stopped interleaved RL controller back to an earlier committed policy checkpoint of its own lineage.

    python3 scripts/neuralese_data/rewind_controller.py CONTROL_OUT --to-policy SHA256 [--reason TEXT] [--reselect]
        [--memetic-every N] [--no-verify-sha] [--dry-run]

SHA256 names a checkpoint the controller committed: the result of an RL update (``completed_update_history``; the state
then is "round K+1 about to start", the interleave window that was due after it not yet run) or of an interleaved SFT
window (``interleave['windows']``; "round R about to collect"). The controller resumes from that checkpoint as if the later
updates, windows and collections had not happened; everything that lives in the checkpoint (optimizer, RNG, SFT exposure)
comes back with it.

What changes (``plan``):

- state: head checkpoint, round/trained_rounds, offset (and the memetic cursor), completed-update history and everything the
  update commit and the collection add per round (trained ids and token counts, accounted stages, attempted/collected ids,
  the task-curriculum history rebuilt by replaying the kept rounds' score files and checked against the live history, spread
  yield, latest reward, last learner progress/parameters, pre-collection history, dated transitions), the interleave windows
  and the memetic-distillation ledger (every package admitted or decided in a rolled-back window is pending again and re-admits at
  the next window; a queued rollout/memetic package whose sampling policy is not in the rewound lineage is refused,
  never admitted), and a recorded transition ``state['rewind_history']`` (from/to, moved directories, ledger effect);
- files: the rolled-back round/window directories, selections and pre-collections are renamed to ``<name>.rolled-back-<ts>``
  (nothing is deleted) and the old state stays as ``state.json.pre-rewind-<ts>``. Held-eval results, the distillation
  queue, packages and checkpoints are measurements/inputs and stay where they are.
- the selection of the round the controller resumes at is kept only when its pre-collection existed (it was sealed before
  the update committed); a selection sealed after the update (or after a window) is moved, so the window the cadence makes
  due runs and the round is selected again from the rewound history (``--reselect`` moves it in every case).

It refuses while the controller holds ``controller.lock``/``restart.lock`` or its unit is active, while a process still names
a directory it would move, for a SHA that is not a committed checkpoint of the lineage, and whenever the state cannot be
reconstructed exactly (non-contiguous history, a replayed curriculum history that disagrees with the live one, a memetic
cursor that cannot be restored). ``--dry-run`` prints the planned state diff and moves and writes nothing.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training/neuralese'))

SCHEMA = 'natlang.rl-rewind/1'
ROUND_DIR = re.compile(r'^round-(\d{6})(?:-attempt-(\d+))?$')
ROUND_PRE = re.compile(r'^round-(\d{6})-pre$')
ROUND_SELECTION = re.compile(r'^round-(\d{6})-selection\.json$')
WINDOW_DIR = re.compile(r'^sft-window-(\d{6})$')
STAGE_ROUND = re.compile(r'/round-(\d{6})(?:-attempt-\d+)?/learner')
DEFAULT_MEMETIC_EVERY = 4
# Dated transition lists: (state key, key holding the round the record belongs to). A record made at or after the round the
# controller resumes at is dropped; the controller records it again when it adopts the default at that round.
TRANSITIONS = (('held_eval_transitions', 'recorded_at_round'), ('rollout_distillation_transitions', 'recorded_at_round'),
               ('task_curriculum_transitions', 'round'), ('actor_engine_transitions', 'round'))
# State keys that exist only between an update launch and its commit, or in a stopped/failed state.
TRANSIENT_KEYS = ('active_update_round', 'update_mode', 'update_rollouts', 'update_stages', 'update_coefficient_policy',
                  'update_policy_loss', 'update_amendment_proof', 'update_code', 'pending_native_checkpoint',
                  'pending_native_round_dir', 'partial_checkpoint', 'reason')


class Refusal(Exception):
    pass


def _load(path):
    return json.loads(Path(path).read_text())


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def _round_dir(out, r, attempt=0):
    return Path(out) / (f'round-{r:06d}' + (f'-attempt-{attempt}' if attempt else ''))


def _selection_ids(out, r):
    path = Path(out) / f'round-{r:06d}-selection.json'
    return {case['id'] for case in _load(path).get('cases') or ()} if path.is_file() else None


def _is_memetic(r, every):
    return (r + 1) % every == 0


def memetic_every(out, override=None):
    """The memetic stage's ``memetic_every`` from the controller's recipe (default 4 when it declares none)."""
    if override is not None:
        return override
    try:
        recipe = _load(_load(Path(out) / 'config.json')['production_recipe'])
        for stage in recipe['stages']:
            if stage.get('kind') == 'native_memetic_rl':
                return int((stage.get('parameters') or {}).get('memetic_every', DEFAULT_MEMETIC_EVERY))
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return DEFAULT_MEMETIC_EVERY


# ---- locating the target ---------------------------------------------------------------------------------------------
def locate(state, sha):
    """The committed checkpoint ``sha``: {'kind', 'R' (round the controller resumes at), 'windows' (kept), 'head'}."""
    for record in state.get('completed_update_history') or []:
        head = record.get('checkpoint') or {}
        if head.get('sha256') == sha:
            windows = sum(1 for w in state['interleave']['windows'] if w['controller_round'] < record['round'] + 1)
            return {'kind': 'update', 'R': record['round'] + 1, 'windows': windows, 'head': head}
    for index, window in enumerate((state.get('interleave') or {}).get('windows') or []):
        if window['checkpoint']['sha256'] == sha:
            return {'kind': 'window', 'R': window['controller_round'], 'windows': index + 1, 'head': window['checkpoint']}
    raise Refusal(f'{sha} is not a committed checkpoint of this lineage (RL update results: '
                  + ', '.join(f"round {r['round']} {r['checkpoint']['sha256'][:12]}" for r in state.get('completed_update_history') or [])
                  + '; SFT window results: '
                  + ', '.join(f"window {w['window']} {w['checkpoint']['sha256'][:12]}" for w in state['interleave']['windows']) + ')')


def _head_sha(state):
    """SHA-256 of the controller's current head checkpoint, from the records that committed it (None if unrecorded)."""
    records = [*(state.get('completed_update_history') or []), *[w for w in state['interleave']['windows']]]
    for record in reversed(records):
        if (record.get('checkpoint') or {}).get('path') == state['checkpoint']:
            return record['checkpoint']['sha256']
    return None


# ---- per-round reconstruction ----------------------------------------------------------------------------------------
def replay_history(out, state, R):
    """The task-curriculum history after round R-1, rebuilt from the kept rounds' score files (the same fold the
    controller applies at each collection), refused unless it agrees with every live observation it did not overwrite."""
    from natlang_neuralese.rl.curriculum import record_visit
    yields = {item['round']: item for item in state.get('task_curriculum_spread_yield') or []}
    live = state.get('task_curriculum_history') or {}
    history = {task: entry for task, entry in live.items() if 'observed_round' not in entry}  # carried from earlier controllers
    for r in range(R):
        item = yields.get(r)
        if item is None:
            raise Refusal(f'no spread-yield record for round {r}: its curriculum observation cannot be replayed')
        score = _load(_round_dir(out, r, item.get('collection_attempt', 0)) / 'score.json')
        groups = [g for g in score.get('groups') or [] if g.get('scored', 0) > 0]
        counted = {g['task_id']: record_visit(history.get(g['task_id']), r) for g in groups}
        for g in groups:
            history[g['task_id']] = {**g, **counted[g['task_id']], 'observed_policy_id': item['policy_id'],
                                     'observed_round': r, 'collection_attempt': item.get('collection_attempt', 0)}
    wrong = sorted(task for task, entry in live.items()
                   if (entry.get('observed_round') is None or entry['observed_round'] < R) and history.get(task) != entry)
    wrong += sorted(task for task in history if task not in live)
    if wrong:
        raise Refusal(f'the replayed curriculum history disagrees with the live one on {len(wrong)} tasks, first {wrong[0]}')
    return history


def _round_task_ids(out, r, attempt=0):
    """Train task ids the round's update trained on (its learner's data receipt)."""
    path = _round_dir(out, r, attempt) / 'learner' / 'rl-data-receipt.json'
    if not path.is_file():
        return None
    return {d['task_id'] for d in _load(path).get('tasks_selected') or () if d.get('split') == 'train'}


def _learner_progress(report_path):
    from natlang_neuralese.train.native_operation_report import control_progress
    return control_progress(_load(report_path), Path(report_path))


# ---- the memetic-distillation ledger -----------------------------------------------------------------------------------
def _file_ref(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': _sha256(path)}


def rewind_ledger(ledger, windows_kept, *, queue, lineage, window_index):
    """(ledger after the rewind or None, summary): every entry decided in a window >= ``windows_kept`` is forgotten, so its
    packages are pending again and re-admit at the next window; a package whose earlier windows kept some of its rows
    returns as partial. Queued rollout/memetic packages whose sampling policy is not in ``lineage`` are refused now (recorded as
    the next window's decision), never admitted."""
    summary = {'rows_forgotten': 0, 'packages_pending_again': 0, 'admissions_dropped': 0, 'refused_off_lineage': []}
    if ledger is None:
        new = None
    else:
        new = copy.deepcopy(ledger)
        for row_id, row in list(new['rows'].items()):
            if row['window'] >= windows_kept:
                del new['rows'][row_id]
                summary['rows_forgotten'] += 1
        retained = {}
        for row in new['rows'].values():
            retained.setdefault(row['package'], []).append(row['window'])
        for sha, package in list(new['packages'].items()):
            if package.get('window', -1) >= windows_kept:
                summary['packages_pending_again'] += 1
                if sha in retained:
                    new['packages'][sha] = {**package, 'status': 'partial', 'window': max(retained[sha])}
                else:
                    del new['packages'][sha]
        before = len(new['admissions'])
        new['admissions'] = [a for a in new['admissions'] if a['window'] < windows_kept]
        summary['admissions_dropped'] = before - len(new['admissions'])
        for key in ('decisions',):
            if key in new:
                new[key] = [d for d in new[key] if d['window'] < windows_kept]
        if 'withdrawals' in new:
            new['withdrawals'] = {k: v for k, v in new['withdrawals'].items() if v.get('window', -1) < windows_kept}
        if new['admissions']:
            settled = Path(new['admissions'][-1]['recipe']['path']).parent / 'recipe-settled.json'
            if not settled.is_file():
                raise Refusal(f'the settled recipe of the last kept admission is missing: {settled}')
            new['recipe'] = _file_ref(settled)
        else:
            new['recipe'] = None
        if not (new['admissions'] or new['packages'] or new['rows'] or new.get('decisions') or new.get('withdrawals')):
            new = None
    seen = set((new or {}).get('packages') or {})
    for item in queue:
        package = Path(item['package'])
        if not package.is_file() or item.get('family') == 'prepared-source':
            continue
        sha = _sha256(package)
        if sha in seen:
            continue
        foreign = sorted(set(_load(package).get('episode_actors') or []) - lineage)
        if foreign:
            if new is None:
                new = {'schema': 'natlang.memetic-distillation-admission/1', 'recipe': None, 'packages': {}, 'rows': {},
                       'admissions': []}
            new['packages'][sha] = {'path': str(package), 'status': 'refused', 'window': window_index,
                                    'reason': 'rewind: episode actors are not checkpoints of this lineage: ' + ', '.join(foreign)}
            summary['refused_off_lineage'].append({'package': str(package), 'actors': foreign})
    return new, summary


# ---- the plan ----------------------------------------------------------------------------------------------------------
def plan(out, sha, *, reselect=False, every=None, now=None, reason='operator rewind'):
    out = Path(out)
    state = _load(out / 'state.json')
    interleave = state.get('interleave') or {}
    if state.get('schema') != 'natlang.rl-production-state/1' or not interleave:
        raise Refusal('state.json is not an interleaved RL production state')
    if state['round'] != state['trained_rounds']:
        raise Refusal(f"round {state['round']} != trained_rounds {state['trained_rounds']}: no-signal or amended rounds are not supported")
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(now))
    point = locate(state, sha)
    R, n = point['R'], point['windows']
    if R > state['round'] or n > len(interleave['windows']):
        raise Refusal('the target is not behind the controller\'s current position')
    history = [h for h in state['completed_update_history'] if h['round'] < R]
    if [h['round'] for h in history] != list(range(R)):
        raise Refusal('the completed-update history is not contiguous up to the target round')
    if point['kind'] == 'update' and history[-1]['checkpoint']['sha256'] != sha:
        raise Refusal('internal: the target update is not the last kept update')
    every = memetic_every(out, every)
    new = copy.deepcopy(state)
    notes = []
    last = history[-1]
    rolled_rounds = range(R, state['round'])
    kept_windows = interleave['windows'][:n]

    # head and commit bookkeeping
    new.update(checkpoint=point['head']['path'], round=R, trained_rounds=R, phase='collect', production_complete=False,
               collection_attempt=0, group_stats=[], actor_policy_pending_load=sha, completed_update_history=history,
               last_completed_update=copy.deepcopy(last))
    for key in TRANSIENT_KEYS:
        new.pop(key, None)
    for key in ('readiness', 'foundation_candidate', 'foundation_qualified'):
        new.pop(key, None)

    # trained counts / accounted stages
    def stage_round(key):
        match = STAGE_ROUND.search(key)
        return int(match.group(1)) if match else None
    new['trained_stage_counts'] = {k: v for k, v in state['trained_stage_counts'].items()
                                   if stage_round(k) is not None and stage_round(k) < R}
    if len(new['trained_stage_counts']) != len(state['trained_stage_counts']) - sum(
            1 for k in state['trained_stage_counts'] if stage_round(k) is None or stage_round(k) >= R):
        raise Refusal('internal: stage count bookkeeping')
    for key in ('supervised_tokens', 'target_tokens'):
        new['trained_' + key] = sum(v[key] for v in new['trained_stage_counts'].values())
    new['accounted_stages'] = [s for s in state['accounted_stages'] if stage_round(s) is not None and stage_round(s) < R]

    # task id sets: forget what only the rolled-back rounds contributed
    attempts = {item['round']: item.get('collection_attempt', 0) for item in state.get('task_curriculum_spread_yield') or []}
    rolled_trained = set()
    kept_trained = set()
    for r in range(state['trained_rounds']):
        ids = _round_task_ids(out, r, attempts.get(r, 0))
        if ids is None:
            raise Refusal(f'round {r} has no learner data receipt: its trained tasks cannot be reconstructed')
        (kept_trained if r < R else rolled_trained).update(ids)
    new['trained_task_ids'] = sorted(set(state['trained_task_ids']) - (rolled_trained - kept_trained))
    kept_sel, rolled_sel = set(), set()
    for r in range(state['round'] + 1):
        ids = _selection_ids(out, r)
        if ids is None:
            if r < state['round']:
                raise Refusal(f'round {r} has no sealed selection: its attempted tasks cannot be reconstructed')
            continue
        (kept_sel if r < R else rolled_sel).update(ids)
    only_rolled = rolled_sel - kept_sel
    new['attempted_task_ids'] = sorted(set(state['attempted_task_ids']) - only_rolled)
    new['collected_task_ids'] = sorted(set(state['collected_task_ids']) - only_rolled)
    if R < state['round']:
        # available_tasks/collected_task_fraction are the last collection's telemetry over the round's own task source
        if _is_memetic(R - 1, every):
            notes.append('available_tasks/collected_task_fraction kept (round %d was a memetic round over its own source)' % (R - 1))
        else:
            try:
                cases = {json.loads(line)['id'] for line in Path(_load(out / 'config.json')['cases']).read_text().splitlines() if line.strip()}
                new['available_tasks'] = len(cases)
                new['collected_task_fraction'] = len(set(new['collected_task_ids']) & cases) / len(cases)
            except (OSError, KeyError, ValueError):
                notes.append('available_tasks/collected_task_fraction kept (the task source could not be read)')

    # curriculum observations, reward and learner telemetry of the last kept round
    new['task_curriculum_history'] = replay_history(out, state, R)
    new['task_curriculum_spread_yield'] = [y for y in state['task_curriculum_spread_yield'] if y['round'] < R]
    kept_attempt = attempts.get(R - 1, 0)
    score = _load(_round_dir(out, R - 1, kept_attempt) / 'score.json')
    new['latest_reward'] = score
    if R == state['round'] and score != state['latest_reward']:
        notes.append('latest_reward differs from the last collection score file; set from the file')
    learner = _round_dir(out, R - 1, kept_attempt) / 'learner'
    new['last_learner_progress'] = _learner_progress(learner / 'report.json')
    parameters = _round_dir(out, R - 1, kept_attempt) / 'learner-parameters.json'
    if parameters.is_file():
        new['update_learner_parameters'] = _load(parameters)
    else:
        notes.append('update_learner_parameters kept (the round has no learner-parameters.json)')
    new['precollection_history'] = [p for p in state.get('precollection_history') or [] if p['round'] < R]
    for key, round_key in TRANSITIONS:
        if key in state:
            new[key] = [t for t in state[key] if not (type(t.get(round_key)) is int and t[round_key] >= R)]

    # offset and the memetic cursor
    memetic = copy.deepcopy(interleave.get('memetic'))
    if R < state['round']:
        selection = out / f'round-{R:06d}-selection.json'
        if selection.is_file() and (_load(selection).get('budget') or {}).get('offset') is not None:
            new['offset'] = _load(selection)['budget']['offset']
        elif not _is_memetic(R - 1, every):
            previous = _load(out / f'round-{R - 1:06d}-selection.json')
            size = sum(1 for line in Path(_load(out / 'config.json')['cases']).read_text().splitlines() if line.strip())
            new['offset'] = (previous['selection'][-1]['row'] + 1) % size
            notes.append('offset derived from the last row of round %d (no sealed selection for round %d)' % (R - 1, R))
        else:
            raise Refusal(f'cannot derive the offset: round {R - 1} was a memetic round and round {R} has no sealed selection')
        bound = [r for r in range(R, state['round'] + 1) if _is_memetic(r, every)]
        if bound and memetic is not None:
            if any(_is_memetic(r, every) for r in range(R)):
                raise Refusal('the rewind crosses a memetic round after an earlier one: its memetic cursor cannot be restored')
            memetic = {'pending': None}
    if memetic is not None and (memetic.get('pending') or {}).get('round', -1) >= R:
        memetic['pending'] = None

    # interleave
    new_interleave = new['interleave']
    new_interleave['windows'] = copy.deepcopy(kept_windows)
    new_interleave['sft_windows'] = n
    new_interleave['rl_rounds_at_last_window'] = kept_windows[-1]['controller_round'] if kept_windows else 0
    new_interleave.pop('active_window', None)
    if memetic is not None:
        new_interleave['memetic'] = memetic
    new_interleave['stage_bindings'] = [b for b in interleave.get('stage_bindings') or [] if b['round'] <= R]
    if kept_windows:
        window = kept_windows[-1]
        complete = window['checkpoint']['sha256'] == window['parent']['sha256'] or (
            window.get('completed_stage') is True and not window.get('new_optimizer_updates'))
    else:
        complete = False
    if complete:
        new_interleave['sft_complete'] = True
    else:
        new_interleave.pop('sft_complete', None)
    if 'memetic_distillation_failures' in new_interleave:
        new_interleave['memetic_distillation_failures'] = [
            f for f in new_interleave['memetic_distillation_failures'] if f['window'] < n]
        if not new_interleave['memetic_distillation_failures']:
            del new_interleave['memetic_distillation_failures']
    lineage = {h['checkpoint']['sha256'] for h in history} | {y['policy_id'] for y in new['task_curriculum_spread_yield']}
    for window in kept_windows:
        lineage |= {window['parent']['sha256'], window['checkpoint']['sha256']}
    lineage.add(sha)
    from natlang_neuralese.rl.memetic_admission import queue_path
    queue = [json.loads(line) for line in queue_path(out).read_text().splitlines() if line.strip()] if queue_path(out).is_file() else []
    ledger, ledger_summary = rewind_ledger(interleave.get('memetic_distillation'), n, queue=queue, lineage=lineage, window_index=n)
    if ledger is None:
        new_interleave.pop('memetic_distillation', None)
    else:
        new_interleave['memetic_distillation'] = ledger

    # files to move
    moves = []
    pre_existed = (out / f'round-{R:06d}-pre').exists()
    round_dir_existed = any(path.is_dir() or path.is_symlink() for path in out.glob(f'round-{R:06d}*') if ROUND_DIR.match(path.name))
    for path in sorted(out.iterdir()):
        name = path.name
        if '.rolled-back-' in name:
            continue
        if (m := ROUND_DIR.match(name)) and int(m.group(1)) >= R:
            moves.append(path)
        elif (m := ROUND_PRE.match(name)) and (int(m.group(1)) > R or (int(m.group(1)) == R and round_dir_existed)):
            moves.append(path)
        elif m := ROUND_SELECTION.match(name):
            r = int(m.group(1))
            if r > R or (r == R and (reselect or not pre_existed)):
                moves.append(path)
        elif (m := WINDOW_DIR.match(name)) and int(m.group(1)) >= n:
            moves.append(path)
    moved = [{'from': str(p), 'to': f'{p}.rolled-back-{stamp}'} for p in moves]
    record = {'schema': SCHEMA, 'transition': f'rewind-{stamp}', 'at': stamp, 'reason': reason, 'kind': point['kind'],
              'from': {'checkpoint': state['checkpoint'], 'checkpoint_sha256': _head_sha(state), 'round': state['round'],
                       'sft_windows': len(interleave['windows'])},
              'to': {'checkpoint': point['head']['path'], 'checkpoint_sha256': sha, 'round': R, 'sft_windows': n},
              'rolled_back_policies': sorted({h['checkpoint']['sha256'] for h in state['completed_update_history'] if h['round'] >= R}
                                             | {w['checkpoint']['sha256'] for w in interleave['windows'][n:]}),
              'moved': moved, 'state_backup': str(out / f'state.json.pre-rewind-{stamp}'), 'memetic_every': every,
              'memetic_distillation': ledger_summary, 'notes': notes}
    new['rewind_history'] = [*(state.get('rewind_history') or []), record]
    return {'state': state, 'new': new, 'moves': [(p, Path(m['to'])) for p, m in zip(moves, moved)], 'record': record,
            'point': point, 'stamp': stamp}


# ---- guards ------------------------------------------------------------------------------------------------------------
def running_problems(out, paths=()):
    """Reasons the controller must not be rewound now (empty: nothing found running)."""
    out = Path(out)
    problems = []
    for name in ('controller.lock', 'restart.lock'):
        path = out / name
        if path.is_file():
            with open(path, 'rb') as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    problems.append(f'{name} is held: the controller (or its restart) is running')
    launch = out / 'launch.json'
    if launch.is_file():
        for unit in [_load(launch).get('unit')]:
            if unit:
                try:
                    result = subprocess.run(['systemctl', '--user', 'is-active', unit], text=True, capture_output=True, timeout=20)
                    if result.stdout.strip() in ('active', 'activating', 'reloading'):
                        problems.append(f'unit {unit} is {result.stdout.strip()}')
                except (OSError, subprocess.SubprocessError):
                    pass
    patterns = [str(out / 'config.json'), *map(str, paths)]
    own = {os.getpid(), os.getppid()}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) in own:
            continue
        try:
            command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except OSError:
            continue
        hit = next((p for p in patterns if p in command), None)
        if hit:
            problems.append(f'process {entry.name} names {hit}')
    return problems


def verify_head(out, point, verify_sha):
    head = point['head']
    path = Path(head['path'])
    if not path.is_file():
        raise Refusal(f'the target checkpoint is not on disk: {path}')
    if head.get('bytes') is not None and path.stat().st_size != head['bytes']:
        raise Refusal(f'the target checkpoint size differs from its record: {path}')
    if verify_sha:
        sha = _sha256(path)
        if sha != head['sha256']:
            raise Refusal(f'the target checkpoint bytes hash to {sha}, not {head["sha256"]}')


# ---- reporting ---------------------------------------------------------------------------------------------------------
def _brief(value, limit=140):
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= limit else f'{text[:limit]}... <{len(text)} bytes>'


def describe(planned):
    old, new = planned['state'], planned['new']
    lines = [f"rewind to {planned['record']['to']['checkpoint_sha256'][:12]} ({planned['point']['kind']} result): round "
             f"{old['round']} -> {new['round']}, sft_windows {len(old['interleave']['windows'])} -> {new['interleave']['sft_windows']}"]
    for key in sorted(set(old) | set(new)):
        if key == 'interleave':
            continue
        if old.get(key) != new.get(key):
            lines.append(f'  state[{key}]: {_brief(old.get(key))}  ->  {_brief(new.get(key))}')
    for key in sorted(set(old['interleave']) | set(new['interleave'])):
        a, b = old['interleave'].get(key), new['interleave'].get(key)
        if a == b:
            continue
        if key == 'memetic_distillation':
            def size(ledger):
                return None if ledger is None else {k: len(ledger[k]) for k in ('packages', 'rows', 'admissions')}
            lines.append(f'  interleave[memetic_distillation]: {size(a)}  ->  {size(b)}')
            lines.append('    ' + _brief(planned['record']['memetic_distillation'], 400))
        elif key == 'windows':
            lines.append(f'  interleave[windows]: {[w["window"] for w in a]}  ->  {[w["window"] for w in b]}')
        else:
            lines.append(f'  interleave[{key}]: {_brief(a)}  ->  {_brief(b)}')
    lines.append('moves (renamed, never deleted):')
    lines += [f'  {src}  ->  {dst}' for src, dst in planned['moves']] or ['  (none)']
    lines.append(f"old state kept as {planned['record']['state_backup']}")
    lines += [f'note: {note}' for note in planned['record']['notes']]
    return '\n'.join(lines)


def apply(out, planned):
    from natlang_neuralese.common.jsonio import write_json_durable
    out = Path(out)
    backup = Path(planned['record']['state_backup'])
    if backup.exists() or any(dst.exists() or dst.is_symlink() for _, dst in planned['moves']):
        raise Refusal('a rewind with this timestamp already left files; retry')
    shutil.copy2(out / 'state.json', backup)
    done = []
    try:
        for src, dst in planned['moves']:
            os.rename(src, dst)
            done.append((src, dst))
        write_json_durable(out / 'state.json', planned['new'])
    except BaseException:
        for src, dst in reversed(done):
            os.rename(dst, src)
        backup.unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('out', type=Path, help='the controller out directory (holds state.json)')
    parser.add_argument('--to-policy', required=True, metavar='SHA256', help='committed checkpoint of the lineage to resume from')
    parser.add_argument('--reason', default='operator rewind')
    parser.add_argument('--reselect', action='store_true', help='also move the sealed selection of the round the controller resumes at')
    parser.add_argument('--memetic-every', type=int, help='override the memetic stage cadence read from the recipe')
    parser.add_argument('--no-verify-sha', action='store_true', help='check the target checkpoint\'s size only, not its hash')
    parser.add_argument('--dry-run', action='store_true', help='print the planned state diff and moves; write nothing')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    try:
        planned = plan(out, args.to_policy, reselect=args.reselect, every=args.memetic_every, reason=args.reason)
        print(describe(planned))
        problems = running_problems(out, [src for src, _ in planned['moves']])
        if problems:
            if not args.dry_run:
                raise Refusal('; '.join(problems))
            print('WARNING (a real run would refuse): ' + '; '.join(problems))
        verify_head(out, planned['point'], verify_sha=not args.no_verify_sha and not args.dry_run)
        if args.dry_run:
            print('dry run: nothing written')
            return 0
        apply(out, planned)
        print(f"rewound; the next controller start resumes at round {planned['new']['round']} "
              f"(restart with restart_controller.py {out} --in-place)")
        return 0
    except Refusal as error:
        print(f'refused: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
