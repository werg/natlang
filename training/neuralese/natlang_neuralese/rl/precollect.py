"""Pop pre-collects the next round's students while the DGX learner updates (owner-approved 2026-11-09, fallow #25).

The learner holds the DGX (about 105 of 121 GiB unified memory) for hours while the Pop actor, already serving the
round's policy N, would idle. When round N's learner starts, the controller therefore seals round N+1's task selection
(the selector state after round N's scores is final then) and runs round N+1's student entries on the Pop slots only,
against the policy-N actor, as an ordinary ``collect`` of its own directory ``round-N+1-pre`` (own collection binding,
policy N, own block bundle). When the learner finishes the collection DRAINS: no new entry is taken, in-flight episodes
finish (never killed: a partial episode cannot continue under another policy), the policy-N blocks are bundled, and only
then does the next round replace the Pop actor. Round N+1 imports the finished episodes: their result directories and
journal rows move into the round's directory (so its dispatch never repeats them), their bundle becomes ``pre-blocks``
and every such rollout keeps ``policy.checkpoint_sha256`` = N. The learner then trains from checkpoint N+1 with a
``natlang.rl-actors/1`` declaration naming policy N and Pop's recorded export: the episodes are one update stale, joined
to the engine-mismatch guard and corrected by the loss's truncated importance weights (rl.native_replay); the exact
pre-SGD replay gate stays with same-policy native-engine episodes.

Staleness is bounded to ONE update: pre-collection starts only when no SFT window can run between round N's update and
round N+1's collection, and the import refuses (recording why, collecting the entries afresh) unless the round, trained
round count and window count are exactly one RL update past the recorded start.
"""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

from ..common.jsonio import read_jsonl, write_json_durable

START = 'pre-start.json'
RESULT = 'pre-collection-result.json'
RECEIPT = 'pre-collection.json'
RECEIPT_SCHEMA = 'natlang.rl-precollection/1'
IMPORT_JOURNAL = 'journal-9000.jsonl'   # generation 0 (legacy) journal name: Generations reads it like any other
IMPORT_QUEUE = 'queue-9000.jsonl'


def pre_directory(round_dir):
    return Path(str(Path(round_dir).resolve()) + '-pre')


def eligible(config, state, interleaved, policy_loss):
    """(ok, reason): may Pop pre-collect the next round while this round's update runs?"""
    if config.get('precollect') is False:
        return False, 'precollect: false'
    if not config.get('remote_endpoints') or not config.get('actor_command'):
        return False, 'no remote actor'
    if config.get('update_objective', 'policy') != 'policy' or config.get('memetic_round') is not None \
            or config.get('coefficient_policy'):
        return False, 'not a plain policy-update round'
    loss = policy_loss or {}
    outcome = loss.get('outcome', loss)
    if loss.get('schema') != 'clipped-ratio/1' and outcome.get('importance', 'none') not in ('tis', 'mis'):
        return False, 'the policy loss has no importance correction'
    if interleaved is not None:
        from .interleave import window_due
        after = {**state, 'round': state['round'] + 1}
        if window_due(interleaved, after):
            return False, 'an SFT window runs between this update and the next collection (staleness would exceed one update)'
    return True, 'eligible'


def seal_next_selection(config, state, cases, selection, out, select_tasks):
    """Seal round N+1's selection exactly as the next loop iteration would choose it (offset and selector history after
    this round's scores). Returns ``(round_number, selection, selected cases)`` or None when nothing is selectable."""
    from ..common.hashing import sha256_file_hex
    number = state['round'] + 1
    path = Path(out) / f'round-{number:06d}-selection.json'
    if path.exists():
        saved = json.loads(path.read_text())
        return (number, saved['selection'], saved['cases']) if saved.get('selection') else None
    budget = {'rows': config.get('rows_per_round', 32), 'group': config.get('group', 4),
              'offset': (selection[-1]['row'] + 1) % len(cases)}
    from .held_eval import exclusion
    held = exclusion(config, cases, number)   # the held-out evaluation tasks (rl.held_eval) are never pre-collected either
    chosen = select_tasks(cases, list(state.get('task_curriculum_history', {}).values()), budget,
                          command=config.get('select_tasks_command'), curriculum=config.get('task_curriculum'),
                          **({'exclude_ids': held} if held else {}))
    if not chosen:
        return None
    selected = [cases[item['row']] for item in chosen]
    write_json_durable(path, {'source_sha256': sha256_file_hex(Path(config['cases'])), 'selection': chosen,
                              'cases': selected, 'budget': budget,
                              **({'curriculum': config['task_curriculum']} if config.get('task_curriculum') is not None else {})})
    return number, chosen, selected


def _jsonable(kwargs):
    return json.loads(json.dumps({k: v for k, v in kwargs.items() if k not in ('control', 'drain')}, default=str))


def begin(*, config, state, round_kwargs, number, selection, selected, out, collect, control=None, log=print):
    """Start round ``number``'s student collection on the remote actor(s) against this round's policy; returns a
    ``Handle`` (None when this round's pre-collection already settled). ``round_kwargs``: this round's collect kwargs."""
    pre = pre_directory(Path(out) / f'round-{number:06d}')
    if (pre / RESULT).exists():
        return None
    remote = list(config['remote_endpoints'])
    spec = (config.get('remote_workers') or {})[remote[0]]
    pre.mkdir(parents=True, exist_ok=True)
    selected_path = pre / 'selected-cases.jsonl'
    if not selected_path.exists():
        selected_path.write_bytes(''.join(json.dumps(row) + '\n' for row in selected).encode('utf-8'))
    kwargs = {**{k: v for k, v in round_kwargs.items() if k != 'control'}, 'cases': selected_path, 'out': pre, 'endpoints': [remote[0]], 'rows': len(selection),
              'offset': 0, 'group': selection[0]['samples'], 'workers': spec['workers'], 'remote_workers': None,
              'late_endpoints': (), 'seed_base': 700000 + number * 1000000, 'teacher_phase': False,
              'lost_block_exclusions': None}
    if not (pre / START).exists():
        write_json_durable(pre / START, {
            'schema': 'natlang.rl-precollection-start/1', 'round': number, 'behavior_policy_id': round_kwargs['policy_id'],
            'trained_rounds': state['trained_rounds'], 'sft_windows': (state.get('interleave') or {}).get('sft_windows', 0),
            'update_round': state['round'], 'kwargs': _jsonable(kwargs)})
    return Handle(pre, kwargs, collect, control, log)


class Handle:
    """One background drainable pre-collection; ``finish`` drains it, waits for its in-flight episodes and never raises."""

    def __init__(self, directory, kwargs, collect, control, log):
        self.directory, self.drain, self.error, self.settle = Path(directory), threading.Event(), None, True
        self.thread = threading.Thread(target=self._run, args=(kwargs, collect, control, log), name='precollect', daemon=True)
        self.thread.start()

    def _run(self, kwargs, collect, control, log):
        try:
            collect(**kwargs, control=control, drain=self.drain)
            if self.settle:
                _record_settled(self.directory)
        except BaseException as error:   # best effort: a failed pre-collection costs only Pop's work, never the round
            self.error = repr(error)[:400]
            log(json.dumps({'event': 'pre-collection failed', 'directory': str(self.directory), 'error': self.error}))

    def finish(self, settle=True):
        """``settle=False`` (the learner failed or was stopped): leave it unsettled, so the resumed update's ``begin``
        reopens the same directory and Pop keeps collecting while the learner reruns."""
        self.settle = settle
        self.drain.set()
        self.thread.join()
        return self.error


def _record_settled(directory):
    """A drained collection with no worker left running is settled; a restart in place leaves workers detached (the
    relaunched controller settles it)."""
    from .detached_processes import restart_requested
    if not restart_requested():
        write_json_durable(Path(directory) / RESULT, {'schema': 'natlang.rl-precollection-result/1', 'settled': True})


def settle(directory, collect, control=None, log=print):
    """Finish a pre-collection left unsettled (the controller restarted during the update): drain it inline."""
    directory = Path(directory)
    if (directory / RESULT).exists() or not (directory / START).exists():
        return
    start = json.loads((directory / START).read_text())
    kwargs = start['kwargs']
    drain = threading.Event()
    drain.set()
    try:
        collect(**{**kwargs, 'cases': Path(kwargs['cases']), 'out': Path(kwargs['out']),
                   'late_endpoints': (), 'control': control, 'drain': drain})
        _record_settled(directory)
    except BaseException as error:
        log(json.dumps({'event': 'pre-collection settle failed', 'directory': str(directory), 'error': repr(error)[:400]}))
        write_json_durable(directory / RESULT, {'schema': 'natlang.rl-precollection-result/1', 'settled': False,
                                                'error': repr(error)[:400]})
        raise


def import_into_round(pre, round_dir, state, *, learner_policy_id):
    """Move the settled pre-collection's finished episodes into ``round_dir`` before its first collect. Returns the
    receipt dict, or ``{'rejected': reason}`` (nothing moves; the round collects those entries afresh)."""
    pre, round_dir = Path(pre).resolve(), Path(round_dir).resolve()
    if (round_dir / RECEIPT).exists():
        return json.loads((round_dir / RECEIPT).read_text())
    if not (pre / START).exists():
        return None
    start = json.loads((pre / START).read_text())
    reason = None
    behavior_policy_id = start['behavior_policy_id']
    if learner_policy_id == behavior_policy_id:
        reason = 'the learner did not update the behaviour policy'
    elif (round_dir / 'collection-binding.json').exists():
        reason = 'the round already started collecting'
    elif (state['round'] != start['round'] or state['trained_rounds'] != start['trained_rounds'] + 1
          or (state.get('interleave') or {}).get('sft_windows', 0) != start['sft_windows']):
        reason = 'more than one update (an SFT window or another round) separates the behaviour policy from the learner'
    result = json.loads((pre / RESULT).read_text()) if (pre / RESULT).exists() else {}
    if reason is None and (not (pre / 'score.json').exists() or result.get('settled') is not True):
        reason = 'the pre-collection did not settle'
    if reason is not None:
        write_json_durable(pre / 'pre-collection-rejected.json', {'reason': reason, 'state_round': state['round']})
        return {'rejected': reason}
    from .collection_generations import Generations
    generations = Generations(pre)
    finished = generations.attempted()
    queue = {entry['key']: entry for entry in read_jsonl(pre / 'queue.jsonl')}
    keys = sorted(key for key in finished if key in queue and not queue[key].get('jolt_role'))
    report = json.loads((pre / 'score.json').read_text())
    round_dir.mkdir(parents=True, exist_ok=True)
    workers = round_dir / 'workers'
    workers.mkdir(exist_ok=True)
    wanted = set(keys)
    journal_rows = [row for path in generations.journals() for row in generations.rows(path) if row.get('key') in wanted]
    sent = generations.finished_endpoints()
    for key in keys:
        entry = queue[key]
        output = Path(entry['output'])
        moved = Path(str(output).replace(str(pre), str(round_dir), 1))
        directory = output.parent
        if directory != pre and directory.is_relative_to(pre):   # the entry's whole row/sample directory (jobs, log, results)
            output, moved = directory, moved.parent
        if output.exists() and not moved.exists():
            moved.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(output), str(moved))
    (workers / IMPORT_JOURNAL).write_text(''.join(json.dumps(row) + '\n' for row in journal_rows))
    (workers / IMPORT_QUEUE).write_text(''.join(json.dumps({'key': key, 'rl_endpoint': sent.get(key)}) + '\n' for key in keys))
    blocks = pre / 'blocks'
    if blocks.exists():
        shutil.move(str(blocks), str(round_dir / 'pre-blocks'))
    actors = report.get('actor_receipts') or {}
    export = next((receipt['actor_export'] for receipt in actors.values() if (receipt or {}).get('actor_export')), None)
    receipt = {'schema': RECEIPT_SCHEMA, 'behavior_policy_id': behavior_policy_id, 'round': start['round'],
               'update_round': start['update_round'], 'staleness_updates': 1, 'keys': keys,
               'endpoints': sorted(actors), 'actor_export': export,
               'blocks_manifest': 'pre-blocks/manifest.json' if blocks.exists() or (round_dir / 'pre-blocks').exists() else None,
               'lineage_step': start['trained_rounds']}
    write_json_durable(round_dir / RECEIPT, receipt)
    return receipt


def stamp_rollouts(path, receipt):
    """Keep each imported episode's behaviour policy on its rollout (the scorer stamped the round's policy)."""
    keys = set(receipt['keys'])
    rows = list(read_jsonl(path))
    stamped = 0
    for row in rows:
        if f"s7-task-{int(row['group']):03d}-{int(row['sample']):02d}" in keys and row.get('jolt_role', 'student') == 'student':
            row['policy'] = {**row['policy'], 'checkpoint_sha256': receipt['behavior_policy_id'],
                             'pre_collected': True, 'staleness_updates': receipt['staleness_updates']}
            stamped += 1
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    return stamped


def actors_declaration(receipt, learner_policy_id):
    """``natlang.rl-actors/1`` for a round holding pre-collected episodes: policy N sampled them on Pop's recorded export."""
    return {'schema': 'natlang.rl-actors/1', 'learner_checkpoint_sha256': learner_policy_id,
            'actors': {receipt['behavior_policy_id']: {'lineage_step': receipt['lineage_step'],
                                                       'export': receipt['actor_export']}}}
