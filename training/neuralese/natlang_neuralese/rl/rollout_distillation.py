"""Self-distillation of the policy's own verified successful RL episodes into the interleaved SFT windows.

After a round's collection is scored, the student episodes that solved their task are lowered to whole-graph
native SFT rows (``native_stage.prepare`` in ``rft`` mode: sampled outputs only carry loss, prompts, tool output and
harness feedback are zero-weight context, every producer/consumer dependency kept) and queued as one
``natlang.memetic-distillation-package/1`` (family ``rollout-distillation``) in the same admission queue the memetic
packages use. The next mandatory SFT window appends them as a recorded source with exact actor provenance
(``rl/memetic_admission.py``: protection recheck, lineage-actor check, share cap, deferral instead of dropping).

Selection (``select``):
- student episodes only: never a JOLT+ teacher twin, never a memetic "improve program" episode, never a protected or
  held task (train split, and the package and admission both recheck the protection union);
- current-policy provenance only: the sampled policy is this round's learner policy, so an older pre-collected policy's
  episode is never admitted;
- one episode per task per round (no padding by repetition); among a task's successes a Neuralese-using one (a real
  between-call state edge in the captured graph, ``task_rollouts.neuralese_exchanges``) comes first, then higher
  return, then fewer model calls; the package keeps that rank order so the share cap defers the weakest rows;
- Neuralese-using episodes carry their return (1 + bonus) as the weight of their sampled tokens, as the rft objective
  does; closed whole graphs only (an episode importing old external actor state is skipped and recorded).

``rollout_distillation: false`` opts out. Continuations adopt it through the recorded controller transition
``adopt_defaults``: a round already collected or in flight keeps its behaviour. Failures are recorded, never raised
into the round: the policy update must not wait on a distillation package.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex
from ..common.jsonio import write_json_durable
from .memetic_admission import ROLLOUT_FAMILY, _lines, queue_path

SCHEMA = 'natlang.rollout-distillation/1'
TRANSITION = 'rollout-distillation-20261110'
RECORD = 'rollout-distillation.json'


def adopt_defaults(config, state, state_path, *, sealed_selection=False, interleaved=True):
    """Interleaved native controllers adopt rollout self-distillation through a recorded transition. The first round
    it applies to is the next unsealed one: a round whose selection is sealed or whose update is active keeps its
    behaviour. ``rollout_distillation: false`` opts out; coefficient-policy actors (adapter-bound episodes) are not
    distilled."""
    if config.get('rollout_distillation') is False or not interleaved or config.get('coefficient_policy'):
        return config
    transitions = state.setdefault('rollout_distillation_transitions', [])
    record = next((item for item in transitions if item.get('transition') == TRANSITION), None)
    if record is None:
        in_flight = bool(sealed_selection or state.get('active_update_round'))
        record = {'schema': 'natlang.rollout-distillation-transition/1', 'transition': TRANSITION,
                  'recorded_at_round': state.get('round', 0),
                  'from_round': state.get('round', 0) + (1 if in_flight else 0)}
        transitions.append(record)
        write_json_durable(state_path, state)
    return {**config, 'rollout_distillation_from_round': record['from_round']}


def due(config, state):
    start = config.get('rollout_distillation_from_round')
    return config.get('rollout_distillation') is not False and start is not None and state.get('round', 0) >= start


def _rank(row):
    cost = float(((row['reward'].get('execution_efficiency') or {}).get('cost')) or 0.)
    return (-row['_neuralese_exchanges'], -float(row['reward']['quality'] + (row['reward'].get('neuralese_bonus') or 0.)),
            cost, row.get('sample') if type(row.get('sample')) is int else 0)


def select(rows, policy_sha256):
    """(selected rollouts in rank order, skipped {reason: count}); one successful student episode per task."""
    from .task_rollouts import neuralese_exchanges
    best, skipped = {}, {}

    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    for row in rows:
        reward = row.get('reward') or {}
        if reward.get('quality') != 1:
            continue
        if row.get('jolt_role', 'student') != 'student':
            skip('teacher episode')
        elif row.get('split') != 'train':
            skip('not a train task')
        elif reward.get('memetic') is not None:
            skip('memetic improvement episode')
        elif (row.get('policy') or {}).get('checkpoint_sha256') != policy_sha256:
            skip('sampled by another policy')
        elif not row.get('turns') or row.get('unscored'):
            skip('no complete captured episode')
        else:
            candidate = {**row, '_neuralese_exchanges': 1 if neuralese_exchanges(row) > 0 else 0}
            held = best.get(row['task_id'])
            if held is None or _rank(candidate) < _rank(held):
                best[row['task_id']] = candidate
    ordered = sorted(best.values(), key=_rank)
    return ordered, skipped


def _geometry(tokenizer_path):
    from transformers import AutoTokenizer
    from ..data.text_corpus import tokenizer_fingerprint
    from ..model.shared_state import ROLE_MARKERS
    tokenizer = AutoTokenizer.from_pretrained(Path(tokenizer_path).resolve(), local_files_only=True)
    return tokenizer_fingerprint(tokenizer), [int(value) for value in tokenizer.convert_tokens_to_ids(list(ROLE_MARKERS))]


def _prepare(selected, binding, checkpoint, bundle):
    """Whole-graph rft lowering of ``selected``: the batch, else (one bad episode) episode by episode."""
    from .native_stage import prepare
    options = dict(checkpoint=checkpoint, tokenizer=binding['rl_tokenizer'], bundle=bundle or None, mode='rft',
                   expected_history_reasoning_policy=binding.get('history_reasoning_policy'),
                   expected_history_binding_sha256=binding.get('history_reasoning_binding_sha256'),
                   expected_history_binding_path=binding.get('history_reasoning_binding_path'))
    clean = [{key: value for key, value in row.items() if key != '_neuralese_exchanges'} for row in selected]
    try:
        return prepare(clean, **options), []
    except Exception:  # noqa: BLE001 - isolate the episode that does not lower
        graphs, rows, failed = [], {}, []
        for row in clean:
            try:
                (one,), objective = prepare([row], **options)
                graphs.append(one)
                rows.update(objective['rows'])
            except Exception as error:  # noqa: BLE001
                failed.append({'task_id': row['task_id'], 'sample': row.get('sample'),
                               'error': f'{type(error).__name__}: {error}'[:300]})
        return (graphs, {'rows': rows}), failed


def graph_rows(selected, graphs, objective, *, policy_sha256, round_number):
    """Native SFT rows (rank order) for the lowered graphs: closed whole graphs only, Neuralese-using episodes
    weighted by their return. Returns (rows, skipped)."""
    by_key = {canonical_json_sha256_hex({key: value for key, value in row.items() if key != '_neuralese_exchanges'}): row
              for row in selected}
    ranked = {key: index for index, key in enumerate(by_key)}
    rows, skipped = [], []
    for graph in sorted(graphs, key=lambda item: ranked[item['id']]):
        source = by_key[graph['id']]
        if graph.get('external_states'):
            skipped.append({'task_id': source['task_id'], 'reason': 'imports old external actor state'})
            continue
        weight = float((objective['rows'][graph['id']]).get('rft_weight') or 1.)
        nodes = [{**node, 'target_weights': [value * weight for value in node['target_weights']]}
                 for node in graph['nodes']]
        rows.append({**graph, 'id': 'rollout:' + graph['id'], 'nodes': nodes,
                     'rollout_distillation': {
                         'schema': SCHEMA, 'task_id': source['task_id'], 'sample': source.get('sample'),
                         'round': round_number, 'actor_checkpoint_sha256': policy_sha256,
                         'neuralese_exchanges': source['_neuralese_exchanges'], 'sampled_token_weight': weight,
                         'rollout_sha256': graph['id']}})
    return rows, skipped


def package_round(config, state, round_dir, rollouts, *, policy_sha256, binding, bundle, protected=None, prepare=_prepare,
                  geometry=_geometry, log=print):
    """Select, lower and queue this round's verified successful student episodes; the record is written next to the
    round's rollouts. ``binding`` is the round's learner tokenizer and checkpoint history profile (``values`` of the
    round: rl_tokenizer, history_reasoning_*). Returns the record (idempotent: a round queued once is never
    queued again)."""
    round_dir = Path(round_dir)
    record_path = round_dir / RECORD
    controller_memetic = Path(config['out']) / 'memetic'
    if any(item.get('manifests') == [str(record_path)] for item in _lines(queue_path(config['out']))):
        return json.loads(record_path.read_text()) if record_path.is_file() else None
    selected, skipped = select(rollouts, policy_sha256)
    record = {'schema': SCHEMA, 'round': state.get('round'), 'policy_sha256': policy_sha256,
              'successful_episodes': sum((row.get('reward') or {}).get('quality') == 1 for row in rollouts),
              'selected_tasks': len(selected),
              'selected_neuralese_using': sum(row['_neuralese_exchanges'] for row in selected),
              'skipped': skipped, 'package': None}
    if selected:
        (graphs, objective), failed = prepare(selected, binding, state['checkpoint'], bundle)
        rows, closed = graph_rows(selected, graphs, objective, policy_sha256=policy_sha256,
                                  round_number=state.get('round'))
        from ..common.custody import ProtectedWorlds
        guard = ProtectedWorlds.load() if protected is None else protected
        kept = [row for row in rows if not guard.related(row)]
        record.update(lowering_failed=failed, not_closed=closed, protected_excluded=len(rows) - len(kept))
        if kept:
            tokenizer, roles = geometry(binding['rl_tokenizer'])
            from .memetic_admission import queue_package
            receipt = queue_package(
                controller_memetic, kept, family=ROLLOUT_FAMILY, tokenizer=tokenizer, roles=roles, protected=guard,
                manifests=[record_path], actors=[policy_sha256], source_name='rollout-distillation-graphs.jsonl',
                scope='verified successful whole student episodes of the current policy, ranked Neuralese-using '
                      'first; one per task; sampled outputs only; adoption is a recorded source append at the next '
                      'mandatory SFT window',
                extra={'round': state.get('round'), 'round_directory': str(round_dir)}, log=log)
            record['package'] = receipt['source']
            record['rows'] = len(kept)
            record['neuralese_using_rows'] = sum(row['rollout_distillation']['neuralese_exchanges'] > 0 for row in kept)
    write_json_durable(record_path, record)
    log(json.dumps({'event': 'rollout_distillation', **{k: v for k, v in record.items() if k != 'package'}}))
    return record


def after_scoring(config, state, round_dir, values, rollouts, policy_sha256, log=print):
    """The production hook: never raises into the round."""
    if not due(config, state):
        return None
    try:
        from ..common.hashing import sha256_file_hex_cached
        if sha256_file_hex_cached(state['checkpoint']) != policy_sha256:
            raise ValueError('controller head is not this round\'s learner policy')
        binding = {key: values[key] for key in ('rl_tokenizer', 'history_reasoning_policy',
                   'history_reasoning_binding_path', 'history_reasoning_binding_sha256') if key in values}
        return package_round(config, state, round_dir, rollouts, policy_sha256=policy_sha256, binding=binding,
                             bundle=values.get('block_bundle'), log=log)
    except Exception as error:  # noqa: BLE001 - recorded for supervision; the update proceeds
        log(json.dumps({'event': 'rollout_distillation_failed', 'round': state.get('round'),
                        'error': f'{type(error).__name__}: {error}'[:500]}))
        return None
