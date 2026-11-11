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
- expert iteration (2026-11-10): every verified successful student episode of the round is a row, not one per task.
  Exact duplicates (the same sampled assistant outputs for one task) are dropped, never repeated, and a row is visited
  once (a distinct sampled graph is an independent example; ``rollout_distillation_max_per_task`` bounds one task's
  share if its samples crowd the rest). The package is in rank order, Neuralese-using episodes first (a real
  between-call state edge in the captured graph, ``task_rollouts.neuralese_exchanges``), then higher return, then
  fewer model calls, so the share cap defers the weakest rows;
- rounds scored before the transition, or by code that queued only one episode per task, are topped up at the next
  scored round (``backfill``): their remaining verified successes are queued once, with each row's own sampling policy
  recorded, provided that policy is an actual checkpoint of this lineage;
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

SCHEMA = 'natlang.rollout-distillation/2'
TRANSITION = 'rollout-distillation-20261110'
RECORD = 'rollout-distillation-all.json'
LEGACY_RECORD = 'rollout-distillation.json'  # one episode per task (the 2026-11-10 first form)


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


def _sampled_outputs(row):
    """Identity of an episode's sampled assistant outputs (never its sampling probabilities): two samples of one task
    with the same outputs are one example."""
    return canonical_json_sha256_hex([turn.get('target') for turn in row['turns']])


def select(rows, policies, *, max_per_task=None, exclude=()):
    """(selected rollouts in rank order, skipped {reason: count}): every verified successful student episode.

    ``policies`` is the policy SHA-256 (or a collection of them) an episode may have been sampled by; ``exclude`` holds
    (task, sample) pairs already queued by an earlier package."""
    from .task_rollouts import neuralese_exchanges
    allowed = {policies} if isinstance(policies, str) else set(policies)
    best, seen, skipped = [], set(), {}

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
        elif (row.get('policy') or {}).get('checkpoint_sha256') not in allowed:
            skip('sampled by another policy')
        elif not row.get('turns') or row.get('unscored'):
            skip('no complete captured episode')
        elif (row['task_id'], row.get('sample')) in exclude:
            skip('already queued')
        else:
            identity = (row['task_id'], _sampled_outputs(row))
            if identity in seen:
                skip('duplicate sampled outputs')
                continue
            seen.add(identity)
            exchanges = row['_neuralese_exchanges'] if '_neuralese_exchanges' in row else int(neuralese_exchanges(row) > 0)
            best.append({**row, '_neuralese_exchanges': 1 if exchanges else 0})
    ordered, per_task = [], {}
    for row in sorted(best, key=_rank):
        if max_per_task is not None and per_task.get(row['task_id'], 0) >= max_per_task:
            skip('over the per-task bound')
            continue
        per_task[row['task_id']] = per_task.get(row['task_id'], 0) + 1
        ordered.append(row)
    return ordered, skipped


def _geometry(tokenizer_path):
    from transformers import AutoTokenizer
    from ..data.text_corpus import tokenizer_fingerprint
    from ..model.shared_state import ROLE_MARKERS
    tokenizer = AutoTokenizer.from_pretrained(Path(tokenizer_path).resolve(), local_files_only=True)
    return tokenizer_fingerprint(tokenizer), [int(value) for value in tokenizer.convert_tokens_to_ids(list(ROLE_MARKERS))]


CHUNK = 4  # episodes lowered (and resident, parsed) at a time: a captured episode is tens of megabytes of JSON


def _prepare(selected, binding, checkpoint, bundle):
    """Whole-graph rft lowering of ``selected`` (full episodes); a batch that does not lower goes episode by episode so
    that one bad episode never costs the rest."""
    from .native_stage import prepare
    options = dict(checkpoint=checkpoint, tokenizer=binding['rl_tokenizer'], bundle=bundle or None, mode='rft',
                   expected_history_reasoning_policy=binding.get('history_reasoning_policy'),
                   expected_history_binding_sha256=binding.get('history_reasoning_binding_sha256'),
                   expected_history_binding_path=binding.get('history_reasoning_binding_path'))
    clean = [_clean(row) for row in selected]
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


def _clean(row):
    """The episode exactly as captured (selection and reading bookkeeping removed)."""
    return {key: value for key, value in row.items() if key not in ('_neuralese_exchanges', '_offset')}


def _slim(row, offset):
    """What selection needs of a captured episode, with where to read the whole episode again."""
    from .task_rollouts import neuralese_exchanges
    return {**{key: value for key, value in row.items() if key not in ('turns', 'execution_graph')},
            'turns': [{'target': turn.get('target')} for turn in row.get('turns') or []],
            '_neuralese_exchanges': int(neuralese_exchanges(row) > 0), '_offset': offset}


def graph_rows(sources, graphs, objective, *, round_number):
    """Native SFT rows (rank order) for the lowered graphs: closed whole graphs only, Neuralese-using episodes
    weighted by their return, each row naming the exact policy that sampled it. ``sources`` is {graph id: selected
    episode} in rank order. Returns (rows, skipped)."""
    ranked = {key: index for index, key in enumerate(sources)}
    rows, skipped = [], []
    for graph in sorted(graphs, key=lambda item: ranked[item['id']]):
        source = sources[graph['id']]
        if graph.get('external_states'):
            skipped.append({'task_id': source['task_id'], 'reason': 'imports old external actor state'})
            continue
        weight = float((objective['rows'][graph['id']]).get('rft_weight') or 1.)
        nodes = [{**node, 'target_weights': [value * weight for value in node['target_weights']]}
                 for node in graph['nodes']]
        rows.append({**graph, 'id': 'rollout:' + graph['id'], 'nodes': nodes,
                     'rollout_distillation': {
                         'schema': SCHEMA, 'task_id': source['task_id'], 'sample': source.get('sample'),
                         'round': round_number, 'actor_checkpoint_sha256': source['policy']['checkpoint_sha256'],
                         'neuralese_exchanges': source['_neuralese_exchanges'], 'sampled_token_weight': weight,
                         'rollout_sha256': graph['id']}})
    return rows, skipped


def policy_binding(config):
    """The binding function of ``package_round`` for earlier rounds: each policy's own actor tokenizer directory and
    its history-reasoning sidecar (``rl/production`` writes them under ``<out>/actor-tokenizers/<policy>``)."""
    from ..common.hashing import sha256_file_hex_cached

    def binding(sha):
        directory = Path(config['out']) / 'actor-tokenizers' / sha
        sidecar = directory / 'natlang-history-reasoning.json'
        if not sidecar.is_file():
            raise ValueError(f'no actor tokenizer directory for policy {sha}')
        return {'rl_tokenizer': str(directory), 'history_reasoning_policy': config.get('history_reasoning_policy'),
                'history_reasoning_binding_path': str(sidecar),
                'history_reasoning_binding_sha256': sha256_file_hex_cached(sidecar)}

    return binding


def queued_episodes(controller_out):
    """{(task, sample)} of every rollout-distillation package already queued, per round: an earlier form of this hook
    (one episode per task) must not be queued again by a top-up."""
    queued = {}
    for item in _lines(queue_path(controller_out)):
        if item.get('family') != ROLLOUT_FAMILY or not Path(item['package']).is_file():
            continue
        receipt = json.loads(Path(item['package']).read_text())
        source = Path((receipt.get('source') or {}).get('path', ''))
        if not source.is_file():
            continue
        for line in source.read_text().splitlines():
            if line.strip():
                info = json.loads(line).get('rollout_distillation') or {}
                queued.setdefault(info.get('round', receipt.get('round')), set()).add((info.get('task_id'), info.get('sample')))
    return queued


def package_round(config, state, round_dir, rollouts, *, policies, binding, bundle, round_number=None, protected=None,
                  prepare=_prepare, geometry=_geometry, checkpoint=None, checkpoints=None, reader=None, log=print):
    """Select, lower and queue this round's verified successful student episodes; the record is written next to the
    round's rollouts. ``checkpoints`` maps each allowed policy to its checkpoint file (the one policy's ``checkpoint``
    when scored live). ``binding`` is the round's learner tokenizer and checkpoint history profile (``values`` of the
    round: rl_tokenizer, history_reasoning_*), or a function of the sampling policy returning it (``policy_binding``:
    an earlier round's policy has its own tokenizer directory and history sidecar). ``reader`` turns a selected
    episode of ``rollouts`` kept slim (``_slim``) back into the whole captured episode, a chunk at a time. ``policies`` are the policies an episode may have been sampled by
    (the round's own learner policy while it is scored; the lineage's checkpoints for an earlier round).
    Returns the record (idempotent: a round queued once is never queued again)."""
    round_dir = Path(round_dir)
    round_number = state.get('round') if round_number is None else round_number
    record_path = round_dir / RECORD
    controller_memetic = Path(config['out']) / 'memetic'
    if any(item.get('manifests') == [str(record_path)] for item in _lines(queue_path(config['out']))):
        return json.loads(record_path.read_text()) if record_path.is_file() else None
    selected, skipped = select(rollouts, policies, max_per_task=config.get('rollout_distillation_max_per_task'),
                               exclude=queued_episodes(config['out']).get(round_number, ()))
    record = {'schema': SCHEMA, 'round': round_number, 'policies': sorted([policies] if isinstance(policies, str) else policies),
              'successful_episodes': sum((row.get('reward') or {}).get('quality') == 1 for row in rollouts),
              'selected_episodes': len(selected), 'selected_tasks': len({row['task_id'] for row in selected}),
              'selected_neuralese_using': sum(row['_neuralese_exchanges'] for row in selected),
              'skipped': skipped, 'package': None}
    if selected:
        # One lowering per sampling policy: the replay binds its checkpoint file and collector bundle to the policy.
        files = checkpoints if checkpoints is not None else {policies: checkpoint or state['checkpoint']} \
            if isinstance(policies, str) else {}
        graphs, objective, failed, sources = [], {'rows': {}}, [], {}
        for sha in sorted({row['policy']['checkpoint_sha256'] for row in selected}):
            group = [row for row in selected if row['policy']['checkpoint_sha256'] == sha]
            if sha not in files:
                failed.extend({'task_id': row['task_id'], 'sample': row.get('sample'),
                               'error': 'the sampling policy checkpoint is no longer on disk'} for row in group)
                continue
            try:
                options = binding(sha) if callable(binding) else binding
            except ValueError as error:
                failed.extend({'task_id': row['task_id'], 'sample': row.get('sample'), 'error': str(error)[:300]}
                              for row in group)
                continue
            for start in range(0, len(group), CHUNK):
                chunk = group[start:start + CHUNK]
                full = [reader(row) for row in chunk] if reader is not None else chunk
                for episode, row in zip(full, chunk, strict=True):
                    sources[canonical_json_sha256_hex(_clean(episode))] = row
                (lowered, partial), missed = prepare(full, options, files[sha], bundle)
                graphs.extend(lowered)
                objective['rows'].update(partial['rows'])
                failed.extend(missed)
                del full
        order = {id(row): index for index, row in enumerate(selected)}
        sources = dict(sorted(sources.items(), key=lambda item: order[id(item[1])]))
        rows, closed = graph_rows(sources, graphs, objective, round_number=round_number)
        from ..common.custody import ProtectedWorlds
        guard = ProtectedWorlds.load() if protected is None else protected
        kept = [row for row in rows if not guard.related(row)]
        record.update(lowering_failed=failed, not_closed=closed, protected_excluded=len(rows) - len(kept))
        if kept:
            actors = sorted({row['rollout_distillation']['actor_checkpoint_sha256'] for row in kept})
            tokenizer, roles = geometry((binding(actors[0]) if callable(binding) else binding)['rl_tokenizer'])
            from .memetic_admission import queue_package
            receipt = queue_package(
                controller_memetic, kept, family=ROLLOUT_FAMILY, tokenizer=tokenizer, roles=roles, protected=guard,
                manifests=[record_path], actors=actors, source_name='rollout-distillation-graphs.jsonl',
                scope='every verified successful whole student episode of the round (expert iteration), ranked '
                      'Neuralese-using first; exact duplicate outputs dropped; sampled outputs only; adoption is a '
                      'recorded source append at the next mandatory SFT window',
                extra={'round': round_number, 'round_directory': str(round_dir)}, log=log)
            record['package'] = receipt['source']
            record['rows'] = len(kept)
            record['neuralese_using_rows'] = sum(row['rollout_distillation']['neuralese_exchanges'] > 0 for row in kept)
    write_json_durable(record_path, record)
    log(json.dumps({'event': 'rollout_distillation', **{k: v for k, v in record.items() if k != 'package'}}))
    return record


def _round_number(directory):
    name = directory.name
    return int(name.split('-')[1]) if name.startswith('round-') and name.count('-') == 1 and name[6:].isdigit() else None


def _student_successes(path):
    """The verified successful student episodes of a round's rollouts file as slim episodes (``_slim``), read one line
    at a time: a round's file holds gigabytes of sampling probabilities, so only what selection needs stays resident."""
    with Path(path).open('rb') as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                return
            if b'"quality": 1.0' not in line and b'"quality":1.0' not in line and b'"quality": 1,' not in line:
                continue
            row = json.loads(line)
            if (row.get('reward') or {}).get('quality') == 1 and row.get('jolt_role', 'student') == 'student':
                yield _slim(row, offset)


def _episode_reader(path):
    def read(slim):
        with Path(path).open('rb') as handle:
            handle.seek(slim['_offset'])
            return json.loads(handle.readline())
    return read


def backfill(config, state, *, current_round, binding=None, log=print, protected=None, prepare=_prepare,
             geometry=_geometry):
    """Queue the verified successful episodes of every scored round before ``current_round`` that has no record of
    this form (rounds scored before the hook existed, or when it queued one episode per task). An episode qualifies
    when its sampling policy is an actual checkpoint of this lineage; its row records that exact policy."""
    from .memetic_admission import lineage_checkpoint_files, lineage_checkpoints
    controller = Path(config['out'])
    policies, files = lineage_checkpoints(config, state), lineage_checkpoint_files(config, state)
    binding = policy_binding(config) if binding is None else binding
    results = []
    for directory in sorted(controller.glob('round-[0-9]*')):
        number = _round_number(directory)
        if number is None or number >= current_round or (directory / RECORD).is_file():
            continue
        rollouts_path, score_path = directory / 'rollouts.jsonl', directory / 'score.json'
        if not rollouts_path.is_file() or not score_path.is_file():
            continue
        score = json.loads(score_path.read_text())
        if not score.get('collection_complete'):
            continue
        # The collector bundle of a round that trained is pruned, and an episode lowers only by replaying the actor states
        # in it: such a round is recorded as lost (nothing is read or lowered) and never retried.
        bundle = score.get('block_bundle')
        if not bundle or not Path(bundle).is_file():
            record = {'schema': SCHEMA, 'round': number, 'package': None, 'rows': 0,
                      'skipped': 'the round\'s collector bundle is no longer on disk; its episodes cannot be replayed'}
            write_json_durable(directory / RECORD, record)
            log(json.dumps({'event': 'rollout_distillation', **record}))
            results.append(record)
            continue
        results.append(package_round(config, state, directory, list(_student_successes(rollouts_path)),
                                     reader=_episode_reader(rollouts_path),
                                     policies=policies, binding=binding, bundle=bundle,
                                     round_number=number, protected=protected, prepare=prepare, geometry=geometry,
                                     checkpoints=files, log=log))
    return results


def after_scoring(config, state, round_dir, values, policy_sha256, log=print):
    """The production hook: never raises into the round. Earlier rounds are topped up first (``backfill``). The round's
    successes are read slim from ``values['rollouts']`` (a round's file holds gigabytes; w5 round 2 held 5.1 GB)."""
    if not due(config, state):
        return None
    try:
        from ..common.hashing import sha256_file_hex_cached
        if sha256_file_hex_cached(state['checkpoint']) != policy_sha256:
            raise ValueError('controller head is not this round\'s learner policy')
        binding = {key: values[key] for key in ('rl_tokenizer', 'history_reasoning_policy',
                   'history_reasoning_binding_path', 'history_reasoning_binding_sha256') if key in values}
    except Exception as error:  # noqa: BLE001 - recorded for supervision; the update proceeds
        log(json.dumps({'event': 'rollout_distillation_failed', 'round': state.get('round'),
                        'error': f'{type(error).__name__}: {error}'[:500]}))
        return None
    try:
        backfill(config, state, current_round=state.get('round', 0), log=log)
    except Exception as error:  # noqa: BLE001
        log(json.dumps({'event': 'rollout_distillation_backfill_failed', 'round': state.get('round'),
                        'error': f'{type(error).__name__}: {error}'[:500]}))
    try:
        path = Path(values['rollouts'])
        return package_round(config, state, round_dir, list(_student_successes(path)), reader=_episode_reader(path),
                             policies=policy_sha256, binding=binding, bundle=values.get('block_bundle'), log=log)
    except Exception as error:  # noqa: BLE001
        log(json.dumps({'event': 'rollout_distillation_failed', 'round': state.get('round'),
                        'error': f'{type(error).__name__}: {error}'[:500]}))
        return None
