"""Window-boundary admission of memetic distillation packages into the joint SFT source.

``memetic_stage.package_distillation`` queues immutable packages (``natlang.memetic-distillation-package/1``) of
verified guided-query producer/consumer graphs in ``<controller out>/memetic/distill/admission-queue.jsonl``. At the
first interleaved SFT window after an unconsumed package exists, ``prepare_window`` appends its admitted rows to the
joint SFT stage as an ordinary authenticated data continuation (the same form as every other source append):

- ``native_graphs`` (and a shared ``foundation_native_graphs``) become a successor manifest = the exact predecessor
  manifest followed by the admitted package leaves (prefix, bindings and order unchanged);
  ``continued_native_graphs`` names that predecessor, so the trainer's ``append_only_rows`` and
  ``continuation_predecessors`` authenticate the append and give the added rows fresh quotas while earned exposure,
  optimizer and RNG stay in the carried frame;
- the CE policy declares each new leaf (``native-row-objectives-v1``) and, under a logical-family objective, extends
  the family map with family ``memetic-guidance`` (declared/1: its own declared scale) through an explicit
  ``objective_transition`` (validated with ``native_resident_ce.extension_policy``);
- a declared share cap bounds the family's CE mass share: admitted memetic rows x their largest possible family scale
  <= ``distillation_max_share`` x train rows. Rows over the cap stay deferred for a later window, never dropped;
- every row is rechecked against the complete current protection union at admission; overlapping rows are excluded
  explicitly; a package whose episode actors are not checkpoints of this controller's actual lineage is refused.

Packages come from three producers through ``queue_package``: memetic guidance graphs (family ``memetic-guidance``),
the policy's own verified successful RL episodes (family ``rollout-distillation``, ``rl/rollout_distillation.py``) and
prepared static sources (family ``prepared-source``: verified native graphs that no rollout produced, queued by
``scripts/neuralese_data/queue_prepared_source.py``). A static package has no episode actor; it instead records its
origin, verification and row selection (``source_provenance``), and its rows pass the same protection recheck, share
cap and deferral. Pending static packages are admitted after the lineage's own (self-distilled) packages, so a static
source only fills the share the policy's own rows leave. Each admitted family declares its own scale; the share cap
bounds all of them together.

The consumed ledger (``state['interleave']['memetic_distillation']``) is committed before the window config is
written, so a package is appended exactly once and a retried boundary reuses its recorded admission. Actor provenance
is student self-distillation: the actors are this lineage's own checkpoints, never a teacher.
"""
from __future__ import annotations

import copy
import json
import math
import shutil
from pathlib import Path

from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import write_json_durable

SCHEMA = 'natlang.memetic-distillation-admission/1'
PACKAGE_SCHEMA = 'natlang.memetic-distillation-package/1'
FAMILY = 'memetic-guidance'
# The other producer of packages: the policy's own verified successful RL episodes (rl/rollout_distillation.py).
ROLLOUT_FAMILY = 'rollout-distillation'
# The third producer: prepared static sources (verified native graphs without an episode actor).
STATIC_FAMILY = 'prepared-source'
FAMILIES = (FAMILY, ROLLOUT_FAMILY, STATIC_FAMILY)
STATIC_PROVENANCE_KEYS = ('origin', 'verification', 'selection')
ENCODING = 'native-row-objectives-v1'
# Conservative: the 2026-11-03 recipe review measured genuine natlang teaching at ~6% and multi-node Neuralese at ~2%
# of quota-aware loss; one new self-distilled family must not dominate them.
DEFAULT_MAX_SHARE = 0.02
DEFAULT_FAMILY_SCALE = 1.0
PROVENANCE = 'student-self-distillation'


def validate_parameters(parameters):
    share = parameters.get('distillation_max_share', DEFAULT_MAX_SHARE)
    if type(share) not in (int, float) or not 0 < share <= 0.25:
        raise ValueError('interleaved RL distillation_max_share is in (0, 0.25]')
    scale = parameters.get('distillation_family_scale', DEFAULT_FAMILY_SCALE)
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('interleaved RL distillation_family_scale must be a positive finite number')


def queue_path(controller_out):
    return Path(controller_out) / 'memetic' / 'distill' / 'admission-queue.jsonl'


def queue_package(out, rows, *, family, tokenizer, roles, protected, manifests, actors, scope, problems=(),
                  source_name='memetic-guidance-graphs.jsonl', extra=None, static_source=None, log=print):
    """Write one immutable ``natlang.memetic-distillation-package/1`` of verified native-graph ``rows`` (the exact
    sampled actor(s) ``actors``) under ``<out>/distill/package-NNNN`` and append it to ``<out>/distill/admission-
    queue.jsonl``: the single producer-side form every distillation source uses (memetic guidance graphs and the
    policy's own successful RL episodes, prepared static sources). A ``prepared-source`` package names no actor and
    carries ``static_source`` (origin, verification, selection, free-form details) instead. Returns the receipt."""
    from ..data.native_sources import NativeGraphSource
    if family not in FAMILIES:
        raise ValueError('distillation package family is not one of ' + ', '.join(FAMILIES))
    if family == STATIC_FAMILY:
        if actors or not isinstance(static_source, dict) or any(not static_source.get(key) for key in STATIC_PROVENANCE_KEYS):
            raise ValueError('a prepared-source package names no actor and records ' + ', '.join(STATIC_PROVENANCE_KEYS))
    elif static_source is not None or not actors:
        raise ValueError('only a prepared-source package carries static provenance; every other package names its actors')
    from ..common.jsonio import write_bytes_durable
    out = Path(out)
    queue = out / 'distill' / 'admission-queue.jsonl'
    number = len(_lines(queue))
    target = out / 'distill' / f'package-{number:04d}'
    target.mkdir(parents=True, exist_ok=False)
    graphs = target / source_name
    write_bytes_durable(graphs, ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows).encode())
    indexed = NativeGraphSource(graphs, tokenizer, required_split='train', role_ids=roles)
    if [entry['id'] for entry in indexed.entries] != [row['id'] for row in rows]:
        raise ValueError('distillation package does not index as exactly its rows')
    receipt = {'schema': PACKAGE_SCHEMA, 'source': {'kind': 'native-graphs', 'path': str(graphs),
                                                    'sha256': sha256_file_hex_cached(graphs)},
               'rows': [{'id': entry['id'], 'binding': entry['binding'], 'disposition': 'added',
                         'communication': bool(entry.get('communication'))} for entry in indexed.entries],
               'family': family, 'manifests': [str(m) for m in manifests],
               'episode_actors': sorted(actors), 'tokenizer_fingerprint': tokenizer, 'role_ids': list(roles),
               'protection_digest': protected.digest, 'problems': list(problems),
               'ce_declaration': {'encoding': ENCODING}, 'training_admission': False, 'scope': scope,
               **({'source_provenance': static_source} if static_source is not None else {}),
               **(extra or {})}
    write_json_durable(target / 'receipt.json', receipt)
    with open(queue, 'a') as handle:
        handle.write(json.dumps({'package': str(target / 'receipt.json'), 'manifests': [str(m) for m in manifests],
                                 'rows': len(rows), 'family': family}) + '\n')
    log(json.dumps({'event': 'distillation_package', 'family': family, 'package': str(target), 'rows': len(rows)}))
    return receipt


def _lines(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _ref(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_file_hex_cached(path)}


def _queued(controller_out):
    return _lines(queue_path(controller_out))


def unseen_packages(controller_out, state):
    """Queued packages this lineage has never processed (a refused or partially deferred package is not unseen)."""
    if not controller_out:
        return []
    seen = set((((state.get('interleave') or {}).get('memetic_distillation') or {}).get('packages') or {}))
    result = []
    for item in _queued(controller_out):
        path = Path(item['package'])
        if path.is_file() and sha256_file_hex_cached(path) not in seen:
            result.append(item)
    return result


def lineage_checkpoints(config, state):
    """SHA-256 of every checkpoint this controller actually held: its initial parent, current head, every completed
    RL update and every interleaved SFT window parent/result."""
    shas = set()
    for path in (config.get('checkpoint'), state.get('checkpoint')):
        if path and Path(path).is_file():
            shas.add(sha256_file_hex_cached(path))
    interleave = state.get('interleave') or {}
    for window in interleave.get('windows') or []:
        shas.update(filter(None, ((window.get('parent') or {}).get('sha256'),
                                  (window.get('checkpoint') or {}).get('sha256'))))
    if interleave.get('active_window'):
        shas.add(interleave['active_window']['parent']['sha256'])
    for record in [*(state.get('completed_update_history') or []), state.get('last_completed_update') or {}]:
        sha = (record.get('checkpoint') or {}).get('sha256')
        if sha:
            shas.add(sha)
    return shas


def _index(path, tokenizer, roles):
    from ..data.native_sources import NativeGraphSource
    return NativeGraphSource(Path(path), tokenizer, required_split='train', role_ids=roles)


def _raw_rows(path):
    """{id: (raw line, row)} in file order."""
    result = {}
    for line in Path(path).read_text().splitlines(keepends=True):
        if line.strip():
            row = json.loads(line)
            result[row['id']] = (line if line.endswith('\n') else line + '\n', row)
    return result


def _check_receipt(receipt, receipt_path):
    if receipt.get('schema') != PACKAGE_SCHEMA:
        return 'package schema is not ' + PACKAGE_SCHEMA
    if receipt.get('family') not in FAMILIES:
        return 'package family is not one of ' + ', '.join(FAMILIES)
    if receipt.get('training_admission') is not False:
        return 'package claims training admission'
    if (receipt.get('ce_declaration') or {}).get('encoding') != ENCODING:
        return 'package CE declaration is not ' + ENCODING
    source = receipt.get('source') or {}
    if source.get('kind') != 'native-graphs' or not Path(source.get('path', '')).is_file():
        return 'package source is not an existing native-graphs file'
    if sha256_file_hex_cached(source['path']) != source.get('sha256'):
        return 'package source bytes differ from its receipt'
    rows = receipt.get('rows')
    if (not isinstance(rows, list) or not rows or any(row.get('disposition') != 'added' for row in rows)
            or len({row.get('id') for row in rows}) != len(rows)):
        return 'package rows are not unique added rows'
    if receipt['family'] == STATIC_FAMILY:
        provenance = receipt.get('source_provenance')
        if receipt.get('episode_actors') != []:
            return 'a prepared-source package names no episode actor'
        if not isinstance(provenance, dict) or any(not provenance.get(key) for key in STATIC_PROVENANCE_KEYS):
            return 'prepared-source package lacks its ' + ', '.join(STATIC_PROVENANCE_KEYS)
    elif not isinstance(receipt.get('episode_actors'), list) or not receipt['episode_actors']:
        return 'package names no episode actor'
    return None


def _geometry_identity(parameters):
    reference = parameters.get('sft_packed_geometry')
    if isinstance(reference, str):
        reference = json.loads(reference)
    if not isinstance(reference, dict) or not Path(reference.get('path', '')).is_file():
        return None
    return json.loads(Path(reference['path']).read_text()).get('identity')


def _ce_update(policy_ref, successor, leaves, family_rows, out, *, family_scale):
    """The successor CE policy: declare the new leaves; extend a logical-family map (each admitted family its own
    declared scale). Returns (binding, bound_scale)."""
    from ..train.native_supervision import load_policy
    from ..train.own_output_supervision import identity, policy_objective
    policy = json.loads(Path(policy_ref['path']).read_text())
    previous = identity(policy_objective(load_policy(policy_ref)))['sha256']
    declared = {item['sha256'] for item in policy['sources']}
    contract = policy.get('source_contract')
    weight = (contract or {}).get('default_raw_context_weight', 0.0)
    for leaf in leaves:
        if leaf['sha256'] in declared:
            raise ValueError('memetic leaf is already declared in the CE policy')
        policy['sources'].append({'sha256': leaf['sha256'], 'encoding': ENCODING, 'context_weight': weight})
    policy['sources'].sort(key=lambda item: item['sha256'])
    if 'inventory' in policy:
        policy['inventory'] = [dict(successor)]
    if isinstance(contract, dict):
        contract['inventory'] = [dict(successor)]
        if isinstance(contract.get('active_sources'), list):
            contract['active_sources'] = sorted([*contract['active_sources'], *(dict(leaf) for leaf in leaves)],
                                                key=lambda item: item['sha256'])
            contract['active_source_count'] = len(contract['active_sources'])
    balance = (policy.get('objective') or {}).get('family_balance')
    bound_scale = 1.0
    if isinstance(balance, dict) and balance.get('key') == 'logical-family':
        old_map = json.loads(Path(balance['map']['path']).read_text())
        new_map = {'schema': 'natlang.logical-family-map/1', 'source_sha256': successor['sha256'],
                   'rows': [*old_map['rows'], *family_rows]}
        map_path = Path(out) / 'families.json'
        write_json_durable(map_path, new_map)
        balance['map'] = _ref(map_path)
        if balance.get('normalization') == 'declared/1':
            if family_scale > balance['max_scale']:
                raise ValueError('distillation_family_scale exceeds the objective max_scale')
            balance['scales'] = {**balance['scales'], **{row['family']: float(family_scale) for row in family_rows}}
            bound_scale = float(family_scale)
        else:
            bound_scale = float(balance['max_scale'])  # bounded-mean-one may raise a small family up to its cap
    elif isinstance(balance, dict):
        bound_scale = float(balance['max_scale'])
    if identity(policy_objective(policy))['sha256'] != previous:
        policy['objective_transition'] = {'from': previous}
    path = Path(out) / 'ce-policy.json'
    write_json_durable(path, policy)
    binding = _ref(path)
    load_policy(binding)
    return binding, bound_scale


def _write_recipes(base_recipe, sft_id, successor, ce, out):
    """(append recipe for the admitting window, settled recipe for later windows). The stage keeps its binding
    names, whose declared content changes (a shared train/foundation binding moves together); the append recipe binds
    the exact predecessor under a new name as ``continued_native_graphs``."""
    from ..train.recipe import load_recipe, resolve_recipe_data
    recipe = copy.deepcopy(resolve_recipe_data(base_recipe))
    stage = next(item for item in recipe['stages'] if item['id'] == sft_id)
    inputs = stage['inputs']
    train_name = inputs['native_graphs']
    predecessor = dict(recipe['input_bindings'][train_name])
    recipe['input_bindings'][train_name] = {'path': successor['path'], 'sha256': successor['sha256']}
    if ce is not None:
        recipe['input_bindings'][inputs['ce_supervision_policy']] = {'path': ce['path'], 'sha256': ce['sha256']}
    settled = copy.deepcopy(recipe)
    prior_name = 'memetic_predecessor_' + predecessor['sha256'][:12]
    recipe['input_bindings'][prior_name] = predecessor
    inputs['continued_native_graphs'] = prior_name
    paths = {}
    for label, value in (('append', recipe), ('settled', settled)):
        path = Path(out) / f'recipe-{label}.json'
        write_json_durable(path, value)
        load_recipe(path)
        paths[label] = path
    return paths['append'], paths['settled']


def _stage(recipe_path, sft_id):
    from ..train.recipe import load_recipe, resolve_stage_inputs, effective_stage_parameters
    recipe = load_recipe(recipe_path)
    stage = next(item for item in recipe['stages'] if item['id'] == sft_id)
    return stage, resolve_stage_inputs(recipe, stage, {}), effective_stage_parameters(stage, [], recipe)


def effective_sft_stage(config, declared, state):
    """The lineage's current SFT recipe and stage (after any admitted append), for windows that admit nothing."""
    ledger = (state.get('interleave') or {}).get('memetic_distillation') or {}
    if not ledger.get('recipe'):
        return None, declared['sft_stage']
    recipe = ledger['recipe']
    if sha256_file_hex_cached(recipe['path']) != recipe['sha256']:
        raise ValueError('admitted memetic successor recipe changed')
    return recipe['path'], _stage(recipe['path'], declared['sft_stage']['id'])[0]


def prepare_window(config, declared, state, state_path, directory, *, protected=None, index=_index, log=print):
    """The recipe for SFT window ``state['interleave']['sft_windows']`` (None: the controller's own) after admitting
    every pending memetic package. Returns (recipe path or None, the SFT stage, admission record ref or None)."""
    interleave = state['interleave']
    window = interleave['sft_windows']
    ledger = interleave.get('memetic_distillation')
    if ledger and ledger['admissions'] and ledger['admissions'][-1]['window'] == window:
        # A retried boundary: this window's admission is already committed; reuse it exactly.
        record = ledger['admissions'][-1]
        if sha256_file_hex_cached(record['recipe']['path']) != record['recipe']['sha256']:
            raise ValueError('committed memetic admission recipe changed')
        return record['recipe']['path'], _stage(record['recipe']['path'], declared['sft_stage']['id'])[0], record['record']
    base_path, base_stage = effective_sft_stage(config, declared, state)
    controller_out = config.get('out')
    queued = _queued(controller_out) if controller_out else []
    rows_ledger = (ledger or {}).get('rows') or {}
    packages_ledger = (ledger or {}).get('packages') or {}
    candidates = [item for item in queued if Path(item['package']).is_file()
                  and packages_ledger.get(sha256_file_hex_cached(item['package']), {}).get('status')
                  in (None, 'partial')]
    # The lineage's own rows (self-distillation, memetic guidance) take the share cap first; prepared static sources
    # fill what is left (stable within each group).
    candidates.sort(key=lambda item: item.get('family') == STATIC_FAMILY)
    if not candidates:
        return base_path, base_stage, None

    from ..common.custody import ProtectedWorlds
    guard = ProtectedWorlds.load() if protected is None else protected
    sft_id = declared['sft_stage']['id']
    base_recipe = base_path or config['production_recipe']
    _, base_inputs, base_parameters = _stage(base_recipe, sft_id)
    geometry = _geometry_identity(base_parameters)
    ancestry = lineage_checkpoints(config, state)
    predecessor = {'path': base_inputs['native_graphs']['path'], 'sha256': base_inputs['native_graphs']['sha256']}
    ledger = copy.deepcopy(ledger) if ledger else {'schema': SCHEMA, 'recipe': None, 'packages': {}, 'rows': {},
                                                   'admissions': []}
    decisions, chosen = [], []  # per package: (item, receipt, ref, [admitted rows])
    for item in candidates:
        receipt_ref = _ref(item['package'])
        receipt = json.loads(Path(receipt_ref['path']).read_text())
        reason = _check_receipt(receipt, receipt_ref['path'])
        if reason is None and geometry is not None and (
                receipt.get('tokenizer_fingerprint') != geometry.get('tokenizer')
                or list(receipt.get('role_ids') or []) != list(geometry.get('roles') or [])):
            reason = 'package tokenizer/role ids differ from the joint SFT stage geometry'
        if reason is None and receipt['family'] != STATIC_FAMILY:
            foreign = sorted(set(receipt['episode_actors']) - ancestry)
            if foreign:
                reason = 'episode actors are not checkpoints of this lineage: ' + ', '.join(foreign)
        indexed = None
        if reason is None:
            indexed = index(receipt['source']['path'], receipt['tokenizer_fingerprint'], receipt['role_ids'])
            if [(entry['id'], entry['binding']) for entry in indexed.entries] != [
                    (row['id'], row['binding']) for row in receipt['rows']]:
                reason = 'package source does not index as exactly its receipt rows'
        if reason is not None:
            ledger['packages'][receipt_ref['sha256']] = {'path': receipt_ref['path'], 'status': 'refused',
                                                         'reason': reason, 'window': window}
            decisions.append({'package': receipt_ref, 'refused': reason})
            log(json.dumps({'event': 'memetic_package_refused', 'package': receipt_ref['path'], 'reason': reason}))
            continue
        raw = _raw_rows(receipt['source']['path'])
        candidates_rows = []
        for row in receipt['rows']:
            previous = rows_ledger.get(row['id'])
            if previous is not None and previous['disposition'] != 'deferred':
                continue
            if previous is not None and previous['package'] != receipt_ref['sha256']:
                continue  # deferred under another package: that package retries it
            _, graph = raw[row['id']]
            if graph.get('split') != 'train':
                why = 'not a train row'
            elif guard.related(graph):
                why = 'related to a protected world in the current protection union'
            elif any(row['id'] == other['id'] for _, _, _, rows in chosen for other in rows):
                why = 'duplicate row id in another pending package'
            else:
                why = None
            if why is not None:
                ledger['rows'][row['id']] = {'package': receipt_ref['sha256'], 'disposition': 'excluded',
                                             'reason': why, 'window': window}
                continue
            candidates_rows.append(row)
        chosen.append((item, receipt, receipt_ref, candidates_rows))

    # Share cap over the successor: (prior + new) x bound <= share x (rows + new).
    share = declared['parameters'].get('distillation_max_share', DEFAULT_MAX_SHARE)
    family_scale = declared['parameters'].get('distillation_family_scale', DEFAULT_FAMILY_SCALE)
    pending_rows = [row for _, _, _, rows in chosen for row in rows]
    admitted, deferred = [], []
    if pending_rows:
        prior_ids = _prior_ids(base_inputs, predecessor, index, chosen[0][1])
        clashes = [row for row in pending_rows if row['id'] in prior_ids]
        for row in clashes:
            package = next(ref for _, _, ref, rows in chosen if row in rows)
            ledger['rows'][row['id']] = {'package': package['sha256'], 'disposition': 'excluded',
                                         'reason': 'row id already in the joint SFT source', 'window': window}
        pending_rows = [row for row in pending_rows if row['id'] not in prior_ids]
        prior_memetic = sum(1 for value in ledger['rows'].values() if value['disposition'] == 'admitted')
        bound = _bound_scale(base_inputs, family_scale)
        total = len(prior_ids)
        if bound <= share:
            budget = len(pending_rows)  # the family's mass share cannot exceed the cap at any row count
        else:
            budget = max(0, math.floor((share * total - prior_memetic * bound) / (bound - share) + 1e-9))
        admitted, deferred = pending_rows[:budget], pending_rows[budget:]
        cap = {'max_share': share, 'bound_scale': bound, 'predecessor_rows': total,
               'prior_memetic_rows': prior_memetic, 'budget_rows': budget}
    else:
        cap = {'max_share': share}
    admitted_ids = {row['id'] for row in admitted}
    for row in deferred:
        package = next(ref for _, _, ref, rows in chosen if row in rows)
        ledger['rows'][row['id']] = {'package': package['sha256'], 'disposition': 'deferred',
                                     'reason': 'family share cap', 'window': window}

    target = Path(directory) / 'memetic-admission'
    if target.exists():
        shutil.rmtree(target)  # an uncommitted earlier attempt: nothing references it
    leaves, family_rows, package_records = [], [], []
    for item, receipt, receipt_ref, rows in chosen:
        mine = [row for row in rows if row['id'] in admitted_ids]
        if mine:
            raw = _raw_rows(receipt['source']['path'])
            if [row['id'] for row in mine] == list(raw):
                leaf_path = Path(receipt['source']['path'])
            else:
                target.mkdir(parents=True, exist_ok=True)
                leaf_path = target / f'admitted-{receipt_ref["sha256"][:16]}.jsonl'
                leaf_path.write_text(''.join(raw[row['id']][0] for row in mine))
            leaf = _ref(leaf_path)
            indexed = index(leaf['path'], receipt['tokenizer_fingerprint'], receipt['role_ids'])
            if [(e['id'], e['binding']) for e in indexed.entries] != [(row['id'], row['binding']) for row in mine]:
                raise ValueError('admitted memetic leaf does not index as exactly its admitted rows')
            leaves.append(leaf)
            family_rows.extend({'id': row['id'], 'binding': row['binding'], 'family': receipt['family']}
                               for row in mine)
            for row in mine:
                ledger['rows'][row['id']] = {'package': receipt_ref['sha256'], 'disposition': 'admitted',
                                             'window': window, 'leaf_sha256': leaf['sha256']}
        statuses = [ledger['rows'].get(row['id'], {}).get('disposition') for row in receipt['rows']]
        status = 'partial' if 'deferred' in statuses else 'admitted' if 'admitted' in statuses else 'consumed-none-admitted'
        ledger['packages'][receipt_ref['sha256']] = {'path': receipt_ref['path'], 'status': status, 'window': window}
        package_records.append({'package': receipt_ref, 'status': status, 'episode_actors': receipt['episode_actors'],
                                'family': receipt['family'], 'source_provenance': receipt.get('source_provenance'),
                                'package_protection_digest': receipt.get('protection_digest'),
                                'admitted': [row['id'] for row in mine],
                                'excluded': {row['id']: ledger['rows'][row['id']]['reason'] for row in receipt['rows']
                                             if ledger['rows'].get(row['id'], {}).get('disposition') == 'excluded'},
                                'deferred': [row['id'] for row in receipt['rows']
                                             if ledger['rows'].get(row['id'], {}).get('disposition') == 'deferred']})
    record = {'schema': SCHEMA, 'window': window, 'controller_round': state.get('round', 0), 'stage': sft_id,
              'packages': package_records, 'refused': [item for item in decisions if 'refused' in item],
              'protection_digest': guard.digest, 'share_cap': cap,
              'actor_provenance': {'kind': PROVENANCE, 'lineage_checkpoints': sorted(ancestry), 'teacher': None},
              'exposure': 'none claimed; added rows earn exposure only through SGD in the window'}
    recipe_path = None
    if leaves:
        target.mkdir(parents=True, exist_ok=True)
        successor_path = target / 'train-sources.json'
        write_json_durable(successor_path, {'schema': 'natlang.native-graph-sources/1', 'sources': [
            {'kind': 'native-graphs', 'path': predecessor['path'], 'sha256': predecessor['sha256']},
            *({'kind': 'native-graphs', **leaf, 'training_priority': 1} for leaf in leaves)]})
        successor = _ref(successor_path)
        ce = None
        if base_inputs.get('ce_supervision_policy'):
            ce, _ = _ce_update(base_inputs['ce_supervision_policy'], successor, leaves, family_rows, target,
                                         family_scale=family_scale)
        append, settled = _write_recipes(base_recipe, sft_id, successor, ce, target)
        _, append_inputs, _ = _stage(append, sft_id)
        from ..train.recipe import _ce_supervision_input_continuation, _data_continuation
        if not (_ce_supervision_input_continuation(base_inputs, append_inputs) if ce is not None
                else _data_continuation(base_inputs, append_inputs)):
            raise ValueError('memetic append is not a declared data continuation of the joint SFT stage')
        if ce is not None:
            from ..train.native_resident_ce import extension_policy
            extension_policy(base_inputs, append_inputs)
        record.update(predecessor=predecessor, successor=successor, leaves=leaves, ce_policy=ce,
                      recipes={'append': _ref(append), 'settled': _ref(settled)},
                      rows_added=len(family_rows))
        recipe_path = str(append)
        ledger['recipe'] = _ref(settled)
    target.mkdir(parents=True, exist_ok=True)
    record_path = target / 'admission.json'
    write_json_durable(record_path, record)
    record_ref = _ref(record_path)
    if recipe_path is not None:
        ledger['admissions'].append({'window': window, 'recipe': _ref(recipe_path), 'record': record_ref})
        interleave.pop('sft_complete', None)  # new rows reopen the joint SFT stage
    else:
        ledger.setdefault('decisions', []).append({'window': window, 'record': record_ref})
    interleave['memetic_distillation'] = ledger
    write_json_durable(state_path, state)
    log(json.dumps({'event': 'memetic_distillation_admission', 'window': window, 'rows_added': len(family_rows),
                    'deferred': len(deferred), 'refused': len(record['refused'])}))
    if recipe_path is None:
        return base_path, base_stage, record_ref
    return recipe_path, _stage(recipe_path, sft_id)[0], record_ref


def _prior_ids(inputs, predecessor, index, receipt):
    """Row ids of the predecessor: from its exact logical-family map when the CE objective binds one (no reparse of a
    large source), else from the source index (a cache hit for the trainer's own parse)."""
    reference = inputs.get('ce_supervision_policy')
    if reference:
        balance = (json.loads(Path(reference['path']).read_text()).get('objective') or {}).get('family_balance') or {}
        if balance.get('key') == 'logical-family':
            from ..train.native_logical_families import _verified_bytes
            value = json.loads(_verified_bytes(balance['map'])[1])
            if value.get('source_sha256') == predecessor['sha256']:
                ids = [row['id'] for row in value['rows']]
                if len(set(ids)) != len(ids):
                    raise ValueError('logical family map repeats a row id')
                return set(ids)
    return set(index(predecessor['path'], receipt['tokenizer_fingerprint'], receipt['role_ids']).bindings)


def _bound_scale(inputs, family_scale):
    """The largest CE scale a memetic row can receive under the current objective (1 without family balance)."""
    reference = inputs.get('ce_supervision_policy')
    if not reference:
        return 1.0
    balance = (json.loads(Path(reference['path']).read_text()).get('objective') or {}).get('family_balance')
    if not isinstance(balance, dict):
        return 1.0
    if balance.get('key') == 'logical-family' and balance.get('normalization') == 'declared/1':
        return float(family_scale)
    return float(balance['max_scale'])
