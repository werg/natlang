"""Exact objective preservation for a resident's additive source transaction."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .native_supervision import load_policy


def _public(policy):
    return {key:value for key,value in policy.items() if not key.startswith('_')}


def _validate_historical_observations(observations, old_observations, declared, old_active,
                                      correction_decl, correction, checked):
    """Keep the separate target-mask and prior-quarantine custody chains exact."""
    remaining = list(observations)
    if correction_decl is not None:
        old_leaf = correction_decl.get('source_leaf_before') or {}
        new_leaf = correction_decl.get('source_leaf_after') or {}
        matched = [item for item in remaining if item.get('path') == old_leaf.get('path')
                   and item.get('sha256') == old_leaf.get('sha256')]
        if matched and (len(matched) != 1 or matched[0] not in old_observations
                or (str(Path(new_leaf.get('path', '')).resolve()), new_leaf.get('sha256')) not in declared):
            raise ValueError('resident CE historical observation is not the authenticated target-mask replacement')
        for item in matched:
            checked(item)
            remaining.remove(item)
    if remaining:
        if not isinstance(correction, dict) or not isinstance(correction.get('quality_receipt'), dict):
            raise ValueError('resident CE historical observations lack their quarantine proof')
        receipt = json.loads(checked(correction['quality_receipt']).read_text())
        if (receipt.get('schema') != 'natlang.nested-semantic-child-quarantine-receipt/1'
                or receipt.get('source_manifest_sha256') != correction.get('historical_source_sha256')
                or receipt.get('successor_manifest_sha256') != correction.get('active_source_sha256')):
            raise ValueError('resident CE historical observation quarantine identities differ')
        replacements = receipt.get('source_replacements') or {}
        for item in remaining:
            replacement = replacements.get(item['path'])
            if (item not in old_observations or not isinstance(replacement, dict)
                    or replacement.get('source_sha256') != item.get('sha256')
                    or (str(Path(replacement.get('successor_path', '')).resolve()),
                        replacement.get('successor_sha256')) not in old_active
                    or (str(Path(replacement.get('successor_path', '')).resolve()),
                        replacement.get('successor_sha256')) not in declared):
                raise ValueError('resident CE historical observation is not an authenticated active replacement')
            checked(item)


def _validate_decision_factor_evidence(policy, prior_objective, candidate_objective, train_binding):
    """Authenticate a new exact target-span objective factor before a resident transition."""
    import json
    from ..common.hashing import sha256_file_hex
    from .own_output_supervision import identity, policy_objective

    evidence = policy.get('decision_factor_evidence')
    source_manifest = evidence.get('source_manifest') if isinstance(evidence, dict) else None
    if (not isinstance(evidence, dict)
            or set(evidence) != {'schema', 'predecessor_objective_sha256', 'source_manifest', 'factors'}
            or evidence.get('schema') != 'natlang.own-output-decision-factor-evidence/1'
            or evidence.get('predecessor_objective_sha256') != identity(policy_objective(prior_objective))['sha256']
            or not isinstance(source_manifest, dict)
            or source_manifest.get('sha256') != train_binding.get('sha256')
            or Path(source_manifest.get('path', '')).resolve() != Path(train_binding['path']).resolve()
            or not isinstance(evidence.get('factors'), list)):
        raise ValueError('new decision factors lack their exact predecessor and source evidence')
    manifest_path = Path(train_binding['path']).resolve(strict=True)
    if sha256_file_hex(manifest_path) != train_binding['sha256']:
        raise ValueError('decision factor evidence source manifest changed')
    from .recipe import native_graph_payload_dependencies
    sources = {source.get('sha256') for source in native_graph_payload_dependencies(manifest_path)}
    old_factors = (prior_objective.get('objective') or {}).get('decision_factors', [])
    new_factors = (candidate_objective.get('objective') or {}).get('decision_factors', [])
    additions = new_factors[len(old_factors):]
    if (new_factors[:len(old_factors)] != old_factors or not additions
            or len(evidence['factors']) != len(additions)):
        raise ValueError('decision factor evidence does not describe a pure additive objective extension')
    for spec, proof in zip(additions, evidence['factors'], strict=True):
        if (not isinstance(proof, dict) or set(proof) != {'spec', 'audit', 'observed_occurrences'}
                or proof['spec'] != spec or type(proof['observed_occurrences']) is not int
                or proof['observed_occurrences'] <= 0):
            raise ValueError('decision factor proof does not bind the exact factor and observed target count')
        audit_binding = proof['audit']
        if (not isinstance(audit_binding, dict) or set(audit_binding) not in ({'path', 'sha256'},
                {'path', 'sha256', 'bytes'}) or ('bytes' in audit_binding and
                type(audit_binding.get('bytes')) is not int)):
            raise ValueError('decision factor audit must bind path and SHA-256, with optional byte size')
        audit_path = Path(audit_binding['path']).resolve(strict=True)
        if (('bytes' in audit_binding and audit_path.stat().st_size != audit_binding['bytes'])
                or sha256_file_hex(audit_path) != audit_binding['sha256']):
            raise ValueError('decision factor audit bytes differ from the evidence binding')
        audit = json.loads(audit_path.read_bytes())
        if (audit.get('schema') != 'natlang.big2-program-template-repetition-audit/1'
                or audit.get('source', {}).get('sha256') not in sources
                or spec.get('row_kind') != 'source_pairing' or spec.get('node_name') != 'producer'):
            raise ValueError('decision factor audit is not a source-bound Big2 producer repetition audit')
        matches = [item for item in audit.get('top_producer_templates', [])
                   if item.get('key') == spec.get('target_sha256')]
        if len(matches) != 1 or matches[0].get('count') != proof['observed_occurrences']:
            raise ValueError('decision factor does not match the exact audited producer target and frequency')


def extension_policy(old, new, *, prior_train=None, candidate_train=None):
    """Load a complete, additive CE declaration without changing prior intent.

    Normal recipe/source/index guards still authenticate the actual append.
    None means the CE binding did not change, not permission to drop a policy.
    """
    role = 'ce_supervision_policy'
    before, after = old.get(role), new.get(role)
    if before == after:
        return None
    same_source = old.get('native_graphs') == new.get('native_graphs')
    if not before or not after:
        raise ValueError('resident CE append requires both prior and successor policies')
    from .recipe import _ce_supervision_input_continuation, native_graph_payload_dependencies
    if not _ce_supervision_input_continuation(old, new):
        raise ValueError('resident CE append is not a declared data continuation')
    prior, policy = load_policy(before), load_policy(after)
    a, b = _public(prior), _public(policy)
    excluded = {'sources', 'source_contract', 'inventory', 'first_row_observations', 'prior_inventory_correction',
                'decision_factor_evidence', 'source_correction'}
    source_correction = b.get('source_correction')
    prior_source_correction = a.get('source_correction')
    if prior_source_correction is not None and source_correction != prior_source_correction:
        raise ValueError('resident CE source correction history was changed or removed')
    correction_decl = None
    if source_correction is not None:
        if (not isinstance(source_correction, dict)
                or set(source_correction) not in ({'path', 'sha256'}, {'path', 'sha256', 'bytes'})):
            raise ValueError('resident source correction reference is malformed')
        from .native_source_correction import source_correction_dependencies
        source_correction_dependencies(source_correction)
        correction_path = Path(source_correction['path']).resolve(strict=True)
        correction_doc = json.loads(correction_path.read_bytes())
        correction_decl = correction_doc.get('target_mask_correction')
        if correction_decl is None:
            raise ValueError('resident source correction has no target-mask transition')
        if correction_doc.get('schema') != 'natlang.native-source-correction/1':
            raise ValueError('resident source correction schema differs')
        if prior_source_correction is None and (
                correction_doc.get('prior_source', {}).get('sha256') != old.get('native_graphs', {}).get('sha256')
                or correction_doc.get('current_source', {}).get('sha256') != new.get('native_graphs', {}).get('sha256')):
            raise ValueError('resident source correction does not bind the exact predecessor and successor manifests')
        if (source_correction.get('sha256') != hashlib.sha256(correction_path.read_bytes()).hexdigest()
                or ('bytes' in source_correction and source_correction['bytes'] != correction_path.stat().st_size)):
            raise ValueError('resident source correction receipt changed')
    old_objective,new_objective=a.get('objective'),b.get('objective')
    declared_transition = ((old_objective or {}).get('family_balance',{}).get('normalization') == 'bounded-mean-one/1' and (new_objective or {}).get('family_balance',{}).get('normalization') == 'declared/1')
    if same_source and not declared_transition:
        raise ValueError('same-source CE change requires an explicit measured family-scale transition')
    if old_objective != new_objective:
        from .own_output_supervision import identity,policy_objective
        from .native_logical_families import _verified_bytes,load_map
        left,right=copy.deepcopy(old_objective),copy.deepcopy(new_objective)
        old_family=(left or {}).get('family_balance') or {};new_family=(right or {}).get('family_balance') or {}
        if old_family.get('key')!='logical-family' or new_family.get('key')!='logical-family':
            raise ValueError('resident CE objective change is not a logical-family map extension')
        old_map=old_family.pop('map',None);new_map=new_family.pop('map',None)
        old_factors = left.pop('decision_factors', [])
        new_factors = right.pop('decision_factors', [])
        factor_addition = old_factors != new_factors
        if factor_addition:
            _validate_decision_factor_evidence(policy, prior, policy, new['native_graphs'])
            if new_factors[:len(old_factors)] != old_factors:
                raise ValueError('resident decision factors must be an append-only exact objective extension')
        elif a.get('decision_factor_evidence') != b.get('decision_factor_evidence'):
            raise ValueError('resident source append changed historical decision-factor evidence')
        if same_source and old_map != new_map:
            raise ValueError('same-source objective transition cannot change family labels')
        # A declared objective transition may replace row-count balancing with
        # explicit measured own-output family scales. Keep all other objective
        # settings and the cap identical; load_policy validates the scales.
        if old_family.get('normalization') == 'bounded-mean-one/1' and new_family.get('normalization') == 'declared/1':
            old_family.pop('normalization');new_family.pop('normalization')
            new_family.pop('scales',None)
        elif not same_source and old_family.get('normalization') == new_family.get('normalization') == 'declared/1':
            old_scales=old_family.pop('scales',None);new_scales=new_family.pop('scales',None)
            if not isinstance(old_scales,dict) or not isinstance(new_scales,dict) or set(old_scales)!=set(new_scales):
                raise ValueError('resident declared-scale refresh changed the family set')
        if left!=right or not old_map or not new_map or (b.get('objective_transition') or {}).get('from')!=identity(policy_objective(prior))['sha256']:
            raise ValueError('resident logical-family extension changed objective law or lacks its exact predecessor')
        previous=json.loads(_verified_bytes(old_map)[1]);current=json.loads(_verified_bytes(new_map)[1])
        if (previous.get('schema')!='natlang.logical-family-map/1' or current.get('schema')!='natlang.logical-family-map/1'
                or previous.get('source_sha256')!=old['native_graphs']['sha256']
                or current.get('source_sha256')!=new['native_graphs']['sha256']
                or not isinstance(previous.get('rows'),list) or not isinstance(current.get('rows'),list)
                or len(current['rows']) < len(previous['rows'])):
            raise ValueError('resident logical-family extension changed prior source row families')
        if correction_decl is not None and prior_source_correction is None:
            # The map proof alone is not enough: bind its exposure ledger to
            # the exact in-place row and every graph appended in this source
            # transaction. This prevents an unrelated mask review from
            # authorizing changed supervision or hidden additions.
            from ..common.jsonio import open_jsonl_text
            from .native_source_correction import _origins
            rows_path = Path(correction_doc['row_bindings_path'])
            if not rows_path.is_absolute():
                rows_path = correction_path.parent / rows_path
            ledger = []
            with open_jsonl_text(rows_path) as stream:
                ledger = [json.loads(line) for line in stream if line.strip()]
            if (not isinstance(previous.get('rows'), list) or not isinstance(current.get('rows'), list)
                    or len(previous['rows']) > len(current['rows'])):
                raise ValueError('target-mask CE transition lacks complete ordered logical-family rows')
            before_ids = [row.get('id') for row in previous['rows']]
            after_ids = [row.get('id') for row in current['rows']]
            before_origin = _origins(correction_doc['prior_source'], before_ids)
            after_origin = _origins(correction_doc['current_source'], after_ids)
            target_id = correction_decl['graph_id']
            target_rows = [row for row in ledger if row.get('disposition') == 'eligibility-decreased']
            added_rows = [row for row in ledger if row.get('disposition') == 'added']
            expected_added = current['rows'][len(previous['rows']):]
            if (len(target_rows) != 1 or target_rows[0].get('id') != target_id
                    or target_id not in before_origin or target_id not in after_origin
                    or target_rows[0].get('old_binding') != correction_decl.get('binding_before')
                    or target_rows[0].get('new_binding') != correction_decl.get('binding_after')
                    or target_rows[0].get('old_source_sha256') != before_origin[target_id]
                    or target_rows[0].get('new_source_sha256') != after_origin[target_id]
                    or target_rows[0].get('old_communication') is not False
                    or [row.get('id') for row in added_rows] != [entry['id'] for entry in expected_added]
                    or any(row.get('new_binding') != entry['binding']
                           or row.get('new_source_sha256') != after_origin.get(entry['id'])
                           for row, entry in zip(added_rows, expected_added, strict=True))
                    or len(ledger) != 1 + len(expected_added)):
                raise ValueError('resident source correction ledger does not bind the exact mask and appended rows')
        prior_map_rows = previous['rows']
        current_prefix = current['rows'][:len(prior_map_rows)]
        if current_prefix != prior_map_rows:
            if correction_decl is None:
                raise ValueError('resident logical-family extension changed prior source row families')
            changed = [index for index, (left_row, right_row) in enumerate(
                zip(prior_map_rows, current_prefix, strict=True)) if left_row != right_row]
            declaration = correction_decl
            if len(changed) != 1:
                raise ValueError('resident source correction changed more than its exact target-mask row')
            index = changed[0]
            before_row, after_row = prior_map_rows[index], current_prefix[index]
            if (before_row.get('id') != declaration.get('graph_id')
                    or after_row.get('id') != declaration.get('graph_id')
                    or before_row.get('binding') != declaration.get('binding_before')
                    or after_row.get('binding') != declaration.get('binding_after')
                    or before_row.get('family') != after_row.get('family')):
                raise ValueError('resident source correction changed an unrelated logical-family row')
        if (prior_train is None)!=(candidate_train is None):raise ValueError('resident logical-family source validation needs both sources')
        if prior_train is not None:
            load_map(old_map,prior_train);load_map(new_map,candidate_train)
        excluded |= {'objective','objective_transition'}
    if ({k:v for k,v in a.items() if k not in excluded} !=
            {k:v for k,v in b.items() if k not in excluded}):
        raise ValueError('resident CE append changed prior masks, held declarations or objective settings')
    previous = {entry['sha256']:entry for entry in a['sources']}
    current = {entry['sha256']:entry for entry in b['sources']}
    if any(current.get(key) != value for key,value in previous.items()):
        raise ValueError('resident CE append changed or removed a prior source coefficient/encoding')
    ac, bc = a.get('source_contract'), b.get('source_contract')
    if not isinstance(ac, dict) or not isinstance(bc, dict):
        raise ValueError('resident CE append requires complete source contracts')
    inventory_fields = {'inventory', 'active_sources', 'active_source_count'}
    if ({k:v for k,v in ac.items() if k not in inventory_fields} !=
            {k:v for k,v in bc.items() if k not in inventory_fields}):
        raise ValueError('resident CE append changed source-contract semantics')
    train = new['native_graphs']
    from .native_resident import bound
    def checked(reference):
        p = bound(reference)
        if 'bytes' in reference and (type(reference['bytes']) is not int
                or reference['bytes'] < 0 or p.stat().st_size != reference['bytes']):
            raise ValueError('resident CE source binding byte count differs')
        return p
    path = checked(train)
    leaves = native_graph_payload_dependencies(path)
    if not leaves:
        leaves = [{'path':str(path), 'sha256':train['sha256']}]
    leaves = [ref for ref in leaves if Path(ref['path']).suffix != '.json']
    def identities(values):
        result = {}
        for ref in values:
            # Authenticate supplied optional bytes too; do not decode graph rows.
            p = checked(ref)
            key = (str(p), ref['sha256'])
            if key in result:
                raise ValueError('resident CE contract repeats a source alias')
            result[key] = ref
        return result
    actual, declared = identities(leaves), identities(bc.get('active_sources', []))
    if (not declared.keys() <= actual.keys()
            or {key[1] for key in actual} != {key[1] for key in declared}
            or bc.get('active_source_count') != len({key[1] for key in actual})):
        raise ValueError('resident CE contract does not bind the complete appended source DAG')
    old_active = identities(ac.get('active_sources', []))
    missing_active = old_active.keys() - declared.keys()
    if missing_active:
        if correction_decl is None:
            raise ValueError('resident CE contract removed prior active source custody')
        old_leaf = correction_decl.get('source_leaf_before') or {}
        new_leaf = correction_decl.get('source_leaf_after') or {}
        expected_old = (str(Path(old_leaf.get('path', '')).resolve()), old_leaf.get('sha256'))
        expected_new = (str(Path(new_leaf.get('path', '')).resolve()), new_leaf.get('sha256'))
        old_matches = {key for key in missing_active if key == expected_old}
        new_matches = {key for key in declared.keys() - old_active.keys() if key == expected_new}
        if len(missing_active) != 1 or len(old_matches) != 1 or len(new_matches) != 1:
            raise ValueError('resident CE correction does not replace exactly its reviewed source leaf')
    old_observations, new_observations = a.get('first_row_observations', []), b.get('first_row_observations', [])
    if not isinstance(old_observations, list) or not isinstance(new_observations, list):
        raise ValueError('resident CE first-row observations are malformed')
    removed_observations = [item for item in old_observations if item not in new_observations]
    if removed_observations:
        if correction_decl is None:
            raise ValueError('resident CE append removed a prior first-row observation')
        old_leaf = correction_decl.get('source_leaf_before') or {}
        if (len(removed_observations) != 1
                or removed_observations[0].get('path') != old_leaf.get('path')
                or removed_observations[0].get('sha256') != old_leaf.get('sha256')):
            raise ValueError('resident CE correction removed an unrelated source observation')
    historical_observations = []
    for item in new_observations:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str):
            raise ValueError('resident CE observation is malformed')
        if (str(Path(item['path']).resolve()), item.get('sha256')) not in declared:
            historical_observations.append(item)
    prior_inventory = a.get('inventory')
    prior_matches = (prior_inventory is None or (len(prior_inventory) == 1 and
        checked(prior_inventory[0]) == checked(old['native_graphs']) and
        prior_inventory[0]['sha256'] == old['native_graphs']['sha256']))
    correction = b.get('prior_inventory_correction')
    if not prior_matches:
        # The earned quality-quarantine transition kept a historical top-level
        # inventory even though its active contract and family map moved. The
        # exact quarantine receipt must authenticate both source identities;
        # this exception never credits rows or changes the source contract.
        if (not isinstance(correction, dict) or correction.get('schema') != 'natlang.resident-ce-prior-inventory-correction/1'
                or correction.get('prior_policy_sha256') != before['sha256']
                or len(prior_inventory) != 1
                or correction.get('historical_source_sha256') != prior_inventory[0]['sha256']
                or correction.get('active_source_sha256') != old['native_graphs']['sha256']):
            raise ValueError('resident CE prior inventory mismatch lacks its exact correction')
        receipt = json.loads(checked(correction['quality_receipt']).read_text())
        prior_map = (a.get('objective') or {}).get('family_balance', {}).get('map') or {}
        from .native_logical_families import _verified_bytes
        if correction.get('prior_family_map_sha256') != prior_map.get('sha256'):
            raise ValueError('resident CE prior inventory correction lacks its exact family map')
        prior_map_doc = json.loads(_verified_bytes(prior_map)[1])
        if (prior_map_doc.get('schema') != 'natlang.logical-family-map/1'
                or prior_map_doc.get('source_sha256') != old['native_graphs']['sha256']):
            raise ValueError('resident CE prior inventory correction family map differs from its active source')
        prior_contract = ac.get('inventory', [])
        if (receipt.get('schema') != 'natlang.nested-semantic-child-quarantine-receipt/1'
                or receipt.get('source_manifest_sha256') != prior_inventory[0]['sha256']
                or receipt.get('successor_manifest_sha256') != old['native_graphs']['sha256']
                or len(prior_contract) != 1 or checked(prior_contract[0]) != checked(old['native_graphs'])
                or prior_contract[0]['sha256'] != old['native_graphs']['sha256']):
            raise ValueError('resident CE prior inventory correction does not bind the actual quarantine transition')
    elif correction is not None and correction != a.get('prior_inventory_correction'):
        raise ValueError('resident CE prior inventory correction is not applicable')
    if historical_observations:
        _validate_historical_observations(historical_observations, old_observations, declared, old_active,
                                          correction_decl, correction, checked)
    top_inventory = b.get('inventory')
    if top_inventory is not None and (len(top_inventory) != 1 or
            checked(top_inventory[0]) != checked(train) or top_inventory[0]['sha256'] != train['sha256']):
        raise ValueError('resident CE inventory differs from its exact train input')
    inventory = bc.get('inventory', [])
    if len(inventory) != 1 or checked(inventory[0]) != path or inventory[0]['sha256'] != train['sha256']:
        raise ValueError('resident CE contract inventory differs from its train input')
    held = {ref['sha256'] for ref in b.get('held_sources', []) + b.get('retained_held_sources', [])}
    if held & {key[1] for key in actual}:
        raise ValueError('resident CE append overlaps held declarations')
    for _, digest in actual:
        declaration = current.get(digest)
        if not declaration or declaration['encoding'] != 'native-row-objectives-v1':
            raise ValueError('resident CE append lacks complete per-row source intent')
        if digest not in previous and declaration['context_weight'] != bc['default_raw_context_weight']:
            raise ValueError('resident CE append changed the declared new-source coefficient')
    return policy


def adopt_policy_update(update, *, current_binding, args, checkpoint_options, global_step):
    """Install a policy loaded before the resident commit, before another batch."""
    if update is None:
        return None
    binding, policy = update
    from .own_output_supervision import continuation_identity
    objective,transition=continuation_identity(checkpoint_options,policy)
    if transition is not None:
        transitions=list(checkpoint_options.get('supervision_objective_history') or [])
        transitions.append({**transition,'at_global_step':int(global_step),'weights_optimizer_rng_exposure':'preserved'})
        checkpoint_options['supervision_objective_history']=transitions
        checkpoint_options['supervision_objective']=objective
    reference = {key:binding[key] for key in ('path', 'sha256')}
    history = list(checkpoint_options.get('ce_supervision_policy_history') or [])
    history.append({'from':copy.deepcopy(current_binding), 'to':reference,
        'at_global_step':int(global_step), 'weights_optimizer_rng_exposure':'preserved',
        'held_ce':'original-corpus-weights', 'kind':'resident-additive-source-coverage'})
    args.ce_supervision_policy = Path(reference['path'])
    checkpoint_options.update(ce_supervision_policy=str(args.ce_supervision_policy),
        ce_supervision_policy_binding=reference, ce_supervision_policy_history=history,
        ce_supervision_active=True)
    return reference, policy


def require_loaded_policy(binding, reference):
    """The running objective must agree with its authoritative resident plan."""
    if (binding is None) != (reference is None) or (binding is not None and
            binding.get('sha256') != reference.get('sha256')):
        raise ValueError('loaded CE policy differs from the resident source plan')
