"""Exact objective preservation for a resident's additive source transaction."""
from __future__ import annotations

import copy
from pathlib import Path

from .native_supervision import load_policy


def _public(policy):
    return {key:value for key,value in policy.items() if not key.startswith('_')}


def extension_policy(old, new):
    """Load a complete, additive CE declaration without changing prior intent.

    Normal recipe/source/index guards still authenticate the actual append.
    None means the CE binding did not change, not permission to drop a policy.
    """
    role = 'ce_supervision_policy'
    before, after = old.get(role), new.get(role)
    if before == after:
        return None
    if old.get('native_graphs') == new.get('native_graphs'):
        raise ValueError('resident CE coverage extension requires an actual source append')
    if not before or not after:
        raise ValueError('resident CE append requires both prior and successor policies')
    from .recipe import _ce_supervision_input_continuation, native_source_dependencies
    if not _ce_supervision_input_continuation(old, new):
        raise ValueError('resident CE append is not a declared data continuation')
    prior, policy = load_policy(before), load_policy(after)
    a, b = _public(prior), _public(policy)
    excluded = {'sources', 'source_contract', 'inventory', 'first_row_observations'}
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
    leaves = native_source_dependencies(path)
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
    if not old_active.keys() <= declared.keys():
        raise ValueError('resident CE contract removed prior active source custody')
    old_observations, new_observations = a.get('first_row_observations', []), b.get('first_row_observations', [])
    if (not isinstance(old_observations, list) or not isinstance(new_observations, list)
            or any(item not in new_observations for item in old_observations)):
        raise ValueError('resident CE append removed a prior first-row observation')
    for item in new_observations:
        if (not isinstance(item, dict) or not isinstance(item.get('path'), str)
                or (str(Path(item['path']).resolve()), item.get('sha256')) not in declared):
            raise ValueError('resident CE observation is outside the exact active source DAG')
    for policy, reference in ((a, old['native_graphs']), (b, train)):
        top_inventory = policy.get('inventory')
        if top_inventory is not None and (len(top_inventory) != 1 or
                checked(top_inventory[0]) != checked(reference) or top_inventory[0]['sha256'] != reference['sha256']):
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
