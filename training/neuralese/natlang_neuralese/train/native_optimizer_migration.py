"""Explicit native Muon→AdamW transition with honest new adaptive history."""
from __future__ import annotations

import copy
import torch

POLICY = 'native-ordinary-to-adamw-new-history-v1'
SCHEMA = 'natlang.native-optimizer-transition/1'


def validate_native_optimizer_migration(value):
    if value not in (None,POLICY):
        raise ValueError('native optimizer migration must explicitly declare '+POLICY)
    return value


def own_transition_history(value):
    """Own tiny immutable prior histories without retaining a full checkpoint mmap."""
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:own_transition_history(v) for k,v in value.items()}
    if isinstance(value,list):return [own_transition_history(v) for v in value]
    if isinstance(value,tuple):return tuple(own_transition_history(v) for v in value)
    return copy.deepcopy(value)


def migrate_native_ordinary_optimizer(optimizer,saved,policy):
    from .optim import FORMAT
    from .optim_restore import OptimizerRestoreError,port_saved_slot_names
    validate_native_optimizer_migration(policy)
    if policy is None:raise ValueError('native optimizer kind transition requires an explicit declared policy')
    if saved.get('format')!=FORMAT:raise OptimizerRestoreError('native algorithm transition requires PortMuonAdamW state')
    prior={x['name']:x for x in saved['schema']};current={x['name']:x for x in optimizer.schema}
    if len(prior)!=len(saved['schema']) or len(current)!=len(optimizer.schema):
        raise OptimizerRestoreError('native transition requires unique exact parameter names')
    removed=prior.keys()-current.keys()
    if removed:raise OptimizerRestoreError('native transition cannot remove existing optimizer parameters')
    changed={name for name in prior if prior[name]['optimizer']!=current[name]['optimizer']}
    if not changed:return saved
    if any(prior[name]['optimizer']!='muon' or current[name]['optimizer']!='adamw'
           or name not in optimizer._force_adamw_names or prior[name]['shape']!=current[name]['shape']
           for name in changed):
        raise OptimizerRestoreError('native transition permits only proven ordinary Muon to AdamW parameter routing')
    if optimizer.auxiliary is None:raise OptimizerRestoreError('native transition has no target AdamW optimizer')
    child=saved.get('muon')
    if child is None:raise OptimizerRestoreError('native transition lacks its actual source Muon state')
    source_names=port_saved_slot_names(saved,'muon',child)
    source_slots=[p for group in child['param_groups'] for p in group['params']]
    if set(child['state'])-set(source_slots):raise OptimizerRestoreError('source Muon state has unnamed orphan slots')
    params={optimizer._parameter_names[id(p)]:p for group in optimizer.auxiliary.param_groups for p in group['params']}
    defaults={optimizer._parameter_names[id(p)]:{k:v for k,v in group.items() if k!='params'}
              for group in optimizer.auxiliary.param_groups for p in group['params']}
    previous_adam=saved.get('adamw')
    adam_groups=[dict(g) for g in previous_adam['param_groups']] if previous_adam is not None else []
    adam_states=dict(previous_adam['state']) if previous_adam is not None else {}
    adam_names=port_saved_slot_names(saved,'adamw',previous_adam) if previous_adam is not None else []
    next_slot=max([p for group in adam_groups for p in group['params']]+[-1])+1
    muon_groups=[];muon_states={};muon_names=[];transitions=[];cursor=0
    for group in child['param_groups']:
        group_names=source_names[cursor:cursor+len(group['params'])];cursor+=len(group['params'])
        kept=[];new_groups=[]
        for slot,name in zip(group['params'],group_names):
            state=child['state'].get(slot,{})
            if name not in changed:
                kept.append(slot);muon_names.append(name)
                if slot in child['state']:muon_states[slot]=state
                continue
            if set(state)-{'momentum_buffer'}:
                raise OptimizerRestoreError('native transition encountered an unqualified source Muon state field')
            momentum=state.get('momentum_buffer')
            if momentum is not None and (not torch.is_tensor(momentum) or list(momentum.shape)!=prior[name]['shape']
                    or not momentum.is_floating_point() or not bool(torch.isfinite(momentum).all())):
                raise OptimizerRestoreError('native transition source Muon momentum has invalid shape or numeric values: '+name)
            parameter=params[name]
            options={**defaults[name],'params':[next_slot],'lr':group['lr'],
                     'weight_decay':group.get('weight_decay',0.)}
            if options.get('capturable') or options.get('fused'):
                step_device=parameter.device
            else:step_device=torch.device('cpu')
            adam_states[next_slot]={'step':torch.zeros((),dtype=torch.float32,device=step_device),
                'exp_avg':torch.zeros_like(parameter),'exp_avg_sq':torch.zeros_like(parameter)}
            if options.get('amsgrad'):adam_states[next_slot]['max_exp_avg_sq']=torch.zeros_like(parameter)
            new_groups.append(options);adam_names.append(name)
            transitions.append({'name':name,'source_contract':prior[name],
                'source_slot':slot,'source_group':own_transition_history(group),
                'source_group_parameter_names':list(group_names),'source_state':own_transition_history(state),
                'new_history':{'step':0,'first_moment':'zero','second_moment':'zero',
                    'historical_adam_variance_available':False,'historical_parameter_age_available':False}})
            next_slot+=1
        if kept:muon_groups.append({**group,'params':kept})
        adam_groups.extend(new_groups)
    if {x['name'] for x in transitions}!=changed:
        raise OptimizerRestoreError('native transition did not cover every changed parameter')
    result={**saved,'schema':[{**entry,'optimizer':'adamw'} if entry['name'] in changed else entry for entry in saved['schema']],
        'muon':{'state':muon_states,'param_groups':muon_groups} if muon_groups else None,
        'adamw':{'state':adam_states,'param_groups':adam_groups},
        'param_names':{**saved.get('param_names',{}),'adamw':adam_names},
        'native_optimizer_transitions':[*own_transition_history(saved.get('native_optimizer_transitions',[])),
            {'schema':SCHEMA,'policy':POLICY,'parameters':transitions}]}
    if muon_groups:result['param_names']['muon']=muon_names
    else:result['param_names'].pop('muon',None)
    return result


def native_optimizer_migration_option(config,name=None):
    base=config.get('native_optimizer_migration',(config.get('options') or {}).get('native_optimizer_migration'))
    if name is not None:base=config.get('models',{}).get(name,{}).get('native_optimizer_migration',base)
    return validate_native_optimizer_migration(base)


def validate_transition_history(history,schema):
    from .optim_restore import OptimizerRestoreError
    actual={item['name']:item for item in schema}
    if not isinstance(history,list):raise OptimizerRestoreError('native optimizer transition history must be an explicit record list')
    for record in history:
        if (not isinstance(record,dict) or record.get('schema')!=SCHEMA or record.get('policy')!=POLICY
                or not isinstance(record.get('parameters'),list) or not record['parameters']):
            raise OptimizerRestoreError('native optimizer transition provenance is malformed')
        names=set()
        for item in record['parameters']:
            name=item.get('name');contract=item.get('source_contract') or {};group=item.get('source_group') or {}
            slots=group.get('params') or [];bound=item.get('source_group_parameter_names') or []
            slot=item.get('source_slot');source=item.get('source_state')
            if (name in names or name not in actual or contract.get('name')!=name or contract.get('optimizer')!='muon'
                    or actual[name]['optimizer']!='adamw' or contract.get('shape')!=actual[name]['shape']
                    or len(slots)!=len(bound) or slot not in slots or bound[slots.index(slot)]!=name
                    or not isinstance(source,dict) or set(source)-{'momentum_buffer'}
                    or item.get('new_history')!={'step':0,'first_moment':'zero','second_moment':'zero',
                        'historical_adam_variance_available':False,'historical_parameter_age_available':False}):
                raise OptimizerRestoreError('native optimizer prior history lost its exact parameter binding')
            names.add(name);buffer=source.get('momentum_buffer')
            if buffer is not None and (not torch.is_tensor(buffer) or list(buffer.shape)!=contract['shape']
                    or not buffer.is_floating_point() or not bool(torch.isfinite(buffer).all())
                    or str(buffer.dtype)!=contract['dtype']):
                raise OptimizerRestoreError('native optimizer retained Muon momentum differs from its original contract')
    return history
