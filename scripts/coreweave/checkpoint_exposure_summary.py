"""Read-only cohort exposure from a checkpoint already decoded by its owner."""
from __future__ import annotations


def summarize_cohorts(state,spec,load_source):
    """Use exact saved row bindings/credits; never infer visits from availability."""
    training=state['training'];stage=spec['stage_id']
    if (training.get('options')or{}).get('stage_id')==stage:
        frame=training
    else:
        frame=(training.get('sft_progress')or{})[stage]
    bindings=frame['source_row_bindings'];counts=frame['exposure_counts'];quotas=frame.get('required_visits')or{}
    result={'schema':'natlang.native-checkpoint-cohort-exposure/1','global_step':state['step'],
            'stage_id':stage,'stage_step':frame.get('step'),'cohorts':{},
            'scope':'Exact saved row credits, not live unsaved updates; no model execution'}
    overlap=spec.get('allow_overlapping_cohorts',False)
    if type(overlap)is not bool:raise ValueError('cohort overlap setting must be an explicit boolean')
    seen={}
    if overlap:result['overlap_policy']='Independent diagnostic views of identical saved credits; never sum overlapping cohorts'
    for name,reference in spec['cohorts'].items():
        source=load_source(reference)
        rows=admitted=completed=visits=required=unvisited=missing=not_admitted=0;local_seen=set()
        for entry in source.entries:
            key=entry['id']
            if key in local_seen:raise ValueError('exposure summary cohort repeats a row')
            local_seen.add(key)
            if key in seen:
                if not overlap:raise ValueError('exposure summary cohorts overlap')
                if seen[key]!=entry['binding']:raise ValueError('overlapping exposure views changed a row binding')
            seen[key]=entry['binding'];rows+=1
            actual_binding=bindings.get(key)
            if actual_binding is not None and actual_binding!=entry['binding']:
                raise ValueError('exposure summary row differs from saved binding')
            quota=quotas.get(key,1)
            if type(quota)is not int or quota<1:raise ValueError('exposure summary quota is invalid')
            actual=counts.get(key,0)
            if type(actual)is not int or actual<0:raise ValueError('exposure summary credit is invalid')
            if actual_binding is None:
                if actual:raise ValueError('exposure summary credit has no saved admitted row')
                not_admitted+=1
            else:
                admitted+=1;visits+=actual;completed+=int(actual>=quota);unvisited+=int(actual==0)
            required+=quota;missing+=max(0,quota-actual)
        result['cohorts'][name]={'source':reference,'rows':rows,'admitted_rows':admitted,
            'not_admitted_rows':not_admitted,'completed_rows':completed,'never_visited_admitted_rows':unvisited,
            'actual_visits':visits,'required_visits':required,'remaining_visits':missing}
        del source
    if spec.get('optimizer_state_inventory')is True:
        result['current_optimizer_inventory']=optimizer_state_inventory(training['optimizer'])
        result['current_optimizer_stage_id']=(training.get('options')or{}).get('stage_id')
    return result


def optimizer_state_inventory(optimizer):
    """Describe actual named saved tensors without cloning, reading values or RNG."""
    import torch
    names=optimizer.get('param_names');schema=optimizer.get('schema')
    if not isinstance(names,dict)or not isinstance(schema,list):
        raise ValueError('optimizer observation needs actual named state and schema')
    declared={row['name']:row for row in schema}
    if len(declared)!=len(schema):raise ValueError('optimizer schema repeats a parameter')
    rows=[];seen=set()
    for route,parameters in names.items():
        child=optimizer.get(route)
        if not isinstance(child,dict):raise ValueError('optimizer route lacks its saved child')
        identifiers=[identifier for group in child['param_groups']for identifier in group['params']]
        if len(identifiers)!=len(parameters)or len(identifiers)!=len(set(identifiers)):
            raise ValueError('optimizer saved names and group identifiers differ')
        for name,identifier in zip(parameters,identifiers):
            if name not in declared or name in seen:raise ValueError('optimizer route loses or repeats its declared parameter')
            seen.add(name);buffers={}
            for key,value in child['state'].get(identifier,{}).items():
                if torch.is_tensor(value):buffers[str(key)]={'dtype':str(value.dtype),'shape':list(value.shape)}
            rows.append({'name':name,'route':route,'parameter_dtype':declared[name]['dtype'],
                'parameter_shape':declared[name]['shape'],'state_present':identifier in child['state'],'buffers':buffers})
    if seen!=set(declared):raise ValueError('optimizer observation lacks declared parameters')
    return {'schema':'natlang.native-saved-optimizer-buffer-inventory/1','parameters':rows,
        'scope':'Actual current saved optimizer metadata only; absent buffers remain absent; no numeric values or model execution'}
