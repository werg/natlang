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
    seen=set()
    for name,reference in spec['cohorts'].items():
        source=load_source(reference)
        rows=admitted=completed=visits=required=unvisited=missing=not_admitted=0
        for entry in source.entries:
            key=entry['id']
            if key in seen:raise ValueError('exposure summary cohorts overlap')
            seen.add(key);rows+=1
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
    return result
