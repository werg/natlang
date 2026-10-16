"""Standalone jointly bound offline preferences, separate from positive SFT."""
from dataclasses import asdict

from .common import Reject


def miroverse_preference_records(limit=None):
    from .acquired_agents import acquired_rows
    from .acquired_granted import miro_pair_trajectories,preference_binding,_digest
    for receipt,rid,row in acquired_rows('D24',limit):
        try:
            if not rid.rsplit(':',1)[0].startswith('dpo/'):
                raise Reject('non-preference source row; original SFT exporter is the consumer')
            binding=preference_binding(receipt,rid,row)
            if binding['source_file'] is None:
                raise Reject('offline preference lacks exact pinned original file binding')
            trajectories={key.rsplit(':',1)[1]:value for key,value in miro_pair_trajectories(receipt,rid,row)}
            yield rid,bound_preference_record(trajectories,binding)
        except (Reject,ValueError,TypeError,KeyError) as error:
            yield rid,Reject(str(error))


def bound_preference_record(trajectories,binding):
    """One joint offline record from two exact normalized source alternatives."""
    from .acquired_granted import _digest
    chosen,rejected=[trajectories[name] for name in ('chosen','rejected')]
    if chosen.split!=rejected.split or set(chosen.split_groups)!=set(rejected.split_groups):
        raise Reject('offline preference alternatives disagree on protected task/split')
    common=0
    for left,right in zip(chosen.messages,rejected.messages):
        if left!=right:break
        common+=1
    if not all(any(message['role']=='assistant' for message in t.messages[common:]) for t in (chosen,rejected)):
        raise Reject('offline preference lacks a divergent assistant continuation in both alternatives')
    return {'schema':'natlang.offline-native-preference/1','id':'offline-preference:'+_digest(binding),
        'source':binding,'split':chosen.split,'source_groups':sorted(set(chosen.split_groups)),
        'common_message_count':common,'source_preference':'chosen-over-rejected',
        'alternatives':{name:asdict(trajectories[name]) for name in ('chosen','rejected')},
        'offline':True,'local_correctness_verified':False,'native_on_policy':False,
        'training_admission':False,'required_objective':'native-offline-pairwise'}


def normalized_preference_records(source):
    """Retain each original normalized session and join only exact adjacent pairs.

    The full source export and this reader are byte-bound by the converter.
    This is an exact join, not a task-similarity or inferred reward policy.
    """
    from .build_acquired_full_session_graphs import sessions
    from .trajectory import Trajectory
    pending=None;seen=set()
    for index,value in enumerate(sessions(source)):
        notes=value['trajectory'].get('lineage',{}).get('notes',{})
        preference=notes.get('source_preference')
        if preference is None:
            if pending is not None:
                yield pending[0],None,Reject('incomplete normalized offline preference pair')
                pending=None
            yield (index,),None,Reject('ordinary source session; positive SFT is its consumer')
            continue
        binding={key:child for key,child in preference.items() if key not in
                 ('alternative','preference','local_correctness_verified')}
        name=preference['alternative']
        if (binding.get('schema')!='natlang.offline-preference-source-binding/1'
                or binding.get('source_file')!=value.get('source_file')
                or binding.get('source_revision')!=value.get('source_revision')
                or value.get('source_row')!=str(binding.get('source_row'))+':'+str(name)):
            raise ValueError('normalized offline preference changed its exact original file/row/revision binding')
        key=binding['pair_sha256'],binding['source_revision'],str(binding['source_row']),binding['source_file']['sha256']
        if name=='chosen':
            if pending is not None:
                yield pending[0],None,Reject('incomplete normalized offline preference pair')
            if key in seen:raise ValueError('normalized source repeats an offline preference occurrence')
            pending=((index,),key,binding,value)
            continue
        if name!='rejected' or pending is None or key!=pending[1] or binding!=pending[2]:
            raise ValueError('normalized offline alternatives lack an exact adjacent original pair binding')
        indices=(*pending[0],index);chosen=pending[3];pending=None;seen.add(key)
        if chosen['trajectory']['outcome'].get('label')=='failed':
            yield indices,None,Reject('source-reported failed trajectory cannot become preferred behavior')
            continue
        chosen_flags=set(chosen['trajectory'].get('lineage',{}).get('notes',{}).get('source_quality_flags') or [])
        rejected_flags=set(notes.get('source_quality_flags') or [])-{'source-rejected-preference-alternative'}
        if chosen_flags or rejected_flags:
            yield indices,None,Reject('source quality exclusions apply to the jointly bound offline pair')
            continue
        try:
            trajectories={'chosen':Trajectory(**chosen['trajectory']),'rejected':Trajectory(**value['trajectory'])}
            record=bound_preference_record(trajectories,binding)
            record['normalized_source_rows']=list(indices)
            yield indices,record,None
        except (Reject,ValueError,TypeError,KeyError) as error:
            yield indices,None,Reject(str(error))
    if pending is not None:yield pending[0],None,Reject('incomplete normalized offline preference pair')
