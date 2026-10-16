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
            chosen,rejected=[trajectories[name] for name in ('chosen','rejected')]
            if chosen.split!=rejected.split or set(chosen.split_groups)!=set(rejected.split_groups):
                raise Reject('offline preference alternatives disagree on protected task/split')
            common=0
            for left,right in zip(chosen.messages,rejected.messages):
                if left!=right:break
                common+=1
            if not all(any(message['role']=='assistant' for message in t.messages[common:]) for t in (chosen,rejected)):
                raise Reject('offline preference lacks a divergent assistant continuation in both alternatives')
            if chosen.messages[:common]!=rejected.messages[:common]:
                raise Reject('offline preference common history changed')
            yield rid,{'schema':'natlang.offline-native-preference/1','id':'offline-preference:'+_digest(binding),
                'source':binding,'split':chosen.split,'source_groups':sorted(set(chosen.split_groups)),
                'common_message_count':common,'source_preference':'chosen-over-rejected',
                'alternatives':{name:asdict(trajectories[name]) for name in ('chosen','rejected')},
                'offline':True,'local_correctness_verified':False,'native_on_policy':False,
                'training_admission':False,'required_objective':'native-offline-pairwise'}
        except (Reject,ValueError,TypeError,KeyError) as error:
            yield rid,Reject(str(error))
