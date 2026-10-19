"""Whole admitted policy geometry for the existing semantic phase gate."""
from __future__ import annotations
import copy
from ..common.hashing import canonical_json_sha256_hex
from .native_semantic_policy import prepare_policy,policy_loss


class PolicyQualification:
    kind='whole-policy-episode'

    def __init__(self,source,participants,vae,training,origin,device):
        self.source,self.participants,self.vae=source,participants,vae
        self.training,self.origin,self.device=training,origin,device
        self.binding=copy.deepcopy({'schema':'natlang.native-semantic-policy-qualification-source/1',
            'sources':source.sources,'declaration':source.declaration,'origin':origin,
            'objectives_sha256':canonical_json_sha256_hex(source.entries)})
        self.current=None;self.prepared=None

    def prepare(self,entry):
        if self.current!=entry['id']:
            self.prepared=None
            self.prepared=prepare_policy(self.source,entry,self.participants,self.vae,
                self.training,self.origin,device=self.device)
            self.current=entry['id']
        return self.prepared

    def geometry(self,entry):
        selected,replays,dependencies,_,_,_,_=self.prepare(entry)
        dimensions=[]
        for name in sorted(self.participants):
            replay=replays.get(name)
            if replay is None:dimensions.extend((0,)*7);continue
            plan=replay.plan
            own=[edge for edge in dependencies if edge.target_participant==name]
            dimensions.extend((plan.token_count,len(plan.slot_sources),len(plan.invocations),
                len(replay.actions),sum(len(edge.target_token_ids)+2 for edge in own),
                sum(len(edge.source_token_ids) for edge in own),len(own)))
        return tuple(dimensions)

    def loss(self,entry,reference):
        loss,details,active=policy_loss(self.source,entry,self.participants,self.vae,
            self.training,self.origin,device=self.device,reference=reference,prepared=self.prepare(entry))
        expected=set(self.prepared[0])
        if active!=expected:raise ValueError('semantic policy qualification changed actual participants')
        return loss,details

    def frontier(self,participants):
        frontiers={};representatives={};count=0
        for entry in self.source.entries:
            selected,replays,dependencies,_,_,_,_=self.prepare(entry)
            active=tuple(sorted(selected))
            if set(active)!=set(participants):continue
            # Separate communication mechanisms and objective branches. A text
            # episode cannot qualify a codec edge or a stop-gradient branch.
            category=(active,(entry['mode'],(entry['advantage']>0)-(entry['advantage']<0),
                any(d.source_node is not None for d in dependencies),
                any(d.external_source is not None for d in dependencies),
                any(node.imports for replay in replays.values() for node in replay.plan.invocations),
                any(edge.stop_gradient for replay in replays.values()
                    for node in replay.plan.invocations for edge in node.imports)))
            dimensions=self.geometry(entry);count+=1
            representatives.setdefault(category,entry)
            frontier=frontiers.setdefault(category,[])
            if any(all(a>=b for a,b in zip(old,dimensions)) for old,_ in frontier):continue
            frontier[:]=[(old,case) for old,case in frontier if not all(a>=b for a,b in zip(dimensions,old))]
            frontier.append((dimensions,entry))
        if not frontiers:raise ValueError('semantic policy capacity source has no actual selected participant episode')
        return [case for frontier in frontiers.values() for case in frontier],representatives,count
