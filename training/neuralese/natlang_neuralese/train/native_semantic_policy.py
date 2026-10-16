"""Exact policy-source accounting for the existing joint semantic stage.

The curriculum declares sources, objective and cadence. This helper verifies
immutable episodes, computes the existing group-relative advantages and tracks
whole committed visits in the stage's existing per-phase exposure ledger.
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex
from ..common.jsonio import iter_jsonl_rows
from .native_readiness import _bound


def validate_policy_training(value,phase_ids=()):
    if value is None:return None
    if (not isinstance(value,dict) or not isinstance(value.get('id'),str) or not value['id']
            or value['id'] in phase_ids or value.get('mode') not in {'grpo','rft','mixed'}
            or not isinstance(value.get('sources'),list) or not value['sources']):
        raise ValueError('semantic policy training needs its own phase ID, actual sources and objective')
    for name in ('epochs','every_semantic_updates','updates_per_refresh'):
        if type(value.get(name)) is not int or value[name]<1:
            raise ValueError('semantic policy training needs positive declared '+name)
    for name in ('clip_low','clip_high'):
        bound=value.get(name,.2 if name=='clip_low' else .28)
        if type(bound) not in (float,int) or not math.isfinite(bound) or not 0<=bound<1:
            raise ValueError('semantic policy clipping must be finite and in [0,1)')
    for source in value['sources']:
        if (not isinstance(source,dict) or not {'rollouts','block_bundle','adapter'}<=set(source)
                or set(source)-{'rollouts','block_bundle','adapter','mode','collector_participant'}
                or source.get('mode',value['mode']) not in {'grpo','rft'}):
            raise ValueError('semantic policy source must bind exact rollouts, block bundle and behavior adapter')
        if 'collector_participant' in source and (not isinstance(source['collector_participant'],str) or not source['collector_participant']):
            raise ValueError('semantic policy collection requires its exact participant scope')
        for binding in (source[key] for key in ('rollouts','block_bundle','adapter')):
            if not isinstance(binding,dict) or set(binding)!={'path','sha256'}:
                raise ValueError('semantic policy source roles require immutable path/SHA bindings')
    return value


class PolicySource:
    """Compact episode/source identities; full trajectories are loaded one at a time."""
    def __init__(self,declaration,*,previous_sources=()):
        self.declaration=validate_policy_training(declaration)
        self.sources=[];self.entries=[];self.by_id={};self.channels={};self.readers={}
        from ..rl.grpo import group_advantages
        groups=defaultdict(list)
        for binding in [*previous_sources,*declaration['sources']]:
            if binding in self.sources:continue
            self.sources.append(binding)
        for source_index,source in enumerate(self.sources):
            path=_bound(source['rollouts']);_bound(source['block_bundle']);_bound(source['adapter'])
            for line,rollout in iter_jsonl_rows(path):
                if not isinstance(rollout,dict):raise ValueError('semantic policy source needs complete episode objects')
                reward=rollout.get('reward')
                if reward is not None and (not isinstance(reward,dict) or reward.get('quality') not in (0,1)):
                    raise ValueError('semantic policy source contains an unsupported/unscored reward')
                key=canonical_json_sha256_hex(rollout)
                if key in self.by_id:raise ValueError('semantic policy sources duplicate an actual episode')
                entry={'id':key,'source':source_index,'line':line,'task_id':rollout.get('task_id'),
                    'mode':source.get('mode',declaration['mode']),
                    'reward':None if rollout.get('reward') is None else {'quality':rollout['reward'].get('quality')}}
                if (rollout.get('split')!='train' or not rollout.get('task_id') or not rollout.get('turns')
                        or rollout.get('training_admission') not in (None,'grpo')):
                    raise ValueError('semantic policy source contains a non-training or unbound episode')
                self.by_id[key]=entry
                groups[(rollout['task_id'],canonical_json_sha256_hex(rollout.get('group')),
                        source['adapter']['sha256'],source.get('collector_participant'),entry['mode'])].append(entry)
        self.omissions={}
        for members in groups.values():
            advantages,stats=group_advantages([entry['reward']['quality'] if entry['reward'] is not None else None for entry in members])
            for entry,advantage in zip(members,advantages):
                reason=('unscored' if entry['reward'] is None else
                    'no-relative-reward-spread' if entry['mode']=='grpo' and not stats['spread'] else
                    'nonpositive-rft' if entry['mode']=='rft' and entry['reward']['quality']!=1 else None)
                if reason:self.omissions[entry['id']]=reason;continue
                entry['advantage']=float(advantage or 0.)
                entry['objective_sha256']=canonical_json_sha256_hex({'mode':entry['mode'],
                    'clip_low':declaration.get('clip_low',.2),'clip_high':declaration.get('clip_high',.28),
                    'advantage':entry['advantage'],'episode':entry['id']})
                self.entries.append(entry)
        # Stable physical source/row order; repeated source scans are avoided by
        # retaining a forward iterator, resetting only when the durable cursor wraps.
        self.entries.sort(key=lambda entry:(entry['source'],entry['line']))
        if not self.entries:
            raise ValueError('semantic policy source has no useful scored episodes; collect new relative-reward groups or verified RFT successes')
        self._stream=None;self._stream_source=None;self._stream_line=-1

    def load(self,entry,*,device):
        source=self.sources[entry['source']]
        path=_bound(source['rollouts'])
        if self._stream_source!=entry['source'] or self._stream_line>=entry['line']:
            self._stream=iter_jsonl_rows(path);self._stream_source=entry['source'];self._stream_line=-1
        rollout=None
        for line,row in self._stream:
            self._stream_line=line
            if line==entry['line']:rollout=row;break
        if rollout is None or canonical_json_sha256_hex(rollout)!=entry['id']:
            raise ValueError('semantic policy source changed under its exact episode cursor')
        adapter_key=source['adapter']['sha256'];bundle_key=source['block_bundle']['sha256']
        if adapter_key not in self.channels:
            from ..model.semantic_channel import SemanticChannel
            self.channels.clear()
            self.channels[adapter_key]=SemanticChannel.load(_bound(source['adapter']),device=device)
        if bundle_key not in self.readers:
            from ..rl.graph_replay import NativeBundleStore
            bundle=_bound(source['block_bundle']);manifest=json.loads(bundle.read_text())
            owners={identity['checkpoint_sha256'] for identity in self.channels[adapter_key].bindings.values()}
            if owners!={manifest.get('policy_id')}:
                raise ValueError('semantic policy archive ownership differs from its exact shared behavior parent')
            self.readers.clear()
            self.readers[bundle_key]=NativeBundleStore(bundle,policy_id=manifest['policy_id'],manifest=manifest)
        return rollout,self.channels[adapter_key],self.readers[bundle_key].state

    def synchronize(self,training):
        phase=self.declaration['id'];counts=training.setdefault('exposure',{}).setdefault(phase,{})
        prior_phase=training.get('policy_training_phase')
        if prior_phase not in (None,phase) and not (training.get('policy_obligations') or {}).get('complete'):
            raise ValueError('a new policy phase cannot discard unfinished earlier episode obligations')
        training['policy_training_phase']=phase
        prior=training.setdefault('policy_objectives',{}).setdefault(phase,{})
        required=training.setdefault('policy_required_visits',{}).setdefault(phase,{})
        cancelled=training.setdefault('policy_cancellations',{}).setdefault(phase,{})
        active={entry['id'] for entry in self.entries}
        for key,quota in required.items():
            visits=counts.get(key,{}).get('episode',0)
            if key in active or visits>=quota or key in cancelled:continue
            reason=self.declaration.get('cancel_stale_reason')
            if not isinstance(reason,str) or not reason.strip():
                raise ValueError('changed policy objective would drop unfinished episodes; declare an explicit supervised cancellation reason')
            cancelled[key]={'step':training['step'],'reason':reason,'remaining_uncredited':quota-visits,
                            'objective_sha256':prior[key]}
        for entry in self.entries:
            key=entry['id'];visits=counts.get(key,{}).get('episode',0)
            if key in cancelled:raise ValueError('cancelled semantic behavior cannot be re-admitted; recollect under current policy')
            old=prior.get(key);current=entry['objective_sha256']
            required[key]=max(required.get(key,0),self.declaration['epochs'])
            if old is not None and old!=current:
                # Earlier visits remain actual earned exposure. A changed loss
                # owes one new complete visit; no reset or fictitious old credit.
                required[key]=max(required[key],visits+1)
                training.setdefault('policy_objective_revisions',[]).append({'step':training['step'],
                    'phase':phase,'row':key,'from':old,'to':current,'retained_visits':visits,
                    'required_visits':required[key]})
            prior[key]=current
        training['policy_source_history']=self.sources
        return self.summary(training)

    def summary(self,training):
        phase=self.declaration['id'];counts=training.get('exposure',{}).get(phase,{})
        required=training.get('policy_required_visits',{}).get(phase,{})
        cancelled=training.get('policy_cancellations',{}).get(phase,{})
        pending=[{'row':key,'visits':counts.get(key,{}).get('episode',0),'required':quota}
                 for key,quota in required.items() if key not in cancelled and counts.get(key,{}).get('episode',0)<quota]
        return {'phase':phase,'complete':not pending,'remaining_episode_visits':sum(row['required']-row['visits'] for row in pending),
                'sources':self.sources,'rows':len(self.entries),'omissions':self.omissions,'missing_exposure':pending,
                'cancelled_uncredited':cancelled,'original_obligations_complete':not pending and not cancelled}

    def next(self,training):
        if not self.entries:return None
        phase=self.declaration['id'];counts=training['exposure'][phase];required=training['policy_required_visits'][phase]
        start=training.setdefault('policy_source_cursors',{}).get(phase,0)%len(self.entries)
        for offset in range(len(self.entries)):
            index=(start+offset)%len(self.entries);entry=self.entries[index]
            if counts.get(entry['id'],{}).get('episode',0)<required[entry['id']]:return index,entry
        return None

    def committed(self,training,index,entry):
        phase=self.declaration['id'];counts=training['exposure'][phase].setdefault(entry['id'],{})
        counts['episode']=counts.get('episode',0)+1
        training.setdefault('policy_source_cursors',{})[phase]=(index+1)%len(self.entries)
        training['policy_obligations']=self.summary(training)

    def cycle(self,training,signature):
        current=training.get('policy_refresh_cycle')
        if current is None or current['signature']!=signature:
            required=min(self.declaration['updates_per_refresh'],self.summary(training)['remaining_episode_visits'])
            current={'signature':signature,'required_updates':required,'completed_updates':0,'started_step':training['step']}
            training['policy_refresh_cycle']=current
        return current


def policy_cycle_complete(cycle):
    return cycle is None or cycle['completed_updates']>=cycle['required_updates']


def checkpoint_progress(training):
    keys=('policy_training_phase','policy_source_history','policy_required_visits','policy_objectives',
          'policy_cancellations','policy_objective_revisions','policy_source_cursors',
          'policy_refresh_cycle','policy_execution_origin','policy_obligations')
    return {key:training[key] for key in keys if key in training}


def policy_loss(source,entry,participants,vae,training,origin,*,device,loaded=None):
    """Use current joint tensors with separately authenticated behavior history."""
    from types import SimpleNamespace
    from ..model.semantic_channel import SemanticChannel
    from ..rl.native_semantic_replay import compile_semantic_episode,semantic_episode_loss
    rollout,behavior,fetch=loaded or source.load(entry,device=device)
    if set(behavior.bindings)-set(participants):
        raise ValueError('semantic policy behavior names an absent current participant')
    if set(participants)-set(origin['models']):
        raise ValueError('a new semantic participant requires an exact refreshed policy execution parent')
    for name in behavior.bindings:
        actual=origin['models'][name]['native_identity'];old=behavior.bindings[name]
        if any(actual[key]!=old[key] for key in ('tokenizer_fingerprint','architecture')):
            raise ValueError('semantic policy continuation changed tokenizer or native architecture')
        if (name not in vae.layouts or name not in behavior.vae.layouts
                or vae.layouts[name]!=behavior.vae.layouts[name]
                or (vae.latent_width,vae.hidden_width)!=(behavior.vae.latent_width,behavior.vae.hidden_width)):
            raise ValueError('semantic policy codec layout changed without an explicit shared-space migration')
    behavior_participants={name:SimpleNamespace(model=participants[name].model,identity=identity,
        tokenizer=behavior.tokenizers[name]) for name,identity in behavior.bindings.items()}
    replays,dependencies,external=compile_semantic_episode(rollout,behavior_participants,channel=behavior,fetch_block=fetch)
    selected={name:SimpleNamespace(model=participants[name].model,tokenizer=participants[name].tokenizer,
        identity=origin['models'][name]['native_identity']) for name in replays}
    channel=SemanticChannel(vae,{name:item['native_identity'] for name,item in origin['models'].items()},
        {name:participants[name].tokenizer for name in origin['models']},adapter_sha256=origin['adapter_sha256'])
    execution={'schema':'natlang.native-semantic-policy-execution/1',
        'parent_checkpoint_sha256':origin['checkpoint_sha256'],'parent_adapter_sha256':origin['adapter_sha256'],
        'parent_step':origin['step'],'step':training['step'],'behavior_adapter_sha256':behavior.adapter_sha256}
    declaration=source.declaration
    loss,details=semantic_episode_loss(selected,replays,dependencies,channel=channel,external_blocks=external,
        advantage=entry['advantage'],mode=entry['mode'],clip_low=declaration.get('clip_low',.2),
        clip_high=declaration.get('clip_high',.28),chunk_tokens=declaration.get('chunk_tokens',128),
        training_execution=execution,behavior_channel=behavior)
    return loss,{**details,'training_execution':execution,'source_row_sha256':entry['id'],
        'behavior_source':source.sources[entry['source']]},set(replays)


def validate_policy_boundary(proof,adapter,curriculum):
    """Bind policy visits/quanta to the actual saved joint optimizer checkpoint."""
    claimed=proof.get('policy_training_progress')
    if claimed is None:
        if curriculum.get('policy_training') or adapter.get('policy_training_progress'):
            raise ValueError('shared boundary omitted declared semantic policy obligations')
        return
    if claimed!=adapter.get('policy_training_progress'):
        raise ValueError('shared adapter and completion changed policy exposure provenance')
    from .checkpoint_policy import CheckpointPolicy
    state=CheckpointPolicy.load(_bound(proof['checkpoint']),semantic_only=True)
    training=state.get('training') or {}
    if (checkpoint_progress(training)!=claimed
            or proof.get('exposure',{}).get(claimed['policy_training_phase'])
                 !=training.get('exposure',{}).get(claimed['policy_training_phase'])):
        raise ValueError('semantic policy boundary differs from actual checkpointed episode/update state')
    declaration=curriculum.get('policy_training')
    if declaration is None:
        declaration=next((item['config']['policy_training'] for item in reversed(training.get('config_history',[]))
            if item['config'].get('policy_training',{}).get('id')==claimed['policy_training_phase']),None)
    if declaration is None:raise ValueError('semantic policy boundary lost its declared curriculum')
    source=PolicySource(declaration,previous_sources=claimed.get('policy_source_history',[]))
    if source.summary(training)!=claimed.get('policy_obligations'):
        raise ValueError('semantic policy boundary misstated remaining actual episode exposure')
    if proof.get('completed') and not claimed['policy_obligations']['complete']:
        raise ValueError('shared completion left mandatory policy episode visits unfinished')
    if proof.get('refresh_ready') and not policy_cycle_complete(claimed.get('policy_refresh_cycle')):
        raise ValueError('shared refresh omitted its declared finite policy quantum')
