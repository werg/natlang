"""Durable phase quanta and rotating held coverage for the shared semantic loop.

The curriculum declares useful work; these helpers only choose exact rows and
account committed updates. Cumulative source exposure is never reset here.
"""
from __future__ import annotations
import itertools

from ..common.hashing import canonical_json_sha256_hex


def initialize(training,config,manifest):
    declaration=config.get('adaptive_refresh')
    if not declaration:return None
    default=declaration.get('updates_per_phase')
    required={phase['id']:(phase.get('options') or {}).get('refresh_updates',default) for phase in config['phases']}
    if any(type(value) is not int or value<=0 for value in required.values()):
        raise ValueError('adaptive semantic refresh requires positive declared updates for every phase')
    signature=canonical_json_sha256_hex({'models':manifest['models'],'config':config,
        **({'semantic_parent':{role:binding['sha256'] for role,binding in manifest['semantic_parent'].items()}} if manifest.get('semantic_parent') else {})})
    current=training.get('refresh_cycle')
    if current is None or current.get('signature')!=signature:
        current={'schema':'natlang.native-semantic-refresh-cycle/1','signature':signature,
            'required_updates':required,'completed_updates':{key:0 for key in required},
            'started_step':training['step'],'participants_signature':manifest['signature']}
        training['refresh_cycle']=current
    return current


def complete(cycle):
    return bool(cycle) and all(cycle['completed_updates'].get(key,0)>=value
                              for key,value in cycle['required_updates'].items())


def work(phases,entries,participants,exposure,cycle,cursors):
    from .native_semantic_stage import needed_pairs
    for phase in phases:
        if cycle['completed_updates'][phase['id']]>=cycle['required_updates'][phase['id']]:continue
        counts=exposure.get(phase['id'],{})
        # Continue from the last committed whole row and wrap to catch older
        # deficits, increased quotas and appended sources without prefix scans.
        start=cursors.get(phase['id'],0)%len(entries)
        for offset in range(len(entries)):
            index=(start+offset)%len(entries);entry=entries[index]
            pairs=needed_pairs(entry,participants,counts.get(entry['id'],{}),phase['epochs'])
            if pairs:return phase,entry['id'],pairs,index
        # Additional adaptation advances through the whole qualified source,
        # rather than repeatedly selecting its first row each refresh cycle.
        start=cursors.get(phase['id'],0)%len(entries)
        for offset in range(len(entries)):
            index=(start+offset)%len(entries);entry=entries[index]
            names=sorted(set(entry['models'])&set(participants))
            pairs=list(itertools.combinations(names,2))
            if pairs:return phase,entry['id'],pairs,index
        raise ValueError('adaptive semantic phase has no qualified participant pair')
    return None


def committed(training,cycle,phase,index,rows):
    cycle['completed_updates'][phase['id']]+=1
    training.setdefault('refresh_source_cursors',{})[phase['id']]=(index+1)%rows


def exposure_summary(phases,source,participants,exposure):
    from .native_semantic_stage import _exposure_complete
    done,missing=_exposure_complete(phases,source,participants,exposure)
    return {'complete':done,'remaining_pair_visits':sum(item['required']-item['visits'] for item in missing),
        'train_source':source.binding,'phases':phases}


def held_evaluation(participants,vae,source,*,phase,options,resolver,coverage=None,rows=None,final=False):
    import math
    import torch
    from .loop import capture_training_rng_state,restore_training_rng_state
    from .native_semantic_stage import _native_modules
    from .native_semantic import joint_loss
    total=len(source.entries)
    if rows is not None and (type(rows) is not int or rows<=0):
        raise ValueError('semantic held_eval_rows must be a positive whole-graph count')
    eligible=[index for index,entry in enumerate(source.entries) if len(set(entry['models'])&set(participants))>=2]
    if not eligible:raise ValueError('ready semantic participants have no paired held evaluation data')
    full=final or rows is None or rows>=len(eligible)
    coverage=dict(coverage or {})
    cursor=0 if full else int(coverage.get('cursor',0))%total
    selected=eligible if full else [index for index in eligible if index>=cursor][:rows]
    if not selected:selected=eligible[:rows]
    cursor=selected[0];stop=total if full else selected[-1]+1
    visits=dict(coverage.get('row_visits') or {})
    losses=[];used=[]
    device=next(vae.parameters()).device
    rng=capture_training_rng_state(device)
    modes={name:entry.model.training for name,entry in participants.items()};vae_mode=vae.training
    try:
        for entry in participants.values():entry.model.eval()
        vae.eval()
        with torch.no_grad(),_native_modules(participants):
            for index,row in enumerate(source.rows()):
                if index<cursor:continue
                if index>=stop:break
                if len(set(row['models'])&set(participants))<2:continue
                value,_=joint_loss(vae,participants,row,phase=phase,options=options,fetch_external=resolver)
                scalar=float(value)
                if not math.isfinite(scalar):raise FloatingPointError('nonfinite semantic held objective: '+row['id'])
                losses.append(scalar);used.append(row['id']);visits[row['id']]=visits.get(row['id'],0)+1
    finally:
        for name,entry in participants.items():entry.model.train(modes[name])
        vae.train(vae_mode);restore_training_rng_state(rng,device)
    if not losses:raise ValueError('ready semantic participants have no paired held evaluation data in declared slice')
    coverage.update(source_sha256=source.sha256,cursor=0 if stop==total else stop,row_visits=visits,
                    total_rows=total,last_row_ids=used,last_phase=phase)
    return {'loss':sum(losses)/len(losses),'rows':len(losses),'scope':'full' if full else 'rotating',
            'source_sha256':source.sha256,'phase':phase,'row_ids':used},coverage


def validate_boundary(proof,adapter,curriculum):
    """Verify complete curriculum or an honestly partial, all-phase refresh."""
    from .native_semantic_stage import PairedSource,_exposure_complete
    if proof.get('completed') is not True and proof.get('refresh_ready') is not True:
        raise ValueError('shared boundary is neither complete nor a qualified all-phase refresh')
    if proof.get('step')!=adapter.get('step'):raise ValueError('shared boundary step differs from exported checkpoint progress')
    from .native_semantic_policy import validate_policy_boundary
    validate_policy_boundary(proof,adapter,curriculum)
    source=PairedSource(proof['source_bindings']['train'],split='train')
    if proof['phases']!=curriculum['phases'] or proof['source_bindings']['train']!=curriculum['pairs']:
        raise ValueError('shared boundary changed its required curriculum/source')
    done,missing=_exposure_complete(proof['phases'],source,adapter['models'],proof['exposure'])
    obligations=exposure_summary(proof['phases'],source,adapter['models'],proof['exposure'])
    if (proof.get('exposure_complete') is not done or proof.get('missing_exposure')!=missing
            or proof.get('semantic_obligations')!=obligations or adapter.get('semantic_obligations')!=obligations):
        raise ValueError('shared boundary lost or misstated mandatory semantic exposure')
    if proof.get('completed') is True:
        held=(proof.get('metrics') or {}).get('held') or {}
        held_source=PairedSource(curriculum['held_pairs'],split='test')
        expected_rows=[entry['id'] for entry in held_source.entries if len(set(entry['models'])&set(adapter['models']))>=2]
        if (not done or held.get('scope')!='full' or held.get('source_sha256')!=curriculum['held_pairs']['sha256']
                or held.get('row_ids')!=expected_rows or held.get('rows')!=len(expected_rows)):
            raise ValueError('shared completion left production data or full final held evaluation unfinished')
    if curriculum.get('adaptive_refresh'):
        cycle=proof.get('refresh_progress') or {}
        required={phase['id']:(phase.get('options') or {}).get('refresh_updates',
            (curriculum.get('adaptive_refresh') or {}).get('updates_per_phase')) for phase in curriculum['phases']}
        if (any(type(value) is not int or value<=0 for value in required.values())
                or cycle.get('schema')!='natlang.native-semantic-refresh-cycle/1'
                or cycle.get('required_updates')!=required or cycle!=adapter.get('refresh_progress')
                or cycle.get('participants_signature')!=proof['participants_signature']
                or set(cycle.get('completed_updates',{}))!=set(required)
                or any(type(value) is not int or value<0 for value in cycle.get('completed_updates',{}).values())
                or type(cycle.get('started_step')) is not int or cycle['started_step']<0
                or not complete(cycle) or proof['step']-cycle['started_step']!=sum(cycle['completed_updates'].values())):
            raise ValueError('shared refresh lacks committed work in every declared phase')
    held=(proof.get('metrics') or {}).get('held') or {}
    import math
    if (held.get('rows',0)<=0 or not math.isfinite(held.get('loss',float('nan')))
            or held.get('source_sha256')!=curriculum['held_pairs']['sha256']):
        raise ValueError('shared boundary lacks current finite held evaluation')
    return done
