"""Teacherless full-stack neuralese warm-up on causally aligned ordinary text.

Gold IDs anchor next-token CE and raw embedding targets. The input/output token
 space is fixed; transformer layers, the input map, and content residual train. Gold text history is teacher-forced; individual neuralese completions use the
shared one-stage gradient policy, never unconditional free-running imitation.
No task, compression, autonomous stopping or transport certificate is issued.
"""
from __future__ import annotations
import argparse, atexit, hashlib, json, math, os, random, time, traceback
from collections import Counter
from array import array
from collections.abc import Sequence
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .execution import (causal_gold_prefix_mask, causal_prefix_metrics,
                        full_depth_projected_feedback_step, prefill_write_context)
from .output_embedding_projection import sha
from .optim_restore import optimizer_param_names
from .trajectory_state import (AsyncAtomicCheckpointWriter, atomic_checkpoint,
                               available_system_memory_bytes, clip_finite_gradients,
                               drop_file_cache, gradient_norm, immutable_cpu_snapshot)
from .foundation_schedule import ProjectionFirstSchedule
from . import warmup_export
from .loop import (Cadence, StopSignal, TrainingLoop, capture_training_rng_state,
                   commit_optimizer_step, restore_training_rng_state)
from .memory_estimator import AdaptiveGraphMemory, backbone_memory_layout
from .memory_policy import (TEXT_WARMUP_READOUT_CHUNKS,
                            TEXT_WARMUP_FFN_CHUNKS,
                            conservative_expanded_readout_prediction,
                            select_text_warmup_chunk_pair,
                            text_warmup_ffn_workspace_delta_bytes,
                            text_warmup_update_geometry_bytes)
from .checkpoint_safety import (CheckpointDiskReserve, CheckpointReserveError,
                                persist_postcommit_recovery,
                                warmup_checkpoint_size_upper_bound)
from .projection_anchor import relative_mse_positions, gold_aligned_projection_errors
from .text_supervision import (ROLE_CODES, TEXT_POSITION_WEIGHT_POLICY,
                               balanced_position_weights, DocumentWindowSampler)

def relative_mse(predicted, target):
    return relative_mse_positions(predicted,target).mean()


TEXT_WARMUP_MEMORY_KIND='text-warmup-complete-update-v1'
TEXT_WARMUP_MEMORY_GEOMETRY='text-warmup-isolated-sequence-v1'
TEXT_WARMUP_MEMORY_HEADROOM=.05
TEXT_WARMUP_OFFLOAD_SAVINGS_ASSUMPTION=.5


_OBJECTIVE_METRIC_SCALARS = (
    'ce', 'text_ce', 'ce_delta', 'relative_mse',
    'text_embedding_mse', 'embedding_mse_delta', 'text_argmax_agreement',
    'gold_accuracy', 'close_targets', 'close_probability', 'close_top1',
    'premature_close_top1', 'supervised_ce', 'supervised_embedding_mse',
    'context_valid_gold_tokens', 'context_valid_gold_fraction',
    'channel_consistency_kl', 'channel_consistency_agreement', 'channel_consistency_tokens',
)

_EVALUATION_CONTEXT_METRICS = (
    'context_valid_gold_ce', 'context_valid_gold_accuracy',
    'context_valid_text_argmax_agreement', 'context_valid_ce_delta',
    'context_valid_last256_target_tokens', 'context_valid_last256_gold_tokens',
    'context_valid_last256_gold_fraction', 'context_valid_last256_gold_ce',
    'context_valid_last256_gold_accuracy', 'context_valid_last256_text_argmax_agreement',
    'context_valid_last256_ce_delta',
)


def objective_metric_scalars(secondary_projection='input_map'):
    if secondary_projection != 'input_map':
        raise ValueError('unknown secondary projection')
    return (*_OBJECTIVE_METRIC_SCALARS, secondary_projection+'_mse',
            'supervised_'+secondary_projection+'_mse')


class LinearModuleCallCounter:
    """Count Linear invocations without retaining tensors or synchronizing devices."""

    def __init__(self, **roots):
        self.counts = Counter()
        self.handles = []
        self.closed = False
        for root_name, root in roots.items():
            for name, module in root.named_modules():
                if not isinstance(module, torch.nn.Linear):
                    continue
                full_name = root_name + ('.' + name if name else '')

                def count_input(mod, inputs, *, module_name=full_name):
                    value = inputs[0] if inputs else None
                    shape = tuple(int(dim) for dim in value.shape) if isinstance(value, torch.Tensor) else ()
                    key = (module_name, type(mod).__name__, shape, torch.is_grad_enabled())
                    self.counts[key] += 1

                self.handles.append(module.register_forward_pre_hook(count_input))

    def close(self):
        if self.closed:
            return
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def rows(self):
        return [dict(module=name, module_class=class_name, input_shape=list(shape),
                     grad_enabled=grad_enabled, calls=count)
                for (name, class_name, shape, grad_enabled), count in sorted(self.counts.items())]


def materialize_objective_metrics(pass_metrics, pass_losses=None, passes=1, *, secondary_projection='input_map'):
    """Extract scalar objective metrics with one device-to-host read.

    The objective stores only reduced scalar tensors here; predictions, token
    losses, and other sequence-sized values are never retained for reporting.
    Evaluation calls this immediately; training defers it until all backwards.
    """
    if not pass_metrics:
        raise ValueError('at least one objective metric packet is required')
    if pass_losses is not None and len(pass_metrics) != len(pass_losses):
        raise ValueError('one scalar loss is required for every training pass')
    metric_names = objective_metric_scalars(secondary_projection)
    evaluation_context_presence = [
        all(name in metric for name in _EVALUATION_CONTEXT_METRICS) for metric in pass_metrics]
    has_evaluation_context = any(evaluation_context_presence)
    if has_evaluation_context and not all(evaluation_context_presence):
        raise ValueError('evaluation context metrics must be present in every pass or none')
    if has_evaluation_context:
        metric_names = (*metric_names, *_EVALUATION_CONTEXT_METRICS)
    packed = [metric[name].detach().reshape(())
              for metric in pass_metrics for name in metric_names]
    if pass_losses is not None:
        packed.extend(loss.detach().reshape(()) for loss in pass_losses)
    values = torch.stack(packed).to(device='cpu').tolist()
    metric_values = values[:len(pass_metrics) * len(metric_names)]
    loss_values = values[len(metric_values):] if pass_losses is not None else []

    materialized = []
    width = len(metric_names)
    for index, metric in enumerate(pass_metrics):
        row = dict(metric)
        scalars = metric_values[index * width:(index + 1) * width]
        row.update(zip(metric_names, scalars))
        close_targets = int(row['close_targets'])
        if close_targets == 0:
            row['close_probability'] = None
            row['close_top1'] = None
        if close_targets == row['tokens']:
            row['premature_close_top1'] = 0.
        materialized.append(row)

    # Keep the original Python accumulation order and division semantics.
    total_loss = None
    if pass_losses is not None:
        total_loss = 0.
        for value in loss_values:
            total_loss += float(value) / passes
    return materialized, total_loss


def _normalize_context_valid_strata(strata):
    """Finalize prefix/tail statistics accumulated for whole and regional rows."""
    for row in strata.values():
        non_metrics={'tokens','context_valid_gold_tokens','context_valid_gold_fraction',
                     'context_valid_last256_target_tokens','context_valid_last256_gold_tokens',
                     'context_valid_windows', 'channel_consistency_tokens'}
        non_metrics.update(n for n in row if n.endswith('_weighted_sum'))
        for n in row.keys()-non_metrics:row[n]/=row['tokens']
        channel_count=row.get('channel_consistency_tokens',0)
        for name in ('channel_consistency_kl','channel_consistency_agreement'):
            total=row.pop(name+'_weighted_sum',0.)
            row[name]=total/channel_count if channel_count else None
        row['context_valid_gold_fraction']=row['context_valid_gold_tokens']/row['tokens']
        row['context_valid_last256_gold_fraction']=(
            row['context_valid_last256_gold_tokens']/row['context_valid_last256_target_tokens']
            if row['context_valid_last256_target_tokens'] else 0.)
        for n in ('context_valid_gold_ce','context_valid_gold_accuracy',
                  'context_valid_text_argmax_agreement','context_valid_ce_delta'):
            total=row.pop(n+'_weighted_sum',0.)
            row[n]=total/row['context_valid_gold_tokens'] if row['context_valid_gold_tokens'] else None
        for n in ('context_valid_last256_gold_ce','context_valid_last256_gold_accuracy',
                  'context_valid_last256_text_argmax_agreement','context_valid_last256_ce_delta'):
            total=row.pop(n+'_weighted_sum',0.)
            row[n]=total/row['context_valid_last256_gold_tokens'] if row['context_valid_last256_gold_tokens'] else None


def _warmup_memory_kind(batch_size, sequence_passes, readout_chunk_tokens=128,
                        ffn_chunk_tokens=1024):
    return (f'{TEXT_WARMUP_MEMORY_KIND}:batch{int(batch_size)}:'
            f'passes{int(sequence_passes)}:readout{int(readout_chunk_tokens)}:'
            f'ffn{int(ffn_chunk_tokens)}')


def _warmup_readout_calibration_state(state, *, default_ffn_chunk_tokens=1024):
    """Assign legacy warmup rows to their historical readout and actual FFN chunk."""
    if not isinstance(state,dict):return state
    migrated=dict(state)
    samples=dict(state.get('samples',{}))
    for key,values in list(samples.items()):
        parts=key.split(':')
        if (len(parts)==5 and ':'.join(parts[:1])==TEXT_WARMUP_MEMORY_KIND and
                parts[1].startswith('batch') and parts[2].startswith('passes')):
            target=':'.join((*parts[:3],'readout128',f'ffn{default_ffn_chunk_tokens}',*parts[3:]))
            samples.setdefault(target,list(values))
        elif (len(parts)==6 and parts[0]==TEXT_WARMUP_MEMORY_KIND and
              parts[1].startswith('batch') and parts[2].startswith('passes') and
              parts[3].startswith('readout')):
            target=':'.join((*parts[:4],f'ffn{default_ffn_chunk_tokens}',*parts[4:]))
            samples.setdefault(target,list(values))
    migrated['samples']=samples
    return migrated


def _warmup_memory_layout(backbone, heads, *, checkpointed):
    """Return full-depth and actual cutoff-depth layouts for warm-up."""
    return (backbone_memory_layout(backbone, checkpointed=checkpointed),
            backbone_memory_layout(backbone, depth=int(heads.cutoff),
                                   checkpointed=checkpointed))


def _seed_warmup_memory_estimator(estimator, train_log, *, prefix_tokens,
                                  full_layout, shallow_layout, cutoff,
                                  vocab_size, batch_size, named, optimizer,
                                  default_ffn_chunk_tokens=1024, channel_consistency=False):
    """Bootstrap calibration only from previously successful update records."""
    path=Path(train_log)
    if not path.is_file():return 0
    seeded=0
    with path.open() as stream:
        for line in stream:
            try:row=json.loads(line)
            except (json.JSONDecodeError,TypeError):continue
            memory=row.get('memory')
            if not isinstance(memory,dict):continue
            offload=memory.get('offload',{})
            # Offloaded peaks are censored by a resource intervention. They
            # remain telemetry, but cannot calibrate the no-offload predictor.
            if (int(memory.get('preflight',{}).get('offload_budget_bytes',0)) > 0 or
                    int(offload.get('offloaded_tensors',0)) > 0):continue
            positions=int(row.get('positions',0));passes=int(row.get('schedule',{}).get('sequence_passes',0))
            count=int(row.get('batch',batch_size))
            preflight=memory.get('preflight',{})
            if bool(preflight.get('channel_consistency',False)) != channel_consistency:continue
            context=int(preflight.get('context_tokens',prefix_tokens+positions-1))
            target=int(preflight.get('target_tokens',positions))
            actual_prefix=context-target+1
            start=int(memory.get('start_allocated_bytes',0));peak=int(memory.get('peak_allocated_bytes',0))
            if positions<1 or target<1 or actual_prefix<1 or passes<1 or count<1 or peak<=start:continue
            readout_chunk=int(preflight.get('readout_chunk_tokens',128))
            ffn_chunk=int(preflight.get('ffn_chunk_tokens',default_ffn_chunk_tokens))
            raw=text_warmup_update_geometry_bytes(actual_prefix,target,passes,count,
                full_layout,shallow_layout,cutoff=cutoff,vocab_size=vocab_size,
                readout_chunk_tokens=readout_chunk,channel_consistency=channel_consistency)
            raw += text_warmup_ffn_workspace_delta_bytes(actual_prefix,target,count,
                full_layout['intermediate'],base_chunk_tokens=default_ffn_chunk_tokens,
                candidate_chunk_tokens=ffn_chunk)
            bootstrap=not bool(row.get('schedule',{}).get('plateau_reached',False))
            raw += _warmup_update_floor_bytes(named,optimizer,bootstrap=bootstrap)
            estimator.observe(_warmup_memory_kind(count,passes,readout_chunk,ffn_chunk),
                              context,target,raw,peak-start)
            seeded+=1
    return seeded


def _warmup_update_floor_bytes(named, optimizer, *, bootstrap):
    """Bound new gradient and lazy optimizer-state allocations for one update."""
    active=[(name,param) for name,param in named if not bootstrap or
            name.startswith(('heads.feedback.','heads.input_map.','heads.content.proj.'))]
    grad_bytes=sum(param.numel()*param.element_size() for _,param in active)
    active_params={id(param):param for _,param in active}
    children=(getattr(optimizer,'muon',None),getattr(optimizer,'auxiliary',None)) if hasattr(optimizer,'muon') else (optimizer,)
    lazy_state_bytes=0
    for child in children:
        if child is None:continue
        is_muon=type(child).__name__.lower().startswith('muon')
        for group in child.param_groups:
            for param in group['params']:
                if id(param) not in active_params or child.state.get(param):continue
                # Muon allocates a momentum tensor; AdamW allocates exp_avg and
                # exp_avg_sq. Use FP32 as a conservative minimum state element
                # size for low-precision parameters.
                slots=1 if is_muon else 2
                lazy_state_bytes += slots*param.numel()*max(4,param.element_size())
    return int(grad_bytes+lazy_state_bytes)


# Options a resumed run may change in place (the rest are recipe; see main's resume check).
# Activation checkpointing trades memory for recomputation with identical math, so it is operational too.
RESUME_OPERATIONAL_OPTIONS=frozenset({'steps','checkpoint_every','checkpoint_minutes','eval_every','device',
                                      'checkpoint_layers','cuda_reserved_cap_gb','optimizer_state','optimizer_added'})


def same_resume_identity(previous, current):
    """Whether an in-place code handoff preserves the saved recipe identity."""
    old_options, new_options = previous.get('options', {}), current.get('options', {})
    changed = {k for k in old_options.keys() & new_options.keys()
               if old_options[k] != new_options[k]}
    old_recipe = {k: v for k, v in previous.items() if k not in {'code', 'options', 'display'}}
    new_recipe = {k: v for k, v in current.items() if k not in {'code', 'options', 'display'}}
    return (old_recipe == new_recipe and not (changed - RESUME_OPERATIONAL_OPTIONS)
            and not (old_options.keys() - new_options.keys()))


def text_supervision_policy():
    return {**TEXT_POSITION_WEIGHT_POLICY,
            'objectives':['full_projection', 'input_map_self_consistency', 'next_token_ce']}


def warmup_display_labels():
    return {
        'secondary_objective': 'neuralese_input_map_self_consistency',
        'secondary_head': 'heads.input_map',
        'secondary_metric': 'input_map',
        'schedule_head': 'input_map',
    }


def display_update_flags(updates):
    # Older mapped checkpoints used "sketch" for this same input-map flag.
    # This is metadata normalization only; no parameter or optimizer state moves.
    result = dict(updates)
    if 'sketch' in result:
        previous = result.pop('sketch')
        if 'input_map' in result and result['input_map'] != previous:
            raise ValueError('conflicting input-map update metadata')
        result['input_map'] = previous
    return result


from ..maple.family import evaluate_members, leading_system_tokens, member_backward, window_labels
from ..maple.model import eager_rms_norm


def chunked_readout(backbone, states, targets, close_id, *, chunk_size=128,
                    gradients=True, position_weights=None):
    """Exact token-mean CE and readout metrics without retaining T x vocab logits.

    Differentiable chunks are non-reentrantly checkpointed, so backward
    recomputes each readout instead of keeping its vocabulary-sized activation.
    Only scalar CE chunks and small detached token metrics survive the loop.
    """
    if states.ndim != 3 or targets.shape != states.shape[:2]:
        raise ValueError('readout expects [batch,time,width] states and aligned token IDs')
    if chunk_size < 1 or states.shape[1] < 1:
        raise ValueError('positive chunk size and nonempty sequence required')
    if position_weights is None:position_weights=torch.ones_like(targets,dtype=torch.float32)
    if (position_weights.shape!=targets.shape or not torch.isfinite(position_weights).all()
            or (position_weights<0).any() or position_weights.sum()<=0):
        raise ValueError("finite nonnegative aligned position weights with nonzero total required")
    losses=[];weighted_losses=[];predictions=[];close_probabilities=[];token_losses=[]
    needs_grad=gradients and torch.is_grad_enabled() and (
        states.requires_grad or any(p.requires_grad for p in backbone.parameters()))
    for start in range(0,states.shape[1],chunk_size):
        stop=min(start+chunk_size,states.shape[1])
        state_chunk=states[:,start:stop]
        target_chunk=targets[:,start:stop]
        weight_chunk=position_weights[:,start:stop]
        def readout(chunk, gold, weights):
            with eager_rms_norm():  # the checkpoint recompute must replay the forward's exact graph
                logits=backbone.logits(chunk).float()
            per_token=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),gold.reshape(-1),reduction='none').reshape_as(gold)
            ce=per_token.sum()
            training_ce=(per_token*weights).sum()
            with torch.no_grad():
                pred=logits.argmax(-1)
                close=(logits[...,close_id]-torch.logsumexp(logits,-1)).exp()
            return ce,training_ce,pred,close,per_token.detach()
        if needs_grad:
            ce,training_ce,pred,close,positions=checkpoint(readout,state_chunk,target_chunk,weight_chunk,use_reentrant=False)
        elif gradients:
            ce,training_ce,pred,close,positions=readout(state_chunk,target_chunk,weight_chunk)
        else:
            with torch.no_grad():
                ce,training_ce,pred,close,positions=readout(state_chunk,target_chunk,weight_chunk)
        losses.append(ce)
        weighted_losses.append(training_ce)
        predictions.append(pred.detach())
        close_probabilities.append(close.detach())
        token_losses.append(positions)
    count=targets.numel()
    return (torch.stack(losses).sum()/count,torch.stack(weighted_losses).sum()/count,
            torch.cat(predictions,dim=1),
            torch.cat(close_probabilities,dim=1),torch.cat(token_losses,dim=1))


def projection_errors(heads, top, secondary_projection, target, *, secondary_target=None):
    """Full projection stays anchored to gold; an input map may have a separate detached target."""
    full_projection = heads.content(torch.zeros_like(top), top)
    secondary_target = target if secondary_target is None else secondary_target
    return gold_aligned_projection_errors(full_projection, secondary_projection, target,
                                          shallow_target_embeddings=secondary_target)


def projection_losses(heads, top, secondary_projection, target):
    """Both separate trainable maps see fixed gold embeddings immediately."""
    full,secondary=projection_errors(heads,top,secondary_projection,target)
    return full.mean(),secondary.mean()


@torch.no_grad()
def alignment_region_metrics(losses, prediction, plain_losses, plain_prediction,
                             embedding, secondary_projection, reference, target, gold,
                             *, secondary_name='sketch'):
    """Score selected positions after the same full-history forward pass."""
    ce=losses.mean();plain_ce=plain_losses.mean()
    full_error=relative_mse(embedding,target);reference_error=relative_mse(reference,target)
    return {'ce':float(ce),'text_ce':float(plain_ce),'ce_delta':float(ce-plain_ce),
            'relative_mse':float(full_error),secondary_name+'_mse':float(relative_mse(secondary_projection,target)),
            'text_embedding_mse':float(reference_error),
            'embedding_mse_delta':float(full_error-reference_error),
            'text_argmax_agreement':float((prediction==plain_prediction).float().mean()),
            'gold_accuracy':float((prediction==gold).float().mean()),'tokens':gold.numel()}


def gold_completion(backbone, heads, prefix_ids, span_ids, *, auxiliary_scale=.05):
    """Gold history anchors both depth-specific next-token projections."""
    if span_ids.ndim != 2 or span_ids.shape[1] < 1 or prefix_ids.shape[1] < 1:
        raise ValueError('nonempty prefixes and gold target tokens required')
    ordinary=backbone.forward_ids(torch.cat([prefix_ids,span_ids[:,:-1]],1),
                                  cutoff=heads.cutoff,logits=False)
    start=prefix_ids.shape[1]-1
    sources=ordinary['h_cut'][:,start:]
    auxiliary=sources.detach()+auxiliary_scale*(sources-sources.detach())
    return {'top':ordinary['h_final'][:,start:],
            'sketches':heads.feedback(auxiliary)}


# Held matched-history consumers. Sketch mode uses pass 0's feedback as pass 1's matched control. Map mode passes an
# independent serving-feedback completion to this diagnostic, so its sketch_projection remains the actual feedback
# channel rather than the training-only input map.
MATCHED_CONSUMERS=('full_projection','sketch_projection','live_greedy')


def mapped_completions(backbone, heads, prefix_ids, span_ids, *, passes=2):
    """Neuralese positions read the token-to-Neuralese input map (``heads.input_map``) of the gold tokens.

    Pass zero reads gold text; pass one reads the mapped history in one parallel full-stack pass. In both passes
    the map's outputs (one per target slot) are fitted to the model's own stop-gradient projection at that slot
    (self-consistency), and the consumer reads the map's values detached: the map models the drift, the model
    learns to read it.
    """
    if span_ids.ndim != 2 or span_ids.shape[1] < 1 or prefix_ids.shape[1] < 1:
        raise ValueError('nonempty prefixes and gold target tokens required')
    start=prefix_ids.shape[1]-1
    def projected(top):
        with torch.no_grad():return heads.content(torch.zeros_like(top),top)
    ordinary=backbone.forward_ids(torch.cat([prefix_ids,span_ids[:,:-1]],1),cutoff=heads.cutoff,logits=False)
    top=ordinary['h_final'][:,start:]
    yield {'top':top,'sketches':heads.input_map(backbone.embed(span_ids)),'secondary_target':projected(top),
           'pass_index':0}
    del ordinary,top
    if passes<2:return
    with torch.no_grad():mapped=heads.input_map(backbone.embed(span_ids[:,:-1]))
    history=heads.read_embeddings(backbone,mapped)
    out=backbone.forward_embeds(torch.cat([backbone.embed(prefix_ids),history],1),cutoff=heads.cutoff,logits=False)
    top=out['h_final'][:,start:]
    yield {'top':top,'sketches':heads.input_map(backbone.embed(span_ids)),'secondary_target':projected(top),
           'pass_index':1}


def autoregressive_feedback_completion(backbone, heads, prefix_ids, span_ids):
    """Train a consumer on a detached, genuinely self-fed full-depth history.

    The producer rolls out ``heads.content`` one position at a time and reads
    each projected payload through every backbone layer. Gold IDs are not fed
    into this history; they remain fixed targets through its first differing
    decision. Later gold targets belong to another history and are masked from
    consumer training, while full-span diagnostic metrics remain available.
    Detaching the producer bounds activation memory and makes this exposure
    objective distinct from full BPTT through the rollout.
    """
    if (prefix_ids.ndim != 2 or span_ids.ndim != 2 or prefix_ids.shape[0] != span_ids.shape[0]
            or prefix_ids.shape[1] < 1 or span_ids.shape[1] < 1):
        raise ValueError('nonempty batched prefix and aligned target tokens required')
    if heads.read_markers:
        raise ValueError('autoregressive text feedback requires the raw read profile')
    with torch.no_grad():
        producer = prefill_write_context(backbone, heads, backbone.embed(prefix_ids))
        top, cache = producer.top, producer.cache
        payloads = []
        predictions = []
        for index in range(span_ids.shape[1]):
            predictions.append(backbone.logits(top).argmax(-1))
            if index + 1 < span_ids.shape[1]:
                payload, top, cache = full_depth_projected_feedback_step(
                    backbone, heads, top, cache)
            else:
                payload = heads.content(torch.zeros_like(top), top)
            payloads.append(payload)
        producer_payloads = torch.stack(payloads, dim=1).detach()
        producer_predictions = torch.stack(predictions, dim=1)
        del producer, cache, top, payloads, predictions
        # Same decisions, ordinary input coordinates. This target is recomputed
        # from the live model; gold span IDs are not fed into this history.
        ordinary_input = torch.cat((backbone.embed(prefix_ids),
                                    backbone.embed(producer_predictions[:, :-1])), dim=1)
        ordinary = backbone.forward_embeds(ordinary_input, cutoff=heads.cutoff, logits=False)
        ordinary_top = ordinary['h_final'][:, prefix_ids.shape[1] - 1:].detach().clone()
        del ordinary, ordinary_input

    # The final payload predicts the final target (including a real close
    # marker, where present) but is not consumed: no post-target state exists.
    history = heads.read_embeddings(backbone, producer_payloads[:, :-1])
    consumer_input = torch.cat((backbone.embed(prefix_ids), history), dim=1)
    consumer = backbone.forward_embeds(consumer_input, cutoff=heads.cutoff, logits=False)
    top = consumer['h_final'][:, prefix_ids.shape[1] - 1:]
    with torch.no_grad():
        secondary_target = heads.content(torch.zeros_like(top), top)
    return {'top': top,
            'sketches': heads.input_map(backbone.embed(span_ids)),
            'secondary_target': secondary_target,
            'producer_payloads': producer_payloads,
            'producer_predictions': producer_predictions,
            'ordinary_generated_top': ordinary_top,
            'gold_prefix_mask': causal_gold_prefix_mask(producer_predictions, span_ids),
            'pass_index': 1}


def consumer_position_weights(weights, gold_prefix_mask=None):
    """Preserve relative position weights inside the context-valid prefix.

    The readout and projection objectives reduce by position count. Rescaling
    retained weights to that count therefore normalizes by valid weighted mass,
    without shrinking the update merely because a rollout diverged early.
    """
    if gold_prefix_mask is None:
        return weights
    if gold_prefix_mask.shape != weights.shape or gold_prefix_mask.dtype != torch.bool:
        raise ValueError('aligned boolean gold prefix mask required')
    retained = weights * gold_prefix_mask.to(weights.dtype)
    mass = retained.sum()
    if not torch.isfinite(mass) or mass <= 0:
        raise ValueError('context-valid prefix needs positive finite weight mass')
    return retained * (weights.numel() / mass)


def text_history_completions(backbone, heads, prefix_ids, span_ids, *, passes,
                             ar_feedback_fixup=False):
    """Run the shared mapped-input objective or its detached self-fed fixup."""
    if ar_feedback_fixup:
        if passes != 2:
            raise ValueError('AR feedback fixup has exactly one gold control and one self-fed consumer pass')
        yield next(mapped_completions(backbone, heads, prefix_ids, span_ids, passes=1))
        yield autoregressive_feedback_completion(backbone, heads, prefix_ids, span_ids)
        return
    yield from mapped_completions(backbone, heads, prefix_ids, span_ids, passes=min(passes, 2))


def text_history_pass_count(schedule_passes, *, ar_feedback_fixup=False):
    """Map depth ramps to two passes; its AR fixup always uses both control and consumer."""
    return 2 if ar_feedback_fixup else min(schedule_passes, 2)


def matched_history_completion(backbone, heads, prefix_ids, span_ids, objective_completion):
    """Use the serving shallow feedback projection for matched-history diagnostics.

    A mapped warm-up's objective completion carries input-map outputs in its
    ``sketches`` slot. The matched-history control instead measures the
    independent serving ``heads.feedback`` projection at the corresponding
    gold-history states.
    """
    return gold_completion(backbone, heads, prefix_ids, span_ids)


def qualification(report, *, max_ce_delta=.1, max_relative_mse=.25,
                  min_agreement=.9):
    """All held strata must pass; no aggregate can hide a failing stratum."""
    strata=report.get('strata',{})
    return bool(strata) and all(r['tokens']>0 and
        r['ce_delta']<=max_ce_delta and r['embedding_mse_delta']<=max_relative_mse and
        r.get('text_ce_delta_from_initial',0.)<=max_ce_delta and
        r['text_argmax_agreement']>=min_agreement for r in strata.values())


def alignment_selection_score(report, *, max_ce_delta=.1, max_relative_mse=.25, min_agreement=.9):
    """Prefer certified alignment, then the worst held gate ratio, never prompt volume."""
    ratios=[]
    for row in report.get('strata',{}).values():
        if row.get('tokens',0)<=0:return (1,math.inf)
        for value,limit in ((row['ce_delta'],max_ce_delta),
                            (row['embedding_mse_delta'],max_relative_mse),
                            (row.get('text_ce_delta_from_initial',0.),max_ce_delta),
                            (1-row['text_argmax_agreement'],1-min_agreement)):
            if not math.isfinite(value):return (1,math.inf)
            ratios.append(max(0.,value)/limit if limit>0 else (0. if value<=0 else math.inf))
    return (0 if report.get('qualified') else 1,max(ratios,default=math.inf))


def _alignment_qualification_pass_depth(*, schedule):
    """Only a plateaued mapped two-pass objective can qualify alignment."""
    return 2 if schedule.plateau_reached else None


def retain_best_checkpoint(out, report):
    """Keep complete optimizer/model state without another GPU serialization.

    Current checkpoints are atomically replaced, so hard links preserve the old
    inode. The receipt is written last and binds both files; consumers verify it
    before using the serving export. Full-state checkpoint remains atomic alone.
    """
    out=Path(out)
    receipt={'schema':'natlang.neuralese-best-warmup-checkpoint/1','step':report['step'],
             'qualification':report,'files':{}}
    for name in ('checkpoint.pt','heads.pt'):
        source=out/name;destination=out/('best-'+name);pending=destination.with_suffix('.pending-link')
        pending.unlink(missing_ok=True)
        os.link(source,pending);pending.replace(destination)
        receipt['files'][destination.name]={'sha256':sha(destination),'bytes':destination.stat().st_size}
    pending=out/'best-checkpoint.json.pending'
    with pending.open('w') as stream:
        json.dump(receipt,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    pending.replace(out/'best-checkpoint.json')


def document_windows(token_ids, *, open_id, close_id, tokens, prefix_tokens, supervised_suffix_start=None,
                     context_tokens=0, target_tokens=None):
    """Prime with real open; supervise each text token and the real close once.

    The first ``context_tokens`` document tokens (the system prompt) are context only: they stay in the
    window as its crisp prefix, but are never targets, so they carry no loss and no held metric. With
    ``target_tokens``, ``tokens`` is the maximum total window and each lazy window advances by at most
    that supervised span, using all available preceding document tokens as context."""
    ids=array('I',[open_id]);ids.extend(token_ids);ids.append(close_id)
    bounded_target = target_tokens is not None
    if bounded_target:
        if type(target_tokens) is not int or target_tokens < 1 or target_tokens >= tokens:
            raise ValueError('target_tokens must be a positive integer smaller than tokens')
        stride=target_tokens
        context_capacity=tokens-target_tokens
    else:
        stride=tokens-prefix_tokens
        context_capacity=prefix_tokens
    context_end=1+max(0,int(context_tokens))
    windows=[]
    for offset in range(0,len(ids),stride):
        start=max(0,offset-context_capacity)
        end=min(len(ids),offset+stride)
        chunk=_TokenWindow(ids,start,end)
        width=max(1 if offset==0 else offset-start,context_end-start)
        if len(chunk)<=width:continue
        window={'ids':chunk,'prefix':width,'offset':offset,'start':start}
        if supervised_suffix_start is not None:
            window['supervised_suffix_start']=max(0,supervised_suffix_start+1-(start+width))
        windows.append(window)
    return windows


class _TokenWindow(Sequence):
    """A bounded view into one document, so short-target long-context windows share storage."""
    __slots__=('_source','_start','_end')

    def __init__(self, source, start, end):
        self._source=source;self._start=start;self._end=end

    def __len__(self):
        return self._end-self._start

    def __getitem__(self, index):
        if isinstance(index,slice):
            start,stop,step=index.indices(len(self))
            return [self._source[self._start+i] for i in range(start,stop,step)]
        if index<0:index+=len(self)
        if index<0 or index>=len(self):raise IndexError(index)
        return self._source[self._start+index]

    def __iter__(self):
        for index in range(self._start,self._end):yield self._source[index]

    def count(self, value):
        return sum(token==value for token in self)

    def __eq__(self, other):
        if isinstance(other,Sequence):return len(self)==len(other) and all(a==b for a,b in zip(self,other))
        return NotImplemented


def chat_roles(ids, *, start_id, role_ids, think_open=None, think_close=None):
    """Per-token chat role of a rendered document: the role named after each ``<|im_start|>``; assistant tokens
    split into reasoning (inside the think block) and reply. Structural start tokens count as 'other'."""
    codes=array('B');role='other';thinking=False;header=False
    for token in ids:
        if token==start_id:
            role='other';thinking=False;header=True;codes.append(0);continue
        if header:
            role=role_ids.get(token,'other');header=False
        if role=='assistant':
            if token==think_open:thinking=True
            name='assistant_reasoning' if thinking else 'assistant_reply'
            if token==think_close:thinking=False
        else:
            name=role
        codes.append(ROLE_CODES.index(name))
    return codes


def prepare_text_windows(engine, rows, *, tokens, prefix_tokens, target_tokens,
                         mask_system_prompt=True):
    """Build split-preserving text windows with the shared chat-role policy.

    System prompt tokens remain in the crisp context but can be excluded from
    supervised targets. Held windows retain per-token role labels for role
    diagnostics and the shared context loss. Labels use one byte per token and
    are shared across overlapping windows, including training windows.
    """
    tokenizer=engine.tokenizer
    backbone=engine.backbone

    def token_id(text):
        try:
            value=tokenizer.convert_tokens_to_ids(text)
            return value if isinstance(value,int) and value!=tokenizer.unk_token_id else None
        except Exception:
            return None

    role_start=token_id('<|im_start|>')
    role_ids=({i:name for name in ('system','user','assistant','tool')
               for i in [token_id(name)] if i is not None} if role_start is not None else {})
    if any('<|im_start|>' in row['text'] for row in rows) and (
            role_start is None or set(role_ids.values())!={'system','user','assistant','tool'}):
        raise ValueError('ChatML text requires effective role parsing before tool-feedback weighting')
    system_code=ROLE_CODES.index('system')
    windows={'train':[],'test':[]}
    masked_system_tokens=0
    for row in rows:
        split=row.get('split')
        if split not in windows:
            raise ValueError('text rows require train/test split')
        row_tokens=row['token_ids'] if 'token_ids' in row else engine._tokens(row['text'])
        labels=None
        context=0
        if role_start is not None:
            role_tokens=array('I',[backbone.controls.open_id]);role_tokens.extend(row_tokens);role_tokens.append(backbone.controls.close_id)
            labels=chat_roles(role_tokens,
                start_id=role_start,role_ids=role_ids,
                think_open=token_id('<think>'),think_close=token_id('</think>'))
            if mask_system_prompt:
                index=1
                while index<len(labels) and labels[index]==0:index+=1
                while index<len(labels) and labels[index]==system_code:index+=1
                context=index-1 if any(c==system_code for c in labels[1:index]) else 0
                masked_system_tokens+=context
        for window in document_windows(row_tokens,
                open_id=backbone.controls.open_id,close_id=backbone.controls.close_id,
                tokens=tokens,prefix_tokens=prefix_tokens,
                supervised_suffix_start=row.get('supervised_suffix_start'),
                context_tokens=context,target_tokens=target_tokens):
            if labels is not None:
                window['roles']=_TokenWindow(labels,window['start'],window['start']+len(window['ids']))
            windows[split].append({**window,
                'document':hashlib.sha256(row['text'].encode()).hexdigest(),
                'cohort':row.get('text_cohort','native'),
                'groups':row['source_groups']})
    receipt={'enabled':bool(mask_system_prompt and role_start is not None),
             'requested':bool(mask_system_prompt),'role_start_id':role_start,
             'masked_system_tokens':masked_system_tokens,'documents':len(rows)}
    return windows,receipt


def evaluation_batches(windows, limit, max_tokens=None):
    """Batch equal geometry within one held stratum; retain every held window.

    With max_tokens, a batch of long windows holds no more tokens than that (at least one window), so held
    evaluation never needs more activation memory than a training step on the same budget."""
    if limit<1:raise ValueError('positive evaluation batch required')
    buckets={}
    for window in windows:
        key=(window.get('cohort','native'),window['prefix'],len(window['ids']),window['offset']==0)
        buckets.setdefault(key,[]).append(window)
    for (_,_,length,_),bucket in buckets.items():
        size=limit if max_tokens is None else max(1,min(limit,max_tokens//max(1,length)))
        for start in range(0,len(bucket),size):yield bucket[start:start+size]


def select_held_document_windows(windows, limit):
    """Select held documents round-robin across a hash-ordered factual-group queue.

    Hashing the canonical complete group tuple removes lexical-prefix preference
    while keeping selection fixed for a given corpus. Within each group tuple,
    document hashes remain sorted; selected documents contribute bounded first
    and last windows. This function never changes row splits or source groups.
    """
    if limit < 1:raise ValueError('positive held document limit required')
    documents={};document_groups={};group_documents={}
    for window in windows:
        document=window['document']
        groups=tuple(sorted(set(window.get('groups') or ()))) or ('<unattributed>',)
        documents.setdefault(document,[]).append(window)
        document_groups.setdefault(document,set()).add(groups)
        group_documents.setdefault(groups,set()).add(document)
    def canonical(groups):return json.dumps(groups,ensure_ascii=False,separators=(',',':'))
    ordered_groups=sorted(group_documents,key=lambda group:(hashlib.sha256(canonical(group).encode()).hexdigest(),canonical(group)))
    group_order=[{'groups':list(group),'sha256':hashlib.sha256(canonical(group).encode()).hexdigest()} for group in ordered_groups]
    queues={group:sorted(group_documents[group]) for group in ordered_groups}
    selected=[];selected_by=[];seen=set()
    while len(selected)<limit and any(queues[group] for group in ordered_groups):
        for group in ordered_groups:
            queue=queues[group]
            while queue and queue[0] in seen:queue.pop(0)
            if queue and len(selected)<limit:
                document=queue.pop(0)
                seen.add(document);selected.append(document);selected_by.append(group)
    held=[];selected_metadata=[]
    for document,selected_group in zip(selected,selected_by):
        values=sorted(documents[document],key=lambda window:(window.get('offset',0),window.get('prefix',0),len(window.get('ids',[]))))
        if not values:continue
        held.append(values[0])
        if len(values)>1:held.append(values[-1])
        source_groups=sorted({name for group in document_groups[document] for name in group if name!='<unattributed>'})
        selected_metadata.append({'document_sha256':document,'source_groups':source_groups,
                                  'selected_group_tuple':list(selected_group),'window_count':len(values),
                                  'selected_window_offsets':[values[0].get('offset',0)] +
                                      ([values[-1].get('offset',0)] if len(values)>1 else [])})
    metadata={'policy':'sha256-ordered-complete-factual-group-tuples-round-robin-v1',
              'limit_documents':limit,'group_order':group_order,'selected_documents':selected_metadata,
              'window_policy':'retain first and last window per selected document; all selected windows flow to evaluation batching'}
    return held,metadata


def load_text_rows(records, pieces=None, text_data=None, *, tokenizer=None):
    """Explicit train/test and factual provenance; exact duplicates stay held out."""
    if text_data:
        rows=[]
        with Path(text_data).open() as source:
            for line in source:
                row=json.loads(line)
                if 'token_ids' in row:
                    ids=row['token_ids']
                    if not isinstance(ids,list) or any(type(i) is not int or not 0<=i<2**32 for i in ids):
                        raise ValueError('text token IDs must be unsigned integer coordinates')
                    row['token_ids']=array('I',ids)
                rows.append(row)
        if any(r.get('split') not in ('train','test') or not isinstance(r.get('text'),str)
               or not r['text'].strip() or not r.get('source_groups') for r in rows):
            raise ValueError('text JSONL needs nonempty text, train/test split and source_groups')
    else:
        import hashlib
        from ..data.text_corpus import gold_text_rows
        records_path=Path(records)
        record_lines=[line for line in records_path.read_bytes().splitlines() if line]
        record_rows=[json.loads(line) for line in record_lines]
        for row,line in zip(record_rows,record_lines):
            row['_source_record_sha256']=hashlib.sha256(line).hexdigest()
        pieces_path=Path(pieces) if pieces else records_path.parent/'pieces.jsonl'
        piece_rows=list(map(json.loads,pieces_path.open())) if pieces_path.is_file() else []
        rows,_,_,_=gold_text_rows(record_rows,piece_rows,tokenizer=tokenizer)
    if any(not isinstance(r.get('text_cohort','native'),str) or not r.get('text_cohort','native') for r in rows):
        raise ValueError('text_cohort must be a nonempty string')
    if any('supervised_suffix_start' in r and 'token_ids' not in r for r in rows):
        raise ValueError('supervised suffix requires native token IDs')
    encoded=[r for r in rows if 'token_ids' in r]
    if encoded:
        from ..data.text_corpus import tokenizer_fingerprint
        if tokenizer is None:raise ValueError('native gold token IDs require the student tokenizer')
        fingerprint=tokenizer_fingerprint(tokenizer)
        vocab_size=len(tokenizer)
        for row in encoded:
            ids=row['token_ids']
            if row.get('tokenizer_sha256')!=fingerprint or not isinstance(ids,(list,array)) or not ids or any(
                    type(i) is not int or i<0 or i>=vocab_size for i in ids):
                raise ValueError('gold token IDs or tokenizer fingerprint mismatch')
            if 'supervised_suffix_start' in row and (type(row['supervised_suffix_start']) is not int or
                    not 0<=row['supervised_suffix_start']<len(ids)):
                raise ValueError('gold supervised suffix coordinate is invalid')
    groups={s:set(g for r in rows if r['split']==s for g in r['source_groups']) for s in ('train','test')}
    if groups['train']&groups['test']:
        raise ValueError('text warm-up factual source groups cross train/test')
    held={r['text'] for r in rows if r['split']=='test'}
    dedup={}; excluded=0
    for row in rows:
        if row['split']=='train' and row['text'] in held:excluded+=1;continue
        ids_digest=hashlib.sha256(array('I',row.get('token_ids',[])).tobytes()).digest()
        key=(row['split'],row['text'],ids_digest,row.get('supervised_suffix_start'))
        if key in dedup and dedup[key].get('text_cohort','native')!=row.get('text_cohort','native'):
            raise ValueError('an identical document belongs to multiple cohorts; resolve ownership before training')
        dedup.setdefault(key,row)
    rows=list(dedup.values())
    if not all(any(r['split']==s for r in rows) for s in ('train','test')):
        raise ValueError('nonempty independent train and held text required')
    return rows, {'policy':'explicit source-disjoint ordinary text; gold targets, no teacher model',
                  'excluded_train_exact_held_duplicates':excluded,'documents':len(rows)}


def same_alignment_data(previous, current):
    """Whether a saved text-CE baseline measures the current held objective.

    Input digests make path-only moves equivalent. The initial text CE is also
    tied to the held window and target-mask policy, however, so reusing it
    across those changes would make ``text_ce_delta_from_initial`` compare
    different measurements. This check only controls that diagnostic baseline;
    it does not discard model, optimizer, or schedule state.
    """
    def fingerprints(identity):
        return {name: (identity['inputs'].get(identity['options'][name])
                       if identity['options'].get(name) else None)
                for name in ('records','pieces','text_data')}
    if fingerprints(previous) != fingerprints(current):
        return False
    # These options determine the tokens, windows, and strata contributing to
    # the held text-CE baseline. Missing fields in legacy checkpoints are
    # intentionally unequal: the old diagnostic's evaluation policy is not
    # authenticated well enough to reuse its value.
    fields = ('mask_system_prompt', 'held_documents', 'tokens', 'prefix_tokens', 'target_tokens',
              'qualification_cohort')
    old_options, new_options = previous.get('options', {}), current.get('options', {})
    if any(key not in old_options or key not in new_options
           for key in fields if key!='target_tokens'):
        return False
    if any(old_options.get(key) != new_options.get(key) for key in fields):
        return False
    for field in ('target', 'text_history', 'supervision_policy'):
        if previous.get(field) != current.get(field):
            return False
    return True


_FOUNDATION_CONTEXT_OPTIONS = (
    # These settings change the depth, token positions, or weighted objective
    # whose held projection plateau and recurrence alignment were measured.
    'cutoff', 'tokens', 'prefix_tokens', 'target_tokens',
    'embedding_weight', 'sketch_weight', 'text_weight',
    'cohort_weights', 'qualification_cohort', 'context_weight', 'feedback_weight',
    'projection_patience', 'projection_min_evals',
    'projection_min_improvement', 'backbone_ramp_evals', 'pass_ramp_evals',
)


def same_foundation_context(previous, current):
    """Whether a full-state handoff keeps the same foundation objective.

    Parameter and optimizer state can be shape-compatible across a changed
    cutoff, while the shallow target states and the projection plateau are
    different. Keep those states only when the declared depth, supervision
    and recurrence policy agree. A newer aligned-text corpus keeps them
    (owner: adopt new data at once): its crisp baseline is re-measured and
    qualification still needs consecutive passing evaluations on its held set.
    """
    fields = ('target', 'text_history', 'sketch_gradient',
              'sketch_target_backbone_scale', 'supervision_policy')
    if any(field not in previous or field not in current for field in fields):
        return False
    if any(previous[field] != current[field] for field in fields):
        return False
    old_options, new_options = previous.get('options', {}), current.get('options', {})
    if any(key not in old_options or key not in new_options
           for key in _FOUNDATION_CONTEXT_OPTIONS if key!='target_tokens'):
        return False
    return not any(old_options.get(key) != new_options.get(key) for key in _FOUNDATION_CONTEXT_OPTIONS)


def configure_student(engine, policy='full', rank=16, secondary_head='feedback'):
    backbone,heads=engine.backbone,engine.heads
    from .backbone_policy import configure_backbone_training
    backbone_named=configure_backbone_training(backbone,policy,rank=rank)
    for p in heads.parameters():p.requires_grad_(False)
    for p in getattr(heads,secondary_head).parameters():p.requires_grad_(True)
    for p in heads.content.proj.parameters():p.requires_grad_(True)
    # Keep vocabulary/embedding coordinates and output normalization stable.
    backbone.hf.eval();heads.eval()
    named=[('backbone.'+n,p) for n,p in backbone_named]
    named += [('heads.'+n,p) for n,p in heads.named_parameters() if p.requires_grad]
    if not any(n.startswith('backbone.') for n,p in named):raise ValueError('student full stack has no trainables')
    return named


def _apply_requested_sketch_cutoff(engine, cutoff, device):
    """Make the CLI cutoff the actual latent-sketch depth on the loaded engine."""
    if engine.heads.profile=='raw-token-v1':
        from .sketch_handoff import install_latent_sketch
        install_latent_sketch(engine,cutoff=cutoff,profile='latent-sketch-v2')
    elif engine.heads.profile=='latent-sketch-v2' and engine.heads.cutoff!=cutoff:
        # A requested cutoff is part of this run's identity. Rebuild the
        # same-shaped projection modules at that depth instead of silently
        # retaining the checkpoint's previous cutoff.
        from ..model.heads import PortHeads
        previous_heads=engine.heads
        requested_heads=PortHeads(engine.backbone,cutoff=cutoff,
            max_length=previous_heads.max_length,stop_source='final',
            stop_position=False,profile='latent-sketch-v2')
        requested_heads.load_state_dict(previous_heads.state_dict(),strict=True)
        requested_heads.to(device=device).eval()
        engine.heads=requested_heads
        engine.max_block=requested_heads.max_length
        engine.dialect=requested_heads.dialect
        proof=dict(getattr(engine,'foundation',None) or {})
        engine.foundation={**proof,'qualified':False,'runtime_qualified':False,
            'autonomous_stopping_qualified':False,'requires_requalification':True}
    if engine.heads.profile!='latent-sketch-v2':
        raise ValueError('requires latent-sketch-v2 or qualified raw foundation')
    if engine.heads.cutoff!=cutoff:
        raise ValueError('loaded latent-sketch cutoff does not match requested cutoff')
    return engine.heads


def load_initial(heads, checkpoint, device, cutoff):
    if checkpoint:
        from ..serve.recurrence_checkpoint import load_recurrence_checkpoint
        engine,parent=load_recurrence_checkpoint(checkpoint,device=device)
    else:
        from ..serve import load_engine
        engine=load_engine(heads_checkpoint=str(heads),device=device);parent=None
    _apply_requested_sketch_cutoff(engine,cutoff,device)
    return engine,parent


def main(argv=None):
    # Installed before any loading: a container's PID 1 ignores signals without a handler, so a stop requested
    # while the model loads would otherwise be dropped. The training loop checks the flag before each update.
    stop=StopSignal().install()
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('heads','records','out'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('pieces','text-data','student-checkpoint','continue-from'):p.add_argument('--'+name,type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,default=4096)
    p.add_argument('--tokens',type=int,default=1024);p.add_argument('--prefix-tokens',type=int,default=32)
    p.add_argument('--target-tokens',type=int,default=None,
                   help='optional bounded supervised span; tokens remains the total context capacity')
    p.add_argument('--cohort-weights',type=json.loads,default=None,
                   help='explicit JSON fractions for text_cohort names in an already admitted assembled input')
    p.add_argument('--qualification-cohort',default='native',
                   help='held cohort used for foundation gates; other held cohorts are reported separately')
    p.add_argument('--context-weight',type=float,default=1.)
    p.add_argument('--feedback-weight',type=float,default=.25,
                   help='relative tool-output weight inside context supervision; reasoning remains fully weighted')
    p.add_argument('--cutoff',type=int,default=4)
    p.add_argument('--batch',type=int,default=2,help='same-shape text rows per optimizer update')
    p.add_argument('--eval-batch',type=int,default=4,help='same-shape held rows per inference batch')
    p.add_argument('--backbone-training',choices=['auto','full','adapters','qat'],default='auto');p.add_argument('--rank',type=int,default=16)
    from .optim_restore import add_optimizer_restore_arguments;add_optimizer_restore_arguments(p)
    p.add_argument('--optimizer',choices=['muon','adamw'],default='muon');p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--sketch-lr',type=float,default=3e-4);p.add_argument('--embedding-weight',type=float,default=1.)
    p.add_argument('--sketch-weight',type=float,default=1.);p.add_argument('--text-weight',type=float,default=.25)
    p.add_argument('--channel-consistency-weight',type=float,default=1.,
                   help='effective update weight of same-generated-history ordinary-to-projected KL in the AR fixup')
    p.add_argument('--projection-patience',type=int,default=3)
    p.add_argument('--projection-min-evals',type=int,default=2)
    p.add_argument('--projection-min-improvement',type=float,default=.01)
    p.add_argument('--backbone-ramp-evals',type=int,default=4)
    p.add_argument('--pass-ramp-evals',type=int,default=2)
    p.add_argument('--checkpoint-every',type=int,default=128);p.add_argument('--eval-every',type=int,default=128)
    p.add_argument('--checkpoint-minutes',type=float,default=10.,
                   help='also save full resumable state when this much wall time passed since the last save')
    p.add_argument('--held-documents',type=int,default=16);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--checkpoint-layers',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--mask-system-prompt',action=argparse.BooleanOptionalAction,default=True,
                   help='the leading system prompt of each chat document is context only: no loss, no held metric')
    p.add_argument('--eval-only',action='store_true',
                   help='run one held evaluation of the restored state (use --continue-from into a fresh --out), '
                        'write eval-only.json and exit: no update, no checkpoint')
    p.add_argument('--ar-control-steps',type=int,default=256,
                   help='held autoregressive controls over this many positions of the first held batch; '
                        '0 disables. Diagnostic, not a gate')
    p.add_argument('--cuda-reserved-cap-gb',type=float,default=None,
                   help='cap the CUDA caching allocator (reserved bytes): at the cap it frees its cache and retries '
                        'instead of growing; on unified memory this keeps cache slack under the run\'s memory budget')
    p.add_argument('--ar-feedback-fixup',action='store_true',
                   help='train the mapped-input consumer on a detached full-depth autoregressive history; '
                        'the gold targets stay fixed')
    p.add_argument('--input-map-kernel',type=int,default=4);p.add_argument('--input-map-rank',type=int,default=64)
    p.add_argument('--qat-latent-lr',type=float,default=0.,
                   help='Maple QAT dense latents get their own AdamW groups at this rate times their matrix ternary scale '
                        '(a code flips after its latent moves ~0.5 of it); 0: Muon at the backbone rate, under which '
                        'codes practically never flip (v10: 3.6e-6 of them)')
    p.add_argument('--member-weight',type=float,default=0.,
                   help='nested-family students (MAPLE_NESTED §4a): after the projection-first phase, each update '
                        'also trains one member (in rotation) on the window: CE + KL(full || member), times this '
                        'weight; members\' private parts become trainable. 0 disables (members still evaluated)')
    p.add_argument('--member-tokens',type=int,default=2048,help='members train and evaluate on the last N window tokens')
    p.add_argument('--member-eval-windows',type=int,default=4)
    p.add_argument('--member-mask-system',action=argparse.BooleanOptionalAction,default=True,
                   help='member windows keep a leading system message as context (as the recurrence trainer)')
    p.add_argument('--read-adapter',action=argparse.BooleanOptionalAction,default=False,
                   help='reader-side Neuralese input adaptation (heads.NeuraleseReadAdapter), trained with the heads')
    p.add_argument('--max-ce-delta',type=float,default=.1);p.add_argument('--max-relative-mse',type=float,default=.25)
    p.add_argument('--min-agreement',type=float,default=.9);p.add_argument('--consecutive-gates',type=int,default=2)
    a=p.parse_args(argv)
    if a.ar_feedback_fixup and not a.continue_from:
        p.error('--ar-feedback-fixup requires a mapped --continue-from checkpoint')
    if min(a.steps,a.tokens,a.prefix_tokens,a.batch,a.eval_batch,a.eval_every,a.checkpoint_every,a.held_documents,a.consecutive_gates)<1 or a.tokens<3:
        p.error('positive bounds and at least three tokens required')
    if min(a.lr,a.sketch_lr,a.embedding_weight,a.sketch_weight,a.text_weight)<=0:
        p.error('invalid schedule or optimizer controls')
    if not math.isfinite(a.channel_consistency_weight) or a.channel_consistency_weight <= 0:
        p.error('channel consistency weight must be finite and positive')
    if not math.isfinite(a.context_weight) or a.context_weight<=0 or not math.isfinite(a.feedback_weight) or not 0<a.feedback_weight<=1:
        p.error('context weight must be positive and feedback weight in (0,1], both finite')
    if a.target_tokens is None and a.prefix_tokens>=a.tokens-1:p.error('prefix must leave at least two target tokens')
    if a.target_tokens is not None and (a.target_tokens<1 or a.target_tokens>=a.tokens):
        p.error('--target-tokens must be positive and smaller than --tokens')
    if not 0<=a.min_agreement<=1 or min(a.max_ce_delta,a.max_relative_mse)<0:p.error('invalid gates')
    torch.set_num_threads(2);torch.manual_seed(a.seed);random.seed(a.seed)
    options={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='out'}
    paths=[x for x in (a.heads,a.records,a.pieces,a.text_data,a.student_checkpoint,a.continue_from) if x]
    package=Path(__file__).parents[1]
    identity={'options':options,'inputs':{str(x.resolve()):sha(x) for x in paths},
              'code':{str(x.relative_to(package)):sha(x) for x in package.rglob('*.py')},
              'target':'E(gold next token), fixed raw input table; no teacher; full-stack next-token CE',
              'text_history':('gold-context control then detached sequential full-depth projected-payload history; '
                              'consumer gold supervision ends after its first differing decision; '
                              'live ordinary-distribution KL supervises the same generated history through close; '
                              'full-span held metrics remain diagnostic' if a.ar_feedback_fixup else
                              'gold seed; detached causal token-to-Neuralese input map; one parallel consumer pass'),
              'channel_consistency':{'target':'live stop-gradient ordinary conditional distribution on the same generated history',
                                     'mask':'through first generated close, inclusive',
                                     'effective_update_weight':a.channel_consistency_weight}
                  if a.ar_feedback_fixup else None,
              'ar_feedback_handoff_optimizer':'restore when parameter groups match; otherwise record a fresh optimizer with its reason'
                  if a.ar_feedback_fixup else None,
              'sketch_gradient':'detached_consumer',
              'sketch_target_backbone_scale':0.,
              'supervision_policy':text_supervision_policy(),
              'display':warmup_display_labels(),
              'checkpoint_selection':'qualified first, then worst held gate ratio; complete best full-state and serving-heads hard links'}
    identity['supervision_policy'].update(context_weight=a.context_weight,feedback_weight=a.feedback_weight,
                                         sampler='cohort-then-document-then-window/1',
                                         cohort_weights=a.cohort_weights)
    from ..common.artifact_paths import artifact_refs
    input_roles=artifact_refs(options,identity['inputs'],
        ('heads','records','pieces','text_data','student_checkpoint','continue_from'))
    state_path=a.out/'checkpoint.pt'
    resumed=torch.load(state_path,map_location='cpu',weights_only=False,mmap=True) if state_path.exists() else None
    was_resumed=resumed is not None
    code_handoffs=[]
    if resumed:
        if resumed.get('schema')!='natlang.neuralese-text-warmup/1':raise ValueError('warm-up resume identity changed')
        # Resuming is how a stopped run continues after a fix (owner): code changes, options that newer code
        # added (at their defaults) and operational options are a logged handoff. Recipe, input or objective
        # changes still refuse; they belong in a --continue-from lineage.
        old_options,new_options=resumed['identity'].get('options',{}),identity['options']
        option_changes=sorted(k for k in old_options.keys()&new_options.keys() if old_options[k]!=new_options[k])
        recipe_changes=[k for k in option_changes if k not in RESUME_OPERATIONAL_OPTIONS]
        if not same_resume_identity(resumed['identity'], identity):
            raise ValueError('warm-up resume identity changed'+(f' (options {recipe_changes})' if recipe_changes else ''))
        code_handoffs=list(resumed.get('code_handoffs',[]))
        old_code,new_code=resumed['identity'].get('code',{}),identity['code']
        added_options=sorted(new_options.keys()-old_options.keys())
        if old_code!=new_code or option_changes or added_options:
            handoff={'event':'code_handoff','step':resumed['step'],
                     'changed':sorted(k for k in old_code.keys()&new_code.keys() if old_code[k]!=new_code[k]),
                     'added':sorted(new_code.keys()-old_code.keys()),'removed':sorted(old_code.keys()-new_code.keys()),
                     'options_added':{k:new_options[k] for k in added_options},
                     'options_changed':{k:[old_options[k],new_options[k]] for k in option_changes}}
            code_handoffs.append(handoff)
            with (a.out/'code-handoffs.jsonl').open('a') as f:f.write(json.dumps(handoff)+'\n')
            print(json.dumps(handoff),flush=True)
    if a.out.exists() and not resumed:
        # Interruptible: an attempt stopped before its first checkpoint left only partial files. Keep them under
        # aborted-<time>/ and start fresh rather than refusing every later restart.
        partial=[x for x in a.out.iterdir() if not x.name.startswith('aborted-')]
        if partial:
            aborted=a.out/time.strftime('aborted-%Y%m%dT%H%M%S');aborted.mkdir()
            for x in partial:x.rename(aborted/x.name)
            print(json.dumps({'event':'partial_attempt_preserved','directory':aborted.name,
                              'files':sorted(x.name for x in partial)}),flush=True)
    continuation=None
    if a.continue_from:
        continuation=torch.load(a.continue_from,map_location='cpu',weights_only=False,mmap=True)
        if continuation.get('schema')!='natlang.neuralese-text-warmup/1':raise ValueError('full text warm-up state required')
        old=continuation['identity']['options']
        if any(old[k]!=options[k] for k in ('optimizer','rank','lr','sketch_lr')):
            raise ValueError('continuation optimizer/parameter policy differs')
        if a.ar_feedback_fixup and not any(str(key).startswith('input_map.') for key in continuation.get('heads',{})):
            raise ValueError('AR feedback fixup requires structurally present mapped-input heads')
    a.out.mkdir(parents=True,exist_ok=True)
    if a.cuda_reserved_cap_gb and a.device.startswith('cuda'):
        index=torch.device(a.device).index
        index=torch.cuda.current_device() if index is None else index
        total=torch.cuda.get_device_properties(index).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.,a.cuda_reserved_cap_gb*2**30/total),index)
    engine,parent=load_initial(a.heads,a.student_checkpoint,a.device,a.cutoff)
    backbone,heads=engine.backbone,engine.heads
    from .backbone_policy import resolve_backbone_policy
    a.backbone_training=resolve_backbone_policy(backbone,a.backbone_training)
    if continuation and resolve_backbone_policy(backbone,continuation['identity']['options']['backbone_training'])!=a.backbone_training:
        raise ValueError('continuation resolved backbone parameter policy differs')
    # Token chunks bound dense FFN temporaries; an MoE re-streams every expert's weights per chunk and its
    # per-token temporaries are small, so Maple runs the whole window (and both isolated streams) in one call.
    backbone.checkpoint_layers=a.checkpoint_layers
    default_ffn_chunk_tokens=1<<16 if getattr(backbone,'ternary',False) else 1024
    ffn_chunk_candidates=((default_ffn_chunk_tokens,) if getattr(backbone,'ternary',False)
                         else TEXT_WARMUP_FFN_CHUNKS)
    backbone.ffn_chunk_tokens=default_ffn_chunk_tokens
    secondary_metric_name='input_map'
    projection_schedule_name='input_map'
    from ..model.input_map import NeuraleseInputMap
    heads.add_module('input_map',NeuraleseInputMap(backbone.embedding_weight.shape[1],
        kernel=a.input_map_kernel,rank=a.input_map_rank).to(a.device))
    if a.read_adapter:
        heads.add_read_adapter()
    named=configure_student(engine,a.backbone_training,a.rank,secondary_head='input_map')
    if a.read_adapter:
        for n,q in heads.read_adapter.named_parameters():
            q.requires_grad_(True);named.append(('heads.read_adapter.'+n,q))
    secondary_prefix='heads.input_map.'
    from ..maple.family import family_members, private_parameters
    family=family_members(backbone)
    if a.member_weight and not family:raise ValueError('--member-weight needs a nested-family student')
    if a.member_weight:
        known={id(q) for _,q in named}
        for n,q in private_parameters(backbone):
            if id(q) not in known:q.requires_grad_(True);named.append(('backbone.'+n,q))
    from .memory_policy import effective_cuda_free_bytes, plan_saved_activation_offload
    full_memory_layout,shallow_memory_layout=_warmup_memory_layout(
        backbone,heads,checkpointed=a.checkpoint_layers)
    memory_geometry_version=(f'{TEXT_WARMUP_MEMORY_GEOMETRY}:k{heads.cutoff}:'
        f'l{backbone.num_layers}:w{backbone.config.hidden_size}:'
        f'i{full_memory_layout["intermediate"]}:kv{full_memory_layout["kv_width"]}:'
        f'skv{shallow_memory_layout["kv_width"]}:'
        f'd{backbone.embedding_weight.element_size()}:v{backbone.embedding_weight.shape[0]}:'
        f'prefix{a.prefix_tokens}:ckpt{int(a.checkpoint_layers)}:channelkl{int(a.ar_feedback_fixup)}')
    restored_memory=resumed or continuation or {}
    memory_estimator=AdaptiveGraphMemory(_warmup_readout_calibration_state(
        restored_memory.get('memory_estimator'),
        default_ffn_chunk_tokens=default_ffn_chunk_tokens),
        margin=0.,geometry_version=memory_geometry_version)
    saved_offload_state=restored_memory.get('activation_offload_state',{})
    # This is a planning assumption, not a measured calibration. Legacy state
    # used the same numeric key before the field was named explicitly.
    assumed_savings=float(saved_offload_state.get(
        'assumed_gpu_bytes_freed_per_cpu_byte',
        saved_offload_state.get('gpu_bytes_freed_per_cpu_byte',TEXT_WARMUP_OFFLOAD_SAVINGS_ASSUMPTION)))
    if not 0<assumed_savings<=1:raise ValueError('invalid saved activation offload assumption')
    offload_observations=list(saved_offload_state.get('observations',[]))[-64:]
    has_saved_memory_estimator=bool(restored_memory.get('memory_estimator'))
    memory_bootstrap_count=0
    previous_options=(continuation or {}).get('identity',{}).get('options',{})
    memory_bootstrap_compatible=bool(continuation and
        same_foundation_context(continuation.get('identity',{}),identity) and
        all(previous_options.get(k)==options.get(k) for k in
            ('prefix_tokens','cutoff','checkpoint_layers','backbone_training','optimizer','rank')))
    codes=None
    if a.backbone_training=='qat':
        from ..maple.ternary import CodeTracker
        codes=CodeTracker(backbone.hf)
    parameters=dict(named)
    backbone_names={n.removeprefix('backbone.') for n,q in named if n.startswith('backbone.')}
    from .optim import PortMuonAdamW
    if a.optimizer=='muon':
        latent_lrs={}
        if a.qat_latent_lr and a.backbone_training=='qat':
            # Each dense QAT latent at its own rate in units of its matrix's ternary scale (see qat_latent_scales).
            from .adapters import qat_latent_scales
            latent_lrs={'backbone.'+n:a.qat_latent_lr*v for n,v in qat_latent_scales(backbone).items()}
            print(json.dumps({'event':'qat_latent_groups','latents':len(latent_lrs),
                              'lr_min':min(latent_lrs.values(),default=None),'lr_max':max(latent_lrs.values(),default=None)}),flush=True)
        optimizer=PortMuonAdamW([(n,q) for n,q in named if n not in latent_lrs],lr=a.lr,vocab_size=backbone.embedding_weight.shape[0])
        for child in (optimizer.muon,optimizer.auxiliary):
            if child is None:continue
            original=dict(child.param_groups[0]); buckets={}
            for q in original['params']:buckets.setdefault(a.sketch_lr if any(q is v for n,v in named if n.startswith('heads.')) else a.lr,[]).append(q)
            first,*rest=buckets.items();child.param_groups[0].update(params=first[1],lr=first[0])
            for rate,qs in rest:child.add_param_group({**original,'params':qs,'lr':rate})
        optimizer.param_groups=optimizer._groups()
        for n,q in named:
            if n in latent_lrs:
                optimizer.add_param_group({'params':[q],'lr':latent_lrs[n],'weight_decay':0.,'qat_latent_lr':latent_lrs[n]})
                optimizer.schema[-1]['name']=n
    else:
        optimizer=torch.optim.AdamW([{'params':[q for n,q in named if n.startswith('backbone.')],'lr':a.lr},
          {'params':[q for n,q in named if n.startswith('heads.')],'lr':a.sketch_lr}],weight_decay=0.)
    rows,receipt=load_text_rows(a.records,a.pieces,a.text_data,tokenizer=engine.tokenizer)
    windows,mask_receipt=prepare_text_windows(engine,rows,tokens=a.tokens,
        prefix_tokens=a.prefix_tokens,target_tokens=a.target_tokens,
        mask_system_prompt=a.mask_system_prompt)
    if not all(windows.values()):raise ValueError('no token windows for a split')
    print(json.dumps({'event':'system_prompt_masking',**mask_receipt}),flush=True)
    sampler=DocumentWindowSampler(windows['train'],a.cohort_weights)
    held_by_cohort={}
    for window in windows['test']:
        held_by_cohort.setdefault(window['cohort'],[]).append(window)
    if set(held_by_cohort)!=set(sampler.cohorts) or a.qualification_cohort not in held_by_cohort:
        raise ValueError('each training cohort needs its own held split, including the qualification cohort')
    held=[];cohort_selections={}
    for cohort,values in held_by_cohort.items():
        selected,selection=select_held_document_windows(values,a.held_documents)
        held.extend(selected);cohort_selections[cohort]=selection
    held_selection=cohort_selections[a.qualification_cohort]
    group_order_sha256=hashlib.sha256(json.dumps(held_selection['group_order'],ensure_ascii=False,
        sort_keys=True,separators=(',',':')).encode()).hexdigest()
    held_selection_eval={k:held_selection[k] for k in ('policy','limit_documents','selected_documents','window_policy')}
    held_selection_eval.update(group_count=len(held_selection['group_order']),group_order_sha256=group_order_sha256)
    held_selection_eval['qualification_cohort']=a.qualification_cohort
    held_selection_eval['other_cohorts']={c:{k:v for k,v in s.items() if k!='group_order'}
                                        for c,s in cohort_selections.items() if c!=a.qualification_cohort}
    receipt.update(windows={s:len(v) for s,v in windows.items()},held_windows=len(held),
                   held_selection=held_selection,
                   serving_heads_export_policy={
                       'policy':'baseline, newly selected best, and final saves export heads.pt; non-best periodic checkpoints do not; emergencies attempt export after committing full state',
                       'lag':'during training heads.pt may represent an earlier step than checkpoint.pt; checkpoint.pt is authoritative for resume; optional emergency export failure is recorded in heads-export-status.json and does not invalidate recovery'},
                   boundaries={'policy':'one actual neuralese open/close token per complete document; no synthetic closes at window edges',
                               'open_id':backbone.controls.open_id,'close_id':backbone.controls.close_id},
                   trainable_parameters={s:sum(q.numel() for n,q in named if n.startswith(s)) for s in ('backbone.','heads.')})
    receipt['training_sampler']=sampler.receipt()
    receipt['memory_preflight']={'policy':'exact-shape geometry plus successful full-update calibration',
        'geometry_version':memory_geometry_version,'geometry_bootstrap_updates':memory_bootstrap_count,
        'predictor_margin':0.,'device_headroom_fraction':TEXT_WARMUP_MEMORY_HEADROOM,
        'chunk_policy':{'readout_candidates':list(TEXT_WARMUP_READOUT_CHUNKS),
            'ffn_candidates':list(ffn_chunk_candidates),
            'selection':'largest FFN tile, then largest readout tile, with independently forecast complete-update memory fitting current reusable capacity',
            'calibration_namespace':'separate per actual readout/FFN tile pair; legacy rows map to historical tile sizes',
            'ffn_workspace':'measured intermediate width, three expanded tensors, conservative float32 bytes for workspace delta above base tile',
            'evaluation_readout_chunk_tokens':128},
        'assumed_gpu_bytes_freed_per_cpu_byte':assumed_savings,
        'offload_trigger':'predicted update increment exceeds live reusable free bytes after headroom',
        'failure_policy':'preflight refusal before forward; no sample skipping or context truncation'}
    (a.out/'plan.json').write_text(json.dumps({'identity':identity,'receipt':receipt},indent=2)+'\n')
    def ids_for(w):
        rows=[w] if isinstance(w,dict) else w
        ids=torch.tensor([list(r['ids']) for r in rows],device=a.device)
        span=ids[:,rows[0]['prefix']:]
        roles=(torch.tensor([list(r['roles'][r['prefix']:]) for r in rows],device=span.device)
               if all('roles' in r for r in rows) else None)
        return ids[:,:rows[0]['prefix']],span,balanced_position_weights(
            span,[r.get('supervised_suffix_start') for r in rows],roles=roles,
            context_weight=a.context_weight,feedback_weight=a.feedback_weight)
    mask_system=bool(a.member_weight and a.member_mask_system)
    if mask_system:
        system_start=engine.tokenizer('<|im_start|>system',add_special_tokens=False).input_ids
        system_end=engine.tokenizer.convert_tokens_to_ids('<|im_end|>')
    def member_window(w):
        """The last --member-tokens of a window, labelled on its supervised (non-context) positions; with
        --member-mask-system a leading system message is context too (as in the recurrence trainer)."""
        prefix,span,_=ids_for(w)
        ids=torch.cat([prefix,span],1)
        context=prefix.shape[1]
        if mask_system:
            context=max(context,leading_system_tokens(ids[0].tolist(),system_start,system_end,
                                                      engine.tokenizer.bos_token_id))
        labels=window_labels(ids,min(max(1,context),ids.shape[1]-1))
        return ids[:,-a.member_tokens:],labels[:,-a.member_tokens:]
    def objective_pass(out,span,baseline,bootstrap,weights,readout_chunk_tokens,projected_observer=None,roles=None,objective_passes=1):
        evaluation=not torch.is_grad_enabled()
        top=out['top']
        weights=consumer_position_weights(weights,out.get('gold_prefix_mask'))
        ce,training_ce,prediction,close_probability,token_losses=chunked_readout(
            backbone,top,span,backbone.controls.close_id,chunk_size=readout_chunk_tokens,
            gradients=not bootstrap,position_weights=weights)
        if evaluation and out['pass_index']==0 and projected_observer is not None:
            projected_observer(out,prediction)
        target=backbone.embed(span).detach()
        embedding_positions,secondary_positions=projection_errors(heads,top,out['sketches'],target,
                                                               secondary_target=out.get('secondary_target'))
        embedding,secondary=embedding_positions.mean(),secondary_positions.mean()
        secondary_name='input_map'
        if out['pass_index']==0:
            with torch.no_grad():
                baseline.update(prediction=prediction,ce=ce.detach(),
                    embedding=relative_mse(heads.content.reference(top.detach()),target))
                if evaluation:
                    baseline.update(token_losses=token_losses.detach(),
                                    tail_reference=heads.content.reference(top[:,-256:].detach()))
        plain_prediction=baseline['prediction'];plain_ce=baseline['ce'];plain_embedding=baseline['embedding']
        # Both separate projections receive full-strength gold supervision from
        # the first update. CE joins only when the backbone is gently unfrozen.
        supervised_embedding=(embedding_positions*weights).mean()
        supervised_secondary=(secondary_positions*weights).mean()
        loss=a.embedding_weight*supervised_embedding+a.sketch_weight*supervised_secondary
        if not bootstrap:
            loss=loss+training_ce+(a.text_weight*training_ce if out['pass_index']==0 else 0.)
        channel_metrics = {name: top.new_zeros((), dtype=torch.float32) for name in
                           ('channel_consistency_kl', 'channel_consistency_agreement', 'channel_consistency_tokens')}
        if 'ordinary_generated_top' in out:
            from .channel_objective import generated_history_channel_loss
            channel_loss, channel_metrics = generated_history_channel_loss(
                backbone, out['ordinary_generated_top'], top, out['producer_predictions'],
                backbone.controls.close_id, chunk_size=readout_chunk_tokens,
                gradients=not bootstrap)
            if not bootstrap:
                # Updates average their objective passes. Compensate here so the
                # declared weight is the coefficient in the complete update.
                loss = loss + objective_passes * a.channel_consistency_weight * channel_loss
        with torch.no_grad():
            ending=span==backbone.controls.close_id
            # The same detached reductions serve training and evaluation. Only
            # their host extraction timing differs; empty selections produce
            # NaN reductions normalized by the shared materializer below.
            stop_metrics={'close_targets':ending.sum().detach(),
              'close_probability':close_probability[ending].mean().detach(),
              'close_top1':(prediction[ending]==backbone.controls.close_id).float().mean().detach(),
              'premature_close_top1':(prediction[~ending]==backbone.controls.close_id).float().mean().detach()}
        secondary_name='input_map'
        mask=out.get('gold_prefix_mask',torch.ones_like(span,dtype=torch.bool))
        if evaluation:
            prefix_stats=causal_prefix_metrics(
                prediction.detach(),span,token_losses.detach(),mask,
                reference_prediction=plain_prediction.detach(),
                reference_token_losses=baseline['token_losses'],
                tail_tokens=256,include_windows=True,
                producer_predictions=out.get('producer_predictions'))
        else:
            # Training retains its existing coverage telemetry without doing
            # evaluation-only CE/agreement reductions or host synchronization.
            prefix_stats={'context_valid_tokens':mask.sum().detach(),
                          'context_valid_fraction':mask.float().mean().detach()}
        metrics={'ce':ce.detach(),'text_ce':plain_ce.detach(),'ce_delta':(ce-plain_ce).detach(),
          'relative_mse':embedding.detach(),secondary_name+'_mse':secondary.detach(),
          'text_embedding_mse':plain_embedding.detach(),
          'embedding_mse_delta':(embedding-plain_embedding).detach(),
          'text_argmax_agreement':(prediction==plain_prediction).float().mean().detach(),
          'gold_accuracy':(prediction==span).float().mean().detach(),
          'tokens':span.numel(),'positions':span.shape[1],**stop_metrics,
          'supervised_ce':training_ce.detach(),
          'supervised_embedding_mse':supervised_embedding.detach(),
          'supervised_'+secondary_name+'_mse':supervised_secondary.detach(),
          'context_valid_gold_tokens':prefix_stats['context_valid_tokens'],
          'context_valid_gold_fraction':prefix_stats['context_valid_fraction']}
        metrics.update(channel_metrics)
        if evaluation:
            metrics.update({
              'context_valid_gold_ce':prefix_stats['context_valid_ce'],
              'context_valid_gold_accuracy':prefix_stats['context_valid_accuracy'],
              'context_valid_text_argmax_agreement':prefix_stats['context_valid_text_argmax_agreement'],
              'context_valid_ce_delta':prefix_stats['context_valid_ce_delta'],
              'context_valid_last256_target_tokens':prefix_stats['tail_target_tokens'],
              'context_valid_last256_gold_tokens':prefix_stats['tail_context_valid_tokens'],
              'context_valid_last256_gold_fraction':prefix_stats['tail_context_valid_fraction'],
              'context_valid_last256_gold_ce':prefix_stats['tail_context_valid_ce'],
              'context_valid_last256_gold_accuracy':prefix_stats['tail_context_valid_accuracy'],
              'context_valid_last256_text_argmax_agreement':prefix_stats['tail_context_valid_text_argmax_agreement'],
              'context_valid_last256_ce_delta':prefix_stats['tail_context_valid_ce_delta']})
            metrics['context_valid_windows']=prefix_stats['windows']
        metrics['pass_index']=out['pass_index']
        if evaluation and span.shape[1]>256:
            with torch.no_grad():
                tail_top=top[:,-256:]
                metrics['regions']={'last256':alignment_region_metrics(
                    token_losses[:,-256:],prediction[:,-256:],baseline['token_losses'][:,-256:],
                    plain_prediction[:,-256:],heads.content(torch.zeros_like(tail_top),tail_top),
                    out['sketches'][:,-256:],baseline['tail_reference'],target[:,-256:],span[:,-256:],
                    secondary_name=secondary_name)}
        if evaluation:
            metrics=materialize_objective_metrics([metrics],secondary_projection=secondary_name)[0][0]
            if roles is not None:
                with torch.no_grad():
                    by_role={}
                    deltas=token_losses-baseline['token_losses']
                    for code,name in enumerate(ROLE_CODES):
                        selected=roles==code;count=int(selected.sum())
                        if not count:continue
                        by_role[name]={'tokens':count,'ce':float(token_losses[selected].float().mean()),
                            'ce_delta':float(deltas[selected].float().mean()),
                            'gold_accuracy':float((prediction==span)[selected].float().mean()),
                            'text_argmax_agreement':float((prediction==plain_prediction)[selected].float().mean())}
                    metrics['roles']=by_role
        return loss,metrics

    def objective(w,passes,bootstrap=False,readout_chunk_tokens=128,projected_observer=None):
        prefix,span,weights=ids_for(w);baseline={}
        rows=[w] if isinstance(w,dict) else w
        roles=(torch.tensor([list(r['roles'][r['prefix']:]) for r in rows],device=span.device)
               if not torch.is_grad_enabled() and all('roles' in r for r in rows) else None)
        completions=text_history_completions(backbone,heads,prefix,span,passes=passes,
            ar_feedback_fixup=a.ar_feedback_fixup)
        for out in completions:
            yield objective_pass(out,span,baseline,bootstrap,weights,readout_chunk_tokens,projected_observer,roles,objective_passes=passes)

    step=0;streak=0;best=None;updates={'backbone':False,'input_map':False,'full_projection':False}
    initial_text_ce={}
    remeasure_text_baseline=False
    schedule=ProjectionFirstSchedule(heads=(projection_schedule_name,'full_depth'),
        min_evals=a.projection_min_evals,patience=a.projection_patience,
        min_relative_improvement=a.projection_min_improvement,
        backbone_ramp_evals=a.backbone_ramp_evals,pass_ramp_evals=a.pass_ramp_evals)
    last_schedule_step=None
    restored=resumed or continuation
    if restored:
        # A continuation may freeze a formerly trained head (e.g. feedback when
        # switching to the input map). Restore its weights independently of the
        # current optimizer's trainable set; it remains part of the same model.
        restore_parameters={**parameters,**{'heads.'+n:q for n,q in heads.named_parameters()},
                            **{'backbone.'+n:q for n,q in private_parameters(backbone)}}
        with torch.no_grad():
            for n,v in restored['student_parameters'].items():
                restore_parameters[n].copy_(v.to(restore_parameters[n]))
        del restore_parameters
        fresh_map=not any(k.startswith('input_map.') for k in restored['heads'])
        if fresh_map:
            # Raw foundation weights acquire the shared mapped-input objective here.
            missing,unexpected=heads.load_state_dict(restored['heads'],strict=False)
            if unexpected or any(not k.startswith('input_map.') for k in missing):
                raise ValueError(f'continuation heads differ beyond the new input map: {missing} {unexpected}')
            print(json.dumps({'event':'input_map_initialized','optimizer_state':'fresh'}),flush=True)
        else:
            missing,unexpected=heads.load_state_dict(restored['heads'],strict=False)
            if unexpected or any(not (a.read_adapter and k.startswith('read_adapter.')) for k in missing):
                raise ValueError(f'restored heads differ beyond a new read adapter: {missing} {unexpected}')
            # A read adapter new at this continuation starts at identity; its optimizer slots are declared added.
            added_prefixes=list(a.optimizer_added)+(['heads.read_adapter.'] if missing else [])
            from .optim_restore import (restore_optimizer_state,declared_added_names,
                                        record_restore_report)
            # Named restore: unchanged groups load exactly as before; a grown/shrunk/rerouted trainable set is
            # refused unless declared (--optimizer-added PREFIX) or the reset is explicit (--optimizer-state fresh).
            restore_report=restore_optimizer_state(optimizer,restored['optimizer'],dict(named),
                declared_added_names(named,added_prefixes),
                saved_names=restored.get('optimizer_param_names'),fresh=a.optimizer_state=='fresh' and not resumed)
            record_restore_report(restore_report,a.out,source=a.continue_from or 'resume')
            if continuation:
                print(json.dumps({'event':'optimizer_state_restored',
                                  'source_step':continuation['step'],
                                  'parameter_groups':len(optimizer.param_groups)}),flush=True)
        step=restored['step'];updates=display_update_flags(restored['updates'])
        updates.setdefault('full_projection',False)
        if resumed:
            streak=resumed['streak'];best=resumed['best'];initial_text_ce=resumed['initial_text_ce']
            schedule.load_state_dict(resumed['schedule'])
            last_schedule_step=resumed['last_schedule_step']
        elif continuation:
            # The initial mapped-to-AR handoff preserves its established phase,
            # but not across a changed sampling or context-supervision policy.
            initial_ar_handoff=(a.ar_feedback_fixup and
                not continuation['identity'].get('options',{}).get('ar_feedback_fixup',False) and
                continuation['identity'].get('supervision_policy')==identity['supervision_policy'] and
                any(str(key).startswith('input_map.') for key in continuation.get('heads',{})))
            same_foundation=same_foundation_context(continuation['identity'],identity) or initial_ar_handoff
            if same_foundation and continuation.get('schedule'):
                # An unchanged objective may continue its plateau/ramp phase.
                saved_schedule_heads=(continuation['schedule'].get('config') or {}).get('heads')
                if a.ar_feedback_fixup and saved_schedule_heads==['shallow','full_depth']:
                    # The map checkpoint predates the shared metric label:
                    # its saved ``shallow`` key is this same input-map metric.
                    # Preserve its exact schedule history through the narrow
                    # alias migration instead of resetting the plateau.
                    schedule.load_mapped_input_handoff_state_dict(continuation['schedule'])
                    print(json.dumps({'event':'mapped_input_schedule_head_renamed',
                                      'source_head':'shallow','current_head':'input_map',
                                      'source_step':continuation['step'],
                                      'preserved_eval_count':schedule.eval_count}),flush=True)
                else:
                    schedule.load_state_dict(continuation['schedule'])
                last_schedule_step=continuation['last_schedule_step']
                print(json.dumps({'event':'foundation_schedule_restored',
                                  'source_step':continuation['step'],
                                  'last_schedule_step':last_schedule_step,
                                  'schedule':schedule.controls()}),flush=True)
            else:
                # A changed depth/supervision objective starts a new plateau
                # and must earn its own update and qualification evidence.
                updates={'backbone':False,'input_map':False,'full_projection':False}
                print(json.dumps({'event':'foundation_schedule_reinitialized',
                                  'source_step':continuation['step'],
                                  'same_foundation_context':same_foundation}),flush=True)
            if same_alignment_data(continuation['identity'],identity) and not a.ar_feedback_fixup:
                initial_text_ce=continuation['initial_text_ce']
            else:
                remeasure_text_baseline=True
        restore_training_rng_state(restored,a.device)
    # Restoring a state_dict intentionally invalidates the cached zero-correction
    # proof. Re-establish it only after trainability flags and restored values
    # are both final; nonzero or trainable references remain on the normal path.
    heads.configure_frozen_reference()
    # All model, optimizer, RNG, schedule, and resource state has now been
    # copied into live objects. Drop mmap-backed parent checkpoint aliases so
    # their file mappings do not survive through the training loop.
    del restored, resumed, continuation, restored_memory, saved_offload_state
    for group in optimizer.param_groups:
        projection=all(any(q is v for n,v in named if n.startswith('heads.')) for q in group['params'])
        group['foundation_base_lr']=group.get('qat_latent_lr') or (a.sketch_lr if projection else a.lr)
        group['foundation_projection']=projection
    if not was_resumed and not has_saved_memory_estimator and a.continue_from and memory_bootstrap_compatible:
        memory_bootstrap_count=_seed_warmup_memory_estimator(
            memory_estimator,a.continue_from.parent/'train.jsonl',
            prefix_tokens=a.prefix_tokens,full_layout=full_memory_layout,
            shallow_layout=shallow_memory_layout,cutoff=heads.cutoff,
            vocab_size=backbone.embedding_weight.shape[0],batch_size=a.batch,
            named=named,optimizer=optimizer,
            default_ffn_chunk_tokens=default_ffn_chunk_tokens,
            channel_consistency=a.ar_feedback_fixup)
        plan_path=a.out/'plan.json'
        if plan_path.is_file():
            plan_doc=json.loads(plan_path.read_text())
            plan_doc['receipt']['memory_preflight']['geometry_bootstrap_updates']=memory_bootstrap_count
            plan_path.write_text(json.dumps(plan_doc,indent=2)+'\n')
    def log(name,value):
        with (a.out/name).open('a') as f:f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    checkpoint_reserve=None
    checkpoint_snapshot_size_bound=None
    checkpoint_writer=AsyncAtomicCheckpointWriter()
    def drain_checkpoint_writer_at_exit():
        try:checkpoint_writer.drain()
        except Exception:traceback.print_exc()
    atexit.register(drain_checkpoint_writer_at_exit)
    last_report=None
    def evaluate(*,observe_schedule=True,baseline_reason=None,baseline_rng_preserved=False):
        nonlocal last_schedule_step,last_report
        evaluation_passes=text_history_pass_count(3, ar_feedback_fixup=a.ar_feedback_fixup)
        strata={};cohort_strata={};matched_history_rows=[];boundaries={'close_targets':0,'close_probability_sum':0.,'close_top1_sum':0.}
        ar_batch=None;ar_fallback=None;role_strata={};cohort_role_strata={}
        with torch.no_grad():
            for batch in evaluation_batches(held,a.eval_batch,a.tokens):
                w=batch[0]
                cohort=w['cohort'];qualifying=cohort==a.qualification_cohort
                active_strata=strata if qualifying else cohort_strata.setdefault(cohort,{})
                prefix,span,_=ids_for(batch)
                if qualifying and ar_fallback is None:ar_fallback=(prefix,span)
                if qualifying and ar_batch is None and 'roles' in batch[0]:
                    # Self-fed rollouts start at the first assistant token: the shared system prompt is memorized
                    # boilerplate and would make every rollout trivially exact.
                    assistant={ROLE_CODES.index('assistant_reasoning'),ROLE_CODES.index('assistant_reply')}
                    first=next((i for i,c in enumerate(batch[0]['roles']) if i>=1 and c in assistant),None)
                    if first is not None and len(batch[0]['ids'])-first>=16:
                        whole=torch.tensor([list(batch[0]['ids'])],device=a.device)
                        ar_batch=(whole[:,:first],whole[:,first:])
                def observe_projected_history(completion,live_tokens):
                    from ..eval.projected_history import projected_history_metrics
                    matched_completion=matched_history_completion(backbone,heads,prefix,span,completion)
                    diagnostic=projected_history_metrics(backbone,heads,prefix,span,
                        completion=matched_completion,live_tokens=live_tokens,
                        consumers=MATCHED_CONSUMERS,per_window=True)
                    for window,window_scores in zip(batch,diagnostic['windows']):
                        matched_history_rows.append({'document_sha256':window['document'],
                            'cohort':cohort,
                            'source_groups':window['groups'],'offset':window['offset'],
                            'prefix_tokens':window['prefix'],'target_tokens':len(window['ids'])-window['prefix'],
                            'scores':window_scores})
                for _,m in objective(batch,evaluation_passes,projected_observer=observe_projected_history):
                    if qualifying and m['pass_index']==1:
                        boundaries['close_targets']+=m['close_targets']
                        if m['close_targets']:
                            boundaries['close_probability_sum']+=m['close_probability']*m['close_targets']
                            boundaries['close_top1_sum']+=m['close_top1']*m['close_targets']
                    for name,values in m.get('roles',{}).items():
                        active_roles=role_strata if qualifying else cohort_role_strata.setdefault(cohort,{})
                        row=active_roles.setdefault('pass-'+str(m['pass_index']),{}).setdefault(name,{'tokens':0})
                        for n,value in values.items():
                            if n!='tokens':row[n]=row.get(n,0.)+value*values['tokens']
                        row['tokens']+=values['tokens']
                    key='pass-'+str(m['pass_index'])+'-length-'+('short' if m['positions']<=32 else 'medium' if m['positions']<=128 else 'long')+'-'+('start' if w['offset']==0 else 'tail')
                    row=active_strata.setdefault(key,{'tokens':0})
                    for n in ('ce','text_ce','ce_delta','relative_mse',secondary_metric_name+'_mse','text_embedding_mse','embedding_mse_delta','text_argmax_agreement','gold_accuracy'):
                        row[n]=row.get(n,0.)+m[n]*m['tokens']
                    row['tokens']+=m['tokens']
                    channel_count=int(m['channel_consistency_tokens'])
                    row['channel_consistency_tokens']=row.get('channel_consistency_tokens',0)+channel_count
                    for name in ('channel_consistency_kl','channel_consistency_agreement'):
                        key_sum=name+'_weighted_sum'
                        row[key_sum]=row.get(key_sum,0.)+m[name]*channel_count
                    valid_count=int(m['context_valid_gold_tokens'])
                    row['context_valid_gold_tokens']=row.get('context_valid_gold_tokens',0)+valid_count
                    row['context_valid_gold_fraction']=row.get('context_valid_gold_fraction',0)+valid_count
                    for n in ('context_valid_gold_ce','context_valid_gold_accuracy',
                              'context_valid_text_argmax_agreement','context_valid_ce_delta'):
                        key_sum=n+'_weighted_sum'
                        row[key_sum]=row.get(key_sum,0.)+m[n]*valid_count
                    tail_targets=int(m['context_valid_last256_target_tokens'])
                    tail_valid=int(m['context_valid_last256_gold_tokens'])
                    row['context_valid_last256_target_tokens']=row.get('context_valid_last256_target_tokens',0)+tail_targets
                    row['context_valid_last256_gold_tokens']=row.get('context_valid_last256_gold_tokens',0)+tail_valid
                    for n in ('context_valid_last256_gold_ce','context_valid_last256_gold_accuracy',
                              'context_valid_last256_text_argmax_agreement','context_valid_last256_ce_delta'):
                        key_sum=n+'_weighted_sum'
                        row[key_sum]=row.get(key_sum,0.)+m[n]*tail_valid
                    window_rows=row.setdefault('context_valid_windows',[])
                    for window,detail in zip(batch,m.get('context_valid_windows',[])):
                        window_rows.append({**detail,'document_sha256':window['document'],
                            'source_groups':window['groups'],'offset':window['offset'],
                            'prefix_tokens':window['prefix']})
                    for region,values in m.get('regions',{}).items():
                        regional=active_strata.setdefault(key+'-'+region,{'tokens':0})
                        for n,value in values.items():
                            if n!='tokens':regional[n]=regional.get(n,0.)+value*values['tokens']
                        regional['tokens']+=values['tokens']
                        if region == 'last256':
                            # This row's target span is exactly the evaluated
                            # right-aligned tail, so use the tail/prefix
                            # intersection computed from that same window.
                            tail_targets=int(m['context_valid_last256_target_tokens'])
                            tail_valid=int(m['context_valid_last256_gold_tokens'])
                            regional['context_valid_gold_tokens']=regional.get('context_valid_gold_tokens',0)+tail_valid
                            regional['context_valid_gold_fraction']=regional.get('context_valid_gold_fraction',0)+tail_valid
                            regional['context_valid_last256_target_tokens']=regional.get('context_valid_last256_target_tokens',0)+tail_targets
                            regional['context_valid_last256_gold_tokens']=regional.get('context_valid_last256_gold_tokens',0)+tail_valid
                            for source_name, target_name in (
                                ('context_valid_last256_gold_ce','context_valid_gold_ce'),
                                ('context_valid_last256_gold_accuracy','context_valid_gold_accuracy'),
                                ('context_valid_last256_text_argmax_agreement','context_valid_text_argmax_agreement'),
                                ('context_valid_last256_ce_delta','context_valid_ce_delta'),
                                ('context_valid_last256_gold_ce','context_valid_last256_gold_ce'),
                                ('context_valid_last256_gold_accuracy','context_valid_last256_gold_accuracy'),
                                ('context_valid_last256_text_argmax_agreement','context_valid_last256_text_argmax_agreement'),
                                ('context_valid_last256_ce_delta','context_valid_last256_ce_delta')):
                                key_sum=target_name+'_weighted_sum'
                                regional[key_sum]=regional.get(key_sum,0.)+m[source_name]*tail_valid
            autoregressive_controls=None
            if ar_batch is None:ar_batch=ar_fallback
            if a.ar_control_steps and ar_batch is not None:
                from ..eval.projected_history import autoregressive_history_metrics
                started_ar=time.perf_counter()
                autoregressive_controls=autoregressive_history_metrics(backbone,heads,*ar_batch,steps=a.ar_control_steps)
                autoregressive_controls['seconds']=time.perf_counter()-started_ar
                autoregressive_controls['start']='first assistant token' if ar_batch is not ar_fallback else 'window start'
        _normalize_context_valid_strata(strata)
        for values in cohort_strata.values():
            _normalize_context_valid_strata(values)
        for key,row in strata.items():
            initial_text_ce.setdefault(key,row['text_ce'])
            row['text_ce_delta_from_initial']=row['text_ce']-initial_text_ce[key]
        projection_rows=[r for k,r in strata.items() if k.startswith('pass-0-') and not k.endswith('-last256')]
        total=sum(r['tokens'] for r in projection_rows)
        errors={projection_schedule_name:sum(r[secondary_metric_name+'_mse']*r['tokens'] for r in projection_rows)/total,
                'full_depth':sum(r['relative_mse']*r['tokens'] for r in projection_rows)/total}
        history_rows=[r for k,r in strata.items() if k.startswith('pass-1-') and not k.endswith('-last256')]
        history_tokens=sum(r['tokens'] for r in history_rows)
        history_ce_delta=sum(r['ce_delta']*r['tokens'] for r in history_rows)/history_tokens if history_tokens else None
        pass_ce_deltas={}
        for key,row in strata.items():
            if key.endswith('-last256'):continue
            index=int(key.split('-')[1]);total_row=pass_ce_deltas.setdefault(index,[0.,0])
            total_row[0]+=row['ce_delta']*row['tokens'];total_row[1]+=row['tokens']
        pass_ce_deltas={index:value/count for index,(value,count) in pass_ce_deltas.items() if count}
        if observe_schedule and (last_schedule_step is None or step>last_schedule_step):
            schedule.observe(errors);last_schedule_step=step
        from .trajectory_state import weights_digest
        report={'step':step,'strata':strata,'cohort_strata':cohort_strata,
                'qualification_cohort':a.qualification_cohort,
                'runtime_qualified':False,'autonomous_stopping_qualified':False,
                'boundary_supervision':boundaries,'text_history_policy':identity['text_history'],
                'held_probe_selection':held_selection_eval,
                'text_ce_baseline_domain':{
                    'schema':'natlang.text-warmup-baseline-domain/1',
                    'metric':'held plain-text next-token CE by pass/length/region',
                    'source_inputs_sha256':{
                        str(path.resolve()):identity['inputs'][str(path.resolve())]
                        for path in (a.records,a.pieces,a.text_data) if path is not None and
                        str(path.resolve()) in identity['inputs']},
                    'system_prompt_mask_requested':bool(a.mask_system_prompt),
                    'system_prompt_mask_effective':mask_receipt['enabled'],
                    'held_selection':held_selection_eval,
                    'window_tokens':a.tokens,'prefix_tokens':a.prefix_tokens,
                    'evaluation_passes':evaluation_passes},
                'weights_digest':weights_digest({n:q for n,q in backbone.hf.named_parameters() if n in backbone_names},heads.state_dict()),
                'updates':display_update_flags(updates),
                'update_state_ids':dict(updates), 'display_labels':identity['display'],
                'schedule':schedule.controls(),'schedule_display_labels':identity['display'],
                'projection_held_errors':{'input_map':errors['input_map'],'full_depth':errors['full_depth']},
                'input_map_history_ce_delta':history_ce_delta,
                'pass_ce_deltas':pass_ce_deltas,'evaluation_passes':evaluation_passes}
        if baseline_reason is not None:
            report['text_ce_baseline_remeasurement']={'reason':baseline_reason,
                'schedule_observation':False,'model_or_optimizer_update':False,
                'training_rng_preserved':bool(baseline_rng_preserved)}
        if autoregressive_controls is not None:report['autoregressive_controls']=autoregressive_controls
        if family and a.member_eval_windows:
            report['family']=evaluate_members(backbone,[member_window(w) for w in held[:a.member_eval_windows]])
        for collection in (role_strata,*cohort_role_strata.values()):
            for roles_of_pass in collection.values():
                for row in roles_of_pass.values():
                    for n in row.keys()-{'tokens'}:row[n]/=row['tokens']
        if role_strata:report['role_strata']=role_strata  # diagnostic: chat-role breakdown, not a gate
        if cohort_role_strata:report['cohort_role_strata']=cohort_role_strata
        matched_summary={}
        for consumer in MATCHED_CONSUMERS:
            matched_summary[consumer]={}
            for region in ('whole','last256'):
                scores=[row['scores'][consumer][region] for row in matched_history_rows
                        if row['cohort']==a.qualification_cohort]
                token_count=sum(score['tokens'] for score in scores)
                history_count=sum(score['history_positions'] for score in scores)
                fields=('ce','gold_accuracy','ce_delta_from_live_greedy','argmax_agreement_with_live_greedy')
                aggregate={'tokens':token_count,'history_positions':history_count}
                for field in fields:
                    aggregate[field]=sum(score[field]*score['tokens'] for score in scores)/token_count
                mse_values=[(score['read_history_mse_vs_live_greedy'],score['history_positions'])
                            for score in scores if score['history_positions']]
                if mse_values:
                    aggregate['read_history_mse_vs_live_greedy']=sum(value*count for value,count in mse_values)/sum(count for _,count in mse_values)
                matched_summary[consumer][region]=aggregate
        report['matched_projected_history']={'schema':'natlang.text-warmup-matched-history/2',
            'weights_digest':report['weights_digest'],'held_probe_selection':held_selection_eval,
            'consumers':list(MATCHED_CONSUMERS),
            'pass_correspondence':{
                'sketch_projection':'independent serving heads.feedback projection over pass-zero states; '
                                    'the training-only input map is not used here',
                'full_projection':'full-depth projected history (deployed channel)'},
            'read_interface':'heads.read_embeddings(backbone, payload[:, :-1])',
            'producer_reuse':'pass-zero gold-history top states and crisp next-token predictions from the same held batch',
            'batch_policy':{'eval_batch':a.eval_batch,'max_window_tokens':a.tokens,
                'selected_windows':len(held),'diagnostic_forward_passes_per_batch':len(MATCHED_CONSUMERS)},
            'future_gold_inputs':False,'windows':matched_history_rows,'weighted_summary':matched_summary,
            'changes_qualification_gates':False}
        report['alignment_gate_passed']=qualification(report,max_ce_delta=a.max_ce_delta,max_relative_mse=a.max_relative_mse,min_agreement=a.min_agreement)
        if codes is not None:report['qat_codes']=codes.update()
        log('eval.jsonl',report);last_report=report;return report
    serving_heads_step=0
    if was_resumed:
        status_path=a.out/'heads-export-status.json'
        if status_path.is_file():
            try:
                prior_status=json.loads(status_path.read_text())
                serving_heads_step=(int(prior_status.get('heads_step',step))
                    if prior_status.get('checkpoint_step')==step else -1)
            except (OSError,ValueError,TypeError):serving_heads_step=-1
        else:
            # The current save policy can leave heads.pt behind checkpoint.pt;
            # without a matching receipt its exact step is unknown.
            serving_heads_step=-1

    def write_heads_export_status(*,export_error=None,emergency=False,
                                  checkpoint_step=None,heads_step=None):
        checkpoint_step=step if checkpoint_step is None else int(checkpoint_step)
        heads_step=serving_heads_step if heads_step is None else int(heads_step)
        return warmup_export.write_heads_export_status(a.out,warmup_export.heads_export_status(
            checkpoint_step=checkpoint_step,heads_step=heads_step,
            export_error=export_error,emergency=emergency))

    def build_heads_export(report=None, *, export_step=None, snapshot=False):
        # Shared serving heads carry explicit backbone deltas, never inherited certification.
        return warmup_export.build_heads_export(
            initial_heads_path=a.heads,heads=heads,backbone=backbone,backbone_names=backbone_names,
            backbone_training=a.backbone_training,rank=a.rank,identity=identity,out=a.out,
            report=report,export_step=step if export_step is None else int(export_step),snapshot=snapshot)

    def export_heads(report=None):
        nonlocal serving_heads_step
        exported=build_heads_export(report)
        atomic_checkpoint(a.out/'heads.pt',exported)
        serving_heads_step=step
        return True

    checkpoint_cadence=Cadence(a.checkpoint_every,a.checkpoint_minutes)
    eval_cadence=Cadence(a.eval_every)
    def save(report=None, *, rng_state=None, emergency_recovery=None, write_export=True,
             retain_best=False, wait=False):
        checkpoint_writer.drain()
        checkpoint_cadence.mark()
        if checkpoint_reserve is not None and checkpoint_reserve.active:
            checkpoint_reserve.release_space()
        current_rng=rng_state or capture_training_rng_state(a.device)
        estimate=(checkpoint_snapshot_size_bound + Path(a.heads).stat().st_size + 64*1024*1024
                  if checkpoint_snapshot_size_bound is not None else None)
        available=available_system_memory_bytes()
        async_write=(not wait and emergency_recovery is None and estimate is not None
                     and available is not None and available >= int(estimate*1.25))
        state={'schema':'natlang.neuralese-text-warmup/1','identity':identity,'artifact_refs':input_roles,'step':step,
          'student_parameters':{n:q.detach() if async_write else q.detach().cpu() for n,q in named},'heads':heads.state_dict(),
          'optimizer':optimizer.state_dict(),'optimizer_param_names':optimizer_param_names(optimizer,dict(named)),'python_rng':current_rng['python_rng'],'torch_rng':current_rng['torch_rng'],
          'cuda_rng':current_rng['cuda_rng'],
          'streak':streak,'best':best,'updates':updates,'qualification':report,
          'initial_text_ce':initial_text_ce,'schedule':schedule.state_dict(),
          'last_schedule_step':last_schedule_step,'code_handoffs':code_handoffs,
          # Resource observations are resumable state, not recipe/model identity.
          'memory_estimator':memory_estimator.state_dict(),
          'activation_offload_state':{'schema':'natlang.text-warmup-offload-policy-telemetry/2',
              'assumed_gpu_bytes_freed_per_cpu_byte':assumed_savings,
              'observations':offload_observations[-64:]}}
        if emergency_recovery is not None:state['emergency_recovery']=emergency_recovery
        export_snapshot=build_heads_export(report,export_step=step,snapshot=async_write) if write_export else None
        report_snapshot=(immutable_cpu_snapshot(report) if async_write else report) if retain_best else None
        saved_step=int(step)
        def after_checkpoint(_snapshot):
            nonlocal serving_heads_step
            export_error=None
            if export_snapshot is not None:
                try:
                    atomic_checkpoint(a.out/'heads.pt',export_snapshot)
                    serving_heads_step=saved_step
                except Exception as error:
                    export_error=error
                    write_heads_export_status(export_error=error,checkpoint_step=saved_step)
                    raise
            write_heads_export_status(export_error=export_error,checkpoint_step=saved_step)
            if retain_best:
                retain_best_checkpoint(a.out,report_snapshot)
        try:
            if async_write:
                checkpoint_writer.submit(state_path,state,after_write=after_checkpoint)
            else:
                checkpoint_writer.write_synchronously(state_path,state,after_write=after_checkpoint)
        except Exception:
            raise
    if a.eval_only:
        report=evaluate(observe_schedule=False)
        (a.out/'eval-only.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'event':'eval_only_done','step':step}),flush=True)
        return None
    if not was_resumed:
        # A continued full state gets a fresh starting-weight evaluation on
        # its current held domain before any update. This evaluates without
        # advancing the restored plateau/ramp schedule or consuming its RNG.
        continuation_rng=(capture_training_rng_state(a.device) if a.continue_from else None)
        try:
            baseline=evaluate(observe_schedule=not bool(a.continue_from),
                baseline_reason='input_or_held_objective_changed' if remeasure_text_baseline else None,
                baseline_rng_preserved=continuation_rng is not None)
        finally:
            if continuation_rng is not None:restore_training_rng_state(continuation_rng,a.device)
        (a.out/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
        best={'step':step,'score':alignment_selection_score(baseline,max_ce_delta=a.max_ce_delta,max_relative_mse=a.max_relative_mse,min_agreement=a.min_agreement),'report':baseline}
        save(baseline,wait=True);retain_best_checkpoint(a.out,baseline)
    checkpoint_snapshot_size_bound=warmup_checkpoint_size_upper_bound(named,heads,optimizer)
    checkpoint_reserve=CheckpointDiskReserve(
        a.out/'.checkpoint-space.reserve',checkpoint_snapshot_size_bound)
    atexit.register(checkpoint_reserve.cleanup)
    try:
        checkpoint_reserve.acquire()
    except CheckpointReserveError as error:
        print(json.dumps({'event':'checkpoint_space_preflight_refused',
                          'training_started':False,'step':step,'error':str(error)}),flush=True)
        checkpoint_reserve.cleanup()
        raise SystemExit(2)
    # Drain pending output before reserve cleanup during interpreter shutdown.
    atexit.register(drain_checkpoint_writer_at_exit)
    def recover_postcommit_persistence_failure(error):
        current_rng=capture_training_rng_state(a.device)
        try:
            recovery=persist_postcommit_recovery(
                save,checkpoint_reserve,step=step,error=error,current_rng=current_rng)
        except Exception as checkpoint_error:
            checkpoint_reserve.cleanup()
            print(json.dumps({'event':'postcommit_emergency_checkpoint_failed',
                'safe_to_resume':False,'last_committed_step':step,
                'persistence_error_type':type(error).__name__,
                'persistence_error':str(error)[:1000],
                'checkpoint_error_type':type(checkpoint_error).__name__,
                'checkpoint_error':str(checkpoint_error)[:1000]}),flush=True)
            raise RuntimeError('post-commit persistence failed and emergency checkpoint could not be saved') from checkpoint_error
        export_error=None
        try:export_heads(last_report if last_report is not None and last_report.get('step')==step else None)
        except Exception as error:
            # The full optimizer/model/RNG state is already safely committed.
            # Serving export is secondary and must not change recovery status.
            export_error=error
        recovery['serving_heads_exported']=export_error is None
        if export_error is not None:
            recovery['serving_heads_export_error']={'type':type(export_error).__name__,
                                                    'message':str(export_error)[:1000]}
        recovery['heads_export_status_written']=write_heads_export_status(
            export_error=export_error,emergency=True)
        checkpoint_reserve.cleanup()
        print(json.dumps({'event':'postcommit_emergency_checkpoint_saved',**recovery}),flush=True)
        raise SystemExit(2)
    def perform_update(batch, passes, bootstrap, controls, *, offload_budget_bytes,
                       memory_start, memory_plan):
        """Run forward/backward and gradient prep without mutating model/optimizer state."""
        started=time.perf_counter();pass_metrics=[];pass_losses=[]
        from .backbone_policy import shared_parametrized_weights
        from .memory import offload_attention_tensors
        persistent=(list(backbone.parameters())+list(backbone.buffers())+
                    list(heads.parameters())+list(heads.buffers())+
                    [q for _,q in named])
        wrapped_started=time.perf_counter()
        with offload_attention_tensors(int(offload_budget_bytes),activations=True,
                                       persistent_tensors=persistent) as offload_stats:
            with shared_parametrized_weights(backbone.hf) as next_pass:
                for loss,metrics in objective(batch,passes,bootstrap,
                        readout_chunk_tokens=int(memory_plan['readout_chunk_tokens'])):
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite warm-up loss')
                    (loss/passes).backward();next_pass()
                    pass_losses.append(loss.detach());pass_metrics.append(metrics)
                family_record=None
                if a.member_weight and not bootstrap:
                    # The family term (MAPLE_NESTED §4a): one member per update, in rotation.
                    member=family[step%len(family)]
                    parts=member_backward(backbone,member,*member_window(batch[0]),weight=a.member_weight);next_pass()
                    family_record={'member':member.key,'ce':parts.ce/max(parts.tokens,1),
                                   'kl':parts.kl/max(parts.tokens,1),'tokens':parts.tokens}
        wrapped_forward_backward_seconds=time.perf_counter()-wrapped_started
        pass_metrics,total_loss=materialize_objective_metrics(pass_metrics,pass_losses,passes,
                                                              secondary_projection=secondary_metric_name)
        metrics=dict(pass_metrics[-1])
        if family_record is not None:metrics['family']=family_record
        backbone_norm=gradient_norm(q for n,q in named if n.startswith('backbone.'))
        secondary_norm=gradient_norm(q for n,q in named if n.startswith(secondary_prefix))
        clip_finite_gradients(parameters.values())
        samples={k:next((q for n,q in named if n.startswith(prefix) and q.grad is not None and q.grad.abs().sum()>0),None)
                 for k,prefix in [('backbone','backbone.'),('input_map',secondary_prefix),
                                  ('full_projection','heads.content.proj.')]}
        before={k:q.detach().clone() for k,q in samples.items() if q is not None}
        return {'metrics':metrics,'pass_metrics':pass_metrics,'total_loss':total_loss,
                'started':started,'memory_start':memory_start,'backbone_norm':backbone_norm,
                'secondary_norm':secondary_norm,'samples':samples,'before':before,'controls':controls,
                'memory_plan':memory_plan,'offload_stats':dict(offload_stats),
                'wrapped_forward_backward_seconds':wrapped_forward_backward_seconds}

    def checkpoint_preupdate_failure(error, *, pre_attempt_rng, pre_attempt_lrs, controls):
        """Save only the last committed update after a failure before optimizer.step."""
        # Drop autograd locals from the failed update before cloning model state
        # or exporting serving weights. This is especially useful after CUDA OOM.
        try:traceback.clear_frames(error.__traceback__)
        except (AttributeError, RuntimeError):pass
        optimizer.zero_grad(set_to_none=True)
        if a.device.startswith('cuda'):torch.cuda.empty_cache()
        # Learning-rate staging is replayed from the saved schedule on resume;
        # restore its old optimizer values so the serialized state is exactly
        # the last committed optimizer state.
        for group,lr in zip(optimizer.param_groups,pre_attempt_lrs):group['lr']=lr
        recovery={'schema':'natlang.text-warmup-emergency-recovery/1',
          'safe_to_resume':True,'failure_stage':'before_optimizer_step',
          'failed_attempt_step':step+1,'last_committed_step':step,
          'optimizer_step_started':False,'partial_gradients_cleared':True,
          'pre_attempt_rng_saved_for_replay':True,'schedule_state_is_last_committed':True,
          'error_type':type(error).__name__,'error':str(error)[:1000],
          'phase':controls.get('phase')}
        save(last_report if last_report is not None and last_report['step']==step else None,
             rng_state=pre_attempt_rng,emergency_recovery=recovery,write_export=False)
        export_error=None
        try:export_heads(last_report if last_report is not None and last_report.get('step')==step else None)
        except Exception as error:export_error=error
        recovery['serving_heads_exported']=export_error is None
        if export_error is not None:
            recovery['serving_heads_export_error']={'type':type(export_error).__name__,
                                                    'message':str(export_error)[:1000]}
        recovery['heads_export_status_written']=write_heads_export_status(
            export_error=export_error,emergency=True)
        if checkpoint_reserve is not None:checkpoint_reserve.cleanup()
        print(json.dumps({'event':'emergency_checkpoint_saved',**recovery}),flush=True)

    def prepare_update_memory(batch, passes, bootstrap):
        """Forecast this exact next batch without a model forward or trial update."""
        first=batch[0]
        prefix=int(first['prefix'])
        target=len(first['ids'])-prefix
        memory_start=None
        effective_free=None
        total=None
        candidate_forecasts=[]
        # Readout and FFN tiles are selected only when the full-update forecast
        # fits current reusable memory. FFN deltas use actual module width.
        if a.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(a.device)
            memory_start=int(torch.cuda.memory_allocated(a.device))
            device_free,total=torch.cuda.mem_get_info(a.device)
            reserved=int(torch.cuda.memory_reserved(a.device))
            effective_free=effective_cuda_free_bytes(int(device_free),int(total),
                reserved,memory_start)
            usable_free=max(0,effective_free-math.ceil(int(total)*TEXT_WARMUP_MEMORY_HEADROOM))
            update_floor=_warmup_update_floor_bytes(named,optimizer,bootstrap=bootstrap)
            predictions={}
            candidates={}
            for chunk in TEXT_WARMUP_READOUT_CHUNKS:
                candidate_geometry=text_warmup_update_geometry_bytes(
                    prefix,target,passes,len(batch),full_memory_layout,shallow_memory_layout,
                    cutoff=heads.cutoff,vocab_size=backbone.embedding_weight.shape[0],
                    readout_chunk_tokens=chunk,channel_consistency=a.ar_feedback_fixup)
                candidate_raw=candidate_geometry+update_floor
                candidate_kind=_warmup_memory_kind(len(batch),passes,chunk,
                                                   default_ffn_chunk_tokens)
                native_prediction=memory_estimator.predict(
                    candidate_kind,prefix+target-1,target,candidate_raw)
                calibration_count=len(memory_estimator.calibration(
                    candidate_kind,prefix+target-1,target))
                if chunk==TEXT_WARMUP_READOUT_CHUNKS[0]:
                    readout_prediction=native_prediction
                    base_geometry=candidate_geometry
                    base_prediction=native_prediction
                elif calibration_count>=3:
                    readout_prediction=native_prediction
                else:
                    readout_prediction=conservative_expanded_readout_prediction(
                        base_prediction,native_prediction,base_geometry,candidate_geometry,
                        has_candidate_calibration=False)
                for ffn_chunk in ffn_chunk_candidates:
                    ffn_delta=text_warmup_ffn_workspace_delta_bytes(
                        prefix,target,len(batch),full_memory_layout['intermediate'],
                        base_chunk_tokens=default_ffn_chunk_tokens,
                        candidate_chunk_tokens=ffn_chunk)
                    geometry=candidate_geometry+ffn_delta
                    raw=geometry+update_floor
                    kind=_warmup_memory_kind(len(batch),passes,chunk,ffn_chunk)
                    native=memory_estimator.predict(kind,prefix+target-1,target,raw)
                    observations=len(memory_estimator.calibration(kind,prefix+target-1,target))
                    if ffn_chunk==default_ffn_chunk_tokens:
                        prediction=readout_prediction
                        basis=('chunk-specific forecast/calibration' if calibration_count>=3 or chunk==128
                               else '128 readout complete-update forecast plus geometry delta')
                    elif observations>=3:
                        prediction=native
                        basis='FFN/readout-specific successful-update calibration'
                    else:
                        prediction=conservative_expanded_readout_prediction(
                            readout_prediction,native,candidate_geometry,geometry,
                            has_candidate_calibration=False)
                        basis='1024 FFN complete-update forecast plus unscaled workspace delta'
                    pair=(chunk,ffn_chunk)
                    predictions[pair]=prediction
                    candidates[pair]=(candidate_geometry,geometry,raw,prediction)
                    candidate_forecasts.append({'readout_chunk_tokens':chunk,
                        'ffn_chunk_tokens':ffn_chunk,'geometry_upper_bound_bytes':geometry,
                        'saved_activation_geometry_upper_bound_bytes':candidate_geometry,
                        'added_ffn_workspace_bytes':ffn_delta,
                        'predicted_update_increment_bytes':prediction,
                        'native_geometry_prediction_bytes':native,
                        'calibration_observations':observations,
                        'prediction_basis':basis,
                        'fits_live_usable_memory':prediction<=usable_free})
            readout_chunk_tokens,ffn_chunk_tokens=select_text_warmup_chunk_pair(
                predictions,usable_free,default=(TEXT_WARMUP_READOUT_CHUNKS[0],
                                                  default_ffn_chunk_tokens))
            saved_activation_geometry,predictor_geometry,predictor_raw,predicted=candidates[(readout_chunk_tokens,ffn_chunk_tokens)]
        else:
            readout_chunk_tokens=TEXT_WARMUP_READOUT_CHUNKS[0]
            ffn_chunk_tokens=default_ffn_chunk_tokens
            raw_geometry=text_warmup_update_geometry_bytes(
                prefix,target,passes,len(batch),full_memory_layout,shallow_memory_layout,
                cutoff=heads.cutoff,vocab_size=backbone.embedding_weight.shape[0],
                readout_chunk_tokens=readout_chunk_tokens,channel_consistency=a.ar_feedback_fixup)
            update_floor=_warmup_update_floor_bytes(named,optimizer,bootstrap=bootstrap)
            predictor_raw=raw_geometry+update_floor
            saved_activation_geometry=raw_geometry
            predictor_geometry=raw_geometry
            kind=_warmup_memory_kind(len(batch),passes,readout_chunk_tokens,ffn_chunk_tokens)
            predicted=memory_estimator.predict(kind,prefix+target-1,target,predictor_raw)
        # The successful-update calibration measures the complete incremental
        # peak, including gradient buffers and lazy optimizer slots. Seed the
        # same floor once in the uncalibrated geometry; do not add it again to
        # the calibrated observation.
        context=prefix+target-1
        kind=_warmup_memory_kind(len(batch),passes,readout_chunk_tokens,ffn_chunk_tokens)
        if a.device.startswith('cuda'):
            plan=plan_saved_activation_offload(predicted,effective_free,int(total),
                saved_activation_geometry,headroom_fraction=TEXT_WARMUP_MEMORY_HEADROOM,
                assumed_gpu_bytes_freed_per_cpu_byte=assumed_savings)
            details={'predicted_update_increment_bytes':predicted,
                'geometry_upper_bound_bytes':predictor_geometry,
                'saved_activation_geometry_upper_bound_bytes':saved_activation_geometry,
                'gradient_optimizer_floor_bytes':update_floor,
                'readout_chunk_tokens':readout_chunk_tokens,
                'channel_consistency':a.ar_feedback_fixup,
                'ffn_chunk_tokens':ffn_chunk_tokens,
                'chunk_policy':'largest FFN tile, then largest readout tile whose independently forecast update fits reusable memory after reserve; 128/1024 fallback then existing offload/refusal',
                'chunk_candidates':candidate_forecasts,
                'device_free_bytes':int(device_free),'effective_free_bytes':effective_free,
                'allocator_reserved_bytes':reserved,'start_allocated_bytes':memory_start,
                'device_total_bytes':int(total),'usable_free_bytes':plan.usable_free_bytes,
                'required_gpu_reduction_bytes':plan.required_gpu_reduction_bytes,
                'offload_budget_bytes':plan.offload_budget_bytes,
                'predicted_residual_overage_bytes':plan.predicted_residual_overage_bytes,
                'predicted_fit':plan.predicted_fit,
                'assumed_gpu_bytes_freed_per_cpu_byte':assumed_savings,
                'headroom_fraction':TEXT_WARMUP_MEMORY_HEADROOM,
                'context_tokens':context,'target_tokens':target,'batch':len(batch),
                'sequence_passes':passes}
            if not plan.predicted_fit:
                raise RuntimeError('warm-up memory preflight refused before forward: '
                    f'predicted update increment {predicted} B, usable free '
                    f'{plan.usable_free_bytes} B after 5% reserve, eligible saved-activation '
                    f'upper bound {saved_activation_geometry} B, residual predicted overage '
                    f'{plan.predicted_residual_overage_bytes} B; reduce allocated GPU workload '
                    'or use a larger-memory device; no sample was skipped and no context was truncated')
        else:
            plan=plan_saved_activation_offload(predicted,2**60,2**60,saved_activation_geometry,
                headroom_fraction=0.,assumed_gpu_bytes_freed_per_cpu_byte=assumed_savings)
            details={'predicted_update_increment_bytes':predicted,
                'geometry_upper_bound_bytes':predictor_geometry,
                'saved_activation_geometry_upper_bound_bytes':saved_activation_geometry,
                'gradient_optimizer_floor_bytes':update_floor,'device':'non-cuda',
                'readout_chunk_tokens':readout_chunk_tokens,
                'channel_consistency':a.ar_feedback_fixup,
                'ffn_chunk_tokens':ffn_chunk_tokens,
                'chunk_policy':'conservative 128 readout / 1024 FFN on non-CUDA device',
                'offload_budget_bytes':0,'predicted_fit':True,
                'context_tokens':context,'target_tokens':target,'batch':len(batch),
                'sequence_passes':passes}
        return {'plan':details,'raw_geometry_bytes':saved_activation_geometry,
                'predictor_raw_bytes':predictor_raw,
                'readout_chunk_tokens':readout_chunk_tokens,
                'channel_consistency':a.ar_feedback_fixup,
                'ffn_chunk_tokens':ffn_chunk_tokens,
                'context_tokens':context,'target_tokens':target,
                'memory_start':memory_start,'offload_budget_bytes':plan.offload_budget_bytes}

    def finish_committed_update(prepared,memory_plan,passes,controls):
        """Persist one already-committed optimizer update; return a stop reason."""
        nonlocal step,streak,best
        step+=1
        m=prepared['metrics'];samples=prepared['samples'];before=prepared['before']
        memory_passes=int(memory_plan.get('sequence_passes',passes))
        for k,v in before.items():updates[k]|=not torch.equal(v,samples[k].detach())
        memory_record=None
        if a.device.startswith('cuda'):
            peak_allocated=int(torch.cuda.max_memory_allocated(a.device))
            peak_reserved=int(torch.cuda.max_memory_reserved(a.device))
            actual_increment=max(0,peak_allocated-int(prepared['memory_start']))
            offload=prepared['offload_stats']
            was_offloaded=(int(memory_plan.get('offload_budget_bytes',0)) > 0 or
                           int(offload.get('offloaded_tensors',0)) > 0)
            if not was_offloaded:
                # Only a successful unoffloaded update measures the predictor's
                # target quantity. An offloaded peak is censored telemetry.
                memory_estimator.observe(_warmup_memory_kind(
                    a.batch,memory_passes,int(memory_plan['readout_chunk_tokens']),
                    int(memory_plan['ffn_chunk_tokens'])),
                    memory_plan['context_tokens'],memory_plan['target_tokens'],
                    memory_plan['predictor_raw_bytes'],actual_increment)
            memory_record={'start_allocated_bytes':prepared['memory_start'],
                'peak_allocated_bytes':peak_allocated,'peak_reserved_bytes':peak_reserved,
                'end_allocated_bytes':int(torch.cuda.memory_allocated(a.device)),
                'actual_incremental_peak_bytes':actual_increment,
                'predictor_calibration_observation':not was_offloaded,
                'offloaded_peak_is_censored':was_offloaded,
                'readout_chunk_tokens':int(memory_plan['readout_chunk_tokens']),
                'ffn_chunk_tokens':int(memory_plan['ffn_chunk_tokens']),
                'objective_passes':passes,'memory_geometry_passes':memory_passes,
                'preflight':prepared['memory_plan'],
                'offload':{**offload,'wrapped_forward_backward_seconds':
                           prepared['wrapped_forward_backward_seconds']}}
            offload_observations.append({'step':step,
                'predicted_update_increment_bytes':prepared['memory_plan'].get('predicted_update_increment_bytes'),
                'actual_incremental_peak_bytes':actual_increment,
                'offloaded_bytes':int(offload['offloaded_bytes']),
                'peak_offloaded_bytes':int(offload['peak_offloaded_bytes']),
                'wrapped_forward_backward_seconds':prepared['wrapped_forward_backward_seconds'],
                'predictor_calibration_observation':not was_offloaded,
                'readout_chunk_tokens':int(memory_plan['readout_chunk_tokens']),
                'ffn_chunk_tokens':int(memory_plan['ffn_chunk_tokens']),
                'assumed_gpu_bytes_freed_per_cpu_byte':assumed_savings})
            offload_observations[:]=offload_observations[-64:]
        m.update(step=step,loss=prepared['total_loss'],seconds=time.perf_counter()-prepared['started'],
                 phase=controls['phase'],schedule=controls,pass_metrics=prepared['pass_metrics'],
                 batch=a.batch,backbone_gradient_norm=float(prepared['backbone_norm']),
                 updates=display_update_flags(updates),
                 update_state_ids=dict(updates),display_labels=identity['display'])
        m[secondary_metric_name+'_gradient_norm']=float(prepared['secondary_norm'])
        m['readout_chunk_tokens']=int(memory_plan['readout_chunk_tokens'])
        m['ffn_chunk_tokens']=int(memory_plan['ffn_chunk_tokens'])
        if memory_record is not None:m['memory']=memory_record
        log('train.jsonl',m)
        report=None
        if eval_cadence.due(step):
            report=evaluate()
            qualification_depth=_alignment_qualification_pass_depth(
                schedule=schedule)
            streak=streak+1 if (qualification_depth is not None and passes==qualification_depth and
                report['alignment_gate_passed'] and all(updates.values())) else 0
            report.update(consecutive_passes=streak,qualified=streak>=a.consecutive_gates,
                          scope='text alignment only; stopping, transport and Natlang tasks unqualified')
            score=alignment_selection_score(report,max_ce_delta=a.max_ce_delta,
                max_relative_mse=a.max_relative_mse,min_agreement=a.min_agreement)
            improved=best is None or score<best['score']
            if improved:best={'step':step,'score':score,'report':report}
            (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            try:
                save(report,write_export=improved,retain_best=improved)
            except Exception as error:
                recover_postcommit_persistence_failure(error)
                return
        elif checkpoint_cadence.due(step):
            try:
                save(write_export=False)
            except Exception as error:
                recover_postcommit_persistence_failure(error)
                return
        if report is not None and report.get('qualified'):
            return 'qualified'
        try:
            if not checkpoint_writer.pending:
                checkpoint_reserve.ensure()
        except CheckpointReserveError as error:
            recover_postcommit_persistence_failure(error)
        return None

    # Restored tensors now live on the device: drop the read checkpoints' page cache (GB10 unified memory).
    for path in (state_path,a.continue_from,a.heads,a.student_checkpoint):
        if path and Path(path).is_file():drop_file_cache(path)
    loop=TrainingLoop(step,a.steps,stop)
    for _ in loop:
        try:
            checkpoint_writer.check()
        except Exception as error:
            recover_postcommit_persistence_failure(error)
            return
        controls=schedule.controls();bootstrap=not schedule.plateau_reached
        passes=text_history_pass_count(controls['sequence_passes'],
                                       ar_feedback_fixup=a.ar_feedback_fixup)
        pre_attempt_rng=capture_training_rng_state(a.device)
        pre_attempt_lrs=[group['lr'] for group in optimizer.param_groups]
        try:
            for name,q in named:
                q.requires_grad_(not bootstrap or name.startswith((secondary_prefix,'heads.content.proj.')))
            for group in optimizer.param_groups:
                group['lr']=group['foundation_base_lr']*(1. if group['foundation_projection'] else controls['backbone_lr_scale'])
            batch=sampler.batch(a.batch,random)
            optimizer.zero_grad(set_to_none=True)
            # Free memory on the GB10 moves with page cache and other jobs: re-measure for up to five minutes
            # before refusing, which would cost a full reload.
            for wait in range(11):
                try:
                    # Reserve an extra pass for the sequential producer and detached
                    # ordinary target. Producer KV is released before the target
                    # and consumer forwards; this is a conservative peak reserve.
                    memory_passes=passes+int(a.ar_feedback_fixup)
                    memory_plan=prepare_update_memory(batch,memory_passes,bootstrap);break
                except RuntimeError as error:
                    if 'preflight refused' not in str(error) or wait==10 or stop.requested:raise
                    if wait==0:print(json.dumps({'event':'preflight_waiting','step':step+1,'error':str(error)[:300]}),flush=True)
                    time.sleep(30)
            backbone.ffn_chunk_tokens=int(memory_plan['ffn_chunk_tokens'])
            # Performance work on a live run: touching <out>/profile-request profiles the next update.
            profile_request=a.out/'profile-request'
            profiling=profile_request.exists()
            module_counting=profiling and profile_request.read_text().strip()=='module-counts'
            module_counter=None
            profile_error=None
            if module_counting:
                module_counter=LinearModuleCallCounter(backbone=backbone,heads=heads)
                profile_start=time.perf_counter()
            elif profiling:
                from torch.profiler import ProfilerActivity, profile
                profiler=profile(activities=[ProfilerActivity.CPU]+([ProfilerActivity.CUDA] if a.device.startswith('cuda') else []))
                profiler.__enter__();profile_start=time.perf_counter()
            try:
                prepared=perform_update(batch,passes,bootstrap,controls,
                    offload_budget_bytes=memory_plan['offload_budget_bytes'],
                    memory_start=memory_plan['memory_start'],memory_plan=memory_plan['plan'])
            except Exception as error:
                profile_error=error
                raise
            finally:
                if module_counting:
                    module_counter.close()
                    payload={'update':step+1,'phase':controls['phase'],'passes':passes,
                             'tokens':sum(len(w['ids']) for w in batch),
                             'wall_seconds':time.perf_counter()-profile_start,
                             'status':'error' if profile_error is not None else 'complete',
                             'counts':module_counter.rows()}
                    if profile_error is not None:
                        payload['error']=f'{type(profile_error).__name__}: {profile_error}'
                    (a.out/f'profile-step{step+1}-module-counts.json').write_text(
                        json.dumps(payload,separators=(',',':'))+'\n')
                    profile_request.unlink(missing_ok=True)
                elif profiling:
                    if a.device.startswith('cuda'):torch.cuda.synchronize()
                    profiler.__exit__(None,None,None);profile_request.unlink(missing_ok=True)
                    sort='cuda_time_total' if a.device.startswith('cuda') else 'cpu_time_total'
                    (a.out/f'profile-step{step+1}.txt').write_text(
                        f'wall {time.perf_counter()-profile_start:.2f} s, passes {passes}, '
                        f'tokens {sum(len(w["ids"]) for w in batch)}\n'
                        +profiler.key_averages().table(sort_by=sort,row_limit=80,max_name_column_width=90))
        except Exception as error:
            checkpoint_preupdate_failure(error,pre_attempt_rng=pre_attempt_rng,
                                         pre_attempt_lrs=pre_attempt_lrs,controls=controls)
            raise
        # Do not place optimizer.step inside the emergency-save handler: an
        # exception here can follow partial parameter or moment mutation, so
        # the only safe resume point is the last already-written checkpoint.
        def discard_partial_optimizer_step():
            # Gradients are not checkpointed. Clear them without writing any
            # state because parameters or optimizer moments may have mutated.
            optimizer.zero_grad(set_to_none=True)
            if a.device.startswith('cuda'):torch.cuda.empty_cache()
            # Complete only a snapshot submitted at an earlier committed
            # boundary. Do not snapshot the possibly partially-mutated live
            # state; keeping this drain here also makes main() safe for callers
            # that catch the optimizer exception without exiting the process.
            try:checkpoint_writer.drain()
            except Exception:traceback.print_exc()
        commit_optimizer_step(optimizer,on_failure=discard_partial_optimizer_step)
        try:
            completion=finish_committed_update(prepared,memory_plan,passes,controls)
        except Exception as error:
            recover_postcommit_persistence_failure(error)
            return
        if completion=='qualified':
            loop.finish('qualified');break
    if stop.requested:
        # Interruptible: on a signal, persist the full resumable state at once. The held evaluation is not needed
        # to resume and would delay the stop past the container's kill timeout.
        try:
            save(last_report if last_report is not None and last_report['step']==step else None,
                 write_export=False,wait=True)
        except Exception as error:
            recover_postcommit_persistence_failure(error)
            return
        print(json.dumps({'event':'checkpointed_on_signal','step':step}),flush=True)
        return
    # A signal during the periodic probe must not repeat the same expensive
    # held evaluation before checkpointing exactly the same weights.
    try:
        report=dict(last_report) if last_report is not None and last_report['step']==step else evaluate()
        report.update(consecutive_passes=streak,qualified=streak>=a.consecutive_gates,
          updates=display_update_flags(updates),status='checkpointed_on_signal' if stop.requested else 'complete',
          scope='text alignment only; stopping, transport and Natlang tasks unqualified')
        score=alignment_selection_score(report,max_ce_delta=a.max_ce_delta,max_relative_mse=a.max_relative_mse,min_agreement=a.min_agreement)
        improved=best is None or score<best['score']
        if improved:best={'step':step,'score':score,'report':report}
        (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        save(report,retain_best=improved,wait=True)
    except Exception as error:
        recover_postcommit_persistence_failure(error)
        return
    checkpoint_writer.drain()
    checkpoint_reserve.cleanup()

if __name__=='__main__':main()
