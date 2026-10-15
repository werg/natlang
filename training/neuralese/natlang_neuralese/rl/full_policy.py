"""Token-level GRPO through the shared serving replay and direct trainer.

This module supplies an objective, never a second optimizer/trainer. Behavior
scores are captured before the first update against the checkpoint which sampled
these rollouts. The capture is durable across resume. No base-KL penalty or stop.
Neuralese block-bearing turns require a durable graph/block replay bundle.
Consumer loss reaches the recorded-value writer through shared producer replay.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from .grpo import group_advantages


def clipped_token_loss(logps, old_logps, advantage, *, clip_low=0.2, clip_high=0.28):
    """DAPO-style asymmetric clipping, with stable float32 ratios and signed gradients."""
    if not 0 <= clip_low < 1 or not 0 <= clip_high:
        raise ValueError('invalid RL clipping range')
    if logps.shape != old_logps.shape or logps.numel() == 0:
        raise ValueError('behavior/current token positions differ or are empty')
    if not torch.isfinite(logps).all() or not torch.isfinite(old_logps).all():
        raise ValueError('nonfinite RL log probabilities')
    ratios = (logps.float() - old_logps.float()).clamp(-20, 20).exp()
    a = torch.as_tensor(advantage, dtype=torch.float32, device=logps.device)
    if not torch.isfinite(a).all():
        raise ValueError('nonfinite advantage')
    clipped = ratios.clamp(1 - clip_low, 1 + clip_high)
    return -torch.minimum(ratios * a, clipped * a)


def contains_blocks(value):
    if isinstance(value, dict):
        return value.get('type') == 'neuralese' or any(contains_blocks(v) for v in value.values())
    if isinstance(value, list):
        return any(contains_blocks(v) for v in value)
    return False


def validate_rollouts(rollouts, policy_id, *, block_bundle=None):
    """Exact admission checks for on-policy replay, independent of model architecture."""
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError('RL requires a collection policy identity')
    groups = {}
    for r in rollouts:
        if r.get('reward') is None:
            continue
        if r.get('training_admission') not in (None, 'grpo'):
            raise ValueError('RFT-only diagnostic rollouts are not on-policy GRPO data')
        if r.get('split') != 'train':
            raise ValueError('RL rejects missing/non-train rollout split')
        quality = r['reward'].get('quality')
        if not isinstance(quality, (int, float)) or not math.isfinite(quality) or quality not in (0, 1):
            raise ValueError('RL reward must be exact binary task success')
        p = r.get('policy') or {}
        if p.get('checkpoint_sha256') != policy_id or p.get('temperature') != 1.0:
            raise ValueError('RL requires the sampled checkpoint and temperature=1')
        if p.get('top_p', 1) != 1 or p.get('top_k', 0) not in (0, -1):
            raise ValueError('RL replay needs untruncated sampling probabilities')
        if not r.get('turns'):
            raise ValueError('scored rollout has no policy turns')
        for turn in r['turns']:
            from .sampling import capture_items
            capture_items(turn.get('sampling') or {}, require_samples=False)
            if not (turn.get('replay') or {}).get('raw_calls_retained'):
                raise ValueError('GRPO needs raw sampled calls, not legacy normalized replay')
            from .graph_replay import block_ids
            if (contains_blocks(turn) or any(block_ids(turn))) and not block_bundle:
                raise ValueError('Neuralese blocks require a durable graph/block replay bundle')
        key = (r['task_id'], r.get('group'))
        groups.setdefault(key, []).append(r)
    return groups


def build_documents(rollouts, *, policy_id, block_bundle=None):
    groups = validate_rollouts(rollouts, policy_id, block_bundle=block_bundle)
    from .graph_replay import producer_turns
    documents = []
    for key, members in groups.items():
        advantages, stats = group_advantages([r['reward']['quality'] for r in members])
        # Constant-reward groups genuinely have no policy-gradient signal.
        # The coordinator routes them to more sampling / RFT rather than
        # pretending an optimizer update taught anything.
        if not stats['spread']:
            continue
        for rollout, advantage in zip(members, advantages):
            producers = producer_turns(rollout)
            for index, turn in enumerate(rollout['turns']):
                if not turn['sampling'].get('samples'):
                    continue  # forced-only writer still participates through producer graph
                documents.append({'task_id': rollout['task_id'], 'split': 'train',
                                  'messages': turn['messages'], 'tools': turn.get('tools'),
                                  'target': turn['target'], 'advantage': advantage,
                                  'sample': rollout.get('sample'), 'turn': index, 'sampling': turn['sampling'],
                                  'producers': producers, 'read_bindings': producers[index]['read_bindings']})
    return documents


class FullPolicyObjective:
    """Exact serving likelihood, shared trainer's updates and checkpoint lifecycle."""
    def __init__(self, engine, spec_path, windows, out):
        from ..serve.grad import GradSession
        self.spec_path = Path(spec_path)
        self.spec = json.loads(self.spec_path.read_text())
        self.session = GradSession(engine)
        if self.spec.get('block_bundle'):
            from .graph_replay import restore_bundle
            restore_bundle(engine, self.spec['block_bundle'], policy_id=self.spec['policy_id'])
            if getattr(engine, 'neuralese_projection', None) is None:
                raise ValueError('direct value-channel RL requires the continued writer projection')
        self.rows = self.spec['documents']
        self.cache_path = Path(out) / 'rl-behavior.pt'
        self.cache = None
        if (Path(out) / 'checkpoint.pt').exists() and not self.cache_path.exists() and any(
                not row.get('sampling') for row in self.rows):
            raise ValueError('resumed RL checkpoint is missing durable behavior scores')
        self.by_hash = {row['document_sha256']: row for row in self.rows}
        if len(self.by_hash) != len(self.rows):
            raise ValueError('RL replay document identities collide; retain per-sample weights')
        seen = set()
        for window in windows:
            key = window['document']
            if key in seen:
                raise ValueError('RL turn spans multiple windows: increase context/target tokens for exact replay')
            seen.add(key)
            if key not in self.by_hash:
                raise ValueError('RL train window has no bound replay document')
        expected = {key for key, row in self.by_hash.items() if row.get('split', 'train') == 'train'}
        if seen != expected:
            raise ValueError('RL document lost during corpus preparation/deduplication')

    def _score(self, row):
        if row.get('sampling'):
            from .sampling import captured_logps
            from .graph_replay import captured_producer_leaves
            leaves = (captured_producer_leaves(self.session, row['sampling'], row.get('producers') or [], read_bindings=row.get('read_bindings'))
                      if self.spec.get('block_bundle') else {})
            now, _, prediction = captured_logps(self.session, row['sampling'], leaves=leaves, return_predictions=True)
            self.last_predictions = prediction.detach()
            self.last_targets = torch.tensor([e['token'] for e in row['sampling']['samples']], device=prediction.device)
            return now
        prompt, target = self.session._target_items(row['messages'], row.get('tools'), row['target'])
        if len(prompt) + len(target) > self.session.engine.context:
            raise ValueError('RL replay exceeds the served model context; no truncation is allowed')
        scored = self.session._score(prompt, target, {}, write_terms=False)
        return scored['token_logp'].reshape(-1)

    def _behavior(self):
        if self.cache is not None:
            return
        binding = hashlib.sha256(self.spec_path.read_bytes()).hexdigest()
        if self.cache_path.exists():
            saved = torch.load(self.cache_path, weights_only=True, map_location='cpu')
            if saved['binding'] != binding:
                raise ValueError('RL behavior cache belongs to different rollout data')
            self.cache = saved['scores']
            return
        # Called before any update, after the shared trainer restored the full
        # collection checkpoint. Capture ALL documents before the first backward.
        with torch.no_grad():
            scores = {key: (torch.tensor([e['logp'] for e in row['sampling']['samples']], dtype=torch.float32)
                            if row.get('sampling') else self._score(row).float().cpu())
                      for key, row in self.by_hash.items()}
        from ..train.trajectory_state import atomic_checkpoint
        atomic_checkpoint(self.cache_path, {'binding': binding, 'scores': scores})
        self.cache = scores

    def normalizer(self, windows):
        self._behavior()
        return sum(self.cache[w['document']].numel() * self.by_hash[w['document']].get('multiplicity', 1) for w in windows)

    def forward(self, windows, accumulation=None, *, gradients=True):
        if not gradients:
            scores, predictions, hits = [], [], 0
            for w in windows:
                scores.append(self._score(self.by_hash[w['document']]))
                predictions.append(self.last_predictions)
                hits += int((self.last_predictions == self.last_targets).sum())
            tokens = sum(s.numel() for s in scores)
            loss = -sum(s.sum() for s in scores) / max(tokens, 1)
            metrics = {'ce': float(loss), 'gold_accuracy': hits / max(tokens, 1)}
            device = loss.device
            return loss, metrics, torch.cat(predictions), torch.zeros((), device=device), {'valid_positions': tokens, 'weight_mass': float(tokens)}
        self._behavior()
        denominator = (accumulation['rl_tokens'] if accumulation is not None
                       else self.normalizer(windows))
        total = None
        tokens = 0
        nll = 0.0
        hits = 0
        for window in windows:
            key = window['document']
            row = self.by_hash[key]
            logps = self._score(row)
            if row.get('sampling'):
                hits += int((self.last_predictions == self.last_targets).sum())
            old = self.cache[key].to(logps.device)
            loss = clipped_token_loss(logps, old, row['advantage'],
                clip_low=self.spec.get('clip_low', 0.2), clip_high=self.spec.get('clip_high', 0.28)).sum() * row.get('multiplicity', 1) / denominator
            total = loss if total is None else total + loss
            tokens += logps.numel()
            nll += float(-logps.detach().sum())
        metrics = {'ce': nll / max(tokens, 1), 'gold_accuracy': hits / max(tokens, 1), 'windows': len(windows),
                   'supervised_tokens': tokens, 'target_tokens': tokens,
                   'real_tokens': sum(len(w['ids']) - 2 for w in windows),
                   'forward_tokens': sum(len(w['ids']) - 2 for w in windows), 'rl': True}
        device = total.device
        return (total, metrics, torch.zeros((), device=device), torch.zeros((), device=device),
                {'valid_positions': tokens, 'weight_mass': float(tokens)})


def exact_replay_windows(engine, spec_path, train_windows, held_windows):
    """One complete captured turn per sample; shape placeholders never reach the model.

    Reuse the shared sampler/memory planner, but do not split policy context,
    duplicate signed advantages, or supervise the text loader's artificial EOS.
    """
    from array import array
    from .sampling import capture_items
    spec = json.loads(Path(spec_path).read_text())
    blocks = (json.loads(Path(spec['block_bundle']).read_text())['blocks']
              if spec.get('block_bundle') else {})
    docs = {d['document_sha256']: d for d in spec['documents']}
    def convert(windows):
        result, seen = [], set()
        for window in windows:
            key = window['document']
            if key not in docs:
                result.append(window)  # preserve EVERY ordinary held window
                continue
            if key in seen:
                continue
            seen.add(key)
            row = docs[key]
            prompt, target, _ = capture_items(row['sampling'])
            def coordinates(items):
                values = []
                for kind, value in items:
                    if kind == 'tok':
                        values.append(value)
                    else:
                        markers = 2 if engine.heads.read_markers else 0
                        values.extend([0] * (blocks[value]['length'] + markers))
                return values
            prefix, suffix = coordinates(prompt), coordinates(target)
            # Writer replay retains transitive graphs in the same backward.
            # Account every reachable writer occurrence conservatively; equal
            # payload IDs never hide distinct producer contexts from admission.
            candidates = {}
            for producer in row.get('producers', []):
                pp, pt, _ = capture_items(producer['sampling'], require_samples=False)
                for at, (kind, bid) in enumerate(pt):
                    if kind == 'block':
                        candidates.setdefault(bid, []).append((producer, pp + pt[:at], at))
            visited = set()
            def producer_tokens(bid):
                total = 0
                for producer, context, at in candidates.get(bid, []):
                    occurrence = (producer.get('graph_turn'), json.dumps(producer['sampling'], sort_keys=True), at)
                    if occurrence in visited:
                        continue
                    visited.add(occurrence)
                    count = len(coordinates(context)) + blocks[bid]['length']
                    if count > engine.context:
                        raise ValueError('RL producer trajectory exceeds served model context')
                    total += count
                    for kind, inner in context:
                        if kind == 'block':
                            total += producer_tokens(inner)
                return total
            producer_footprint = sum(producer_tokens(bid) for kind, bid in prompt + target if kind == 'block')
            if len(prefix) + len(suffix) > engine.context:
                raise ValueError('complete RL turn exceeds the served model context')
            ids = array('I', [engine.backbone.controls.open_id])
            ids.extend(prefix)
            ids.extend(suffix)
            ids.append(engine.backbone.controls.close_id)
            result.append({'ids': ids, 'prefix': 1, 'offset': 0, 'start': 0,
                'supervised_suffix_start': len(prefix), 'document': key,
                'document_id': key, 'cohort': 's7-grpo', 'groups': window['groups'],
                'sampling_group': key,
                'visible_forward_tokens': len(prefix) + len(suffix),
                'producer_forward_tokens': producer_footprint,
                'replay_forward_tokens': len(prefix) + len(suffix) + producer_footprint})
        return result
    train, held = convert(train_windows), convert(held_windows)
    return train, list(train), held
