#!/usr/bin/env python3
"""Lower verified-successful student episodes of RL rounds to per-call token rows (exact actor prompt ids + sampled ids).

Experiment 0 (plans/neuralese/APPROACH_REVIEW_20261111.md section 4): every model call of a successful episode is one
(prompt token ids as zero-weight context, sampled output token ids as the loss target) row, read from the captured
rollouts (`turns[*].sampling`), never re-rendered. Calls whose prompt carries native Neuralese state are skipped and
counted; sampled decision calls (no token stream) are lowered separately by ``decision_turns`` (GRPO trains them). Read-only on the rollout files.

    lower_rollout_turns.py --out turns.pkl ROUNDDIR [ROUNDDIR ...]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def episode_turns(row, stats):
    """(rows of one episode's usable calls). A row: dict(invocation, index, prompt int32[], sample int32[], logp f32[])."""
    rows = []
    for index, turn in enumerate(row['turns']):
        sampling = turn.get('sampling') or {}
        if 'samples' not in sampling:
            stats['decision_calls' if sampling.get('decision_sample') else 'skip_no_sample_stream'] += 1
            continue
        segments = sampling['prompt']
        if any(segment['kind'] != 'tokens' for segment in segments):
            stats['skip_native_state_prompt'] += 1
            continue
        if any(any(segment.get('producing') or []) for segment in segments):
            stats['skip_producing_prompt'] += 1
            continue
        if any(item['input_role'] != 'ordinary' for item in sampling['samples']):
            stats['skip_non_ordinary_sample'] += 1
            continue
        # A forced call (`constrained`, template readout): the tokens fed before the first sample are context, not
        # samples; they extend the prompt. Forced tokens after a sample (a write's suffix) are not lowered.
        inputs = sampling.get('generated_inputs') or []
        lead = 0
        while lead < len(inputs) and inputs[lead].get('sample_index') is None:
            lead += 1
        if any(item.get('sample_index') is None for item in inputs[lead:]):
            stats['skip_forced_after_sample'] += 1
            continue
        if lead:
            stats['forced_prefix_calls'] += 1
        prompt = np.fromiter((i for segment in segments for i in segment['token_ids']), dtype=np.int32)
        if lead:
            prompt = np.concatenate([prompt, np.array([item['token_id'] for item in inputs[:lead]], dtype=np.int32)])
        sample = np.array([item['token_id'] for item in sampling['samples']], dtype=np.int32)
        logp = np.array([item['logp'] for item in sampling['samples']], dtype=np.float32)
        rows.append({'invocation': turn['invocation_id'], 'index': index, 'prompt': prompt, 'sample': sample, 'logp': logp,
                     'policy': sampling['policy']['checkpoint_sha256']})
    return rows


def decision_turns(row, stats):
    """(rows of one episode's sampled decision calls). A decision readout scores every option (the closed continuation
    after the prompt: ``</think>``, the JSON value, ``<|im_end|>``) by its summed log-probability s_o and samples its answer
    from softmax(s / T) (``natlang.decision-sample/1``, ``rl/decision_policy.py``). A row: dict(invocation, index,
    prompt int32[], options [int32[]], chosen, temperature, scores f32[] (the actor's s_o), behavior_logq (log q(chosen))).
    The prompt is the exact captured token prompt; an argmax readout (no receipt) has no behaviour density and is skipped."""
    from natlang_neuralese.rl.decision_policy import validate_decision_sample
    rows = []
    for index, turn in enumerate(row['turns']):
        sampling = turn.get('sampling') or {}
        if not sampling.get('decision_sample'):
            continue
        if any(segment['kind'] != 'tokens' for segment in sampling['prompt']):
            stats['skip_decision_native_state_prompt'] += 1
            continue
        sample = validate_decision_sample(sampling)
        rows.append({'invocation': turn['invocation_id'], 'index': index,
                     'prompt': np.fromiter((i for segment in sampling['prompt'] for i in segment['token_ids']), dtype=np.int32),
                     'options': [np.array(ids, dtype=np.int32) for ids in sample['continuations']],
                     'chosen': sample['chosen'], 'temperature': sample['temperature'],
                     'scores': np.array(sampling['decision_sample']['log_probs'], dtype=np.float32),
                     'behavior_logq': float(sample['behavior_logps'][sample['chosen']]),
                     'policy': sampling['policy']['checkpoint_sha256']})
    stats['decision_rows'] += len(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rounds', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--role', default='student', help="episode role to lower ('teacher': JOLT+ privileged attempts)")
    args = parser.parse_args()
    stats, episodes, seen = collections.Counter(), [], set()
    for directory in args.rounds:
        with (directory / 'rollouts.jsonl').open('rb') as handle:
            for line in handle:
                if b'"quality": 1.0' not in line:
                    continue
                row = json.loads(line)
                reward = row.get('reward') or {}
                if reward.get('quality') != 1 or row.get('jolt_role', 'student') != args.role:
                    continue
                if row.get('split') != 'train' or row.get('unscored') or not row.get('turns'):
                    stats['skip_episode_not_train_or_unscored'] += 1
                    continue
                identity = (row['task_id'], hashlib.sha256(json.dumps([t.get('target') for t in row['turns']],
                                                                       sort_keys=True).encode()).hexdigest())
                if identity in seen:
                    stats['skip_duplicate_episode'] += 1
                    continue
                seen.add(identity)
                rows = episode_turns(row, stats)
                stats['episodes'] += 1
                stats['turns_total'] += len(row['turns'])
                stats['turns_kept'] += len(rows)
                episodes.append({'round': directory.name, 'task_id': row['task_id'], 'sample': row['sample'],
                                 'family': row['family'], 'rows': rows})
                print(directory.name, row['task_id'][:60], row['sample'], len(rows), '/', len(row['turns']),
                      file=sys.stderr, flush=True)
    stats['prompt_tokens'] = sum(len(r['prompt']) for e in episodes for r in e['rows'])
    stats['sample_tokens'] = sum(len(r['sample']) for e in episodes for r in e['rows'])
    with args.out.open('wb') as handle:
        pickle.dump({'episodes': episodes, 'stats': dict(stats)}, handle, protocol=4)
    print(json.dumps(dict(stats)))


if __name__ == '__main__':
    main()
