#!/usr/bin/env python3
"""Lower verified-successful student episodes of RL rounds to per-call token rows (exact actor prompt ids + sampled ids).

Experiment 0 (plans/neuralese/APPROACH_REVIEW_20261111.md section 4): every model call of a successful episode is one
(prompt token ids as zero-weight context, sampled output token ids as the loss target) row, read from the captured
rollouts (`turns[*].sampling`), never re-rendered. A call whose prompt imports native Neuralese state (a consumer) is
lowered with its sampled reply as the target and the import as ``imports``: a state whose producer document the episode
recorded (``write_sources``) is computed in the learner's own forward from that document (the pass-through,
plans/neuralese/NEURALESE_PASSTHROUGH_20261117.md); any other needs a ``NativeStates`` reader of the actor's block store
(fixed context exactly as the actor fed it) and is skipped and counted without one. Sampled decision calls (no token stream) are lowered separately by ``decision_turns`` (GRPO trains them), and a reasoned
decision's action reasoning by ``decision_reasoning_turns`` (GRPO trains it as an ordinary sampled call). Read-only on the rollout files.

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


class NativeStates:
    """Read-only access to an actor's native block store (``serve_native_checkpoint --store-dir``: ``blocks/<id[5:7]>/<id>
    .safetensors``, content-addressed and never collected during a run). The SGLang actor imports a block by handing the
    fork this very file (``sglang_actor._sglang_import``), so the store is the exact source of what the model saw. Blocks
    are verified against their id from the file (``read_native_state_file``) and decoded once, cached per instance."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.cache = {}

    def path(self, block_id):
        return self.directory / 'blocks' / block_id.split('_', 1)[1][:2] / f'{block_id}.safetensors'

    def exists(self, block_id):
        return block_id in self.cache or self.path(block_id).exists()

    def __call__(self, block_id):
        if block_id not in self.cache:
            self.cache[block_id] = self.load(block_id)
        return self.cache[block_id]

    def load(self, block_id):
        """The decoded, verified block, not cached (a reader that needs each block briefly, e.g. the GRPO learner)."""
        from natlang_neuralese.serve.store import read_native_state_file
        return read_native_state_file(self.path(block_id), block_id).load()


def write_sources(row, stats=None):
    """{block id: producer document} of an episode's Neuralese writes, from its ``execution_graph`` ``block_write`` events:
    ``source_capture`` is the exact input of the actor's capture (a text write: ``[produce_open] + text ids`` from
    position 0; ``sglang_actor._sglang_write_text``, ``engine._native_write_text_impl``), the state the rows
    [payload_start, payload_stop). A document is {'ids': int32[: payload_stop], 'start', 'stop'}. A producer whose
    context itself imported a stored state (an operator call under its soft prompt state, ``engine.capture_prompt``)
    records it as a ``native_state`` segment of ``token_count`` rows: the document holds placeholder ids there and
    ``imports`` [(position, state id)], read as fixed stored context (``lfm3_fast_train.passthrough_plan``). A
    ``native_state`` segment without its row count is not a source (counted)."""
    sources = {}
    for event in row.get('execution_graph') or ():
        capture = event.get('source_capture') if event.get('kind') == 'block_write' else None
        if not capture or not event.get('block'):
            continue
        ids, imports, lowerable = [], [], True
        for segment in capture['segments']:
            if segment.get('kind') == 'tokens':
                ids.extend(int(i) for i in segment['token_ids'])
            elif segment.get('kind') == 'native_state' and type(segment.get('token_count')) is int and segment.get('id'):
                imports.append((len(ids), segment['id']))
                ids.extend([0] * segment['token_count'])   # placeholders: the import replaces what these rows feed
            else:
                lowerable = False
                break
        if not lowerable:
            if stats is not None:
                stats['source_not_tokens'] += 1
            continue
        start, stop = int(capture['payload_start']), int(capture['payload_stop'])
        if not 0 < start <= stop <= len(ids) or any(at >= start for at, _ in imports):
            if stats is not None:
                stats['source_bad_span'] += 1
            continue
        sources[event['block']] = {'ids': np.array(ids[:stop], dtype=np.int32), 'start': start, 'stop': stop,
                                   **({'imports': imports} if imports else {})}
        if imports and stats is not None:
            stats['source_stored_imports'] += len(imports)
    return sources


def prompt_imports(segments, states, stats, sources=None):
    """(prompt int32[], imports, used sources) of a captured prompt, or None when a state is not lowerable.

    The SGLang actor feeds an exact imported state (``kind: native_state``) as ``len(block.token_ids)`` placeholder
    positions between the consume markers (which are ordinary token segments); at those positions every attention
    layer's keys (after the K norm, before RoPE) and values and every conv layer's input B*x are the stored ones
    (``sglang-natlang`` ``models/lfm3.py`` ``NatlangImport``), and the placeholder rows' own outputs are never read. The
    lowered prompt carries the block's own token ids at those positions (in vocabulary; their embeddings reach nothing)
    and ``imports`` [(start, key)]. A block with a recorded producer document (``sources``, ``write_sources``) is a
    pass-through import: the learner computes its state in-graph (``lfm3_fast_train.passthrough_plan``; a
    ``stop_gradient`` import under the key ``<id>#frozen``, detached). Otherwise the stored state is read from the actor
    store (``states``) as fixed context (``lfm3_fast_train.import_tensors``)."""
    ids, imports, used = [], [], {}
    for segment in segments:
        if segment['kind'] == 'tokens':
            ids.extend(segment['token_ids'])
            continue
        if segment['kind'] != 'native_state':
            stats['skip_native_state_prompt'] += 1
            return None
        source = (sources or {}).get(segment['id'])
        if source is not None:
            key = segment['id'] + ('#frozen' if segment.get('stop_gradient') else '')
            used[key] = {**source, 'frozen': True} if segment.get('stop_gradient') else source
            imports.append((len(ids), key))
            ids.extend(int(i) for i in source['ids'][source['start']:source['stop']])
            stats['passthrough_imports'] += 1
            continue
        if states is None:
            stats['skip_native_state_prompt'] += 1
            return None
        if not states.exists(segment['id']):
            stats['skip_native_state_missing'] += 1
            return None
        block = states(segment['id'])
        imports.append((len(ids), segment['id']))
        ids.extend(int(i) for i in block.token_ids)
        stats['stored_state_imports'] += 1
    return np.array(ids, dtype=np.int32), imports, used


def episode_turns(row, stats, states=None):
    """(rows of one episode's usable calls). A row: dict(invocation, index, prompt int32[], sample int32[], logp f32[],
    imports [(start, block_id)]). ``states``: a ``NativeStates`` reader; without it calls importing native state are skipped."""
    rows = []
    sources = write_sources(row, stats)
    for index, turn in enumerate(row['turns']):
        sampling = turn.get('sampling') or {}
        if 'samples' not in sampling:
            stats['decision_calls' if sampling.get('decision_sample') else 'skip_no_sample_stream'] += 1
            continue
        lowered = sampled_call(turn['invocation_id'], index, sampling, stats, states, sources)
        if lowered is not None:
            rows.append(lowered)
    return rows


def sampled_call(invocation, index, sampling, stats, states=None, sources=None, stop=None):
    """One captured generation (``sampling``: an RL capture with ``samples``) as a row of ``episode_turns``, or None when
    it is not lowerable (counted in ``stats``). ``stop``: the number of leading samples the row keeps (None: all)."""
    segments = sampling['prompt']
    lowered = prompt_imports(segments, states, stats, sources)
    if lowered is None:
        return None
    if any(any(segment.get('producing') or []) for segment in segments):
        stats['skip_producing_prompt'] += 1
        return None
    samples = sampling['samples'][:stop]
    if any(item['input_role'] != 'ordinary' for item in samples):
        stats['skip_non_ordinary_sample'] += 1
        return None
    # A forced call (`constrained`, template readout): the tokens fed before the first sample are context, not
    # samples; they extend the prompt. Forced tokens after a sample (a write's suffix) are not lowered.
    inputs = sampling.get('generated_inputs') or []
    lead = 0
    while lead < len(inputs) and inputs[lead].get('sample_index') is None:
        lead += 1
    if any(item.get('sample_index') is None for item in inputs[lead:]):
        stats['skip_forced_after_sample'] += 1
        return None
    if lead:
        stats['forced_prefix_calls'] += 1
    prompt, imports, used = lowered
    if imports:
        stats['native_state_calls'] += 1
    if lead:
        prompt = np.concatenate([prompt, np.array([item['token_id'] for item in inputs[:lead]], dtype=np.int32)])
    sample = np.array([item['token_id'] for item in samples], dtype=np.int32)
    logp = np.array([item['logp'] for item in samples], dtype=np.float32)
    law = sampling.get('sampling_law') or {}
    return {'invocation': invocation, 'index': index, 'prompt': prompt, 'sample': sample, 'logp': logp,
            'imports': imports, 'sources': used, 'policy': sampling['policy']['checkpoint_sha256'],
            # the weights actually served: after in-place updates ``policy`` still names the start checkpoint
            'served': (sampling.get('actor_export') or {}).get('sha256'),
            'temperature': law.get('temperature'), 'guidance': bool(law.get('guidance')),
            # the draw's nucleus (1.0: untruncated); `logp` is the untruncated law either way
            'top_p': float(law.get('top_p', 1.)), 'probabilities': law.get('probabilities')}


def decision_reasoning_turns(row, stats, states=None, think_close=None):
    """(rows of one episode's reasoned-decision action reasonings), shaped as ``episode_turns`` rows. A reasoned readout
    (DECISIONS 2026-11-23 "reasoned decisions") samples k reasonings and scores its options after each; the sample whose
    choice the program used (its action entry, ``decision_reasoning.action``) carries its generation's exact capture
    (``reasoning_sampling``, ``rl.task_rollouts.rollout_turns``). Its reasoning is part of the sampled action: the row is
    that generation's own prompt ids and sampled ids with their behaviour log-probabilities, kept through the first
    ``think_close`` (``</think>``) included; the free reply after it was never used and is not kept. A reasoning cut
    before any close keeps every sample. The other samples only estimated the readout's confidence and give no row.
    Counted: ``decision_reasoning_rows``/``_tokens``/``_unclosed``, and whether the action's scored prompt continues the
    reasoning's own prompt with exactly the kept ids (``decision_reasoning_ids_exact``/``_rendered``: the engines
    re-render the reasoning text when they score)."""
    from natlang_neuralese.data.decision_graph import capture_prompt_ids
    rows = []
    sources = None
    for index, turn in enumerate(row['turns']):
        reasoned = turn.get('decision_reasoning')
        sampling = reasoned.get('reasoning_sampling') if isinstance(reasoned, dict) else None
        if not isinstance(sampling, dict) or reasoned.get('action') is not True:
            continue
        if not isinstance(sampling.get('samples'), list) or not sampling['samples']:
            stats['skip_decision_reasoning_no_samples'] += 1
            continue
        if think_close is None:
            stats['skip_decision_reasoning_no_close_id'] += 1
            continue
        ids = [int(item['token_id']) for item in sampling['samples']]
        stop = ids.index(think_close) + 1 if think_close in ids else None
        if sources is None:
            sources = write_sources(row)
        lowered = sampled_call(turn['invocation_id'], index, sampling, stats, states, sources, stop)
        if lowered is None:
            stats['skip_decision_reasoning_lowering'] += 1
            continue
        lowered['decision_reasoning'] = True
        rows.append(lowered)
        stats['decision_reasoning_rows'] += 1
        stats['decision_reasoning_tokens'] += len(lowered['sample'])
        if stop is None:
            stats['decision_reasoning_unclosed'] += 1
        scored = capture_prompt_ids(turn.get('sampling'))
        own = capture_prompt_ids(sampling)
        kept = ids[:stop]
        exact = scored is not None and own is not None and scored[:len(own) + len(kept)] == own + kept
        stats['decision_reasoning_ids_exact' if exact else 'decision_reasoning_ids_rendered'] += 1
    return rows


def decision_turns(row, stats, states=None):
    """(rows of one episode's sampled decision calls). A decision readout scores every option (the closed continuation
    after the prompt: ``</think>``, the JSON value, ``<|im_end|>``) by its summed log-probability s_o and samples its answer
    from softmax(s / T) (``natlang.decision-sample/1``, ``rl/decision_policy.py``). A row: dict(invocation, index,
    prompt int32[], options [int32[]], chosen, temperature, scores f32[] (the actor's s_o), behavior_logq (log q(chosen))).
    The prompt is the exact captured prompt (imported native state as in ``episode_turns``: ``imports``); an argmax readout
    (no receipt) has no behaviour density and is skipped."""
    from natlang_neuralese.rl.decision_policy import validate_decision_sample
    rows = []
    sources = write_sources(row)
    for index, turn in enumerate(row['turns']):
        sampling = turn.get('sampling') or {}
        if not sampling.get('decision_sample'):
            continue
        skipped = collections.Counter()
        lowered = prompt_imports(sampling['prompt'], states, skipped, sources)
        if lowered is None:
            for key, count in skipped.items():
                stats['skip_decision_' + key[len('skip_'):]] += count
            continue
        sample = validate_decision_sample(sampling)
        rows.append({'invocation': turn['invocation_id'], 'index': index, 'prompt': lowered[0], 'imports': lowered[1],
                     'sources': lowered[2],
                     'options': [np.array(ids, dtype=np.int32) for ids in sample['continuations']],
                     'chosen': sample['chosen'], 'temperature': sample['temperature'],
                     'scores': np.array(sampling['decision_sample']['log_probs'], dtype=np.float32),
                     'behavior_logq': float(sample['behavior_logps'][sample['chosen']]),
                     'policy': sampling['policy']['checkpoint_sha256'],
                     # the weights actually served (``episode_turns``)
                     'served': (sampling.get('actor_export') or {}).get('sha256')})
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
    from natlang_neuralese.common import data_registry
    data_registry.record_dataset(data_registry.path_id('turns', args.out), args.out, kind='chains',
                                 format='lowered per-call turns pickle {episodes, stats} (exp0_sft --turns, grpo --probe-turns)',
                                 inputs=[str(d / 'rollouts.jsonl') for d in args.rounds], rows=len(episodes),
                                 license='project-generated (own LFM3 policy samples)')
    print(json.dumps(dict(stats)))


if __name__ == '__main__':
    main()
