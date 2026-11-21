#!/usr/bin/env python3
"""Filled-in reasoning for verified teacher trajectories (STaR-style rationalization; DECISIONS 2026-11-20 "filled
reasoning"). Teacher think blocks are context in SFT (sft-v3's -16 pts came from training them), so a teacher row
trains its action after reasoning the student never wrote: an empty block or the teacher's summary. This pass replaces
each teacher call's think block with reasoning drawn from the current student that leads to the teacher's action, step
by step along whole trajectories (each call's filled reasoning is part of the history of the later calls, as
preserve_thinking renders it), and writes them as SFT rows that train the reasoning and the action together.

Per teacher step t (owner's design, 2026-11-20):
  1. candidates from one shared prefix: ``--unhinted`` plain student samples on the real context, and ``--hinted``
     samples with a training guidance turn that shows the teacher's next replies and what came back from them (actions
     and tool responses t .. t + ``--hint-horizon``). The hint turn goes right before the step's assistant opening, so
     the hinted prompt shares the whole history with the unhinted one in SGLang's radix cache.
  2. every candidate is scored by prefill only, on the unhinted trajectory: the student's per-token log-likelihood of the
     teacher's actions t .. t + ``--score-horizon`` after it, minus the same with no reasoning (an empty think block):
     the per-action-token improvement over the no-reasoning baseline.
  3. gates: the block closed; a hinted block does not quote or name the hinted actions or responses (a 40-character
     window of them, or a short action whole, found in the reasoning and not in the context) nor refer to the hint;
     the unhinted student's own p(``</think>``) where the block closes is at least ``--close-floor`` (a close the hint
     forced); the improvement is above ``--min-improvement``.
  4. selection against the future-agreement bias (Strauss, Fowl & Razin, arXiv 2611.23397; DECISIONS 2026-11-20 16:30):
     a sequence score favours blocks that close right after naming the action. Candidates are compared only within a
     length band drawn from the student's own distribution at this step: one closed unhinted sample's length is the
     anchor, the band [``--band``] times it, and the best-scoring admitted candidate inside it wins (else the next
     anchor). The selected set's lengths therefore follow the student's own; ``report`` checks lengths and the
     ``</think>`` hazard against the unhinted samples.
  5. no admitted candidate: the call keeps an empty think block (context) and its action trains, as today.
The chosen block (or the empty one), the teacher action and its tool output extend the context the next step reasons
on; its scoring prefill already left that history in the radix cache, so a step costs only its new tokens.

    filled_reasoning.py generate --out DIR [--work WORK ...] --sglang URL [--front URL] [--polite] ...
    filled_reasoning.py status   --out DIR [--remaining-from WORK ...]
    filled_reasoning.py build    --out DIR --corpus OUT       # chains.pkl for sft_sources build --aux filled=...

Generation is resumable and incremental: DIR/fills.jsonl holds one line per finished episode (the key, its WORK, the
policy identity served while it ran, per call the chosen block, the policy that drew it and every candidate's
statistics); a rerun takes only episodes without a line, so new teacher episodes are covered as the campaigns add them.
An episode in progress keeps its finished steps in DIR/partial/ (one file per episode, rewritten after every step), so
a stopped generation (the expert-iteration cycle runs it in windows that end when the engine is needed elsewhere)
resumes it at the step it reached instead of redrawing up to two hours of steps (``--no-partial`` opts out). ``build`` re-renders each
episode from its WORK (``sft_sources.call_row``, checked against the recorded digest), rebuilds the prompts with the
filled history and writes OUT/chains.pkl (rounds ``filled:<episode key>``; per-row ``sampled`` flags: the block was
sampled by the student, the action was not, ``exp0_sft.sampled_targets``) and OUT/report.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO), str(HERE), str(REPO / 'scripts/rl_tasks')]

from natlang_neuralese.common import data_registry  # noqa: E402
from natlang_neuralese.common.hashing import sha256_file_hex as sha256_file  # noqa: E402

SCHEMA = 'natlang.filled-reasoning/1'
MIN_ROOM = 256            # reasoning tokens a step must have room for, else it keeps an empty block
HINT_HEADER = ('Verified reference for this call and the calls after it (training only):\n\n'
               'The replies that follow your reasoning, and what came back from each:\n')
HINT_PHRASES = re.compile(r'verified reference|training only|\bthe reference\b|reference (?:reply|replies|answer|action)|'
                          r'replies that follow|what came back|'
                          r'(?:given|provided|upcoming) (?:reply|replies|answer|action)s?\b|\bthe hint\b', re.I)
TOOL_TAGS = re.compile(r'</?(?:tool_call|tool_response|function(?:=[^>]*)?|parameter(?:=[^>]*)?)>')
LEAK_WINDOW, LEAK_STEP, LEAK_SHORT = 40, 8, 12


# ---- pure helpers (tested) -------------------------------------------------------------------------------------------
def norm(text):
    return ' '.join(TOOL_TAGS.sub(' ', text).split())


def leak(reasoning, hinted, context):
    """The hinted text the reasoning quotes or names, or None: a ``LEAK_WINDOW``-character window (every ``LEAK_STEP``,
    so any verbatim span of 48+ characters is caught) of a hinted part, or a whole hinted part shorter than a window
    (at least ``LEAK_SHORT`` characters: a short action named outright), found in the reasoning and not already in the
    context (quoting what the student has seen is not a leak). Texts are whitespace- and tool-tag-normalized."""
    reasoning, context = norm(reasoning), norm(context)
    if HINT_PHRASES.search(reasoning):
        return 'hint_phrase'
    for part in hinted:
        part = norm(part)
        if len(part) < LEAK_WINDOW:
            if len(part) >= LEAK_SHORT and part in reasoning and part not in context:
                return part
            continue
        for start in range(0, len(part) - LEAK_WINDOW + 1, LEAK_STEP):
            window = part[start:start + LEAK_WINDOW]
            if window in reasoning and window not in context:
                return window
    return None


def assistant_opening(context, im_start):
    """Where the context's last turn opening starts (the generation prompt ``<|im_start|>assistant\\n<think>``)."""
    for i in range(len(context) - 1, -1, -1):
        if context[i] == im_start:
            return i
    raise ValueError('the context has no turn opening')


def hint_context(context, hint_ids, im_start):
    """The hinted prompt: ``hint_ids`` (a whole system turn) inserted right before the step's assistant opening, so the
    history before it is the unhinted context's own prefix (radix-cache shared)."""
    at = assistant_opening(context, im_start)
    return list(context[:at]) + list(hint_ids) + list(context[at:])


def improvement(scores, baseline, lengths):
    """Per-action-token log-likelihood improvement over the no-reasoning baseline on the spans both cover."""
    n = min(len(scores), len(baseline))
    if not n:
        return None
    total = sum(lengths[:n])
    return sum((scores[i] - baseline[i]) * lengths[i] for i in range(n)) / total


def select(candidates, band, rng):
    """(chosen candidate or None, anchor length or None). Anchors are the closed unhinted samples' lengths in random
    order (the student's own length at this step); each admits the gated candidates whose length lies in
    [band[0], band[1]] times it, and the best improvement among them wins; the first anchor with one decides."""
    anchors = [c['length'] for c in candidates if c['origin'] == 'unhinted' and c['closed']]
    rng.shuffle(anchors)
    for anchor in anchors:
        inside = [c for c in candidates if c['admitted'] and band[0] * anchor <= c['length'] <= band[1] * anchor]
        if inside:
            return max(inside, key=lambda c: c['improvement']), anchor
    return None, None


def calls_of(chain, close_id):
    """A chain's calls (``exp0_sft.chains_of``): prompt, sample, weight, cut (after the first ``</think>``, None when the
    reply has none), action ids and weights, and the suffix to the next call's prompt (tool output, next opening)."""
    ids, out = chain['ids'], []
    segs = chain['segs']
    for i, (start, length, row) in enumerate(segs):
        sample = np.asarray(row['sample'])
        weight = np.asarray(row['weight'], np.float32) if 'weight' in row else np.ones(length, np.float32)
        found = np.flatnonzero(sample == close_id)
        cut = int(found[0]) + 1 if len(found) and found[0] + 1 < len(sample) else None
        end = segs[i + 1][0] if i + 1 < len(segs) else start + length
        out.append({'prompt': ids[:start], 'sample': sample, 'weight': weight, 'cut': cut,
                    'action': sample[cut:] if cut is not None else sample,
                    'action_weight': weight[cut:] if cut is not None else weight,
                    'suffix': ids[start + length:end], 'row': row})
    return out


def chain_digest(chain):
    return hashlib.sha256(np.asarray(chain['ids'], np.int32).tobytes()).hexdigest()[:24]


def filled_rows(calls, fills, close_id, name):
    """Rows of one chain with the filled history: each call's prompt is the context so far; its sample the chosen block
    and the teacher action (block trained, ``sampled``; action at its row weights), or ``</think>`` and the action
    (context) without one; a call with no think block keeps its reply."""
    context, rows = list(calls[0]['prompt']), []
    for i, (call, fill) in enumerate(zip(calls, fills)):
        if call['cut'] is None:
            sample, weight = list(call['sample']), list(call['weight'])
            sampled = [False] * len(sample)
        else:
            block = list(fill['reasoning']) if fill and fill.get('reasoning') else [close_id]
            action, action_weight = [int(t) for t in call['action']], [float(w) for w in call['action_weight']]
            trained = 1. if fill and fill.get('reasoning') and any(action_weight) else 0.
            sample = block + action
            weight = [trained] * len(block) + action_weight
            sampled = [bool(fill and fill.get('reasoning'))] * len(block) + [False] * len(action)
        rows.append({'id': f'{name}#{call["row"].get("index", i)}', 'invocation': call['row'].get('invocation', name),
                     'index': call['row'].get('index', i), 'prompt': np.asarray(context, np.int32),
                     'sample': np.asarray(sample, np.int32), 'weight': np.asarray(weight, np.float32),
                     'sampled': np.asarray(sampled, bool)})
        context = context + [int(t) for t in sample] + [int(t) for t in call['suffix']]
    return rows


# ---- sources ---------------------------------------------------------------------------------------------------------
def default_works():
    return data_registry.role_paths('grpo-sd-teacher')


def teacher_episodes(works, held_manifest, funnel):
    """[{key, work, name, task_id, teacher}] of every verified teacher success of WORKs (first WORK wins a key), held
    closure excluded (``TeacherDemos``' selection without the per-task cap: every episode is data)."""
    from sft_sources import held_closure, held_excluded_ids, verified_episodes
    found, tasks = {}, {}
    for work in works:
        if not (Path(work) / 'replay.jsonl').exists():
            continue
        for episode, record in verified_episodes(work, funnel, sources={'teacher'}):
            if episode['key'] in found or not record.get('name'):
                continue
            tasks[episode['task_id']] = episode['task']
            found[episode['key']] = {'key': episode['key'], 'work': str(work), 'name': record['name'],
                                     'task_id': episode['task_id'],
                                     'teacher': (episode.get('teacher') or {}).get('model_id') or ''}
    excluded = held_excluded_ids(held_manifest, tasks, held_closure(held_manifest))
    funnel['teacher']['held_excluded_filled'] = sum(e['task_id'] in excluded for e in found.values())
    return [e for e in found.values() if e['task_id'] not in excluded]


class Renderer:
    """sft_sources' rendering of a WORK's replayed episode into chains (``call_row`` per call, ``chains_of``) with one
    tokenizer directory (``TemplateRenderer``: tokenizer, chat template, model type), thread-safe."""

    def __init__(self, tokenizer_dir):
        from sft_sources import TemplateRenderer
        self.tokenizer_dir = str(tokenizer_dir)
        self.renderer = TemplateRenderer(self.tokenizer_dir)
        self.lock = threading.Lock()

    def chains(self, item):
        from exp0_sft import chains_of
        from sft_sources import call_row, replayed_turns
        with self.lock:
            rows = [call_row(self.renderer, 'teacher', item['key'], index, turn)[0]
                    for index, turn in enumerate(replayed_turns(item['work'], item['name']))]
        return chains_of([{'task_id': item['task_id'], 'round': item['key'], 'rows': rows}]), self.renderer


# ---- the actor -------------------------------------------------------------------------------------------------------
class Actor:
    """SGLang ``/generate`` with an admission gate: ``polite`` (beside a training run) starts a draw only while the
    engine has no waiting request and a free running slot, keeping at most ``min_inflight`` draws otherwise; a
    dedicated actor takes up to ``max_inflight``."""

    def __init__(self, url, front, polite, max_inflight, min_inflight, max_running):
        self.url, self.front, self.polite = url.rstrip('/'), (front or '').rstrip('/'), polite
        self.max_inflight, self.min_inflight, self.max_running = max_inflight, min_inflight, max_running
        self.cond = threading.Condition()
        self.inflight = 0
        self.load_at, self.load = 0., None
        self.policy_at, self.policy_value = 0., None
        self.counts = collections.Counter()

    def post(self, path, body, timeout=7200, down_for=4 * 3600):
        """POST to SGLang; a refused connection (a training run releases its actor during the learner phase, and
        restarts it at resumes) waits and retries for up to ``down_for`` seconds."""
        deadline = time.time() + down_for
        while True:
            request = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                             headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read())
            except urllib.error.URLError as error:
                refused = isinstance(getattr(error, 'reason', None), (ConnectionRefusedError, ConnectionResetError))
                if not refused or time.time() > deadline:
                    raise
                self.counts['actor_down_waits'] += 1
                time.sleep(15)

    def get(self, url, timeout=10):
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read())

    def _spare(self):
        if time.time() - self.load_at > 2:
            try:
                loads = self.get(self.url + '/v1/loads')['loads'][0]
                self.load = (loads['num_running_reqs'], loads['num_waiting_reqs'])
            except Exception:
                self.load = None
            self.load_at = time.time()
        if self.load is None:
            return False
        running, waiting = self.load
        return waiting == 0 and running < self.max_running - 1

    def acquire(self):
        with self.cond:
            while True:
                limit = self.max_inflight if (not self.polite or self._spare()) else self.min_inflight
                if self.inflight < limit:
                    self.inflight += 1
                    return
                self.cond.wait(1)

    def release(self):
        with self.cond:
            self.inflight -= 1
            self.cond.notify_all()

    def policy(self):
        """The served policy's identity (the front's ``rl_policy_binding``; GRPO rebinds it in place every step)."""
        if not self.front:
            return None
        if time.time() - self.policy_at > 30:
            try:
                info = self.get(self.front + '/v1/neuralese/info')
                binding = info.get('rl_policy_binding') or {}
                self.policy_value = {'checkpoint_sha256': binding.get('checkpoint_sha256'),
                                     'native_checkpoint_sha256': (info.get('native_state') or {}).get('checkpoint_sha256')}
            except Exception as error:
                self.policy_value = {'error': repr(error)[:200]}
            self.policy_at = time.time()
        return self.policy_value


# ---- generation ------------------------------------------------------------------------------------------------------
class Filler:
    def __init__(self, args, actor, tokenizer, stops, banned):
        self.args, self.actor, self.tok = args, actor, tokenizer
        self.close = tokenizer.token_to_id('</think>')
        self.im_start = tokenizer.token_to_id('<|im_start|>')
        self.stops = sorted({self.close, *stops})
        self.banned = sorted(banned)
        self.counts = collections.Counter()
        self.lock = threading.Lock()

    def text(self, ids, special=False):
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=not special)

    def response_text(self, suffix):
        text = self.text(suffix)
        text = re.sub(r'\s*assistant\s*(<think>)?\s*$', '', text).strip()
        text = re.sub(r'^\s*(tool|user)\n', '', text)
        cap = self.args.hint_response_chars
        return text if len(text) <= cap else text[:cap // 2] + '\n...\n' + text[-cap // 2:]

    def hint(self, calls, t):
        """(hint turn ids, hinted texts for the leak check): the teacher's replies t .. t + hint horizon and the tool
        output after each, as text."""
        parts, hinted = [HINT_HEADER], []
        for k, s in enumerate(range(t, min(len(calls), t + 1 + self.args.hint_horizon))):
            action = self.text(calls[s]['action']).strip()
            hinted.append(action)
            parts.append(f'\n### Reply {k + 1}\n{action}\n')
            if s + 1 < len(calls) and len(calls[s]['suffix']):
                response = self.response_text(calls[s]['suffix'])
                if response:
                    hinted.append(response)
                    parts.append(f'\n### What came back {k + 1}\n{response}\n')
        body = list(self.tok.encode(''.join(parts), add_special_tokens=False).ids)[:self.args.hint_max_tokens]
        head = list(self.tok.encode('<|im_start|>system\n', add_special_tokens=False).ids)
        foot = list(self.tok.encode('<|im_end|>\n', add_special_tokens=False).ids)
        return head + body + foot, hinted

    def draw(self, context, room, seed):
        from natlang_neuralese.serve.sglang_actor import TOP_K_KEY, decode_record
        params = {'temperature': 1.0, 'max_new_tokens': room, 'stop_token_ids': self.stops,
                  'skip_special_tokens': False, 'no_stop_trim': True, 'sampling_seed': seed,
                  'custom_params': {TOP_K_KEY: 1}}
        if self.args.top_p < 1:
            params['top_p'] = float(self.args.top_p)
        if self.banned:
            params['logit_bias'] = {str(t): -1e30 for t in self.banned}
        self.actor.acquire()
        try:
            result = self.actor.post('/generate', {'input_ids': list(context), 'sampling_params': params})
        finally:
            self.actor.release()
        ids, records = result.get('output_ids'), (result.get('meta_info') or {}).get(TOP_K_KEY)
        if not isinstance(ids, list) or not isinstance(records, list) or len(ids) != len(records):
            raise RuntimeError('SGLang returned no per-token natlang_top_k records')
        return [int(i) for i in ids], [decode_record(r, 1)[0] for r in records]

    def prefill(self, ids):
        self.actor.post('/generate', {'input_ids': list(ids), 'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1}})

    def score(self, context, block, calls, t):
        """One prefill of context + block + the teacher's next actions (``grpo.future_ids``): (each action span's mean
        token logp, span lengths, the block's unhinted token logps, p(``</think>``) before each block token)."""
        from grpo import future_ids
        ids, spans = future_ids(list(context) + list(block), calls, t, self.close, self.args.score_horizon,
                                self.args.max_len)
        start = len(context) - 1
        result = self.actor.post('/generate', {'input_ids': ids, 'sampling_params': {'temperature': 0.0,
                                                                                    'max_new_tokens': 1},
                                               'return_logprob': True, 'logprob_start_len': start,
                                               'token_ids_logprob': [self.close]})
        meta = result['meta_info']
        values = [v[0] for v in meta['input_token_logprobs']]
        closes = meta.get('input_token_ids_logprobs') or []
        offset = len(ids) - len(values)
        means = []
        for position, length in spans:
            got = values[position - offset:position - offset + length]
            if len(got) != length or any(v is None for v in got):
                raise RuntimeError('SGLang returned no logprob for an action token')
            means.append(float(sum(got) / length))
        block_logp = [values[len(context) + k - offset] for k in range(len(block))]
        close_p = []
        coff = len(ids) - len(closes)
        for k in range(len(block)):
            entry = closes[len(context) + k - coff] if 0 <= len(context) + k - coff < len(closes) else None
            lp = next((float(e[0]) for e in (entry or ()) if int(e[1]) == self.close and e[0] is not None), None)
            close_p.append(math.exp(lp) if lp is not None else float('nan'))
        return means, [length for _, length in spans], block_logp, close_p

    def _evaluate(self, context, calls, t, candidates, baseline, hinted, context_text):
        """Score (one prefill each, on the unhinted trajectory) and gate the closed candidates in place."""
        a = self.args
        scored = [c for c in candidates if c['closed'] and 'improvement' not in c]
        with ThreadPoolExecutor(max(1, len(scored))) as pool:
            for c, (means, span_lengths, block_logp, close_p) in zip(
                    scored, pool.map(lambda c: self.score(context, c['ids'], calls, t), scored)):
                c['improvement'] = improvement(means, baseline, span_lengths)
                c['action_logp'] = means[0]
                c['policy_logp'] = float(np.nansum(np.asarray(block_logp, float)))
                body = np.asarray(close_p[:-1], float)
                c['hazard'] = float(np.nanmean(body) * 1000) if len(body) else None
                c['close_p'] = close_p[-1]
                if c['origin'] == 'hinted':   # per-token log ratio student / hinted proposal (how in-distribution)
                    c['log_ratio'] = (c['policy_logp'] - c['sample_logp']) / max(1, c['length'])
        for c in candidates:
            if 'admitted' in c:
                continue
            reasons = []
            if not c['closed']:
                reasons.append('unclosed' if not c.get('capped') else 'past_band')
            else:
                c['leak'] = leak(self.text(c['ids'][:-1]), hinted, context_text)
                if c['leak'] and c['origin'] == 'hinted':
                    reasons.append('leak')
                if not (c['close_p'] >= a.close_floor):
                    reasons.append('forced_close')
                if c['improvement'] is None or c['improvement'] <= a.min_improvement:
                    reasons.append('no_improvement')
                if c['origin'] == 'hinted' and a.ratio_floor is not None and c['log_ratio'] < a.ratio_floor:
                    reasons.append('off_distribution')
            c['admitted'], c['rejected'] = not reasons, reasons

    def _candidates(self, jobs, room, rng):
        """Draw ``jobs`` [(origin, context, max tokens)] in parallel: candidate dicts (``capped``: stopped at a band cap
        below the step's room, so unclosed means it decoded past the band)."""
        seeds = [rng.randrange(1, 2**31) for _ in jobs]
        with ThreadPoolExecutor(max(1, len(jobs))) as pool:
            drawn = list(pool.map(lambda j: self.draw(j[1][1], j[1][2], j[0]), zip(seeds, jobs)))
        out = []
        for (origin, _, cap), (ids, logp) in zip(jobs, drawn):
            out.append({'origin': origin, 'ids': ids, 'length': len(ids), 'closed': bool(ids) and ids[-1] == self.close,
                        'sample_logp': float(np.sum(logp)) if len(logp) else 0., 'capped': cap < room})
        return out

    def _result(self, candidates, chosen, anchor, baseline, room, **extra):
        stats = [{k: (round(v, 5) if isinstance(v, float) else (bool(v) if k == 'leak' else v))
                  for k, v in c.items() if k != 'ids'} for c in candidates]
        for s_, c in zip(stats, candidates):
            s_['selected'] = c is chosen
        return {'kind': 'filled' if chosen else 'fallback', 'reasoning': chosen['ids'] if chosen else None,
                'origin': chosen['origin'] if chosen else None, 'anchor': anchor, 'baseline': baseline,
                'room': room, 'candidates': stats, 'generated_tokens': sum(c['length'] for c in candidates), **extra}

    def step(self, context, calls, t, context_text, rng, trend=0):
        """Candidates, scores and the choice for step t (the module docstring, 1-5). ``--draw staged`` (default) spends
        draws only where they can still change the choice; ``parallel`` draws every candidate at once (before
        2026-11-21). ``trend``: consecutive drawn steps of this trajectory that fell back."""
        a, call = self.args, calls[t]
        room = min(a.max_tokens, a.max_len - len(context) - len(call['action']) - 1)
        if call['cut'] is None:
            return {'kind': 'no_think'}
        if room < MIN_ROOM:
            return {'kind': 'context_full'}
        if getattr(a, 'draw', 'staged') == 'parallel':
            return self._step_parallel(context, calls, t, context_text, rng, room)
        # 1. the no-reasoning baseline (one prefill; it also leaves the history in the radix cache for the draws). An
        #    action already near-certain after an empty block cannot be made likelier: no draws (``--skip-certain-nll``)
        baseline, _, _, _ = self.score(context, [self.close], calls, t)
        if a.skip_certain_nll and -baseline[0] < a.skip_certain_nll:
            return {'kind': 'certain', 'baseline': baseline, 'room': room, 'candidates': [], 'generated_tokens': 0}
        hint_ids, hinted = self.hint(calls, t)
        # 2. the anchor: a plain student sample with the whole room; its length L is the student's own at this step
        #    and sets the band. An unclosed anchor (cut at the room) is no length: draw one more
        candidates, anchor = [], None
        for _ in range(max(1, a.unhinted)):
            got = self._candidates([('unhinted', context, room)], room, rng)[0]
            got['anchor'] = True
            candidates.append(got)
            if got['closed']:
                anchor = got
                break
        if anchor is None:
            return self._result(candidates, None, None, baseline, room)
        self._evaluate(context, calls, t, [anchor], baseline, hinted, context_text)
        if anchor['admitted']:
            return self._result(candidates, anchor, anchor['length'], baseline, room)
        lo, hi = a.band[0] * anchor['length'], a.band[1] * anchor['length']
        cap = min(room, int(math.ceil(hi)) + 1)   # a block still open past the band cannot be chosen: stop it there
        # 3. waves of the remaining budget, band-capped, until one admitted candidate lies in the band. After
        #    ``--trend-window`` fallbacks in a row the trajectory's steps draw only the anchor (its teacher actions
        #    are far from what the student reasons toward), until a step fills again
        if a.trend_window and trend >= a.trend_window:
            return self._result(candidates, None, anchor['length'], baseline, room, trend_skip=True)
        unhinted_left = max(0, a.unhinted - sum(c['origin'] == 'unhinted' for c in candidates))
        waves = [[('unhinted', context, cap)] * unhinted_left + [('hinted', None, cap)] * min(1, a.hinted),
                 [('hinted', None, cap)] * max(0, a.hinted - 1)]
        hinted_context = None
        for wave in waves:
            if not wave:
                continue
            if any(origin == 'hinted' for origin, _, _ in wave) and hinted_context is None:
                hinted_context = hint_context(context, hint_ids, self.im_start)
                self.prefill(hinted_context)   # once, before its parallel draws
            jobs = [(origin, ctx if origin == 'unhinted' else hinted_context, c) for origin, ctx, c in wave]
            drawn = self._candidates(jobs, room, rng)
            candidates += drawn
            self._evaluate(context, calls, t, drawn, baseline, hinted, context_text)
            inside = [c for c in drawn if c['admitted'] and lo <= c['length'] <= hi]
            if inside:
                return self._result(candidates, max(inside, key=lambda c: c['improvement']), anchor['length'],
                                    baseline, room)
        return self._result(candidates, None, anchor['length'], baseline, room)

    def _step_parallel(self, context, calls, t, context_text, rng, room):
        """Every candidate drawn at once with the whole room, then chosen (``--draw parallel``)."""
        a = self.args
        hint_ids, hinted = self.hint(calls, t)
        hinted_context = hint_context(context, hint_ids, self.im_start)
        # shared prefixes once (LFM3's conv state is cached at track points: concurrent copies would each prefill)
        self.prefill(context)
        self.prefill(hinted_context)
        jobs = [('unhinted', context, room)] * a.unhinted + [('hinted', hinted_context, room)] * a.hinted
        with ThreadPoolExecutor(2) as pool:
            base_future = pool.submit(self.score, context, [self.close], calls, t)
            candidates = self._candidates(jobs, room, rng)
            baseline, _, _, _ = base_future.result()
        self._evaluate(context, calls, t, candidates, baseline, hinted, context_text)
        chosen, anchor = select(candidates, a.band, rng)
        return self._result(candidates, chosen, anchor, baseline, room)

    def episode(self, item, renderer):
        started = time.time()
        rng = random.Random(f"{self.args.seed}:{item['key']}")
        chains, _ = renderer.chains(item)
        policy_start = self.actor.policy()
        directory = getattr(self.args, 'partial_dir', None)
        current = json.loads(json.dumps(settings(self.args)))
        saved = load_partial(directory, item['key'], current)
        out_chains, resumed = [], 0
        for chain in chains:
            calls = calls_of(chain, self.close)
            digest = chain_digest(chain)
            done = saved.get(digest, [])[:len(calls)]
            context = [int(i) for i in calls[0]['prompt']]
            context_text = self.text(context, special=True)
            fills = []
            trend = 0   # consecutive drawn steps of this trajectory that fell back
            for t, call in enumerate(calls):
                if t < len(done):     # finished before a stop: its block is the history the later steps reason on
                    fill = done[t]
                    resumed += 1
                else:
                    fill = self.step(context, calls, t, context_text, rng, trend)
                    fill['policy'] = (self.actor.policy() or {}).get('checkpoint_sha256')
                fills.append(fill)
                if fill.get('kind') == 'filled':
                    trend = 0
                elif fill.get('kind') == 'fallback':
                    trend += 1
                block = fill.get('reasoning') or ([self.close] if call['cut'] is not None else [])
                new = block + [int(i) for i in call['action']] if call['cut'] is not None else \
                    [int(i) for i in call['sample']]
                new += [int(i) for i in call['suffix']]
                context += new
                context_text += self.text(new, special=True)
                if directory is not None and t >= len(done):
                    save_partial(directory, item['key'], current, out_chains + [{'digest': digest, 'calls': fills}])
            out_chains.append({'digest': digest, 'calls': fills})
        return {'schema': SCHEMA, 'key': item['key'], 'work': item['work'], 'name': item['name'],
                'task_id': item['task_id'], 'teacher': item['teacher'], 'render_tokenizer': renderer.tokenizer_dir,
                'policy_start': policy_start,
                'policy_end': self.actor.policy(), 'seconds': round(time.time() - started, 1), 'finished': time.time(),
                'resumed_steps': resumed,
                'settings': settings(self.args), 'chains': out_chains}


def load_rates(paths):
    """{task id: (successes, scored)} from rate files ({id: {successes, scored}}); later files win."""
    rates = {}
    for path in paths:
        for task_id, v in json.loads(Path(path).read_text()).items():
            if v.get('scored'):
                rates[task_id] = (v['successes'], v['scored'])
    return rates


def prefer(episodes, rates):
    """Episodes of tasks the student fails first (measured 0), then unmeasured tasks, then by pass rate (stable)."""
    def rank(e):
        s, n = rates.get(e['task_id'], (None, 0))
        if not n:
            return (1, 0.)
        return (0, 0.) if s == 0 else (2, s / n)
    return sorted(episodes, key=rank)


def settings(args):
    keys = ('unhinted', 'hinted', 'hint_horizon', 'score_horizon', 'max_tokens', 'band', 'close_floor', 'min_improvement',
            'ratio_floor', 'top_p', 'max_len', 'draw', 'skip_certain_nll', 'trend_window')
    return {k: getattr(args, k, None) for k in keys}


def partial_path(directory, key):
    return Path(directory) / (hashlib.sha1(key.encode()).hexdigest()[:24] + '.json')


def load_partial(directory, key, settings_now):
    """{chain digest: finished step fills} of an episode's partial file, if it was drawn under the same settings."""
    if directory is None:
        return {}
    try:
        data = json.loads(partial_path(directory, key).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if data.get('key') != key or data.get('settings') != settings_now:
        return {}
    return {c['digest']: c['calls'] for c in data.get('chains', ())}


def save_partial(directory, key, settings_now, chains):
    path = partial_path(directory, key)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'key': key, 'settings': settings_now, 'chains': chains}))
    os.replace(tmp, path)


def fill_policies(record):
    """The policy shas a fills record was drawn by (episode start/end and every step's)."""
    shas = {(record.get(k) or {}).get('checkpoint_sha256') for k in ('policy_start', 'policy_end')}
    shas |= {f.get('policy') for c in record.get('chains', ()) for f in c.get('calls', ())}
    return {s for s in shas if s}


def done_keys(out):
    path = Path(out) / 'fills.jsonl'
    keys = set()
    if path.exists():
        for line in path.open():
            if line.strip():
                try:
                    keys.add(json.loads(line)['key'])
                except json.JSONDecodeError:   # a line cut by a kill: that episode reruns
                    pass
    return keys


def generate(args):
    from tokenizers import Tokenizer
    from natlang_neuralese.common.code_provenance import code_provenance
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    works = args.work or default_works()
    held = args.held_manifest or data_registry.role_path('held-eval')
    data_registry.consume([*works, held], run=f'filled-reasoning:{out}', how='teacher episodes to fill')
    export = Path(args.export)
    tokenizer = Tokenizer.from_file(str(export / 'tokenizer.json'))
    # the rendering tokenizer must be the served one (ids pass through unchanged)
    if sha256_file(Path(args.render_tokenizer) / 'tokenizer.json') != sha256_file(export / 'tokenizer.json'):
        raise SystemExit(f'{args.render_tokenizer}: rendering tokenizer differs from the served one ({export})')
    manifest = export / 'natlang-export.json'
    banned = (json.loads(manifest.read_text()).get('serving') or {}).get('banned_token_ids', []) if manifest.exists() else []
    stops = [i for i in (tokenizer.token_to_id('<|im_end|>'), tokenizer.token_to_id('<|endoftext|>')) if i is not None]
    funnel = collections.defaultdict(collections.Counter)
    episodes = teacher_episodes(works, held, funnel)
    done = done_keys(out)
    todo = [e for e in episodes if e['key'] not in done]
    random.Random(args.seed).shuffle(todo)
    if args.prefer_rates:
        todo = prefer(todo, load_rates(args.prefer_rates))
    if args.limit:
        todo = todo[:args.limit]
    args.partial_dir = None if args.no_partial else out / 'partial'
    if args.partial_dir is not None:
        args.partial_dir.mkdir(exist_ok=True)
    actor = Actor(args.sglang, args.front, args.polite, args.max_inflight, args.min_inflight, args.max_running)
    filler = Filler(args, actor, tokenizer, stops, banned)
    renderer = Renderer(args.render_tokenizer)
    run = {'event': 'start', 'schema': SCHEMA, 'code': code_provenance(REPO), 'argv': sys.argv, 'works': works,
           'held': held, 'episodes': len(episodes), 'done': len(done), 'todo': len(todo), 'export': str(export),
           'policy': actor.policy(), 'funnel': {k: dict(v) for k, v in funnel.items()}, 'time': time.time()}
    log = (out / 'generate.log').open('a')
    log.write(json.dumps(run) + '\n')
    log.flush()
    record_fills(out, works, held)   # registered from the start: a cycle stops generation without a clean exit
    print(json.dumps({k: run[k] for k in ('episodes', 'done', 'todo', 'policy')}), flush=True)
    write_lock = threading.Lock()
    fills = (out / 'fills.jsonl').open('a')
    failures = collections.Counter()

    def one(item):
        try:
            record = filler.episode(item, renderer)
        except Exception as error:
            failures['episodes'] += 1
            with write_lock:
                log.write(json.dumps({'event': 'episode failed', 'key': item['key'], 'error': repr(error)[:400],
                                      'time': time.time()}) + '\n')
                log.flush()
            if failures['episodes'] > args.max_failures:
                os._exit(3)
            return
        with write_lock:
            fills.write(json.dumps(record) + '\n')
            fills.flush()
        if args.partial_dir is not None:
            partial_path(args.partial_dir, item['key']).unlink(missing_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, todo))
    log.write(json.dumps({'event': 'end', 'failures': dict(failures), 'time': time.time()}) + '\n')
    log.close()
    fills.close()
    record_fills(out, works, held)


def record_fills(out, works, held):
    data_registry.record_dataset(data_registry.dataset_id(out, 'filled-reasoning'), out, kind='trajectories',
                                 include=['fills.jsonl', 'generate.log'], inputs=[*works, held],
                                 format=f'{SCHEMA}: fills.jsonl (per verified teacher episode: the student\'s filled '
                                        'think block per call and every candidate\'s statistics; filled_reasoning.py)',
                                 license='derived: teacher providers\' licenses (registered with the WORKs)',
                                 producer={'script': 'scripts/neuralese_data/filled_reasoning.py'}, closed=False)


# ---- status, build and report ----------------------------------------------------------------------------------------
def read_fills(out):
    for line in (Path(out) / 'fills.jsonl').open():
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def quantiles(values, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    values = sorted(v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v)))
    if not values:
        return None
    return {f'p{int(q * 100)}': values[min(len(values) - 1, int(q * len(values)))] for q in qs} | {
        'mean': round(sum(values) / len(values), 4), 'n': len(values)}


def report_of(records):
    """Lengths, ``</think>`` hazard, leakage, admission and score of the candidates per origin and of the selected
    blocks against the student's own (unhinted) samples; the teacher action's per-token NLL without reasoning and after
    the filled block; throughput."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    kinds, reasons, steps = collections.Counter(), collections.Counter(), 0
    nll_empty, nll_filled, generated, seconds = [], [], 0, 0.
    for record in records:
        seconds += record.get('seconds', 0.)
        for chain in record['chains']:
            for fill in chain['calls']:
                kinds[fill['kind']] += 1
                steps += 1
                generated += fill.get('generated_tokens', 0)
                for c in fill.get('candidates', ()):
                    groups = [c['origin']] + (['selected', f"selected_{c['origin']}"] if c.get('selected') else [])
                    for g in groups:
                        b = by[g]
                        b['length'].append(c['length'])
                        b['closed'].append(float(c['closed']))
                        if c['closed']:
                            b['improvement'].append(c.get('improvement'))
                            b['hazard'].append(c.get('hazard'))
                            b['leak'].append(float(bool(c.get('leak'))))
                            b['admitted'].append(float(c['admitted']))
                            b['action_logp'].append(c.get('action_logp'))
                            if 'log_ratio' in c:
                                b['log_ratio'].append(c['log_ratio'])
                    if not c.get('selected'):
                        for r in c.get('rejected', ()):
                            reasons[f"{c['origin']}:{r}"] += 1
                if fill.get('baseline'):
                    nll_empty.append(-fill['baseline'][0])
                    chosen = [c for c in fill.get('candidates', ()) if c.get('selected')]
                    if chosen:
                        nll_filled.append(-chosen[0]['action_logp'])
    groups = {}
    for g, b in by.items():
        groups[g] = {k: quantiles(v) if k in ('length', 'improvement', 'hazard', 'action_logp', 'log_ratio')
                     else round(sum(v) / len(v), 4) if v else None for k, v in b.items()}
    filled = kinds['filled']
    return {'episodes': len(records) if isinstance(records, list) else None, 'steps': steps, 'kinds': dict(kinds),
            'filled_share': round(filled / steps, 4) if steps else None, 'candidates': groups,
            'rejections': dict(reasons.most_common()),
            'action_nll_per_token': {'empty_think_all_steps': quantiles(nll_empty),
                                     'filled_steps_after_block': quantiles(nll_filled)},
            'throughput': {'generated_tokens': generated, 'episode_seconds_sum': round(seconds),
                           'generated_tokens_per_filled_step': round(generated / filled) if filled else None}}


def status(args):
    records = list(read_fills(args.out))
    report = report_of(records)
    log = [json.loads(l) for l in (Path(args.out) / 'generate.log').open() if l.strip()]
    starts = [r for r in log if r.get('event') == 'start']
    if records and starts:   # the latest generate run: episodes and steps it finished per hour, the projection
        t0 = starts[-1]['time']
        mine = [r for r in records if r.get('finished', 0) >= t0]
        elapsed = max(1., max([r['finished'] for r in mine], default=t0) - t0)
        steps = sum(len(c['calls']) for r in mine for c in r['chains'])
        per_hour = len(mine) * 3600 / elapsed
        report['rate'] = {'run_hours': round(elapsed / 3600, 2), 'episodes': len(mine),
                          'episodes_per_hour': round(per_hour, 1), 'steps_per_hour': round(steps * 3600 / elapsed, 1),
                          'remaining_episodes': starts[-1]['todo'] - len(mine),
                          'projected_hours': round((starts[-1]['todo'] - len(mine)) / per_hour, 1) if mine else None}
    print(json.dumps(report, indent=1))


LENGTH_BINS = (0, 64, 128, 256, 512, 1024, 2048)


def length_bin(n):
    return max(i for i, edge in enumerate(LENGTH_BINS) if n >= edge)


def length_weights(records, cap):
    """Per length bin: the student's own share (closed unhinted candidates) over the selected blocks' share, capped
    at ``cap`` (None: no reweighting). The improvement gate admits long blocks less often (sft-v7's first 30 episodes:
    16% admitted at 512-1024 tokens vs 62% at 64-128; selected p50 191 tokens vs the student's 308), so unweighted the
    filled set would teach shorter reasoning than the student's own, the sft-v3 failure mode."""
    if not cap:
        return None
    own, chosen = collections.Counter(), collections.Counter()
    for record in records:
        for chain in record['chains']:
            for fill in chain['calls']:
                cands = fill.get('candidates', ())
                staged = any('anchor' in c for c in cands)   # staged draws: only anchors drew with the whole room
                for c in cands:
                    if c['origin'] == 'unhinted' and c['closed'] and (c.get('anchor') or not staged):
                        own[length_bin(c['length'])] += 1
                    if c.get('selected'):
                        chosen[length_bin(c['length'])] += 1
    n_own, n_chosen = sum(own.values()), sum(chosen.values())
    if not n_own or not n_chosen:
        return None
    return {b: min(cap, (own[b] / n_own) / (chosen[b] / n_chosen)) for b in chosen}


def reweight(rows, weights):
    """Scale each row's trained reasoning tokens (``sampled``) by its block length's weight (``length_weights``)."""
    for row in rows:
        n = int(np.asarray(row['sampled']).sum())
        if n and weights:
            row['weight'] = np.where(row['sampled'], row['weight'] * weights.get(length_bin(n), 1.), row['weight'])


def build(args):
    import pickle
    from exp0_sft import chains_of
    from natlang_neuralese.common.code_provenance import code_provenance
    corpus = Path(args.corpus)
    corpus.mkdir(parents=True, exist_ok=True)
    starts = [json.loads(l) for l in (Path(args.out) / 'generate.log').open() if '"event": "start"' in l]
    record_fills(args.out, sorted({w for r in starts for w in r['works']}), starts[-1]['held'])   # the fills as built
    records = list(read_fills(args.out))
    excluded = set(args.exclude_policy)
    weights = length_weights(records, args.length_reweight_cap)
    renderers = {}
    chains, counts = [], collections.Counter()
    close = None
    for record in records:
        if fill_policies(record) & excluded:   # drawn by a policy the cycle rejected as failed (expert_iteration gate)
            counts['excluded_policy'] += 1
            continue
        item = {k: record[k] for k in ('key', 'work', 'name', 'task_id')}
        try:
            if record['render_tokenizer'] not in renderers:
                renderers[record['render_tokenizer']] = Renderer(record['render_tokenizer'])
            source, r = renderers[record['render_tokenizer']].chains(item)
        except Exception as error:
            counts['render_error'] += 1
            print(json.dumps({'event': 'render error', 'key': record['key'], 'error': repr(error)[:300]}), flush=True)
            continue
        if close is None:
            close = r.tokenizer.convert_tokens_to_ids('</think>')
        if [chain_digest(c) for c in source] != [c['digest'] for c in record['chains']]:
            counts['digest_mismatch'] += 1
            continue
        rows = []
        name = f"filled:{record['key']}"
        for chain, filled in zip(source, record['chains']):
            rows += filled_rows(calls_of(chain, close), filled['calls'], close, name)
        reweight(rows, weights)
        built = chains_of([{'task_id': record['task_id'], 'round': name, 'rows': rows}])
        for chain in built:
            chain['policy'] = record.get('policy_start')
            for _, _, row in chain['segs']:
                row.pop('prompt', None)
        chains += built
        counts['episodes'] += 1
        counts['rows'] += len(rows)
    report = report_of(records)
    report.update({'length_weights': weights, 'schema': SCHEMA, 'code': code_provenance(REPO), 'fills': str(args.out), 'counts': dict(counts),
                   'chains': len(chains), 'forward_tokens': int(sum(len(c['ids']) for c in chains)),
                   'weighted_target_tokens': int(sum(float(row['weight'].sum()) for c in chains
                                                     for _, _, row in c['segs'])),
                   'policies': dict(collections.Counter(json.dumps(r.get('policy_start'), sort_keys=True)
                                                        for r in records))})
    with (corpus / 'chains.pkl.partial').open('wb') as handle:
        pickle.dump({'chains': chains, 'schema': 'natlang.sft-chains/1'}, handle, protocol=4)
    os.replace(corpus / 'chains.pkl.partial', corpus / 'chains.pkl')
    (corpus / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    data_registry.record_dataset(data_registry.dataset_id(corpus, 'sft-rows'), corpus, kind='sft-rows',
                                 include=['chains.pkl', 'report.json'], inputs=[str(args.out)],
                                 format='natlang.sft-chains/1: chains.pkl (exp0_sft --chains; sft_sources build '
                                        '--aux filled=PATH,share=S,replaces=teacher), report.json',
                                 license='derived: teacher providers\' licenses (registered with the WORKs)',
                                 producer={'script': 'scripts/neuralese_data/filled_reasoning.py'})
    print(json.dumps({k: report[k] for k in ('counts', 'chains', 'forward_tokens', 'weighted_target_tokens',
                                             'filled_share')}), flush=True)



# ---- teacher reasoning audit (per model; owner 2026-11-20) -----------------------------------------------------------
TOOL_SYNTAX = re.compile(r'<tool_call>|<function=|<parameter=|"name"\s*:\s*"(?:eval|return_result)"|return_result\s*\(\s*\{')
TREATMENT_SCHEMA = 'natlang.teacher-think-treatment/1'


def action_text(target):
    """A reply's action as text: its content and its tool calls' names and arguments."""
    parts = [target.get('content') or '']
    for call in target.get('tool_calls') or ():
        function = call.get('function') or {}
        parts.append(f"{function.get('name', '')} {function.get('arguments', '')}")
    return '\n'.join(p for p in parts if p)


def static_call(target):
    """(presence, characters, tool syntax inside think, the action quoted in think) of one reply."""
    reasoning = target.get('reasoning_content')
    if reasoning is None:
        return 'none', 0, False, False
    if not reasoning.strip():
        return 'empty', 0, False, False
    return ('present', len(reasoning), bool(TOOL_SYNTAX.search(reasoning)),
            bool(leak(reasoning, [action_text(target)], '')))


def choose_treatment(row, student, gates):
    """'train' | 'context' | 'filled' for one model's audit row against the student's own (``gates``): the length gate
    first (sft-v3's -16 pts came from training short teacher blocks), then presence, format, in-distribution NLL and
    whether the actions are likelier after the teacher's reasoning than after none."""
    reasons = []
    present = row['presence'].get('present', 0.)
    if present < gates['min_present']:
        reasons.append(f'present {present:.2f} < {gates["min_present"]}')
        return 'filled', reasons
    chars = (row['chars'] or {}).get('p50') or 0
    floor = gates['min_length_ratio'] * student['chars']['p50']
    if chars < floor:
        reasons.append(f'reasoning p50 {chars} chars < {gates["min_length_ratio"]} x student {student["chars"]["p50"]}')
        return 'filled', reasons
    if row['tool_syntax'] > gates['max_tool_syntax']:
        reasons.append(f'tool syntax in think {row["tool_syntax"]:.2f}')
    nll, own = (row.get('reasoning_nll') or {}).get('mean'), (student.get('reasoning_nll') or {}).get('mean')
    if nll is None or own is None or nll > gates['max_nll_ratio'] * own:
        reasons.append(f'reasoning NLL {nll} vs student {own} (x{gates["max_nll_ratio"]})')
    after, empty = (row.get('action_nll_after') or {}).get('mean'), (row.get('action_nll_empty') or {}).get('mean')
    if after is None or empty is None or after > empty:
        reasons.append(f'action NLL after teacher reasoning {after} vs empty {empty}')
    return ('train' if not reasons else 'filled'), reasons


def audit(args):
    """Per teacher model: episodes, reasoning presence, length against the student's own, tool syntax and the action
    quoted inside think (all verified episodes), and on a sample of calls through the served student (prefill only):
    the ``</think>`` position, the student's per-token NLL on the teacher's reasoning, and the action's NLL after the
    teacher's reasoning vs after an empty block. Writes OUT/audit.json and the treatment table OUT/treatment.json
    (``sft_sources.ThinkTreatment``; register it with ``scripts/data_registry.py promote ID --role
    teacher-think-treatment``)."""
    from tokenizers import Tokenizer
    from natlang_neuralese.common.code_provenance import code_provenance
    from sft_sources import call_row, replayed_turns, verified_episodes
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    works = args.work or default_works()
    data_registry.consume(works, run=f'teacher-reasoning-audit:{out}', how='per-model reasoning audit')
    funnel = collections.defaultdict(collections.Counter)
    stats = collections.defaultdict(lambda: {'episodes': 0, 'calls': 0, 'presence': collections.Counter(), 'chars': [],
                                             'tool_syntax': 0, 'quoted_action': 0, 'providers': collections.Counter(),
                                             'samples': []})
    seen = set()
    rng = random.Random(args.seed)
    for work in works:
        for episode, record in verified_episodes(work, funnel, sources={'teacher', 'rl-own'}):
            if episode['key'] in seen:
                continue
            seen.add(episode['key'])
            teacher = episode.get('teacher') or {}
            model = 'student' if episode['source'] == 'rl-own' else (teacher.get('model_id') or 'unknown')
            row = stats[model]
            row['episodes'] += 1
            row['providers'][teacher.get('provider') or ('student' if model == 'student' else 'unknown')] += 1
            for index, turn in enumerate(episode.get('turns') or ()):
                presence, chars, syntax, quoted = static_call(turn.get('target') or {})
                row['calls'] += 1
                row['presence'][presence] += 1
                if presence == 'present':
                    row['chars'].append(chars)
                    row['tool_syntax'] += syntax
                    row['quoted_action'] += quoted
                    # reservoir of calls with reasoning for the NLL sample
                    item = (work, record.get('name'), episode['key'], episode['source'], index)
                    if record.get('name'):
                        n = len(row['chars'])
                        if len(row['samples']) < args.per_model:
                            row['samples'].append(item)
                        elif rng.random() < args.per_model / n:
                            row['samples'][rng.randrange(args.per_model)] = item
    export = Path(args.export)
    tokenizer = Tokenizer.from_file(str(export / 'tokenizer.json'))
    close = tokenizer.token_to_id('</think>')
    actor = Actor(args.sglang, None, True, args.max_inflight, 1, args.max_running)
    renderer = Renderer(args.render_tokenizer)

    def measure(item):
        try:
            return measure_one(item)
        except Exception as error:   # a render error or a call the replay no longer has: not measured
            print(json.dumps({'event': 'audit measure failed', 'key': item[2], 'error': repr(error)[:200]}), flush=True)
            return None

    def measure_one(item):
        work, name, key, source, index = item
        turn = replayed_turns(work, name)[index]
        with renderer.lock:
            row, _ = call_row(renderer.renderer, source, key, index, turn, 'train', 'train')
        prompt, sample = [int(i) for i in row['prompt']], [int(i) for i in row['sample']]
        if close not in sample or len(prompt) + len(sample) > args.max_len:
            return None
        cut = sample.index(close) + 1
        actor.acquire()
        try:
            full = actor.post('/generate', {'input_ids': prompt + sample, 'return_logprob': True,
                                            'logprob_start_len': len(prompt) - 1,
                                            'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1}})
            empty = actor.post('/generate', {'input_ids': prompt + [close] + sample[cut:], 'return_logprob': True,
                                             'logprob_start_len': len(prompt),
                                             'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1}})
        finally:
            actor.release()
        v = [x[0] for x in full['meta_info']['input_token_logprobs']][-len(sample):]
        e = [x[0] for x in empty['meta_info']['input_token_logprobs']][-(len(sample) - cut):]
        think, act = v[:cut], v[cut:]
        return {'think_tokens': cut, 'reasoning_nll': -float(np.mean(think)), 'action_nll_after': -float(np.mean(act)),
                'action_nll_empty': -float(np.mean(e))}
    table = {}
    for model, row in sorted(stats.items(), key=lambda kv: -kv[1]['episodes']):
        with ThreadPoolExecutor(args.max_inflight) as pool:
            measured = [m for m in pool.map(measure, row['samples']) if m]
        calls = max(1, row['calls'])
        present = max(1, row['presence']['present'])
        table[model] = {'episodes': row['episodes'], 'calls': row['calls'], 'providers': dict(row['providers']),
                        'presence': {k: round(v / calls, 4) for k, v in row['presence'].items()},
                        'chars': quantiles(row['chars']), 'tool_syntax': round(row['tool_syntax'] / present, 4),
                        'quoted_action': round(row['quoted_action'] / present, 4), 'measured_calls': len(measured),
                        **{k: quantiles([m[k] for m in measured]) for k in ('think_tokens', 'reasoning_nll',
                                                                            'action_nll_after', 'action_nll_empty')}}
        print(json.dumps({'model': model, **{k: table[model][k] for k in ('episodes', 'presence', 'measured_calls')}}),
              flush=True)
    gates = {'min_present': args.min_present, 'min_length_ratio': args.min_length_ratio,
             'max_tool_syntax': args.max_tool_syntax, 'max_nll_ratio': args.max_nll_ratio}
    student = table.get('student')
    treatment = {'schema': TREATMENT_SCHEMA, 'default': 'context', 'models': {}}
    for model, row in table.items():
        if model == 'student' or student is None:
            continue
        row['treatment'], row['reasons'] = choose_treatment(row, student, gates)
        if model != 'unknown':
            treatment['models'][model] = row['treatment']
    report = {'schema': 'natlang.teacher-reasoning-audit/1', 'code': code_provenance(REPO), 'works': works,
              'policy': Actor(args.sglang, args.front, False, 1, 1, 1).policy(), 'gates': gates, 'models': table,
              'funnel': {k: dict(v) for k, v in funnel.items()}}
    (out / 'audit.json').write_text(json.dumps(report, indent=1) + '\n')
    (out / 'treatment.json').write_text(json.dumps(treatment, indent=1, sort_keys=True) + '\n')
    data_registry.record_dataset(data_registry.dataset_id(out, 'teacher-think-treatment'), out, kind='index',
                                 include=['audit.json', 'treatment.json'], inputs=works,
                                 format=f'{TREATMENT_SCHEMA}: treatment.json (sft_sources.ThinkTreatment), audit.json',
                                 license='derived', producer={'script': 'scripts/neuralese_data/filled_reasoning.py'})
    print(json.dumps({'treatment': treatment['models']}), flush=True)


def main():
    args = build_parser().parse_args()
    {'generate': generate, 'status': status, 'build': build, 'audit': audit}[args.command](args)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate')
    g.add_argument('--out', required=True)
    g.add_argument('--work', action='append', default=[], help='sft_sources WORKs (default: registry role grpo-sd-teacher)')
    g.add_argument('--held-manifest', default=None)
    g.add_argument('--sglang', required=True, help='SGLang server URL (the actor, not the front)')
    g.add_argument('--front', default=None, help='the actor front, to record the served policy identity')
    g.add_argument('--render-tokenizer', required=True,
                   help='sft_sources tokenizer dir the rows are rendered with (an sft WORK\'s tokenizer/)')
    g.add_argument('--export', required=True, help='the served export dir (tokenizer.json, natlang-export.json)')
    g.add_argument('--polite', action='store_true', help='beside a training run: draw only into spare engine slots')
    g.add_argument('--max-inflight', type=int, default=8)
    g.add_argument('--min-inflight', type=int, default=2)
    g.add_argument('--max-running', type=int, default=96, help='the engine\'s max running requests (polite mode)')
    g.add_argument('--workers', type=int, default=4, help='episodes in parallel')
    g.add_argument('--unhinted', type=int, default=2)
    g.add_argument('--hinted', type=int, default=2)
    g.add_argument('--hint-horizon', type=int, default=3, help='later teacher replies shown in the hint')
    g.add_argument('--hint-response-chars', type=int, default=1500)
    g.add_argument('--hint-max-tokens', type=int, default=6000)
    g.add_argument('--score-horizon', type=int, default=1, help='later teacher actions in the score')
    g.add_argument('--max-tokens', type=int, default=4096, help='reasoning cap per candidate')
    g.add_argument('--max-len', type=int, default=32000)
    g.add_argument('--band', type=float, nargs=2, default=(0.67, 1.5))
    g.add_argument('--draw', choices=('staged', 'parallel'), default='staged',
                   help='staged: the anchor first, then band-capped waves until one admitted candidate is in band; '
                        'parallel: every candidate at once with the whole room')
    g.add_argument('--skip-certain-nll', type=float, default=0.,
                   help='no draws where the teacher action\'s NLL per token after an empty block is below this (default '
                        'off: on sft-v7\'s first fills those steps were the cheapest kept ones)')
    g.add_argument('--trend-window', type=int, default=3,
                   help='after this many fallbacks in a row a trajectory draws only anchors until a step fills (0: off)')
    g.add_argument('--close-floor', type=float, default=0.01)
    g.add_argument('--min-improvement', type=float, default=0.0)
    g.add_argument('--ratio-floor', type=float, default=None)
    g.add_argument('--top-p', type=float, default=0.95)
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--limit', type=int, default=0)
    g.add_argument('--prefer-rates', action='append', default=[],
                   help='task pass rates ({id: {successes, scored}}): episodes of failed tasks first, then unmeasured')
    g.add_argument('--max-failures', type=int, default=200)
    g.add_argument('--no-partial', action='store_true',
                   help='do not keep finished steps of episodes in progress (OUT/partial/): a stop redraws them')
    s = sub.add_parser('status')
    s.add_argument('--out', required=True)
    b = sub.add_parser('build')
    b.add_argument('--out', required=True)
    b.add_argument('--corpus', required=True)
    b.add_argument('--length-reweight-cap', type=float, default=4.,
                   help='reweight trained blocks so their lengths follow the student\'s own (per-bin ratio, capped; 0 opts out)')
    b.add_argument('--exclude-policy', action='append', default=[], metavar='SHA',
                   help='leave out episodes any step of which this policy drew (a held-failed candidate)')
    a = sub.add_parser('audit')
    a.add_argument('--out', required=True)
    a.add_argument('--work', action='append', default=[])
    a.add_argument('--render-tokenizer', required=True)
    a.add_argument('--export', required=True)
    a.add_argument('--sglang', required=True)
    a.add_argument('--front', default=None)
    a.add_argument('--per-model', type=int, default=60, help='calls with reasoning measured through the student')
    a.add_argument('--max-inflight', type=int, default=4)
    a.add_argument('--max-running', type=int, default=96)
    a.add_argument('--max-len', type=int, default=32000)
    a.add_argument('--min-present', type=float, default=0.8)
    a.add_argument('--min-length-ratio', type=float, default=0.5,
                   help='teacher reasoning p50 characters at least this times the student\'s own')
    a.add_argument('--max-tool-syntax', type=float, default=0.1)
    a.add_argument('--max-nll-ratio', type=float, default=1.5)
    a.add_argument('--seed', type=int, default=0)
    return p


if __name__ == '__main__':
    main()
