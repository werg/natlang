#!/usr/bin/env python3
"""Subagent -> parent handoffs of agentic research datasets as Neuralese, in the datasets' own agentic format
(plans/neuralese/SUBAGENT_NEURALESE_20261124.md, the DECOMPOSITION; owner 2026-11-24: "It could just be a source of
diversity in our dataset that would still train neuralese").

A source bundle is one lead agent's conversation plus the subagent conversations it dispatched. Every conversation stays
as recorded (its system prompt, messages, tools and tool calls), rendered by the serving engine's chat template for the
student (``serve.targets.target_items``). Only the handoff changes: each subagent report the lead receives in a tool
message becomes a Neuralese value written from the report's text and read by the lead.

* ``--read-mode splice`` (default): the learner's producer/consumer pass-through. The lead's prompt carries
  ``consume_open``, the report's written token ids as placeholder rows, ``consume_close``, and an import at those rows;
  the producer document is the text write ``[produce_open] + tokens`` (``lower_rollout_turns.write_sources``,
  ``sglang_actor._sglang_write_text``), whose state the learner computes in the same forward, so the lead's loss trains
  the producer computation and the marker rows (plans/neuralese/NEURALESE_PASSTHROUGH_20261117.md).
* ``--read-mode inline``: the report's written token ids inline with no markers, the engines' read of a text write
  since 2026-11-23 (``engine.reads_in_context``).

Targets: every assistant reply of the lead and of each recorded subagent (actions, answers, reports); teacher reasoning
and lead-in text are zero-weight context (``sft_sources.think_context``), as are prompts, tool outputs and the spans.
Admission: the source's own verification (WideSeek ``item_f1``; SearchSwarm's gold answer contained in the lead's final
answer), the held closure by id/group, and a content check of each question against every held set's inputs.

    subagent_neuralese.py build --source wideseek|searchswarm --input PATH --tokenizer TOK --out OUT
    subagent_neuralese.py scope --source wideseek|searchswarm|evidence --input PATH --out FILE.json
"""
from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures
import hashlib
import json
import os
import pickle
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO), str(Path(__file__).resolve().parent)]

SCHEMA = 'natlang.sft-sources-subagent-neuralese/1'
READ_MODES = ('splice', 'inline')
LICENSES = {'wideseek': 'apache-2.0 (WideSeek-R1/WideSeek-R1-SFT-data 44c3e177)',
            'searchswarm': 'mit (SearchSwarm/SearchSwarm-SFT 4bad3397)'}


# ---- reading the sources ---------------------------------------------------------------------------------------------
THINK = re.compile(r'\s*<think>(.*?)</think>\s*', re.S)
TOOL_CALL = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.S)


def split_reply(content):
    """(reasoning, text, [tool calls]) of a recorded Qwen-format assistant reply; None when a call does not parse."""
    reasoning, rest = '', content or ''
    match = THINK.match(rest)
    if match:
        reasoning, rest = match.group(1).strip(), rest[match.end():]
    elif '</think>' in rest:
        reasoning, rest = rest.split('</think>', 1)
        reasoning = reasoning.replace('<think>', '').strip()
    calls = []
    for raw in TOOL_CALL.findall(rest):
        try:
            call = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(call, dict) or not isinstance(call.get('name'), str) or not isinstance(call.get('arguments'), dict):
            return None
        calls.append(call)
    return reasoning, TOOL_CALL.sub('', rest).strip(), calls


def chat_messages(messages):
    """Recorded messages as structured chat messages (reasoning_content, tool_calls); None when a reply does not parse.
    A tool message keeps its text; content parts are added by ``handoff``."""
    out = []
    for message in messages:
        if message['role'] != 'assistant':
            out.append({'role': message['role'], 'content': message['content']})
            continue
        parsed = split_reply(message['content'])
        if parsed is None:
            return None
        reasoning, text, calls = parsed
        reply = {'role': 'assistant', 'content': text}
        if reasoning:
            reply['reasoning_content'] = reasoning
        if calls:
            reply['tool_calls'] = [{'id': f'call_{len(out)}_{k}', 'type': 'function',
                                    'function': {'name': c['name'], 'arguments': c['arguments']}}
                                   for k, c in enumerate(calls)]
        out.append(reply)
    return out


SUBTASK = re.compile(r'^# Subtask (\d+):\n', re.M)


def wideseek_handoffs(text):
    """[(brief, report start, report stop)] of a planner's create_sub_agents tool message, in order; None if unparsed."""
    starts = list(SUBTASK.finditer(text))
    note = re.search(r'\n*Your next answer will be on turn \d+\.', text)
    end = note.start() if note else len(text)
    out = []
    for i, start in enumerate(starts):
        stop = starts[i + 1].start() if i + 1 < len(starts) else end
        body = text[start.end():stop]
        if '\n# Result:\n' not in body:
            return None
        brief, report = body.split('\n# Result:\n', 1)
        lo = start.end() + len(brief) + len('\n# Result:\n')
        hi = lo + len(report.rstrip())
        out.append((brief.strip(), lo, hi))
    return out or None


def wideseek_bundles(path):
    """WideSeek-R1 sessions grouped by trajectory (the file keeps a planner and its workers together)."""
    from sft_sources import zstd_lines
    group, current = [], None
    for line in zstd_lines(path):
        if not line.strip():
            continue
        row = json.loads(line)
        trajectory = row['metadata']['trajectory_id']
        if trajectory != current and group:
            yield group
            group = []
        current = trajectory
        group.append(row)
    if group:
        yield group


def wideseek_bundle(sessions, stats, min_width_f1):
    """One WideSeek trajectory: {'key', 'question', 'verified', 'lead': conversation, 'subagents': [...]}, or None."""
    planners = [s for s in sessions if s['metadata']['role'] == 'planner']
    if len(planners) != 1:
        stats['no_single_lead'] += 1
        return None
    planner, meta = planners[0], planners[0]['metadata']
    f1 = meta.get('item_f1')
    verified = f1 is not None and (f1 >= 1.0 if meta['dataset_type'] == 'depth' else f1 >= min_width_f1)
    question = re.sub(r'\n+This is your first turn.*\Z', '', planner['messages'][1]['content'], flags=re.S)
    question = re.sub(r'\A# Task\nYour task is:\n', '', question).strip()
    messages = chat_messages(planner['messages'])
    if messages is None:
        stats['call_not_json'] += 1
        return None
    handoffs = []      # (message index, report start, report stop, brief)
    for i, message in enumerate(messages):
        if message['role'] == 'tool' and i > 0 and any(c['function']['name'] == 'create_sub_agents'
                                                       for c in messages[i - 1].get('tool_calls') or ()):
            found = wideseek_handoffs(message['content'])
            briefs = [a.get('prompt', '').strip() for c in messages[i - 1]['tool_calls']
                      for a in c['function']['arguments'].get('sub_agents') or ()]
            if found is None or [b for b, _, _ in found] != briefs:
                stats['handoff_unparsed'] += 1
                return None
            handoffs += [(i, lo, hi, brief) for brief, lo, hi in found]
    subagents = []
    for worker in (s for s in sessions if s['metadata']['role'] == 'worker'):
        conversation = chat_messages(worker['messages'])
        if conversation is None:
            stats['subagent_call_not_json'] += 1
            continue
        subagents.append({'session': worker['metadata']['session_id'], 'messages': conversation,
                          'tools': json.loads(worker['tools'])})
    return {'key': f"wideseek:{meta['trajectory_id']}", 'source': 'wideseek', 'question': question,
            'verified': verified, 'verification': {'kind': 'source item_f1', 'item_f1': f1, 'type': meta['dataset_type']},
            'answer_type': meta['dataset_type'],
            'lead': {'messages': messages, 'tools': json.loads(planner['tools']), 'handoffs': handoffs},
            'subagents': subagents,
            'source_ids': [f"wideseek:{meta['dataset_type']}:{meta['source_idx']}"],
            'source_groups': [f"wideseek-r1:{meta['dataset_type']}:{meta['source_idx']}"]}


TOOL_RESPONSE = re.compile(r'<tool_response>\n?(.*?)\n?</tool_response>', re.S)
SWARM_HEAD = re.compile(r'A sub-agent dispatched for goal "(.*?)" returned the following report:\n\n', re.S)
SWARM_SEPARATOR = '\n\n---\n\n'
ANSWER = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.S)


SWARM_TOOLS = re.compile(r'\n*# Tools\n\nYou may call one or more functions to assist with the user query\.\n\n'
                         r'You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n(.*?)\n'
                         r'</tools>\n\nFor each function call, return a json object with function name and arguments '
                         r'within <tool_call></tool_call> XML tags:\n<tool_call>\n.*?\n</tool_call>\s*\Z', re.S)


def swarm_tools(system):
    """(system prompt, tools) with the Qwen tool block the release wrote into the system prompt taken out as structured
    tools, which the student's chat template renders with its own call format; (system, None) when there is none."""
    match = SWARM_TOOLS.search(system)
    if not match:
        return system, None
    try:
        tools = [json.loads(line) for line in match.group(1).splitlines() if line.strip()]
    except ValueError:
        return system, None
    return system[:match.start()], tools


def swarm_messages(messages, with_tools=False):
    """SearchSwarm messages with each ``<tool_response>`` user message split into tool messages (the release folded
    tool results into user turns when it normalized roles), replies structured; None when a reply does not parse.
    ``with_tools``: (messages, tools), the system prompt's tool block as structured tools (``swarm_tools``)."""
    out, tools = [], None
    for message in messages:
        if message['role'] == 'system' and with_tools:
            content, tools = swarm_tools(message['content'])
            out.append({'role': 'system', 'content': content})
            continue
        content = message['content']
        if message['role'] == 'user' and content.lstrip().startswith('<tool_response>'):
            out += [{'role': 'tool', 'content': body} for body in TOOL_RESPONSE.findall(content)]
        else:
            out.append(message)
    structured = chat_messages(out)
    return (structured, tools) if with_tools else structured


def swarm_handoffs(text):
    """[(goal, report start, report stop)] of a call_sub_agent result, in order; None if none."""
    heads = list(SWARM_HEAD.finditer(text))
    out = []
    for i, head in enumerate(heads):
        stop = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        if i + 1 < len(heads) and text[:stop].endswith(SWARM_SEPARATOR):
            stop -= len(SWARM_SEPARATOR)
        body = text[head.end():stop]
        out.append((head.group(1), head.end(), head.end() + len(body.rstrip())))
    return out or None


def qa_norm(text):
    text = (text or '').lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(re.findall(r'\w+', text))


def swarm_answer(messages):
    """The lead's final answer: the last reply's ``<answer>``, else its text."""
    last = messages[-1]
    if last['role'] != 'assistant' or last.get('tool_calls'):
        return None
    found = ANSWER.findall(last.get('content') or '')
    return found[-1] if found else last.get('content')


def searchswarm_rows(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
        yield from batch.to_pylist()


def searchswarm_bundle(row, index, stats):
    messages, lead_tools = swarm_messages(row['messages'], with_tools=True)
    if messages is None:
        stats['call_not_json'] += 1
        return None
    stats['tools_structured' if lead_tools else 'tools_in_system_text'] += 1
    subs = collections.defaultdict(list)
    for k, sub in enumerate(row['subagents'] or ()):
        subs[sub['question'].strip()].append(k)
    handoffs, kept = [], []
    for i, message in enumerate(messages):
        if message['role'] != 'tool' or i == 0:
            continue
        calls = [c for c in messages[i - 1].get('tool_calls') or () if c['function']['name'] == 'call_sub_agent']
        if not calls or not SWARM_HEAD.search(message['content']):
            continue
        prompts = [p for c in calls for p in c['function']['arguments'].get('prompts') or ()]
        found = swarm_handoffs(message['content'])
        if found is None or len(found) != len(prompts):
            stats['handoff_unparsed'] += 1
            return None
        for prompt, (goal, lo, hi) in zip(prompts, found):
            handoffs.append((i, lo, hi, prompt.get('prompt', '')))
            matches = subs.get((prompt.get('prompt') or '').strip()) or []
            if matches:
                kept.append(matches.pop(0))
    answer = swarm_answer(messages)
    gold = row.get('answer')
    verified = bool(answer and gold and qa_norm(gold) and qa_norm(gold) in qa_norm(answer))
    subagents = []
    for k in kept:
        conversation, tools = swarm_messages(row['subagents'][k]['messages'], with_tools=True)
        if conversation is None:
            stats['subagent_call_not_json'] += 1
            continue
        subagents.append({'session': f'{k:03d}', 'messages': conversation, 'tools': tools})
    stats['dispatched'] += len(handoffs)
    stats['dispatched_with_subagent'] += len(kept)
    key = hashlib.sha256(row['question'].encode()).hexdigest()[:16]
    return {'key': f'searchswarm:{row["source"]}:{index}:{key}', 'source': 'searchswarm', 'question': row['question'],
            'verified': verified, 'verification': {'kind': 'gold answer in the final answer', 'gold': gold},
            'answer_type': row['source'],
            'lead': {'messages': messages, 'tools': lead_tools, 'handoffs': handoffs}, 'subagents': subagents,
            'source_ids': [f'searchswarm:{key}'], 'source_groups': [f'searchswarm:{key}']}


def bundles(source, path, stats, args):
    if source == 'wideseek':
        for sessions in wideseek_bundles(path):
            stats['source_bundles'] += 1
            bundle = wideseek_bundle(sessions, stats, args.min_width_f1)
            if bundle is not None:
                yield bundle
        return
    for index, row in enumerate(searchswarm_rows(path)):
        stats['source_bundles'] += 1
        bundle = searchswarm_bundle(row, index, stats)
        if bundle is not None:
            yield bundle


# ---- lowering --------------------------------------------------------------------------------------------------------
def state_id(text):
    """A native-state id for a written report (content-addressed; the renderer needs the id form only)."""
    return 'nzs1_' + base64.b32encode(hashlib.sha256(text.encode()).digest()).decode().lower().rstrip('=')[:52]


class Lowering:
    """The serving engine's rendering (``TemplateRenderer``) plus the Neuralese reads of written reports."""

    def __init__(self, tokenizer, read_mode):
        from sft_sources import TemplateRenderer
        from natlang_neuralese.model.shared_state import CONSUME_CLOSE, CONSUME_OPEN, PRODUCE_OPEN
        self.renderer = TemplateRenderer(tokenizer)
        tok = self.renderer.tokenizer
        self.consume_open, self.consume_close, self.produce_open = (
            int(tok.convert_tokens_to_ids(t)) for t in (CONSUME_OPEN, CONSUME_CLOSE, PRODUCE_OPEN))
        self.read_mode = read_mode

    def written(self, text):
        """The token ids a text write of ``text`` holds (``_sglang_write_text``: escaped template tokens)."""
        from natlang_neuralese.serve.chat import escape_specials
        nonce = hashlib.sha256(text.encode()).hexdigest()
        return [int(i) for i in self.renderer._template_tokens(escape_specials(text, self.renderer.specials, nonce), nonce)]

    def call(self, messages, tools, target, bodies):
        """(prompt int32, sample int32, imports, sources, zero-weight lead) of one recorded reply."""
        import numpy as np
        from natlang_neuralese.serve.targets import target_items
        from sft_sources import think_context
        before, rest = target_items(self.renderer, list(messages), tools, target)
        prompt, imports, sources = [], [], {}
        for kind, value in before:
            if kind == 'tok':
                prompt.append(int(value))
                continue
            block = value['id'] if isinstance(value, dict) else value
            read(prompt, imports, sources, block, self.written(bodies[block]), self.read_mode,
                 (self.consume_open, self.consume_close, self.produce_open))
        if any(kind != 'tok' for kind, _ in rest):
            raise ValueError('a Neuralese read inside a target')
        sample = [int(i) for _, i in rest]
        lead = think_context(self.renderer, 'teacher', target, sample)
        return np.array(prompt, np.int32), np.array(sample, np.int32), imports, sources, lead


def read(prompt, imports, sources, block, ids, mode, markers):
    """Append one Neuralese read of the written ``ids`` to ``prompt``. ``splice``: consume_open, the ids as placeholder
    rows with an import of ``block`` at them, consume_close (``sglang_actor._sglang_import``), and the producer document
    ``[produce_open] + ids`` with span [1, n+1) (``lower_rollout_turns.write_sources``). ``inline``: the ids alone
    (``engine.reads_in_context``)."""
    import numpy as np
    consume_open, consume_close, produce_open = markers
    if mode == 'inline':
        prompt += ids
        return
    prompt.append(consume_open)
    imports.append((len(prompt), block))
    prompt += ids
    prompt.append(consume_close)
    sources[block] = {'ids': np.array([produce_open] + list(ids), np.int32), 'start': 1, 'stop': len(ids) + 1}


def handoff_messages(lead):
    """The lead's messages with each report of a handoff as a native_state part; {state id: report}."""
    messages = [dict(m) for m in lead['messages']]
    bodies = {}
    by_message = collections.defaultdict(list)
    for index, lo, hi, _ in lead['handoffs']:
        by_message[index].append((lo, hi))
    for index, spans in by_message.items():
        text, parts, cursor = messages[index]['content'], [], 0
        for lo, hi in sorted(spans):
            report = text[lo:hi]
            block = state_id(report)
            bodies[block] = report
            parts += [{'type': 'text', 'text': text[cursor:lo]}, {'type': 'native_state', 'id': block}]
            cursor = hi
        parts.append({'type': 'text', 'text': text[cursor:]})
        messages[index] = {**messages[index], 'content': [p for p in parts if p.get('type') != 'text' or p['text']]}
    return messages, bodies


def conversation_rows(lowering, invocation, messages, tools, bodies, stats):
    import numpy as np
    rows = []
    for i, message in enumerate(messages):
        if message['role'] != 'assistant' or i == 0:
            continue
        try:
            prompt, sample, imports, sources, lead = lowering.call(messages[:i], tools, message, bodies)
        except Exception as error:
            stats['render_error'] += 1
            stats['render_error:' + type(error).__name__ + ':' + str(error)[:60]] += 1
            return None
        row = {'invocation': invocation, 'index': i, 'prompt': prompt, 'sample': sample, 'imports': imports,
               'sources': sources}
        if lead:
            row['weight'] = (np.arange(len(sample)) >= lead).astype(np.float32)
        rows.append(row)
    return rows


def lower_bundle(lowering, bundle, stats):
    """The bundle's episode {'task_id', 'round', 'rows'} or None: the lead's calls (consumers) and every recorded
    subagent's calls (the producers of their reports, trained on their own replies)."""
    messages, bodies = handoff_messages(bundle['lead'])
    rows = conversation_rows(lowering, 'lead', messages, bundle['lead']['tools'], bodies, stats)
    if rows is None:
        return None
    for sub in bundle['subagents']:
        more = conversation_rows(lowering, 'sub:' + sub['session'], sub['messages'], sub['tools'], {}, stats)
        if more is None:
            stats['subagent_dropped'] += 1
            continue
        rows += more
    return {'task_id': bundle['task_id'], 'round': f"subagent-neuralese:{bundle['key']}", 'rows': rows}


# ---- held separation by content --------------------------------------------------------------------------------------
def norm_words(text):
    return re.findall(r'[0-9a-z]+', (text or '').lower())


def held_texts():
    """Normalized strings of every held set's task inputs (held_eval.held_manifests: held-v3, held-v4, and what they also exclude)
    and their 10-word shingles."""
    from natlang_neuralese.common.jsonio import read_jsonl
    from natlang_neuralese.rl.held_eval import held_manifests
    exact, shingles = set(), set()

    def walk(value):
        if isinstance(value, str):
            words = norm_words(value)
            if len(words) >= 4:
                exact.add(' '.join(words))
            shingles.update(' '.join(words[k:k + 10]) for k in range(len(words) - 9))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    read, manifests = set(), [Path(p) for _, p in held_manifests()]
    for path in list(manifests):    # the earlier held sets a manifest still excludes (held-v3: held-v2)
        try:
            manifests += [Path(e['path']) for e in json.loads(path.read_text()).get('also_excludes') or ()]
        except (ValueError, AttributeError, TypeError):
            pass
    for path in manifests:
        cases = path.parent / 'cases.jsonl'    # a held set's own rows sit beside its manifest
        if not cases.exists() or cases in read:
            continue
        read.add(cases)
        for row in read_jsonl(cases):
            semantics = row.get('semantics') or {}
            walk(semantics.get('inputs'))
            walk(semantics.get('folder_files'))
    return exact, shingles


def held_content_hit(question, held):
    exact, shingles = held
    words = norm_words(question)
    if ' '.join(words) in exact:
        return 'question'
    grams = [' '.join(words[k:k + 10]) for k in range(len(words) - 9)]
    if grams and sum(g in shingles for g in grams) >= max(1, len(grams) // 2):
        return 'shingles'
    return None


# ---- build -----------------------------------------------------------------------------------------------------------
_LOWERING = None


def _init(tokenizer, read_mode):
    global _LOWERING
    _LOWERING = Lowering(tokenizer, read_mode)


def _work(bundle):
    """(chains, per-call summaries, counters) of one bundle, lowered in a worker; the rows themselves stay here."""
    from exp0_sft import MAX_LEN, chains_of
    stats = collections.Counter()
    episode = lower_bundle(_LOWERING, bundle, stats)
    if episode is None:
        return None, [], stats
    calls = [{'invocation': r['invocation'], 'call': r['index'], 'imports': len(r['imports']),
              'states': sorted(r['sources']), 'prompt_tokens': int(len(r['prompt'])),
              'target_tokens': int(len(r['sample'])),
              'weighted_target_tokens': float(r['weight'].sum()) if 'weight' in r else float(len(r['sample'])),
              'too_long': len(r['prompt']) + len(r['sample']) > MAX_LEN} for r in episode['rows']]
    return chains_of([episode]), calls, stats


def task_id(bundle):
    digest = hashlib.sha256(json.dumps([bundle['key'], bundle['question']], ensure_ascii=False).encode()).hexdigest()[:8]
    return f"subagent:{bundle['key']}:{digest}"


def build(args):
    from natlang_neuralese.common import data_registry
    from natlang_neuralese.common.code_provenance import code_provenance
    from exp0_sft import source_tokens
    from sft_sources import ZstdWriter, held_closure, held_hit, sha256_file
    data_registry.require_registered([str(args.input)])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats, funnel = collections.Counter(), collections.Counter()
    held, held_content = held_closure(args.held_manifest), held_texts()
    questions = collections.Counter()

    def admitted():
        for bundle in bundles(args.source, args.input, funnel, args):
            if args.limit and funnel['admitted'] >= args.limit:
                return
            bundle['task_id'] = task_id(bundle)
            if not bundle['verified']:
                funnel['unverified'] += 1
                continue
            if not bundle['lead']['handoffs']:
                funnel['no_handoff'] += 1
                continue
            if held_hit(bundle['task_id'], bundle['source_ids'], bundle['source_groups'], held):
                funnel['held_excluded'] += 1
                continue
            hit = held_content_hit(bundle['question'], held_content)
            if hit:
                funnel[f'held_content_{hit}'] += 1
                continue
            key = ' '.join(norm_words(bundle['question']))
            if questions[key] >= args.max_per_question:
                funnel['over_question_cap'] += 1
                continue
            questions[key] += 1
            funnel['admitted'] += 1
            yield bundle

    answer_types = collections.Counter()

    def collect(future, meta):
        task, key, kind = meta
        got, summaries, more = future.result()
        stats.update(more)
        if got is None:
            funnel['lowering_failed'] += 1
            return
        chains.extend(got)
        calls.extend(summaries)
        answer_types[kind] += 1
        cases[task] = {'id': task, 'source': f'subagent-neuralese:{args.source}', 'key': key}
        for summary in summaries:
            rows_out.write({'episode': key, 'task_id': task, **summary})

    chains, cases, calls = [], {}, []
    rows_out = ZstdWriter(out / 'rows.jsonl.zst')
    with concurrent.futures.ProcessPoolExecutor(args.workers, initializer=_init,
                                                initargs=(args.tokenizer, args.read_mode)) as pool:
        work = admitted()
        pending = {}
        for bundle in work:   # bounded in flight: bundles stream from the source
            pending[pool.submit(_work, bundle)] = (bundle['task_id'], bundle['key'], bundle['answer_type'])
            if len(pending) >= args.workers * 4:
                done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    collect(future, pending.pop(future))
        for future in concurrent.futures.as_completed(list(pending)):
            collect(future, pending.pop(future))
    rows_out.close()
    random.Random(args.seed).shuffle(chains)
    with (out / 'chains.pkl.partial').open('wb') as handle:
        pickle.dump({'chains': chains, 'schema': 'natlang.sft-chains/1'}, handle, protocol=4)
    os.replace(out / 'chains.pkl.partial', out / 'chains.pkl')
    report = {
        'schema': SCHEMA, 'code': code_provenance(), 'argv': sys.argv, 'source': args.source,
        'inputs': {'source': str(args.input), 'source_sha256': sha256_file(args.input),
                   'held_manifest': args.held_manifest, 'tokenizer': str(args.tokenizer)},
        'policy': {'read_mode': args.read_mode, 'min_width_f1': args.min_width_f1,
                   'max_per_question': args.max_per_question, 'seed': args.seed,
                   'targets': 'every recorded assistant reply of the lead and of each recorded subagent; teacher '
                              'reasoning and lead-in text zero weight (think_context); prompts, tool outputs and the '
                              'Neuralese report spans zero weight'},
        'funnel': dict(funnel), 'lowering': dict(stats), 'episodes': len(cases), 'chains': len(chains),
        'importing_chains': sum(1 for c in chains if c['imports']),
        'calls': len(calls), 'lead_calls': sum(1 for r in calls if r['invocation'] == 'lead'),
        'consumer_calls': sum(1 for r in calls if r['imports']),
        'neuralese_reads': sum(r['imports'] for r in calls),
        'distinct_states': len({k for r in calls for k in r['states']}),
        'chain_neuralese_reads': sum(len(c['imports']) for c in chains),
        'forward_tokens': int(sum(len(c['ids']) + source_tokens(c) for c in chains)),
        'producer_document_tokens': int(sum(source_tokens(c) for c in chains)),
        'target_tokens': int(sum(r['target_tokens'] for r in calls)),
        'weighted_target_tokens': int(sum(r['weighted_target_tokens'] for r in calls)),
        'lead_weighted_target_tokens': int(sum(r['weighted_target_tokens'] for r in calls if r['invocation'] == 'lead')),
        'too_long_calls': sum(r['too_long'] for r in calls),
        'distinct_questions': len(questions), 'answer_types': dict(answer_types),
    }
    (out / 'build.json').write_text(json.dumps(report, indent=1) + '\n')
    (out / 'cases.jsonl').write_text(''.join(json.dumps(cases[t]) + '\n' for t in sorted(cases)))
    data_registry.record_dataset(data_registry.dataset_id(out, 'sft-rows'), out, kind='sft-rows',
                                 format=f'{SCHEMA}: chains.pkl (exp0_sft --chains, sft_sources build --aux), '
                                        'rows.jsonl.zst, cases.jsonl, build.json',
                                 inputs=[str(args.input), args.held_manifest], license=LICENSES[args.source])
    print(json.dumps({k: v for k, v in report.items() if k not in ('code', 'argv', 'inputs', 'policy')}, indent=1))


def call_counts(messages):
    """(tool calls, tool responses, {tool name: calls}) of one recorded conversation."""
    calls, responses, names = 0, 0, collections.Counter()
    for message in messages:
        if message['role'] == 'assistant':
            for raw in TOOL_CALL.findall(message['content'] or ''):
                calls += 1
                try:
                    names[json.loads(raw)['name']] += 1
                except (ValueError, KeyError, TypeError):
                    names['<unparsed>'] += 1
        elif message['role'] == 'tool':
            responses += 1
        elif message['role'] == 'user':
            responses += (message['content'] or '').count('<tool_response>')
    return calls, responses, names


def answer_type(answer):
    if answer is None:
        return 'none'
    words = len(answer.split())
    cjk = re.search(r'[぀-ヿ一-鿿가-힯]', answer) is not None
    kind = 'number' if re.fullmatch(r'[-+]?[\d.,%]+\s*\w{0,12}', answer.strip()) else \
        'short' if words <= 6 else 'phrase' if words <= 25 else 'long'
    return ('cjk-' if cjk else '') + kind


def scope(args):
    """Read-only counts for a natlang conversion (owner 2026-11-24, part 2): per item, (a) a checkable final answer and
    (b) a recorded tool output for every tool call of the lead and of every dispatched subagent."""
    stats, rows = collections.Counter(), collections.Counter()
    tools = collections.Counter()
    if args.source == 'searchswarm':
        for index, row in enumerate(searchswarm_rows(args.input)):
            gold = row.get('answer')
            kind = f"{row['source']}:{answer_type(gold)}"
            calls, responses, names = call_counts(row['messages'])
            tools.update({'lead:' + k: v for k, v in names.items()})
            complete = calls == responses
            dispatched = 0
            for message in row['messages']:
                for raw in TOOL_CALL.findall(message['content'] or '') if message['role'] == 'assistant' else ():
                    try:
                        call = json.loads(raw)
                    except ValueError:
                        continue
                    if call.get('name') == 'call_sub_agent':
                        dispatched += len(call['arguments'].get('prompts') or ())
            subs = row['subagents'] or []
            for sub in subs:
                c, r, n = call_counts(sub['messages'])
                tools.update({'sub:' + k: v for k, v in n.items()})
                complete = complete and c == r
            all_subs = dispatched == len(subs)
            final = swarm_answer(swarm_messages(row['messages']) or [{'role': 'user'}])
            right = bool(final and gold and qa_norm(gold) in qa_norm(final))
            # the gold in the recorded tool outputs (lead and subagents): a frozen fixture could answer the item
            evidence = qa_norm(' '.join(m['content'] for conv in [row['messages'], *[x['messages'] for x in subs]]
                                        for m in conv if m['role'] in ('user', 'tool') and '<tool_response>' in m['content']
                                        and 'A sub-agent dispatched for goal' not in m['content']))
            found = bool(gold and qa_norm(gold) and qa_norm(gold) in evidence)
            rows[(kind, 'gold', 'teacher_right' if right else 'teacher_wrong',
                  'gold_in_evidence' if found else 'gold_not_in_evidence',
                  'no_dispatch' if not dispatched else 'complete' if complete and all_subs else
                  'subagents_missing' if complete else 'outputs_missing')] += 1
            stats['items'] += 1
            stats['dispatched'] += dispatched
            stats['subagents_recorded'] += len(subs)
    elif args.source == 'wideseek':
        for sessions in wideseek_bundles(args.input):
            planner = [s for s in sessions if s['metadata']['role'] == 'planner'][0]
            meta = planner['metadata']
            dispatched = 0
            for message in planner['messages']:
                for raw in TOOL_CALL.findall(message['content'] or '') if message['role'] == 'assistant' else ():
                    try:
                        dispatched += len(json.loads(raw)['arguments'].get('sub_agents') or ())
                    except (ValueError, KeyError):
                        pass
            workers = [s for s in sessions if s['metadata']['role'] == 'worker']
            complete = True
            for worker in workers:
                c, r, n = call_counts(worker['messages'])
                tools.update({'sub:' + k: v for k, v in n.items()})
                complete = complete and c == r
            f1 = meta.get('item_f1') or 0
            right = f1 >= 1.0 if meta['dataset_type'] == 'depth' else f1 >= args.min_width_f1
            rows[(meta['dataset_type'], 'no_gold_in_release', 'item_f1_pass' if right else 'item_f1_fail',
                  'no_dispatch' if not dispatched else 'complete' if complete and dispatched == len(workers) else
                  'subagents_missing' if complete else 'outputs_missing')] += 1
            stats['items'] += 1
            stats['dispatched'] += dispatched
            stats['subagents_recorded'] += len(workers)
    else:   # evidence-subagent: child-only conversations
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(args.input).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                stats['items'] += 1
                if stats['items'] == 1:
                    stats['columns:' + ','.join(sorted(row))] += 1
                messages = row.get('messages') or row.get('conversations') or []
                c, r, n = call_counts(messages) if messages and isinstance(messages[0], dict) and 'role' in messages[0] \
                    else (0, 0, collections.Counter())
                tools.update(n)
                gold = next((row[k] for k in ('answer', 'gold', 'ground_truth', 'label') if row.get(k)), None)
                rows[('evidence', 'gold' if gold else 'no_gold', 'complete' if c == r else 'outputs_missing')] += 1
    report = {'source': args.source, 'input': str(args.input), 'stats': dict(stats),
              'items': {' / '.join(k): v for k, v in sorted(rows.items())}, 'tools': dict(tools.most_common())}
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=1, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--source', choices=('wideseek', 'searchswarm'), required=True)
    b.add_argument('--input', type=Path, required=True)
    b.add_argument('--tokenizer', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--read-mode', choices=READ_MODES, default='splice')
    b.add_argument('--min-width-f1', type=float, default=0.9,
                   help='WideSeek width items: the least source item F1 admitted (depth items need 1.0)')
    b.add_argument('--max-per-question', type=int, default=1, help='bundles admitted per distinct question')
    b.add_argument('--held-manifest', default=None, help='default: the registry role held-eval')
    b.add_argument('--workers', type=int, default=8)
    b.add_argument('--limit', type=int, default=0)
    b.add_argument('--seed', type=int, default=0)
    s = sub.add_parser('scope', help='read-only counts for a natlang conversion')
    s.add_argument('--source', choices=('wideseek', 'searchswarm', 'evidence'), required=True)
    s.add_argument('--input', type=Path, required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--min-width-f1', type=float, default=0.9)
    args = p.parse_args()
    if args.command == 'scope':
        return scope(args)
    if args.held_manifest is None:
        from grpo import default_held
        args.held_manifest = str(default_held())
    build(args)


if __name__ == '__main__':
    main()
