"""Neuralese hindsight hints: verified runs as writer input (plans/neuralese/RESTORE_MEMETIC_JOLT_HINDSIGHT_20261117.md
section 2, NEURALESE_HINDSIGHT_HINTS.md).

A verified successful run tau* of a training task (a teacher-generation success or a GRPO run's own success, as
``scripts/neuralese_data/sft_sources.py extract`` writes them) becomes a ``TrajectoryCall`` value
(``applications/guidance-writer/types.ts``). A JOLT+ teacher attempt of the same task carries it in its run controls
(``rl/jolt.py guidance_controls(trajectory=...)``); the runtime's ``writeHint`` (the policy model, a natlang program)
reads it with eval and writes the hint the teacher's root call imports. This module is crisp data only: the value
builder and a per-task index the GRPO collector reads lazily.

    python -m natlang_neuralese.rl.hindsight index --episodes WORK/episodes.jsonl.zst [...] --out INDEX.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

PROMPT_CHARS = 20000
OUTPUT_CHARS = 4000
CODE_CHARS = 8000
REASONING_CHARS = 2000
INDEX_SCHEMA = 'natlang.hindsight-trajectory/1'


def _clip(text, limit):
    text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + f' <<cut off: {len(text) - limit} more characters>>'


def _content(message):
    value = message.get('content')
    if isinstance(value, list):
        return '\n'.join(part.get('text', '') for part in value if isinstance(part, dict))
    return value or ''


def _status(output):
    head = output.lstrip()[:200].lower()
    return 'error' if head.startswith(('error', 'typeerror', 'referenceerror', 'syntaxerror')) or 'error:' in head[:40] \
        else 'ok'


def call_steps(messages, target):
    """The steps of one call from its last turn's transcript (assistant tool calls, each with the tool output that
    answered it) and its final reply ``target``, in the ``TrajectoryStep`` shape."""
    outputs = {m.get('tool_call_id'): _content(m) for m in messages if m.get('role') == 'tool'}
    steps, turn = [], 0
    for message in [*messages, *([target] if isinstance(target, dict) else [])]:
        if message.get('role') != 'assistant':
            continue
        reasoning = message.get('reasoning_content')
        calls = message.get('tool_calls') or []
        for number, call in enumerate(calls):
            function = call.get('function') or {}
            try:
                arguments = json.loads(function.get('arguments') or '{}')
            except (TypeError, json.JSONDecodeError):
                arguments = {'raw': function.get('arguments')}
            if not isinstance(arguments, dict):
                arguments = {'value': arguments}
            output = outputs.get(call.get('id'), '')
            step = {'turn': turn, 'tool': function.get('name') or 'call',
                    'arguments': {k: v for k, v in arguments.items() if k != 'code'},
                    'status': _status(output), 'output': _clip(output, OUTPUT_CHARS)}
            if isinstance(arguments.get('code'), str):
                step['code'] = _clip(arguments['code'], CODE_CHARS)
            if reasoning and number == 0:
                step['reasoning'] = _clip(reasoning, REASONING_CHARS)
            steps.append(step)
        if not calls:
            text = _content(message)
            if text.strip() or reasoning:
                steps.append({'turn': turn, 'tool': 'reply', 'arguments': {}, 'status': 'completed',
                              'output': _clip(text, OUTPUT_CHARS),
                              **({'reasoning': _clip(reasoning, REASONING_CHARS)} if reasoning else {})})
        turn += 1
    return steps


def _result(steps):
    """How a verified call finished: its last ``return_result`` step's status and value (the writer's hint check
    rejects a hint that copies that value), else success with no value."""
    final = next((step for step in reversed(steps) if step['tool'] == 'return_result'), None)
    arguments = final['arguments'] if final else {}
    result = {'status': arguments.get('status') if isinstance(arguments.get('status'), str) else 'success'}
    if 'value' in arguments:
        result['value'] = arguments['value']
    return result


def trajectory_value(episode):
    """``TrajectoryCall`` of a verified episode (``sft_sources`` episode or a scored rollout row: ``turns`` with
    ``messages``, ``target`` and ``invocation_id``). The root is the first turn's invocation; every other invocation
    becomes a child of it in first-appearance order (the extracted turns carry no caller edges, so the tree is one level
    deep). Each call's steps come from its last turn, whose transcript holds all its earlier actions."""
    turns = [turn for turn in episode.get('turns') or () if isinstance(turn.get('messages'), list)]
    if not turns:
        raise ValueError('a hindsight trajectory needs turns with their messages')
    order, last = [], {}
    for turn in turns:
        name = turn.get('invocation_id') or 'root'
        if name not in last:
            order.append(name)
        last[name] = turn
    function = episode.get('family') or episode.get('task_id') or 'task'

    def call(name, role):
        turn = last[name]
        prompt = '\n'.join(_content(m) for m in turn['messages'] if m.get('role') == 'user')
        steps = call_steps(turn['messages'], turn.get('target'))
        return {'call': name, 'function': function if role == 'root' else name.rsplit('/', 1)[-1], 'role': role,
                'prompt': _clip(prompt, PROMPT_CHARS), 'steps': steps, 'result': _result(steps), 'children': []}
    root = call(order[0], 'root')
    root['children'] = [call(name, 'child') for name in order[1:]]
    return root


def size_of(episode):
    """Sampled size of an episode (completion tokens when recorded, else characters of its targets / 4)."""
    total = 0
    for turn in episode.get('turns') or ():
        tokens = turn.get('completion_tokens')
        total += tokens if isinstance(tokens, int) else len(json.dumps(turn.get('target') or '')) // 4
    return total


class TrajectoryIndex(Mapping):
    """``task_id -> TrajectoryCall`` over an index file, read on demand (offsets kept, values not)."""

    def __init__(self, path):
        self.path = Path(path)
        self.offsets = {}
        with self.path.open('rb') as handle:
            offset = 0
            for line in handle:
                if line.strip():
                    head = json.loads(line)
                    if head.get('schema') != INDEX_SCHEMA:
                        raise ValueError(f'{self.path} is not a {INDEX_SCHEMA} index')
                    self.offsets[head['task_id']] = offset
                offset += len(line)

    def __getitem__(self, task_id):
        with self.path.open('rb') as handle:
            handle.seek(self.offsets[task_id])
            return json.loads(handle.readline())['trajectory']

    def __iter__(self):
        return iter(self.offsets)

    def __len__(self):
        return len(self.offsets)


def load_index(path):
    return TrajectoryIndex(path)


def build_index(episodes, *, excluded=frozenset()):
    """{task_id: record} with the shortest verified episode of each task not in ``excluded``. ``episodes`` yields
    ``sft_sources`` episode dicts (``source``, ``task_id``, ``task``, ``turns``)."""
    best = {}
    for episode in episodes:
        task_id = episode.get('task_id')
        if not task_id or task_id in excluded or (episode.get('task') or {}).get('split') != 'train':
            continue
        size = size_of(episode)
        if task_id in best and best[task_id][0] <= size:
            continue
        try:
            value = trajectory_value(episode)
        except ValueError:
            continue
        best[task_id] = (size, {'schema': INDEX_SCHEMA, 'task_id': task_id, 'source': episode.get('source'),
                                'key': episode.get('key'), 'teacher': episode.get('teacher'), 'size': size,
                                'trajectory': value})
    return {task_id: record for task_id, (_, record) in best.items()}


def _episodes(paths):
    import io
    import zstandard
    for path in paths:
        with open(path, 'rb') as raw:
            stream = zstandard.ZstdDecompressor().stream_reader(raw) if str(path).endswith('.zst') else raw
            for line in io.TextIOWrapper(stream, encoding='utf-8'):
                if line.strip():
                    yield json.loads(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    index = sub.add_parser('index', help='per-task shortest verified run as a TrajectoryCall (JSONL)')
    index.add_argument('--episodes', action='append', required=True, help='sft_sources episodes*.jsonl.zst (repeatable)')
    index.add_argument('--held-manifest', help='held manifest whose closure is excluded (default: the registry role held-eval)')
    index.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    from .held_eval import exclusion_ids
    manifest = args.held_manifest
    if manifest is None:
        from ..common.data_registry import role_path
        manifest = role_path('held-eval')
    # two streaming passes: the held closure needs only each task's id and split group
    cases, count = {}, 0
    for episode in _episodes(args.episodes):
        count += 1
        task = episode.get('task')
        if isinstance(task, dict):
            cases[episode['task_id']] = {'id': task.get('id'),
                                         'curriculum': {'split_group': (task.get('curriculum') or {}).get('split_group')}}
    excluded = exclusion_ids(manifest, list(cases.values()))
    episodes = range(count)
    records = build_index(_episodes(args.episodes), excluded=excluded)
    out = Path(args.out)
    partial = out.with_suffix(out.suffix + '.partial')
    with partial.open('w') as handle:
        for task_id in sorted(records):
            handle.write(json.dumps(records[task_id], ensure_ascii=False) + '\n')
    partial.replace(out)
    from ..common import data_registry
    data_registry.record_dataset(data_registry.dataset_id(out.parent, 'hindsight-index') if out.parent.name.startswith(
        'hindsight-index') else data_registry.path_id('hindsight-index', out), out, kind='index',
        format='natlang.hindsight-trajectory/1 JSONL: per task the shortest verified run as a TrajectoryCall',
        inputs=[*args.episodes, manifest], license='derived from the episodes it indexes')
    sources = {}
    for record in records.values():
        sources[record['source']] = sources.get(record['source'], 0) + 1
    print(json.dumps({'episodes': len(episodes), 'tasks': len(records), 'excluded_held': len(excluded),
                      'sources': sources, 'out': str(out)}))


if __name__ == '__main__':
    main()
