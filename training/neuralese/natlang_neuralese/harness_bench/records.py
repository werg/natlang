"""Harness-bench training records (plans/neuralese/HARNESS_BENCH.md §2): agent trajectories as native training turns.

A teacher's trajectory, normalized to a dialect (openhands.py), becomes the pipeline's own record format,
`natlang.teacher_training_turn.native/1`, which both students' stages read: the text warm-up through the shared
renderer, and recurrence/trajectory training directly. One record per supervised assistant turn:

- messages: the harness's system prompt as a soft piece (pi's own prompt, exported by `natlang run applications/pi --
  surface`), the user's task, then every earlier turn: assistant turns with their reasoning (`reasoning_content`), text
  and tool calls, and tool results;
- target: the assistant turn itself; tools: the harness's tool schemas.

A tool result longer than `digest_chars` is a **digest part** (DECISIONS 43): `source` is the full output, `preview`
is the output as the teacher saw it (cut at `preview_chars`, the companion's crisp head-and-tail shape beyond), and the
digest is conditioned on the call's intent: the reasoning and text of the assistant turn that made the call, and the
call (`instructions`). `note` says how the agent gets the whole output (`recall`). Text renderings show the preview,
so the text warm-up trains on the teacher's own view; recurrence training writes the digest through the port, and the
reader's loss plus self-distillation from the preview (`--distill`) train the writer to keep what the following
actions need.

Splits are by repository (no repository in both); source groups are the repository and the task instance.
Admission is recorded per record by explicit criteria (`admission`); the builder never approves on its own.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

from ..common.hashing import sha256_hex
from ..digest import INSTRUCTIONS as DIGEST_INSTRUCTIONS
from .openhands import DIALECT, Normalized, normalize, rows

VERSION = 'natlang.teacher_training_turn.native/1'
BUILDER = 'natlang.harness_bench.records/1'
DIGEST_CHARS = 2000
PREVIEW_CHARS = 40_000
SHAPE_HEAD, SHAPE_TAIL = 2500, 2000


def text_digest(text: str) -> str:
    return sha256_hex(text.encode())


def recall_note(call_id: str) -> str:
    return f'  // digest of the output; recall("{call_id}") returns all of it'


def companion_shape(text: str, call_id: str) -> str:
    """The companion's crisp shape of a long output (applications/pi/extensions/companion shapeOutput)."""
    elided = len(text) - SHAPE_HEAD - SHAPE_TAIL
    return (f'{text[:SHAPE_HEAD]}\n[… {elided} characters elided by the companion; recall("{call_id}") returns the full output …]\n'
            f'{text[-SHAPE_TAIL:]}')


def split_of(repo: str, test_percent: int) -> str:
    """Repositories, not trajectories, are split: the same code never sits on both sides."""
    return 'test' if int(text_digest(f'harness-bench-split:{repo}')[:8], 16) % 100 < test_percent else 'train'


def _arguments_text(arguments: Any) -> str:
    return json.dumps(arguments, ensure_ascii=False, separators=(',', ':'))


def intent(assistant: dict[str, Any], call: dict[str, Any]) -> str:
    """What the agent wanted from a tool call, as it was known when the call was made: the turn's reasoning and text,
    then the call. The digest of the call's output is conditioned on this, never on later turns."""
    reasoning = '\n'.join(part.get('thinking', '') for part in assistant['content'] if part['type'] == 'thinking').strip()
    text = '\n'.join(part.get('text', '') for part in assistant['content'] if part['type'] == 'text').strip()
    lines = ['The agent made this tool call and reads its output next:', f'{call["name"]} {_arguments_text(call["arguments"])}']
    if reasoning or text:
        lines += ['Its reasoning when it made the call:', '\n'.join(part for part in (reasoning, text) if part)]
    return '\n'.join(lines)


def native_messages(transcript: Normalized, system_piece: str, digest_chars: int, preview_chars: int) -> tuple[list[dict], list[int], int]:
    """The transcript as native messages; also the index of every assistant message and the number of digest parts."""
    out: list[dict] = [{'role': 'system', 'content': [{'type': 'soft', 'name': system_piece}]}]
    assistants: list[int] = []
    calls: dict[str, tuple[dict, dict]] = {}
    digests = 0
    for message in transcript.messages:
        role = message['role']
        if role == 'user':
            out.append({'role': 'user', 'content': message['content'] if isinstance(message['content'], str) else
                        ''.join(part.get('text', '') for part in message['content'])})
        elif role == 'assistant':
            content = message['content']
            entry: dict[str, Any] = {'role': 'assistant',
                                     'content': '\n'.join(part['text'] for part in content if part['type'] == 'text')}
            reasoning = '\n'.join(part['thinking'] for part in content if part['type'] == 'thinking')
            if reasoning:
                entry['reasoning_content'] = reasoning
            tool_calls = [part for part in content if part['type'] == 'toolCall']
            if tool_calls:
                entry['tool_calls'] = [{'id': part['id'], 'type': 'function',
                                        'function': {'name': part['name'], 'arguments': _arguments_text(part['arguments'])}}
                                       for part in tool_calls]
                for part in tool_calls:
                    calls[part['id']] = (message, part)
            assistants.append(len(out))
            out.append(entry)
        elif role == 'toolResult':
            text = ''.join(part.get('text', '') for part in message['content'] if part.get('type') == 'text')
            call_id = message['toolCallId']
            if len(text) > digest_chars and call_id in calls:
                assistant, call = calls[call_id]
                preview = text if len(text) <= preview_chars else companion_shape(text, call_id)
                content: Any = [{'type': 'digest', 'name': f'digest:{text_digest(text)[:12]}', 'holder': f'recall("{call_id}")',
                                 'value_type': 'string', 'source': text, 'preview': preview,
                                 'instructions': intent(assistant, call), 'note': recall_note(call_id)}]
                digests += 1
            else:
                content = text
            out.append({'role': 'tool', 'tool_call_id': call_id, 'content': content})
    return out, assistants, digests


def choose_targets(messages: list[dict], assistants: list[int], targets: str, per_trajectory: int, rng: random.Random) -> list[int]:
    """Which assistant turns become records. `all`: every turn (whole-trajectory supervision). `sample`: up to
    `per_trajectory`, first the turns that read a digested output, then others at random, in trajectory order."""
    if targets == 'all':
        return assistants
    readers = [index for index in assistants
               if index > 0 and messages[index - 1]['role'] == 'tool' and isinstance(messages[index - 1]['content'], list)]
    rest = [index for index in assistants if index not in readers]
    rng.shuffle(readers)
    rng.shuffle(rest)
    return sorted((readers + rest)[:per_trajectory])


def admission(transcript: Normalized, replay: dict[str, Any] | None) -> dict[str, Any]:
    """The record's admission by explicit criteria: a resolved task; observations in pi's own results, replayed and
    verified (the record holds only steps before any divergence); every kept step mapped exactly to the dialect."""
    reasons = ([] if transcript.resolved else ['task not resolved']) + ([] if replay else ['observations not replayed']) + \
        (['lossy steps'] if transcript.lossy and not replay else [])
    return {'kind': 'harness-bench-criteria/2', 'criteria': ['resolved', 'pi-replayed-verified', 'exact-mapping'],
            'approved': not reasons, **({'held': reasons} if reasons else {})}


def replayed(line: dict[str, Any]) -> tuple[Normalized, dict[str, Any] | None]:
    """A replayed trajectory (`natlang run applications/pi -- replay`) as a transcript, cut before the step where
    replay diverged: only verified steps, all in pi's own results, become records."""
    report = line.get('replay')
    messages = line['messages']
    if report and report.get('diverged'):
        cut = report['diverged']['message']
        # Back to the assistant turn that made the diverging call: neither it nor anything after it is kept.
        while cut > 0 and messages[cut - 1]['role'] != 'assistant':
            cut -= 1
        messages = messages[:max(cut - 1, 0)]
    fields = {name: line.get(name) for name in ('id', 'instance_id', 'repo', 'resolved', 'mapping', 'goal', 'cwd', 'lossy', 'patch', 'views')}
    fields['messages'] = messages
    fields['lossy'] = fields['lossy'] or []
    fields['views'] = fields['views'] or []
    return Normalized(**fields), report


def build(row: dict[str, Any] | None, *, system_piece: str, surface_sha: str, tools: list[dict], corpus: str, test_percent: int,
          digest_chars: int, preview_chars: int, targets: str, per_trajectory: int, seed: int,
          transcript: Normalized | None = None, replay: dict[str, Any] | None = None, row_sha: str | None = None) -> Iterable[dict]:
    transcript = transcript or normalize(row, 'pi')
    messages, assistants, digests = native_messages(transcript, system_piece, digest_chars, preview_chars)
    rng = random.Random(f'{seed}:{transcript.id}')
    split = split_of(transcript.repo, test_percent)
    admitted = admission(transcript, replay)
    groups = [f'swe-rebench-repo:{transcript.repo}', f'swe-rebench-instance:{transcript.instance_id}']
    row_sha = row_sha or text_digest(json.dumps((row or {}).get('trajectory'), sort_keys=True, default=str))
    for index in choose_targets(messages, assistants, targets, per_trajectory, rng):
        prefix = messages[:index]
        yield {
            'version': VERSION, 'id': f'harness:{DIALECT}:{transcript.id}:turn:{index:04d}',
            'source_ref': {'trajectory_id': transcript.id, 'instance_id': transcript.instance_id, 'repo': transcript.repo,
                           'source_row_sha256': row_sha, 'message_index': index},
            'provenance': {'builder': BUILDER, 'corpus': corpus, 'dialect': DIALECT, 'mapping': 'pi', 'harness': 'pi',
                           'surface_sha256': surface_sha, 'teacher': 'Qwen3-Coder-480B-A35B-Instruct (OpenHands 0.54)',
                           'digest_chars': digest_chars, 'preview_chars': preview_chars,
                           'observations': 'pi-replayed' if replay else 'teacher-recorded',
                           **({'replay': replay} if replay else {})},
            'task': {'kind': 'agent_trajectory', 'instance_id': transcript.instance_id, 'repo': transcript.repo,
                     'goal': transcript.goal},
            'family': 'harness_bench', 'task_family': 'swe', 'task_kind': 'agent_trajectory', 'task_modality': 'code',
            'license': 'CC-BY-4.0', 'split': split, 'source_groups': groups, 'source_ids': [transcript.id],
            'outcome': {'resolved': transcript.resolved}, 'training_admission': admitted,
            'neuralese_conversion': {'version': 'natlang.harness-bench-conversion/1',
                                     'sites': {'digest': {'converted': sum(1 for m in prefix if m['role'] == 'tool' and isinstance(m['content'], list))},
                                               'prompt': {'converted': 1}}},
            'messages': prefix, 'target': messages[index], 'tools': tools,
            'digests_in_trajectory': digests,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--trajectories', help='SWE-rebench OpenHands trajectories parquet (teacher-recorded observations)')
    source.add_argument('--replayed', help='`natlang run applications/pi -- replay` output (pi-replayed observations)')
    parser.add_argument('--surface', required=True, help='`natlang run applications/pi -- surface --companion` output')
    parser.add_argument('--corpus', required=True, help='the source corpus id in training/neuralese_corpora.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=0, help='trajectories to read (0: all)')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--targets', choices=['all', 'sample'], default='sample')
    parser.add_argument('--per-trajectory', type=int, default=8)
    parser.add_argument('--digest-chars', type=int, default=DIGEST_CHARS)
    parser.add_argument('--preview-chars', type=int, default=PREVIEW_CHARS)
    parser.add_argument('--test-percent', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    surface = json.loads(Path(args.surface).read_text())
    system_piece = f'prompt:pi-agent#sha256:{text_digest(surface["system"])}'
    tools = [{'type': 'function', 'function': {'name': t['name'], 'description': t['description'], 'parameters': t['parameters']}}
             for t in surface['tools']]
    surface_sha = text_digest(json.dumps(surface, sort_keys=True))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {'builder': BUILDER, 'corpus': args.corpus, 'surface_sha256': surface_sha, 'trajectories': 0, 'records': 0,
               'by_split': {'train': 0, 'test': 0}, 'admitted': 0, 'held': {}, 'digest_parts': 0}
    def sources() -> Iterable[dict[str, Any]]:
        if args.replayed:
            with open(args.replayed) as lines:
                for line in lines:
                    if line.strip():
                        transcript, report = replayed(json.loads(line))
                        yield {'transcript': transcript, 'replay': report, 'row': None, 'row_sha': json.loads(line).get('source_row_sha256')}
        else:
            columns = ['trajectory_id', 'instance_id', 'repo', 'trajectory', 'model_patch', 'resolved']
            for row in rows(args.trajectories, columns):
                yield {'transcript': None, 'replay': None, 'row': row, 'row_sha': None}

    with (out / 'records.jsonl').open('w') as handle:
        for number, item in enumerate(sources()):
            if number < args.offset:
                continue
            if args.limit and number >= args.offset + args.limit:
                break
            summary['trajectories'] += 1
            for record in build(item['row'], transcript=item['transcript'], replay=item['replay'], row_sha=item['row_sha'],
                                system_piece=system_piece, surface_sha=surface_sha, tools=tools, corpus=args.corpus,
                                test_percent=args.test_percent, digest_chars=args.digest_chars,
                                preview_chars=args.preview_chars, targets=args.targets,
                                per_trajectory=args.per_trajectory, seed=args.seed):
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                summary['records'] += 1
                summary['by_split'][record['split']] += 1
                if record['training_admission']['approved']:
                    summary['admitted'] += 1
                for reason in record['training_admission'].get('held', []):
                    summary['held'][reason] = summary['held'].get(reason, 0) + 1
                summary['digest_parts'] += record['neuralese_conversion']['sites']['digest']['converted']
    pieces = [{'name': system_piece, 'kind': 'system-prompt', 'text': surface['system']},
              {'name': 'prompt:digest', 'kind': 'system-prompt', 'text': DIGEST_INSTRUCTIONS}]
    (out / 'pieces.jsonl').write_text(''.join(json.dumps(piece, ensure_ascii=False) + '\n' for piece in pieces))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
