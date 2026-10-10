"""Harness-bench training records (plans/neuralese/HARNESS_BENCH.md §2): agent trajectories as native training turns.

A teacher's trajectory, normalized to a dialect (openhands.py), becomes the pipeline's own record format,
`natlang.teacher_training_turn.native/1`, which both students' stages read: the text warm-up through the shared
renderer, and recurrence/trajectory training directly. One record per supervised assistant turn:

- messages: the harness's system prompt as a soft piece (pi's own prompt, exported by `natlang run applications/pi --
  surface`), the user's task, then every earlier turn: assistant turns with their reasoning (`reasoning_content`), text
  and tool calls, and tool results;
- target: the assistant turn itself; tools: the harness's tool schemas.

A tool result longer than `view_chars` is a **view part** (DECISIONS.md 2026-10-09, one summarizer family): `source`
is the full output, `preview` is the output as the teacher saw it (cut at `preview_chars`, the companion's crisp
head-and-tail shape beyond), and the view is the builtin `view(source, instructions)` with the call's intent as its
instructions: the reasoning and text of the assistant turn that made the call, and the call (`instructions`). `note`
says how the agent gets the whole output (`recall`). Text renderings show the preview, so the text warm-up trains on
the teacher's own view; recurrence training writes the view through the port at view's template write site
(view.py), and the reader's loss plus self-distillation from the preview (`--distill`) train the writer to keep what
the following actions need. Records before natlang.harness-bench-conversion/2 named these parts `digest`; the trainer
rejects them and scripts/neuralese_data/digest_to_view.py converts them.

Splits are by repository (no repository in both); source groups are the repository and the task instance. Published
corpora that hold the same repositories cannot move, so a repository already placed by one keeps its split: its
component (the repository's and its instances' keys, under this corpus's and S1's group names) is placed through the
published corpora's cross-corpus indexes (scripts/neuralese_data/cross_corpus.py `place`; default S1's registered
index, cross-corpus-index-s1-full-final-20261003-v1); a repository they do not hold is split by `split_of`; one they place in several splits is left
out. Each record says how its split was decided (provenance.split_placement).
Admission is recorded per record by explicit criteria (`admission`); the builder never approves on its own.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

from ..common.hashing import sha256_hex
from ..view import INSTRUCTIONS as VIEW_INSTRUCTIONS
from .openhands import DIALECT, Normalized, normalize, rows

VERSION = 'natlang.teacher_training_turn.native/1'
BUILDER = 'natlang.harness_bench.records/1'
VIEW_CHARS = 2000
PREVIEW_CHARS = 40_000
SHAPE_HEAD, SHAPE_TAIL = 2500, 2000
# Published corpora whose splits this corpus follows: registered cross-corpus indexes (scripts/neuralese_data/cross_corpus.py,
# resolved by cross_corpus_registry).
PLACED_BY = ('cross-corpus-index-s1-full-final-20261003-v1',)


def text_digest(text: str) -> str:
    return sha256_hex(text.encode())


def recall_note(call_id: str) -> str:
    return f'  // view of the output; recall("{call_id}") returns all of it'


def companion_shape(text: str, call_id: str) -> str:
    """The companion's crisp shape of a long output (applications/pi/extensions/companion shapeOutput)."""
    elided = len(text) - SHAPE_HEAD - SHAPE_TAIL
    return (f'{text[:SHAPE_HEAD]}\n[… {elided} characters elided by the companion; recall("{call_id}") returns the full output …]\n'
            f'{text[-SHAPE_TAIL:]}')


def split_of(repo: str, test_percent: int) -> str:
    """Repositories, not trajectories, are split: the same code never sits on both sides."""
    return 'test' if int(text_digest(f'harness-bench-split:{repo}')[:8], 16) % 100 < test_percent else 'train'


def _neuralese_data():
    """scripts/neuralese_data (cross-corpus indexes and the shared split-group key), imported from the checkout."""
    scripts = str(Path(__file__).resolve().parents[4] / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import neuralese_data.cross_corpus as cross_corpus
    import neuralese_data.cross_corpus_registry as cross_corpus_registry
    import neuralese_data.records as data_records
    return cross_corpus, cross_corpus_registry, data_records


def repository_groups(repo: str, instances: Iterable[str]) -> list[str]:
    """A repository's split-group keys under this corpus's names and S1's (`repo:`, `swe-instance:`)."""
    _, _, data_records = _neuralese_data()
    keys = [f'swe-rebench-repo:{repo}', data_records.group_key('repo', repo)]
    for instance in sorted(set(instances)):
        keys += [f'swe-rebench-instance:{instance}', data_records.group_key('swe-instance', instance)]
    return keys


def placements(transcripts: Iterable[tuple[str, str]], index_refs: list, test_percent: int) -> tuple[dict, dict]:
    """Each repository's split from (repo, instance) pairs: the published split when the indexes (directories or
    registered index ids) place it (cross_corpus.place over the repository's component), else `split_of`; None when
    they place it in several. Returns ({repo: {'split', 'rule', 'touched'?}}, report)."""
    cross_corpus, cross_corpus_registry, _ = _neuralese_data()
    resolved = [cross_corpus_registry.resolve(ref) for ref in index_refs]
    indexes = [cross_corpus.Index(root, groups_only=True) for root, _ in resolved]
    instances: dict[str, set] = {}
    for repo, instance in transcripts:
        instances.setdefault(repo, set()).add(instance)
    placed = cross_corpus.place({repo: repository_groups(repo, insts) for repo, insts in instances.items()}, indexes)
    out, moved = {}, {}
    for repo in sorted(instances):
        own = split_of(repo, test_percent)
        if repo not in placed:
            out[repo] = {'split': own, 'rule': 'split_of'}
            continue
        split = placed[repo]['split']
        out[repo] = {'split': split, 'rule': 'published' if split else 'unplaceable', 'touched': placed[repo]['touched']}
        if split and split != own:
            moved[f'{own}->{split}'] = moved.get(f'{own}->{split}', 0) + 1
    report = {'indexes': [{'corpus_id': ix.corpus_id, **({'registry_id': rid} if rid else {'path': str(ix.root)}),
                           'records': ix.meta['records']} for ix, (_, rid) in zip(indexes, resolved)],
              'repositories': len(out), 'published': sum(1 for p in out.values() if p['rule'] == 'published'),
              'split_of': sum(1 for p in out.values() if p['rule'] == 'split_of'),
              'unplaceable': sorted(repo for repo, p in out.items() if p['rule'] == 'unplaceable'),
              'moved_from_split_of': dict(sorted(moved.items())),
              'by_split': {name: sum(1 for p in out.values() if p['split'] == name) for name in ('train', 'validation', 'test')}}
    return out, report


def _arguments_text(arguments: Any) -> str:
    return json.dumps(arguments, ensure_ascii=False, separators=(',', ':'))


def intent(assistant: dict[str, Any], call: dict[str, Any]) -> str:
    """What the agent wanted from a tool call, as it was known when the call was made: the turn's reasoning and text,
    then the call. The view of the call's output is written for this, never for later turns."""
    reasoning = '\n'.join(part.get('thinking', '') for part in assistant['content'] if part['type'] == 'thinking').strip()
    text = '\n'.join(part.get('text', '') for part in assistant['content'] if part['type'] == 'text').strip()
    lines = ['The agent made this tool call and reads its output next:', f'{call["name"]} {_arguments_text(call["arguments"])}']
    if reasoning or text:
        lines += ['Its reasoning when it made the call:', '\n'.join(part for part in (reasoning, text) if part)]
    return '\n'.join(lines)


def native_messages(transcript: Normalized, system_piece: str, view_chars: int, preview_chars: int) -> tuple[list[dict], list[int], int]:
    """The transcript as native messages; also the index of every assistant message and the number of view parts."""
    out: list[dict] = [{'role': 'system', 'content': [{'type': 'soft', 'name': system_piece}]}]
    assistants: list[int] = []
    calls: dict[str, tuple[dict, dict]] = {}
    views = 0
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
            if len(text) > view_chars and call_id in calls:
                assistant, call = calls[call_id]
                preview = text if len(text) <= preview_chars else companion_shape(text, call_id)
                content: Any = [{'type': 'view', 'name': f'view:{text_digest(text)[:12]}', 'holder': f'recall("{call_id}")',
                                 'value_type': 'string', 'source': text, 'preview': preview,
                                 'instructions': intent(assistant, call), 'note': recall_note(call_id)}]
                views += 1
            else:
                content = text
            out.append({'role': 'tool', 'tool_call_id': call_id, 'content': content})
    return out, assistants, views


def choose_targets(messages: list[dict], assistants: list[int], targets: str, per_trajectory: int, rng: random.Random) -> list[int]:
    """Which assistant turns become records. `all`: every turn (whole-trajectory supervision). `sample`: up to
    `per_trajectory`, first the turns that read a viewed output, then others at random, in trajectory order."""
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
          view_chars: int, preview_chars: int, targets: str, per_trajectory: int, seed: int,
          transcript: Normalized | None = None, replay: dict[str, Any] | None = None, row_sha: str | None = None,
          placement: dict[str, Any] | None = None) -> Iterable[dict]:
    """`placement`: the repository's split decision (`placements`); without one the split is `split_of`."""
    transcript = transcript or normalize(row, 'pi')
    messages, assistants, views = native_messages(transcript, system_piece, view_chars, preview_chars)
    rng = random.Random(f'{seed}:{transcript.id}')
    placement = placement or {'split': split_of(transcript.repo, test_percent), 'rule': 'split_of'}
    split = placement['split']
    if split is None:
        raise ValueError(f'repository {transcript.repo} is placed in several published splits: {placement.get("touched")}')
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
                           'view_chars': view_chars, 'preview_chars': preview_chars,
                           'observations': 'pi-replayed' if replay else 'teacher-recorded',
                           'split_placement': placement,
                           **({'replay': replay} if replay else {})},
            'task': {'kind': 'agent_trajectory', 'instance_id': transcript.instance_id, 'repo': transcript.repo,
                     'goal': transcript.goal},
            'family': 'harness_bench', 'task_family': 'swe', 'task_kind': 'agent_trajectory', 'task_modality': 'code',
            'license': 'CC-BY-4.0', 'split': split, 'source_groups': groups, 'source_ids': [transcript.id],
            'outcome': {'resolved': transcript.resolved}, 'training_admission': admitted,
            'neuralese_conversion': {'version': 'natlang.harness-bench-conversion/2',
                                     'sites': {'view': {'converted': sum(1 for m in prefix if m['role'] == 'tool' and isinstance(m['content'], list))},
                                               'prompt': {'converted': 1}}},
            'messages': prefix, 'target': messages[index], 'tools': tools,
            'views_in_trajectory': views,
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
    parser.add_argument('--view-chars', type=int, default=VIEW_CHARS)
    parser.add_argument('--preview-chars', type=int, default=PREVIEW_CHARS)
    parser.add_argument('--test-percent', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--placed-by', action='append', default=None,
                        help='cross-corpus index (registry id or directory) of a published corpus whose repository '
                             'splits this corpus follows (repeatable; default: ' + ', '.join(PLACED_BY) + ')')
    parser.add_argument('--no-placement', action='store_true', help='split by split_of alone (records before v4)')
    args = parser.parse_args(argv)
    index_dirs = [] if args.no_placement else (args.placed_by or list(PLACED_BY))
    surface = json.loads(Path(args.surface).read_text())
    system_piece = f'prompt:pi-agent#sha256:{text_digest(surface["system"])}'
    tools = [{'type': 'function', 'function': {'name': t['name'], 'description': t['description'], 'parameters': t['parameters']}}
             for t in surface['tools']]
    surface_sha = text_digest(json.dumps(surface, sort_keys=True))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {'builder': BUILDER, 'corpus': args.corpus, 'surface_sha256': surface_sha, 'trajectories': 0, 'records': 0,
               'by_split': {'train': 0, 'validation': 0, 'test': 0}, 'admitted': 0, 'held': {}, 'view_parts': 0}
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

    def selected() -> Iterable[dict[str, Any]]:
        for number, item in enumerate(sources()):
            if number < args.offset:
                continue
            if args.limit and number >= args.offset + args.limit:
                break
            if item['transcript'] is None:
                item['transcript'] = normalize(item['row'], 'pi')
            yield item

    if index_dirs:
        # Two passes: a repository is one component, so every trajectory is known before any split is decided.
        placed, summary['placement'] = placements(((item['transcript'].repo, item['transcript'].instance_id)
                                                   for item in selected()), index_dirs, args.test_percent)
    else:
        placed, summary['placement'] = {}, {'indexes': [], 'rule': 'split_of'}
    summary['unplaceable_trajectories'] = 0
    with (out / 'records.jsonl').open('w') as handle:
        for item in selected():
            placement = placed.get(item['transcript'].repo)
            if placement and placement['split'] is None:
                summary['unplaceable_trajectories'] += 1
                continue
            summary['trajectories'] += 1
            for record in build(item['row'], transcript=item['transcript'], replay=item['replay'], row_sha=item['row_sha'],
                                placement=placement,
                                system_piece=system_piece, surface_sha=surface_sha, tools=tools, corpus=args.corpus,
                                test_percent=args.test_percent, view_chars=args.view_chars,
                                preview_chars=args.preview_chars, targets=args.targets,
                                per_trajectory=args.per_trajectory, seed=args.seed):
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                summary['records'] += 1
                summary['by_split'][record['split']] += 1
                if record['training_admission']['approved']:
                    summary['admitted'] += 1
                for reason in record['training_admission'].get('held', []):
                    summary['held'][reason] = summary['held'].get(reason, 0) + 1
                summary['view_parts'] += record['neuralese_conversion']['sites']['view']['converted']
    pieces = [{'name': system_piece, 'kind': 'system-prompt', 'text': surface['system']},
              {'name': 'prompt:view', 'kind': 'system-prompt', 'text': VIEW_INSTRUCTIONS}]
    (out / 'pieces.jsonl').write_text(''.join(json.dumps(piece, ensure_ascii=False) + '\n' for piece in pieces))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
