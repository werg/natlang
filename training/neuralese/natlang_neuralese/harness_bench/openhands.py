"""OpenHands trajectories as pi transcripts (plans/neuralese/HARNESS_BENCH.md §1.2, the `openhands-v0.54` dialect).

A teacher trajectory becomes the messages a pi agent would have produced: user, assistant (text, thinking, tool calls)
and tool results, in pi-ai's message shapes. Two mappings:

- ``pi``: the teacher's tools map onto pi's coding tools where the meaning is the same (``str_replace_editor view`` of a
  file is ``read``, ``create`` is ``write``, ``str_replace`` is ``edit``, ``execute_bash`` is ``bash``, and a directory
  view is the ``find`` it describes, through ``bash``). ``think`` becomes
  a thinking block and ``finish`` the final answer. Steps without an exact pi equivalent (inserts, undo, interactive
  input, the task tracker) keep their own tool name and are listed in ``lossy``, so a consumer can drop
  the trajectory or register a dialect tool for them.
- ``native``: every tool call keeps the teacher's name and arguments; only paths are made relative and the harness's
  system prompt and phase instructions are stripped.

Observations are the teacher's own tool output, unchanged: replaying in pi's tools (§1.3) is what checks them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

DIALECT = 'openhands-v0.54'
_WORKSPACE = re.compile(r'^/workspace/[^/]+/?')
_ISSUE = re.compile(r'<issue_description>\s*(.*?)\s*</issue_description>', re.S)
_UPLOADED = re.compile(r'<uploaded_files>\s*(\S+)\s*</uploaded_files>')


@dataclass
class Normalized:
    """One trajectory as a pi transcript."""
    id: str
    instance_id: str
    repo: str
    resolved: bool
    mapping: str
    goal: str
    cwd: str
    messages: list[dict[str, Any]]
    lossy: list[dict[str, Any]] = field(default_factory=list)
    patch: str = ''

    def record(self) -> dict[str, Any]:
        return {'id': self.id, 'instance_id': self.instance_id, 'repo': self.repo, 'resolved': self.resolved,
                'dialect': DIALECT, 'mapping': self.mapping, 'goal': self.goal, 'cwd': self.cwd,
                'messages': self.messages, 'lossy': self.lossy, 'patch': self.patch}


def _relative(path: Any, cwd: str) -> Any:
    """A workspace path relative to the repository root (the agent's cwd in pi)."""
    if not isinstance(path, str):
        return path
    if cwd and (path == cwd or path.startswith(cwd.rstrip('/') + '/')):
        return path[len(cwd.rstrip('/')) + 1:] or '.'
    return _WORKSPACE.sub('', path) or '.' if path.startswith('/workspace/') else path


def _quote(path: str) -> str:
    import shlex
    return shlex.quote(path)


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or '{}')
    except (TypeError, ValueError):
        return {'_unparsed': raw}
    return value if isinstance(value, dict) else {'_value': value}


def _goal(content: str) -> str:
    """The issue itself, without OpenHands' phase instructions (programme decision: strip harness conventions)."""
    found = _ISSUE.search(content)
    return found.group(1).strip() if found else content.strip()


def _pi_call(name: str, args: dict[str, Any], cwd: str) -> tuple[str, dict[str, Any], str | None]:
    """The pi tool and arguments for a teacher call, and why it is lossy (None when exact)."""
    if name == 'execute_bash':
        if args.get('is_input') in (True, 'true'):
            return name, args, 'interactive input to a running command'
        out = {'command': args.get('command', '')}
        if isinstance(args.get('timeout'), (int, float)):
            out['timeout'] = args['timeout']
        return 'bash', out, None
    if name == 'str_replace_editor':
        command, path = args.get('command'), _relative(args.get('path'), cwd)
        if command == 'view':
            view_range = args.get('view_range')
            if isinstance(view_range, list) and len(view_range) == 2 and all(isinstance(x, int) for x in view_range):
                start, end = view_range
                out = {'path': path, 'offset': start}
                if end != -1:
                    out['limit'] = max(1, end - start + 1)
                return 'read', out, None
            # A directory view has no pi tool of its own; a file view without a range is a plain read.
            if isinstance(path, str) and ('.' in path.rsplit('/', 1)[-1]) and path != '.':
                return 'read', {'path': path}, None
            # OpenHands lists "files and directories up to 2 levels deep, excluding hidden items": the same as this find.
            target = path if isinstance(path, str) else '.'
            return 'bash', {'command': f"find {_quote(target)} -maxdepth 2 -not -path '*/.*'"}, None
        if command == 'create':
            return 'write', {'path': path, 'content': args.get('file_text', '')}, None
        if command == 'str_replace':
            return 'edit', {'path': path, 'edits': [{'oldText': args.get('old_str', ''), 'newText': args.get('new_str', '')}]}, None
        return name, {**args, 'path': path}, f'str_replace_editor {command} (no pi equivalent)'
    return name, args, f'{name} (no pi tool)'


def resolved(value: Any) -> bool:
    """The dataset's resolved flag, which it stores as text ('1', '0', '1.0')."""
    if isinstance(value, str):
        try:
            return float(value) >= 1
        except ValueError:
            return value.strip().lower() == 'true'
    return bool(value)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(part.get('text', '') for part in content if isinstance(part, dict))
    return '' if content is None else str(content)


def normalize(row: dict[str, Any], mapping: str = 'pi') -> Normalized:
    """One dataset row as a pi transcript."""
    if mapping not in ('pi', 'native'):
        raise ValueError(f'unknown mapping {mapping}')
    trajectory = row['trajectory']
    first_user = next((m for m in trajectory if m.get('role') == 'user'), None)
    uploaded = _UPLOADED.search(_text(first_user.get('content')) if first_user else '')
    cwd = uploaded.group(1).rstrip('/') if uploaded else ''
    if cwd and not cwd.startswith('/'):
        cwd = f'/workspace/{cwd}'
    messages: list[dict[str, Any]] = []
    lossy: list[dict[str, Any]] = []
    # Tool calls the transcript dropped (think), whose results are dropped too.
    dropped: set[str] = set()
    names: dict[str, str] = {}
    goal = ''
    for index, message in enumerate(trajectory):
        role = message.get('role')
        if role == 'system':
            continue
        if role == 'user':
            text = _text(message.get('content'))
            if not goal:
                goal = _goal(text)
                text = goal
            messages.append({'role': 'user', 'content': text})
            continue
        if role == 'assistant':
            content: list[dict[str, Any]] = []
            text = _text(message.get('content'))
            if text.strip():
                content.append({'type': 'text', 'text': text})
            final = None
            for call in message.get('tool_calls') or []:
                name = call['function']['name']
                args = _arguments(call['function'].get('arguments'))
                if mapping == 'pi' and name == 'think':
                    content.append({'type': 'thinking', 'thinking': str(args.get('thought', ''))})
                    dropped.add(call['id'])
                    continue
                if mapping == 'pi' and name == 'finish':
                    final = str(args.get('message', ''))
                    dropped.add(call['id'])
                    continue
                if mapping == 'pi':
                    name, args, reason = _pi_call(name, args, cwd)
                    if reason:
                        lossy.append({'message': len(messages), 'step': index, 'reason': reason})
                elif 'path' in args:
                    args = {**args, 'path': _relative(args['path'], cwd)}
                names[call['id']] = name
                content.append({'type': 'toolCall', 'id': call['id'], 'name': name, 'arguments': args})
            if final is not None:
                content.append({'type': 'text', 'text': final})
            has_calls = any(part['type'] == 'toolCall' for part in content)
            messages.append({'role': 'assistant', 'content': content, 'stopReason': 'toolUse' if has_calls else 'stop'})
            continue
        if role == 'tool':
            call_id = message.get('tool_call_id')
            if call_id in dropped:
                continue
            text = _text(message.get('content'))
            messages.append({'role': 'toolResult', 'toolCallId': call_id, 'toolName': names.get(call_id, message.get('name') or ''),
                             'content': [{'type': 'text', 'text': text}],
                             'isError': text.startswith('ERROR:') or text.startswith('Error')})
    return Normalized(id=str(row.get('trajectory_id') or row.get('instance_id')), instance_id=str(row.get('instance_id', '')),
                      repo=str(row.get('repo', '')), resolved=resolved(row.get('resolved')), mapping=mapping, goal=goal,
                      cwd=cwd, messages=messages, lossy=lossy, patch=str(row.get('model_patch') or ''))


def rows(path: str, columns: Iterable[str] | None = None) -> Iterator[dict[str, Any]]:
    """The parquet file's rows, one row group at a time."""
    import pyarrow.parquet as pq
    source = pq.ParquetFile(path)
    for group in range(source.metadata.num_row_groups):
        yield from source.read_row_group(group, columns=list(columns) if columns else None).to_pylist()


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('parquet')
    parser.add_argument('--out', required=True, help='JSONL of normalized transcripts')
    parser.add_argument('--mapping', choices=['pi', 'native'], default='pi')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--resolved-only', action='store_true')
    parser.add_argument('--exact-only', action='store_true', help='skip trajectories with lossy steps')
    args = parser.parse_args(argv)
    written = seen = skipped_lossy = 0
    with open(args.out, 'w', encoding='utf-8') as out:
        for row in rows(args.parquet, ['trajectory_id', 'instance_id', 'repo', 'trajectory', 'model_patch', 'resolved']):
            seen += 1
            if args.resolved_only and not resolved(row.get('resolved')):
                continue
            item = normalize(row, args.mapping)
            if args.exact_only and item.lossy:
                skipped_lossy += 1
                continue
            out.write(json.dumps(item.record(), ensure_ascii=False) + '\n')
            written += 1
            if args.limit and written >= args.limit:
                break
    print(json.dumps({'seen': seen, 'written': written, 'skipped_lossy': skipped_lossy, 'mapping': args.mapping}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
