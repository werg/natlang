"""The current serving prompt for recorded natlang tool-loop calls (training rows that match what ts-host sends).

Recorded trajectories carry the system prompt and tool schemas of the runtime version that produced them. ts-host now
sends, for the same call, the entry point's base prompt (``ENTRY_POINTS``; nl-depth-limited when applicable) + ``FUNCTION_TOOLS_PROMPT`` when the program
has its own code + the directory-reducer prompt of its file-tool surface + the program's guidance, and the native tool
schemas of that surface (``NativeToolAgent.toolsScope``). ``serving_messages`` rebuilds that request from a recorded one:

1. Find every known prompt piece (current texts and every git revision's texts, ``serving-surface.json``, plus the
   training piece bank) in the recorded system prompt, longest first, without overlaps.
2. The call is a tool-loop call iff an interpreter piece occurs. Its surface follows from the generic pieces found:
   depth limit (``interpreter/depth-limit``), own code (``function-tools``), folder access and file-tool surface
   (``directory-reducer/<surface>[/read-only][/depth-limit]``).
3. The first composed piece (interpreter, function tools, directory reducer) is replaced by the current composition
   for that surface (base + function tools + directory reducer); the other composed pieces and the teacher-only
   appendices (approach schematic, generation guidance: ts-host never serves them) are removed. Everything else
   (program declarations, guidance, caller additions, other runtime notes such as a compaction handover) stays
   verbatim in place.
4. Tools become the surface's current schemas in the runtime's order. ``return_result`` keeps the recorded call's own
   declared-result fields of ``value`` (its type is the call's, never invented); a recorded call without tool schemas
   gets ts-host's own schema of the declared result type in its opening (``ts-host/scripts/return-schemas.mjs``,
   ``return_schema: derived``), or keeps its recording when that cannot be derived.

A record that is not a tool-loop call (no interpreter piece: harness agents, decision readouts, predicates) or whose
messages call a tool the current surface does not offer is returned unchanged with the reason; callers decide whether
to keep its recorded rendering. Nothing here grants admission or changes split, source or target bytes.

The surface file is exported from a ts-host build by ``ts-host/scripts/export-serving-surface.mjs``.
"""
from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from .prompt_rendering import find_pieces, default_pieces

SURFACE_SCHEMA = 'natlang.serving-prompt-surface/1'
DEFAULT_SURFACE = Path(__file__).resolve().parents[4] / 'training' / 'prompt-pieces' / 'serving-surface.json'
# The two real tool-loop entry points (serving-surface.json ``bases``): ``collector`` is teacher/collector.ts
# defaultSystemPrompt (TOOLS_PROMPT + GENERATION_GUIDANCE), used by eval probes, RL and teacher collection; ``runtime``
# is the NativeToolAgent default (TOOLS_PROMPT), used by natlang run.
ENTRY_POINTS = ('collector', 'runtime')


# ``mix``: each record's entry point is drawn deterministically from its id with these shares. Eval probes, RL and
# teacher collection use the collector prompt, so it dominates; natlang run's prompt (no generation-guidance
# appendix) keeps the served-runtime form in the distribution.
ENTRY_SHARES = (('collector', 0.85), ('runtime', 0.15))


def entry_for(record_id: str, entry: str) -> str:
    """The entry point of one record: ``entry`` itself, or for ``mix`` a deterministic draw from ``ENTRY_SHARES``."""
    if entry != 'mix':
        return entry
    draw = int.from_bytes(hashlib.sha256(('serving-entry\0' + record_id).encode()).digest()[:8], 'big') / 2 ** 64
    for name, share in ENTRY_SHARES:
        if draw < share:
            return name
        draw -= share
    return ENTRY_SHARES[-1][0]


def system_prompt_mode(entry: str) -> str:
    return f'serving-{entry}'
INTERPRETERS = ('interpreter', 'interpreter/depth-limit')
# Teacher-only appendices (ts-host teacher/cli.ts --approach-guide, tune/pool.ts generation guidance): never served.
TEACHER_ONLY = ('approach', 'generation-guidance')


def composed_piece(logical: str) -> bool:
    """A generic piece of the tool-loop system prompt composition (agent.ts systemPrompt())."""
    return (logical in INTERPRETERS or logical == 'function-tools'
            or (logical.startswith('directory-reducer/') and logical != 'directory-reducer/declaration'))


@lru_cache(maxsize=4)
def load_surface(path: str | None = None) -> dict:
    path = Path(path) if path else DEFAULT_SURFACE
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get('schema') != SURFACE_SCHEMA:
        raise ValueError(f'{path} is not a serving prompt surface')
    current = {p['id']: p for p in value['pieces']}
    bank = []
    for piece in value['pieces']:
        bank.append({**piece, 'logical': piece['id'], 'historical': False})
    known = {p['text'] for p in bank}
    for piece in list(value['history']) + list(default_pieces()):
        if piece['text'] in known or not piece['text']:
            continue
        known.add(piece['text'])
        bank.append({**piece, 'logical': logical_id(piece['id'], current), 'historical': True})
    value['_current'] = current
    value['_bank'] = bank
    value['_identity'] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                          'generated_from_commit': value.get('generated_from_commit')}
    return value


def logical_id(piece_id: str, current) -> str:
    """``historical/<logical id>/<suffix>`` -> the logical id (the longest current id it names), else the id."""
    if not piece_id.startswith('historical/'):
        return piece_id
    rest = piece_id[len('historical/'):]
    return max((key for key in current if rest.startswith(key + '/')), key=len, default=rest.rsplit('/', 1)[0])


def surface_identity(surface: dict) -> dict:
    return dict(surface['_identity'])


def call_surface(found) -> dict | None:
    """The tool-loop call surface named by the generic pieces of a recorded system prompt (None: not a tool loop)."""
    logical = [p['logical'] for _, _, p in found if p['scope'] == 'generic']
    interpreters = [lid for lid in logical if lid in INTERPRETERS]
    if not interpreters:
        return None
    depth = interpreters[0] == 'interpreter/depth-limit'
    folder, file_tools = None, 'all'
    for lid in logical:
        if lid.startswith('directory-reducer/') and lid != 'directory-reducer/declaration':
            parts = lid.split('/')
            file_tools = parts[1]
            folder = 'read' if 'read-only' in parts else 'write'
            depth = depth or parts[-1] == 'depth-limit'
            break
    return {'depthLimited': depth, 'ownCode': 'function-tools' in logical, 'folder': folder,
            'fileTools': file_tools}


def composed_prompt(surface: dict, call: dict, entry: str = 'collector') -> str:
    """ts-host agent systemPrompt() for a call surface and entry point, without the program's guidance."""
    current = surface['_current']
    text = surface['bases'][entry]['depth_limited' if call['depthLimited'] else 'default']
    if call['ownCode']:
        text += current['function-tools']['text']
    if call['folder']:
        key = 'directory-reducer/' + call['fileTools']
        if call['folder'] == 'read':
            key += '/read-only'
        if call['depthLimited']:
            key += '/depth-limit'
        if key not in current:
            raise ValueError(f'no current directory-reducer prompt {key}')
        text += current[key]['text']
    return text


def surface_tools(surface: dict, call: dict, recorded_tools) -> tuple[list, str]:
    key = json.dumps({'depthLimited': call['depthLimited'], 'ownCode': call['ownCode'],
                      'folder': call['folder'], 'fileTools': call['fileTools'] if call['folder'] else 'all'},
                     separators=(',', ':'))
    names = surface['surfaces'].get(key)
    if names is None:
        raise ValueError(f'no current tool surface {key}')
    bucket = surface['tools']['depth_limited' if call['depthLimited'] else 'default']
    recorded = {}
    for item in recorded_tools or []:
        fn = item.get('function', item) if isinstance(item, dict) else {}
        if isinstance(fn.get('name'), str):
            recorded[fn['name']] = fn
    tools, schema = [], 'untyped'
    for name in names:
        definition = copy.deepcopy(bucket[name])
        if name == 'return_result' and 'return_result' in recorded:
            value = ((recorded['return_result'].get('parameters') or {}).get('properties') or {}).get('value')
            if isinstance(value, dict):
                properties = definition['function']['parameters']['properties']
                description = properties['value']['description']
                properties['value'] = {**{k: v for k, v in value.items() if k != 'description'},
                                       'description': description}
                schema = 'recorded'
        tools.append(definition)
    return tools, schema


class _ReturnSchemas:
    """ts-host's own return_result value schema (``schemaOf`` of the declared result type) for a call opening, from a
    persistent ``ts-host/scripts/return-schemas.mjs`` child (one per process). ``NATLANG_TS_DIST`` names the ts-host
    build (default ts-host/dist), ``NATLANG_NODE`` the node binary (default: node on PATH)."""
    def __init__(self):
        self.child, self.cache = None, {}

    def __call__(self, user: str):
        key = user.split('\n\nInstructions:', 1)[0]
        if key not in self.cache:
            import os
            import shutil
            import subprocess
            if self.child is None or self.child.poll() is not None:
                repo = Path(__file__).resolve().parents[4]
                node = os.environ.get('NATLANG_NODE') or shutil.which('node')
                if not node:
                    raise RuntimeError('return schemas need node (NATLANG_NODE or PATH)')
                dist = os.environ.get('NATLANG_TS_DIST') or str(repo / 'ts-host' / 'dist')
                self.child = subprocess.Popen([node, str(repo / 'ts-host/scripts/return-schemas.mjs'), dist],
                                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
            self.child.stdin.write(json.dumps({'user': key}) + '\n')
            self.child.stdin.flush()
            line = self.child.stdout.readline()
            if not line:
                raise RuntimeError('return-schemas child exited')
            answer = json.loads(line)
            self.cache[key] = answer.get('schema'), answer.get('error')
        return self.cache[key]


RETURN_SCHEMAS = _ReturnSchemas()


def _called_tools(messages) -> set[str]:
    names = set()
    for message in messages:
        for call in (message.get('tool_calls') or []) if isinstance(message, dict) else []:
            fn = call.get('function', call) if isinstance(call, dict) else {}
            if isinstance(fn.get('name'), str):
                names.add(fn['name'])
    return names


def serving_messages(messages, tools, *, target=None, surface: dict | None = None, entry: str = 'collector'):
    """``(messages, tools, report)``: the current serving request of ``entry`` for a recorded call (module docstring)."""
    if entry not in ENTRY_POINTS:
        raise ValueError(f'unknown serving entry point {entry!r} (declared: {ENTRY_POINTS})')
    surface = surface or load_surface()
    report = {'surface': surface_identity(surface), 'entry': entry}
    if not messages or not isinstance(messages[0], dict) or messages[0].get('role') != 'system' \
            or not isinstance(messages[0].get('content'), str):
        return messages, tools, {**report, 'mode': 'recorded', 'reason': 'no plain-text system turn'}
    text = messages[0]['content']
    found = find_pieces(text, surface['_bank'])
    call = call_surface(found)
    if call is None:
        return messages, tools, {**report, 'mode': 'recorded', 'reason': 'not a tool-loop call'}
    new_tools, return_schema = surface_tools(surface, call, tools)
    if return_schema == 'untyped':
        # Recorded without tool schemas: the value schema ts-host derives from the call's declared result type.
        user = next((m.get('content') for m in messages if isinstance(m, dict) and m.get('role') == 'user'), None)
        try:
            schema, error = RETURN_SCHEMAS(user) if isinstance(user, str) else (None, 'no call opening')
        except (OSError, RuntimeError, ValueError) as exc:
            schema, error = None, f'{type(exc).__name__}: {exc}'
        if schema is None:
            return messages, tools, {**report, 'mode': 'recorded', 'reason': 'return schema unavailable',
                                     'detail': str(error)[:200]}
        value = new_tools[-1]['function']['parameters']['properties']['value']
        new_tools[-1]['function']['parameters']['properties']['value'] = {**schema, 'description': value['description']}
        return_schema = 'derived'
    offered = {t['function']['name'] for t in new_tools}
    missing = sorted(_called_tools(list(messages) + ([target] if isinstance(target, dict) else [])) - offered)
    if missing:
        return messages, tools, {**report, 'mode': 'recorded', 'reason': 'calls tools the surface lacks',
                                 'tools': missing}
    out, cursor, placed = '', 0, False
    replaced, removed, legacy = [], [], []
    for start, end, piece in found:
        if piece['scope'] != 'generic' or not (composed_piece(piece['logical']) or piece['logical'] in TEACHER_ONLY):
            if piece['historical'] and piece['logical'] not in surface['_current']:
                legacy.append(piece['logical'])
            continue  # program declarations and other runtime notes stay verbatim
        out += text[cursor:start]
        if not placed and composed_piece(piece['logical']):
            out += composed_prompt(surface, call, entry)
            placed = True
            replaced.append(piece['logical'])
        else:
            removed.append(piece['logical'])
        cursor = end
    out += text[cursor:]
    report.update({'mode': system_prompt_mode(entry), 'call_surface': call, 'return_schema': return_schema,
                   'replaced': replaced, 'removed': removed, 'legacy_program_pieces': sorted(set(legacy)),
                   'recorded_system_sha256': hashlib.sha256(text.encode()).hexdigest(),
                   'serving_system_sha256': hashlib.sha256(out.encode()).hexdigest()})
    return [{**messages[0], 'content': out}, *messages[1:]], new_tools, report


def program_header_messages(messages, *, surface: dict | None = None):
    """ts-host's program-header rendering (``renderSystemPrompt``) of a rebuilt serving request: generic runtime prose
    removed with the current and historical piece table, program declarations kept; an empty header has no system
    turn. Tool schemas stay out of the prompt but still type the calls (``native_gold_packet``), as ts-host parses
    header-mode calls against out-of-band schemas."""
    from .prompt_rendering import apply_prompt_rendering
    surface = surface or load_surface()
    rendered, _ = apply_prompt_rendering(messages, None, 'program-header', surface['_bank'])
    return rendered
