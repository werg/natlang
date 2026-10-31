"""Shared case construction for the broad external RL task mix (plans/neuralese/BROAD_RL_TASKS_20261031.md).

Every external item becomes an ordinary `natlang.program/2` curriculum case: a typed root `.nl` function whose
documents, tables, databases and services are its inputs, a declared answer type, an oracle in `semantics.oracle`
(checked by the collector's shared `checkOracle`), and a reference that the collector replays. The `curriculum`
block is oracle metadata the model never sees; `curriculum.external` keeps source/family/shape/instruction
identities for diversity measurement and the per-source qualification checks (meaningful wrong answers that
must be rejected, alternative gold forms that must be accepted).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_hex  # noqa: E402
from natlang_neuralese.common.paths import resolve, resolve_str  # noqa: E402

CURRICULUM_VERSION = 'natlang.inline_curriculum/1'
BUILDER = 'natlang.broad_external_rl_tasks/1'
SPLITS = ('train', 'dev', 'confirmation')
# Logical locations (common.paths roots): downloaded sources, the raw archive, the build's asset copy and the
# registered task source's assets (the default a case's service reads; NATLANG_RL_TASK_ASSETS overrides it).
SOURCES_DIR = resolve_str('data_nvme', 'external-rl-sources')
ARCHIVE_RAW = resolve('archive', 'raw')
BUILD_ASSETS = resolve_str('data_nvme', 'broad-rl-tasks-20261031', 'assets')
TASK_ASSETS = resolve_str('repo', 'runs', 'natlang-rl-broad-external-20261031', 'assets')


def text_digest(text: str) -> str:
    return sha256_hex(text.encode())


def slug(text: str, limit: int = 60) -> str:
    return re.sub(r'[^a-z0-9]+', '-', str(text).lower()).strip('-')[:limit] or 'x'


def pick(options, key: str):
    """Deterministic choice among phrasings, keyed by the split unit so related items share one phrasing."""
    return options[int(text_digest(key)[:8], 16) % len(options)]


def unit_split(unit: str, upstream_split: str, *, salt: str = 'broad-rl-20261031') -> str:
    """Split of one split unit (document/db/repo group). Upstream evaluation splits become our held dev and
    confirmation halves; upstream train units stay train except a 3% dev/confirmation carve-out for sources
    without an upstream evaluation split."""
    bucket = int(text_digest(salt + '|' + unit)[:8], 16) % 100
    if upstream_split in ('validation', 'dev', 'test', 'eval'):
        return 'dev' if bucket < 50 else 'confirmation'
    if upstream_split == 'train-carve':
        return 'train' if bucket < 94 else ('dev' if bucket < 97 else 'confirmation')
    return 'train'


def nl_file(args: dict, returns: str, body: str, kind: str | None = None) -> str:
    lines = ['---']
    if kind:
        lines.append(f'kind: {kind}')
    if args:
        lines.append('args:')
        for name, type_text in args.items():
            lines.append(f'  {name}: {json.dumps(type_text)}')
    else:
        lines.append('args: {}')
    lines.append(f'returns: {json.dumps(returns)}')
    lines.append('---')
    return '\n'.join(lines) + '\n' + body.strip() + '\n'


def make_case(*, source: str, family: str, shape: str, upstream: str, upstream_id: str, upstream_split: str,
              unit: str, groups: list[str], root_name: str, args: dict, returns: str, instruction: str,
              inputs: dict, expected, oracle, level: int, license_: str, license_note: str = '',
              services: dict | None = None, extra_files: dict | None = None, slice_: str = 'nested_scoped',
              domain: str = 'other', reference_root: list | None = None, wrong: list | None = None,
              aliases: list | None = None, neuralese_room: str = '', revision: str = '', difficulty=None,
              extra_semantics: dict | None = None, split: str | None = None, instruction_id: str | None = None,
              neuralese: dict | None = None, framing: str | None = None):
    """`neuralese`: an optional Neuralese framing of the same task: {helper, args, returns, instruction, root_note}.
    The case then uses either the plain framing or a callable folder helper whose declared result is Neuralese<...>
    (per-document/per-item views fanned in to the parent's decision), chosen per split unit unless `framing` is
    given; `curriculum.external.framing` records which. The reward is the same outcome oracle either way."""
    split = split or unit_split(unit, upstream_split)
    if neuralese and framing is None:
        framing = pick(['plain', 'neuralese-helper'], unit + '|framing')
    framing = framing or 'plain'
    if framing == 'neuralese-helper':
        instruction = instruction.rstrip() + '\n\n' + neuralese['root_note'].strip()
        extra_files = {**(extra_files or {}), f'{root_name}/{neuralese["helper"]}.nl':
                       nl_file(neuralese['args'], neuralese['returns'], neuralese['instruction'])}
    case_id = f'ext:{source}:{slug(shape, 40)}:{slug(upstream_id, 80)}:{text_digest(upstream_id + shape + framing)[:8]}'
    root = f'{root_name}.nl'
    files = {root: nl_file(args, returns, instruction)}
    files.update(extra_files or {})
    semantics = {'root': root, 'files': files, 'inputs': inputs, 'expected': expected, 'oracle': oracle}
    if services:
        semantics['services'] = services
    semantics.update(extra_semantics or {})
    ref = reference_root or [['return_result', {'status': 'success', 'value': expected}]]
    groups = sorted(set([f'ext-unit:{source}:{unit}', *groups]))
    family_id = f'ext_{source}_{family}'
    return {
        'version': 'natlang.program/2', 'id': case_id, 'kind': 'lambda_source', 'family': family_id,
        'source': f'external-rl:{source}', 'split': split,
        'source_ids': [f'{upstream}:{upstream_id}'], 'source_groups': groups,
        'source_revisions': [x for x in (BUILDER, f'{upstream}@{revision}' if revision else upstream) if x],
        'license': license_, 'gold_sources': ['upstream-dataset-gold'],
        'generation': {'generator': BUILDER, 'teacher_generation': False,
                       'method': 'external dataset item converted to a typed natlang root function; gold from the '
                                 'upstream annotation or execution; no sampled trajectory',
                       **({'license_note': license_note} if license_note else {})},
        'curriculum': {
            'version': CURRICULUM_VERSION, 'family': family_id, 'family_version': 1, 'shape': shape,
            'variant': 'base', 'pair_group': None, 'split_group': f'ext-unit:{source}:{unit}', 'slice': slice_,
            'domain': domain, 'mode': 'single_call', 'inline': 'optional',
            'evidence': {'world': [], 'retrieved': [], 'background': []}, 'assumptions': [], 'decisive': [],
            'plausible_actions': [], 'minimum_sequence': [], 'reference': {'root': ref},
            'rl': {'level': int(level), 'family': family_id, 'basis': 'external-dataset-gold-reference-verified',
                   **({'difficulty': difficulty} if difficulty is not None else {})},
            'external': {'source': source, 'upstream': upstream, 'upstream_id': upstream_id,
                         'upstream_split': upstream_split, 'unit': unit, 'shape': shape,
                         'instruction_sha256': text_digest(instruction)[:16],
                         'instruction_id': instruction_id or text_digest(instruction)[:12],
                         'neuralese_room': neuralese_room, 'framing': framing,
                         'checks': {'wrong_answers': wrong or [], 'alias_answers': aliases or []}},
        },
        'semantics': semantics,
    }


def jsonl_write(path, rows):
    with open(path, 'w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def iter_jsonl(path):
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def distinct_wrong(gold, pool, n=2, key=lambda v: json.dumps(v, sort_keys=True).lower()):
    """Meaningful wrong answers: other items' gold values of the same source that differ from this gold."""
    gold_key, seen, out = key(gold), set(), []
    for candidate in pool:
        k = key(candidate)
        if k == gold_key or k in seen:
            continue
        seen.add(k)
        out.append(candidate)
        if len(out) >= n:
            break
    return out
