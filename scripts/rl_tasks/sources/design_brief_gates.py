"""Exact behavioral RL cases from the frozen design-brief application gates.

The packet is authored finite task data, not sampled teacher trajectories. Calls
from the same application remain in the same release split.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..common import make_case, resolve_str

PACKET = Path(resolve_str('data_nvme', 'luna-generation-20261026',
                          'design-briefs-pr4-review', 'packet-v3'))
SPLITS = (('train', 'train'), ('dev', 'dev'), ('test', 'confirmation'))


def _rows(packet: Path):
    for filename, split in SPLITS:
        with (packet / f'{filename}.gates.ir.jsonl').open() as stream:
            for line in stream:
                if line.strip():
                    yield split, json.loads(line)


def cases(packet: str | Path | None = None):
    """Convert the 300 immutable gate records to ordinary exact-outcome RL cases."""
    root = Path(packet) if packet else PACKET
    out = []
    for split, row in _rows(root):
        sem = row['semantics']
        original_root = sem['root']
        source_id = row['id']
        app_group = next(g for g in row['source_groups'] if g.startswith('design-briefs:application-'))
        app_num = app_group.rsplit('-', 1)[-1]
        upstream_split = {'train': 'train', 'dev': 'validation', 'confirmation': 'test'}[split]
        # Keep the exact frozen application, root and inputs. The regular RL
        # verifier executes the actual Natlang program and checks this exact gate.
        case = make_case(
            source='design-brief-gates', family=row['family'], shape=source_id.rsplit(':', 1)[-1],
            upstream='design-briefs-100-distinct-applications-20261026-v3',
            upstream_id=source_id, upstream_split=upstream_split,
            unit=app_group, groups=[app_group, row['curriculum']['world_group']],
            root_name='gate', args={}, returns='string', instruction='Execute the application gate using its supplied evidence and policy inputs.',
            inputs=sem['inputs'], expected=sem['expected'], oracle='exact', level=1,
            license_='user-supplied-design; repository-authored-implementation',
            domain='other', slice_='nested_scoped', split=split,
            reference_root=[['return_result', {'status': 'success', 'value': sem['expected']}]],
            revision='|'.join(row['source_revisions']))
        case['id'] = f'ext:design-brief-gates:{source_id}'
        case['source_ids'] = list(row['source_ids']) + [source_id]
        case['source_revisions'] = ['natlang.design-brief-freezer/1', *row['source_revisions']]
        case['semantics']['root'] = original_root
        case['semantics']['files'] = dict(sem['files'])
        case['semantics']['oracle'] = sem['oracle']
        case['generation'].update({
            'teacher_generation': False,
            'method': 'frozen exact behavioral gate and independently authored finite fixture; no sampled trajectory',
            'upstream_teacher_trajectory': False,
        })
        case['gold_sources'] = list(row.get('gold_sources', []))
        case['curriculum']['family'] = f'ext_design_brief_{app_num}'
        case['curriculum']['rl'].update({'family': f'ext_design_brief_{app_num}',
                                         'basis': 'frozen-exact-gate'})
        case['curriculum']['external'].update({
            'application': app_num, 'world_group': row['curriculum']['world_group'],
            'upstream_record_id': source_id,
            'source_record_generation': row['generation'],
            'checks': {'wrong_answers': [], 'alias_answers': []},
        })
        out.append(case)
    ids = [case['id'] for case in out]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate design-brief gate ids')
    # Other finite outcomes from the same gate operation are meaningful negative
    # controls: the verifier must reject a valid label applied to the wrong facts.
    pools = {}
    for case in out:
        key = (case['split'], case['curriculum']['external']['application'], case['semantics']['root'])
        pools.setdefault(key, [])
        if case['semantics']['expected'] not in pools[key]:
            pools[key].append(case['semantics']['expected'])
    for case in out:
        key = (case['split'], case['curriculum']['external']['application'], case['semantics']['root'])
        gold = case['semantics']['expected']
        case['curriculum']['external']['checks']['wrong_answers'] = [v for v in pools[key] if v != gold]
    return out
