#!/usr/bin/env python3
"""Create bounded, corrected source variants from already verified train cases.

This intentionally reads only the verified cases in the pinned broad task build;
it does not rescan or resample upstream corpora. Exact source groups and task
inputs are retained, and the result must pass verify_source.sh again.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.rl_tasks.sources.sql import sql_profile


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sql_rows(source: str, path: Path, cap_per_db: int):
    by_db = defaultdict(list)
    for row in rows(path):
        if row.get('split') != 'train':
            continue
        ext = row['curriculum']['external']
        sql = ext.get('gold_sql', '')
        profile = sql_profile(sql)
        # Require multiple SQL operations and real composition. Keep one set of
        # structurally distinct queries per upstream database.
        if not profile['compositional'] or profile['score'] < 3:
            continue
        db = next((g.split(':', 2)[2] for g in row['source_groups'] if g.startswith(f'sql-db:{source}:')), None)
        if db:
            by_db[db].append((profile, row))
    out = []
    for db, candidates in sorted(by_db.items()):
        seen = set()
        picked = []
        for profile, row in sorted(candidates, key=lambda x: (-x[0]['score'], x[1]['id'])):
            signature = (min(profile['joins'], 3), min(profile['nested_selects'], 2), profile['group_by'],
                         profile['having'], profile['set_operation'], profile['order_by'], profile['aggregate'],
                         min(profile['where_predicates'], 3))
            if signature in seen:
                continue
            seen.add(signature)
            picked.append((profile, row))
            if len(picked) >= cap_per_db:
                break
        for profile, original in picked:
            row = json.loads(json.dumps(original))
            old_id = row['id']
            row['id'] = f'{old_id}:compositional-v1'
            row['source'] = f'external-rl:{source}-hard'
            row['source_ids'] = [f'{source}-hard:{row["source_ids"][0]}']
            row['source_revisions'] = list(row.get('source_revisions', [])) + [source + '-hard@verified-case-derived-v1']
            row['family'] = f'ext_{source}_compositional_sql'
            row['curriculum']['family'] = row['family']
            row['curriculum']['shape'] = f'{source}-compositional-sql'
            row['curriculum']['variant'] = 'compositional-sql-v1'
            row['curriculum']['rl']['level'] = max(2, min(3, row['curriculum']['rl'].get('level', 2)))
            ext = row['curriculum']['external']
            ext['source'] = f'{source}-hard'
            ext['shape'] = row['curriculum']['shape']
            ext['sql_profile'] = profile
            ext['source_case_id'] = old_id
            row['semantics']['oracle'] = {'level': 'normalized', 'normalization': 'result-set',
                                          **({'ordered': True} if profile['order_by'] else {})}
            if profile['order_by']:
                ext['checks'].pop('alias_answers', None)
            out.append(row)
    return out


def _guided_rows(source: str, path: Path, cap: int):
    eligible = []
    for row in rows(path):
        if row.get('split') != 'train':
            continue
        ext = row['curriculum']['external']
        shape = row['curriculum']['shape']
        if source == 'babi' and shape != 'babi-three-supporting-facts':
            continue
        if source == 'clutrr' and int(re.search(r'(\d+)hops', shape).group(1)) < 4:
            continue
        eligible.append(row)
    eligible.sort(key=lambda r: r['id'])
    out = []
    for original in eligible[:cap]:
        row = json.loads(json.dumps(original))
        old_id = row['id']
        family = f'ext_{source}_guided'
        row['id'] = f'{old_id}:guided-v1'
        row['source'] = f'external-rl:{source}-guided'
        row['source_ids'] = [f'{source}-guided:{row["source_ids"][0]}']
        row['source_revisions'] = list(row.get('source_revisions', [])) + [source + '-guided@verified-case-derived-v1']
        row['family'] = family
        row['curriculum']['family'] = family
        row['curriculum']['shape'] += '-guided'
        row['curriculum']['variant'] = 'complete-evidence-guidance-v1'
        ext = row['curriculum']['external']
        ext['source'] = f'{source}-guided'
        ext['source_case_id'] = old_id
        ext['framing'] = 'neuralese-helper'
        if source == 'babi':
            ext['neuralese_room'] = 'complete numbered event history, temporal reference resolution, and repeated state changes'
            root = 'answer_from_facts.nl'
            helper = 'answer_from_facts/track_query_history.nl'
            row['semantics']['files'][root] = row['semantics']['files'][root].replace(
                'facts is a numbered story, in order. Answer question from it, taking into account that later facts can change earlier ones. Return only the answer word or words (comma-separated when there are several).',
                'Use the entire numbered event history in order. Track state changes for the named entity, preserve fact ids, and answer the exact temporal question (including before/after references) from the decisive earlier event. Return only the answer word or words.') + '\n\nCall `const timeline = await nl(facts, question);` once with the complete list. Check the decisive fact ids and the requested time; never truncate or substitute the final state for an earlier state.\n'
            row['semantics']['files'][helper] = ('---\nargs:\n  facts: "string[]"\n  question: "string"\nreturns: "Neuralese<string>"\n---\n'
                'Read the complete numbered history and question. Keep an ordered timeline for relevant entities and transitions, preserving fact ids. For a before/after query, state the value at that exact point. Distinguish movement, object possession, and location. Do not omit repeated changes or infer unstated events.\n')
            ext['neuralese_room'] = 'One awaited helper call tracks the complete ordered history; parent checks temporal target and decisive fact ids.'
        else:
            ext['neuralese_room'] = 'Per-sentence direct-edge extraction followed by an explicit target-to-anchor path composition.'
            root = 'find_relation.nl'
            helper = 'find_relation/extract_relation_edge.nl'
            row['semantics']['files'][root] = row['semantics']['files'][root] + ('\n\nSplit the story into its stated sentences. For every sentence call `const edge = await nl(sentence);`, retain each direct edge, and compose only the path from target to anchor. Check direction and do not infer marriage or an unstated parent edge. Return one relation word.\n')
            row['semantics']['files'][helper] = ('---\nargs:\n  sentence: "string"\nreturns: "Neuralese<string>"\n---\n'
                'Extract only a direct family relation stated in this sentence. Return “PERSON_A is PERSON_B’s RELATION”, preserving direction. Do not infer unstated relations.\n')
        out.append(row)
    return out


def build(base: Path, out: Path):
    if out.exists():
        raise FileExistsError(f'{out} already exists; builds are immutable')
    out.mkdir(parents=True)
    result = {}
    for name, cap in [('spider', 4), ('bird', 4)]:
        result[name + '-hard'] = _sql_rows(name, base / name / 'verified/cases.jsonl', cap)
    result['babi-temporal3'] = _guided_rows('babi', base / 'babi/verified/cases.jsonl', 48)
    result['clutrr-guided'] = _guided_rows('clutrr', base / 'clutrr/verified/cases.jsonl', 48)
    for name, items in result.items():
        target = out / name
        target.mkdir()
        (target / 'raw-cases.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in items))
    manifest = {'schema': 'natlang.verified-successor-build/1', 'base_build': str(base),
                'outputs': {k: {'rows': len(v), 'cases_sha256': hashlib.sha256(
                    (out / k / 'raw-cases.jsonl').read_bytes()).hexdigest()} for k, v in result.items()},
                'method': 'bounded transformations over pinned verified train cases only; upstream facts, source groups, and split retained'}
    (out / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    import sys
    print(json.dumps(build(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
