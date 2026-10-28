#!/usr/bin/env python3
"""Build a prioritized joint-SFT mixture of native graphs from idiom scores (LFM3 joint SFT, LFM3_RECIPE_20261028.md).

Inputs (a declared config JSON, recorded in the output with every file's sha256):
  - ``renders``: native graph views of verified student renders (``<render>/native-graphs/<view>/{train,test}``, views
    program-header and whole-trajectory, from student_rerender.py), each with the idiom score files of its sources;
  - ``teacher_graphs``: verified teacher-package graphs (rows joined to scores by teacher_source row id + turn index);
  - ``communication``: genuine producer/consumer graph files (native-graphs or semantic-pairs), included completely;
  - ``scores``: idiom_scores.py outputs (one row per decision: category, positive, signals);
  - ``allocation``: the recipe's category shares (delegation 35, exact 20, iteration 15, scope 15, broad 15) and the
    communication share; ``held_packets``: protected held/dev case files whose source groups never train.

Selection (all deterministic):
  1. A whole-trajectory document is a candidate when every decision in it is a positive target; its category is the
     rarest category among its decisions. A program-header document is a candidate for a positive decision that no
     candidate whole-trajectory document covers. Decisions are taken once across all renders (the first render in the
     config order that has them); exact duplicate graphs (same token ids and weights) are kept once.
  2. Train rows sharing a source group with any held row of the inputs or with the protected packets are dropped.
  3. Per category, the target row count is the delegation candidates times the allocation ratio (never more than the
     category has); rows are taken round-robin over task families, each family ordered by fit (idiom signal strength)
     and then by sha256 of the row id. Communication rows are all taken.
  4. Each category is written as n equal cohort files (n proportional to its share: 5% per cohort), so the shared
     trainer's own-outputs/1 ``family_balance`` (key ``source-cohort``) gives each category its share of the loss in
     expectation; ``max_scale`` bounds the upweighting of a category with fewer rows than its share needs.
Outputs: OUT/cohorts/*.jsonl.zst, OUT/held/*.jsonl.zst, OUT/train-sources.json and OUT/held-sources.json
(natlang.native-graph-sources/1), OUT/mixture.json (composition: rows, decisions, supervised own-output reply/reasoning
tokens and processed tokens per category and cohort; selection omissions by reason), then the CE policy is built by
build_native_ce_supervision_policy.py from those manifests (separate step, see --policy).

  build_idiom_mixture.py --config CONFIG.json --out OUT
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
sys.path.insert(0, str(REPO / 'scripts' / 'neuralese_data'))
from natlang_neuralese.common.hashing import sha256_file_hex  # noqa: E402

SCHEMA = 'natlang.idiom-mixture/1'
ORDER = ('iteration', 'delegation', 'exact', 'scope', 'broad')  # rarest first: a document's category
FIT = {'per_item_nl_promise_all': 2., 'per_item_nl_for_of': 2., 'iterate': 2., 'nl_awaited': 1., 'sort': 1.,
       'filter': .5, 'exact_ops': .5, 'typed_child_answer': 1., 'scope_direct': .5, 'finish': .25}


def iter_lines(path: Path):
    proc = subprocess.Popen(['zstd', '-dcq', '--long=31', str(path)], stdout=subprocess.PIPE) \
        if path.suffix == '.zst' else None
    stream = proc.stdout if proc else open(path, 'rb')
    for number, raw in enumerate(stream):
        if raw.strip():
            yield number, raw
    if proc and proc.wait():
        raise RuntimeError(f'zstd failed on {path}')


class Writer:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = path.with_name(path.name + '.tmp')
        self.raw = open(self.tmp, 'wb')
        self.proc = subprocess.Popen(['zstd', '-q', '-5', '-T2', '-c'], stdin=subprocess.PIPE, stdout=self.raw)
        self.rows = 0

    def write(self, line: bytes):
        self.proc.stdin.write(line if line.endswith(b'\n') else line + b'\n')
        self.rows += 1

    def close(self) -> dict:
        self.proc.stdin.close()
        if self.proc.wait():
            raise RuntimeError(f'zstd failed for {self.path}')
        self.raw.close()
        self.tmp.replace(self.path)
        return {'path': str(self.path), 'rows': self.rows, 'bytes': self.path.stat().st_size,
                'sha256': sha256_file_hex(self.path)}


def load_scores(files):
    """{decision id: (category, positive, fit, family)} and the exclusion census."""
    scores, census = {}, collections.Counter()
    for path in files:
        for _, raw in iter_lines(Path(path)):
            row = json.loads(raw)
            fit = sum(weight for key, weight in FIT.items() if row['scores'].get(key))
            family = row.get('curriculum_family') or row.get('family') or 'unknown'
            previous = scores.get(row['id'])
            if previous is not None and not previous[1]:
                continue  # the same decision scored in two sources: one non-positive reading excludes it
            # Scores written before idiom_scores stopped filtering on row approval: the render's own admission
            # (row approval or registry admission) decides that, so that reason alone does not exclude.
            positive = not [r for r in row['exclusions'] if r != 'decision not approved']
            scores[row['id']] = (row['category'], positive, fit, family)
            census[f'{row["category"]}:{"positive" if positive else "excluded"}'] += 1
            for reason in row['exclusions']:
                if reason != 'decision not approved':
                    census['exclusion:' + reason[:60]] += 1
    return scores, census


def held_groups_of(config) -> tuple[set, dict]:
    groups, sources = set(), {}
    for packet in config.get('held_packets', []):
        n = 0
        for _, raw in iter_lines(REPO / packet):
            case = json.loads(raw)
            groups.update(case.get('source_groups') or [])
            n += 1
        sources[packet] = {'sha256': sha256_file_hex(REPO / packet), 'cases': n}
    return groups, sources


def graph_files(render: str, split: str):
    for view in ('whole-trajectory', 'program-header'):  # whole trajectories first: they cover chains
        path = REPO / render / 'native-graphs' / view / split / 'native-graphs.jsonl.zst'
        if path.exists():
            yield view, path


def content_key(row) -> str:
    digest = hashlib.sha256()
    for node in row['nodes']:
        digest.update(json.dumps([node['token_ids'], node['target_weights']], separators=(',', ':')).encode())
    return digest.hexdigest()


def candidates(config, scores, held):
    """Pass 1: candidate train rows [(category, fit, family, key, file, line, decisions)] and omission census."""
    out, omitted, covered, seen = [], collections.Counter(), set(), set()
    inputs = []
    for render in config['renders']:
        for view, path in graph_files(render, 'train'):
            inputs.append(('render', view, path))
    for path in config.get('teacher_graphs', []):
        inputs.append(('teacher', 'teacher-turn', REPO / path))
    for kind, view, path in inputs:
        for number, raw in iter_lines(path):
            row = json.loads(raw)
            if row.get('split') != 'train':
                omitted['not train split'] += 1
                continue
            if set(row.get('source_groups') or ()) & held:
                omitted['shares a held source group'] += 1
                continue
            conversion = row.get('conversion') or {}
            if kind == 'teacher':
                source = conversion.get('teacher_source') or {}
                decisions = [f'{source.get("source_row_id")}#{source.get("teacher_turn_index")}']
            else:
                decisions = list(conversion.get('source_record_ids') or [])
            if not decisions:
                omitted['no source decision ids'] += 1
                continue
            known = [scores.get(d) for d in decisions]
            if any(k is None for k in known):
                omitted[f'{view}: decision without an idiom score'] += 1
                continue
            if not all(k[1] for k in known):
                omitted[f'{view}: holds a non-positive decision'] += 1
                continue
            if any(d in covered for d in decisions):
                omitted[f'{view}: decision already taken'] += 1
                continue
            key = content_key(row)
            if key in seen:
                omitted['exact duplicate graph'] += 1
                continue
            seen.add(key)
            covered.update(decisions)
            category = min((k[0] for k in known), key=ORDER.index)
            fit = sum(k[2] for k in known) / len(known)
            family = collections.Counter(k[3] for k in known).most_common(1)[0][0]
            out.append((category, fit, family, row['id'], str(path), number, len(decisions)))
    return out, omitted


def select(cands, allocation, *, anchor='delegation'):
    by = collections.defaultdict(list)
    for item in cands:
        by[item[0]].append(item)
    base = len(by.get(anchor, [])) / allocation[anchor] if by.get(anchor) else max(
        len(v) / allocation[c] for c, v in by.items())
    chosen, targets = [], {}
    for category in ORDER:
        rows = by.get(category, [])
        target = min(len(rows), round(base * allocation[category]))
        targets[category] = {'available': len(rows), 'target': round(base * allocation[category]), 'taken': target}
        families = collections.defaultdict(list)
        for item in rows:
            families[item[2]].append(item)
        for family in families.values():
            family.sort(key=lambda item: (-item[1], hashlib.sha256(item[3].encode()).hexdigest()))
        queues = sorted(families.values(), key=lambda f: (-len(f), f[0][2]))
        taken, depth = [], 0
        while len(taken) < target:
            progressed = False
            for family in queues:
                if depth < len(family) and len(taken) < target:
                    taken.append(family[depth])
                    progressed = True
            if not progressed:
                break
            depth += 1
        chosen.extend(taken)
    return chosen, targets


def objective(config):
    from transformers import AutoTokenizer
    from natlang_neuralese.train.own_output_supervision import resolve, role_tokens_from_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['objective_tokenizer'], local_files_only=True,
                                              trust_remote_code=False)
    template = Path(config['objective_tokenizer']) / 'chat_template.jinja'
    return resolve({'name': 'own-outputs/1', 'reasoning_share': config['reasoning_share'],
                    'role_tokens': role_tokens_from_tokenizer(
                        tokenizer, tokenizer_fingerprint=config['expanded_tokenizer_fingerprint'],
                        chat_template=template.read_text() if template.exists() else None)})


def own_tokens(row, resolved) -> tuple[int, int, int]:
    """(reply, reasoning, processed) token counts of a graph under the own-outputs objective."""
    from natlang_neuralese.train.own_output_supervision import graph_weights, REPLY, REASONING
    weights, categories = graph_weights(row, resolved)
    reply = sum(1 for w, c in zip(weights, categories) if w > 0 and c == REPLY)
    reasoning = sum(1 for w, c in zip(weights, categories) if w > 0 and c == REASONING)
    return reply, reasoning, sum(len(node['token_ids']) for node in row['nodes'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    config = json.loads(a.config.read_text())
    out = a.out
    if (out / 'mixture.json').exists():
        raise SystemExit(f'{out} already holds a mixture; write a new one')
    allocation = {k: float(v) for k, v in config['allocation'].items()}
    natlang = {k: allocation[k] for k in ORDER}
    scale = sum(natlang.values())
    natlang = {k: v / scale for k, v in natlang.items()}
    per_cohort = config.get('cohort_share', 0.05)
    comm_share = allocation.get('communication', 0.) / 100.
    cohorts = {c: max(1, round((1 - comm_share) * natlang[c] / per_cohort)) for c in ORDER}
    comm_cohorts = round(comm_share / per_cohort) if config.get('communication') else 0

    scores, score_census = load_scores(config['scores'])
    held, packet_sources = held_groups_of(config)
    # Held rows of every input supply held groups too (source-group-disjoint custody across corpora).
    held_rows = []
    for render in config['renders']:
        for view, path in graph_files(render, 'test'):
            held_rows.append(path)
            for _, raw in iter_lines(path):
                held.update(json.loads(raw).get('source_groups') or [])
    for item in config.get('teacher_graphs_test', []):
        held_rows.append(REPO / item)
        for _, raw in iter_lines(REPO / item):
            held.update(json.loads(raw).get('source_groups') or [])

    cands, omitted = candidates(config, scores, held)
    chosen, targets = select(cands, natlang)
    resolved = objective(config)

    # Pass 2: write the chosen rows into their category cohorts, counting own-output tokens.
    assign = {}
    for category in ORDER:
        rows = sorted((c for c in chosen if c[0] == category), key=lambda c: hashlib.sha256(c[3].encode()).hexdigest())
        for index, item in enumerate(rows):
            assign[(item[4], item[5])] = (category, index % cohorts[category], item[6])
    writers = {(c, k): Writer(out / 'cohorts' / f'{c}-{k:02d}.jsonl.zst') for c in ORDER for k in range(cohorts[c])}
    composition = {c: collections.Counter() for c in ORDER}
    families = {c: collections.Counter() for c in ORDER}
    by_file = collections.defaultdict(dict)
    for (path, number), value in assign.items():
        by_file[path][number] = value
    for path, lines in by_file.items():
        for number, raw in iter_lines(Path(path)):
            if number not in lines:
                continue
            category, cohort, decisions = lines[number]
            row = json.loads(raw)
            reply, reasoning, processed = own_tokens(row, resolved)
            writers[(category, cohort)].write(raw)
            composition[category].update(rows=1, decisions=decisions, reply_tokens=reply,
                                         reasoning_tokens=reasoning, supervised_tokens=reply + reasoning,
                                         processed_tokens=processed)
    outputs = {key: writer.close() for key, writer in writers.items()}

    # Communication graphs: every row, round-robin into their cohorts (bytes unchanged).
    comm = collections.Counter()
    comm_outputs = []
    if comm_cohorts:
        comm_writers = [Writer(out / 'cohorts' / f'communication-{k:02d}.jsonl.zst') for k in range(comm_cohorts)]
        kinds = set()
        index = 0
        for item in config['communication']:
            kinds.add(item['kind'])
            for _, raw in iter_lines(REPO / item['path']):
                row = json.loads(raw)
                if row.get('split') != 'train':
                    comm['not train split'] += 1
                    continue
                comm_writers[index % comm_cohorts].write(raw)
                index += 1
                graphs = [v['graph'] for v in row['models'].values()] if 'models' in row else [row]
                for graph in graphs:
                    reply, reasoning, processed = own_tokens(graph, resolved)
                    comm.update(rows=1, graphs=1, reply_tokens=reply, reasoning_tokens=reasoning,
                                supervised_tokens=reply + reasoning, processed_tokens=processed,
                                nodes=len(graph['nodes']))
        if len(kinds) != 1:
            raise ValueError('communication inputs mix native-graphs and semantic-pairs; give them separately')
        comm_kind = kinds.pop()
        comm_outputs = [w.close() for w in comm_writers]

    # Held: test rows of the selected inputs, unchanged (evaluation only, never training exposure).
    held_writer = Writer(out / 'held' / 'native-graphs.jsonl.zst')
    held_seen = set()
    for path in held_rows:
        for _, raw in iter_lines(path):
            row = json.loads(raw)
            key = content_key(row)
            if row.get('split') == 'test' and key not in held_seen:
                held_seen.add(key)
                held_writer.write(raw)
    held_output = held_writer.close()

    train_sources = [{'kind': 'native-graphs', 'path': str(Path(o['path']).relative_to(out)), 'sha256': o['sha256']}
                     for o in outputs.values() if o['rows']]
    train_sources += [{'kind': comm_kind, 'path': str(Path(o['path']).relative_to(out)), 'sha256': o['sha256']}
                      for o in comm_outputs if o['rows']]
    (out / 'train-sources.json').write_text(json.dumps({'schema': 'natlang.native-graph-sources/1',
                                                        'sources': train_sources}, indent=1) + '\n')
    (out / 'held-sources.json').write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': 'held/native-graphs.jsonl.zst', 'sha256': held_output['sha256']}]},
        indent=1) + '\n')

    cohort_counts = {**{c: sum(1 for (cc, _), o in outputs.items() if cc == c and o['rows']) for c in ORDER},
                     **({'communication': len([o for o in comm_outputs if o['rows']])} if comm_outputs else {})}
    total_cohorts = sum(cohort_counts.values())
    report = {
        'schema': SCHEMA, 'config': {'path': str(a.config), 'sha256': sha256_file_hex(a.config), 'value': config},
        'code': {name: sha256_file_hex(REPO / name) for name in (
            'scripts/neuralese_data/build_idiom_mixture.py', 'scripts/neuralese_data/idiom_scores.py')},
        'allocation_declared': allocation,
        'cohort_weights': {c: n / total_cohorts for c, n in cohort_counts.items()},
        'cohorts': cohort_counts,
        'selection_targets': targets,
        'composition': {c: dict(v) for c, v in composition.items()},
        'communication': dict(comm),
        'score_census': dict(score_census),
        'selection_omissions': dict(omitted),
        'held': {'rows': held_output['rows'], 'held_source_groups': len(held), 'packets': packet_sources},
        'train_sources': {'path': 'train-sources.json', 'sha256': sha256_file_hex(out / 'train-sources.json')},
        'held_sources': {'path': 'held-sources.json', 'sha256': sha256_file_hex(out / 'held-sources.json')},
        'outputs': [*outputs.values(), *comm_outputs, held_output],
    }
    (out / 'mixture.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
    print(json.dumps({k: report[k] for k in ('cohort_weights', 'composition', 'communication', 'selection_targets')},
                     indent=1))


if __name__ == '__main__':
    main()
