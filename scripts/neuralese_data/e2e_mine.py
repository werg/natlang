#!/usr/bin/env python3
"""Mine every native-turn corpus for end-to-end Neuralese producer->consumer pairs and whole chains.

Owner 2026-10-12 (e2e data, "include everything"): every multi-call trajectory of every registered native-turn source
is converted (ts-host neuralese-convert-trajectories.mjs, conversion /15, with --admit-host-results so a child's own
return of a host-computed value is a producer too, marked host_result) and every authenticated producer->consumer
edge (train/trajectories.authenticated_recurrence_context_view, the rule of the v1 cohort) is kept together with its
whole trajectory (the chain). Exclusions: held/protected eval groups, exact duplicates, records that fail conversion
(registered with their reason in omissions.jsonl, never dropped silently). Records are tokenizer-agnostic native
records, so one corpus serves both lines.

Stages (each resumable; run every stage under the memory ledger):
  e2e_mine.py shard SOURCE_NAME FILE [FILE...] --work DIR [--target-mb 150]
      split a source by trajectory into shards (plain or .zst JSONL), so a trajectory's rows stay together.
  e2e_mine.py convert --work DIR [--source NAME]
      convert each unconverted shard and keep the records of trajectories with at least one edge.
  e2e_mine.py assemble --work DIR --id CORPUS_ID [--held-union DIR]
      merge all shard outputs: dedupe, split rules, piece conflicts, chains, per-family statistics.
"""
from __future__ import annotations

import argparse, collections, glob, hashlib, io, json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex  # noqa: E402
from natlang_neuralese.common.paths import resolve_str  # noqa: E402

TS_HOST = Path(os.environ.get('E2E_MINE_TS_HOST', ROOT / 'ts-host'))
LFM_TOKENIZER = resolve_str('repo', 'models/hf/hub/models--LiquidAI--LFM2.5-350M/snapshots/9e6c6ccf47cd318696e137d381a7ded8fe4df09f')


def open_lines(path):
    if str(path).endswith('.zst'):
        proc = subprocess.Popen(['zstd', '-dc', str(path)], stdout=subprocess.PIPE)
        yield from io.TextIOWrapper(proc.stdout, encoding='utf-8')
        proc.wait()
    else:
        with open(path, encoding='utf-8') as f:
            yield from f


def trajectory_of(row):
    ref = row.get('source_ref') or {}
    return ref.get('trajectory_id') or row.get('teacher_trajectory_id') or str(row.get('id', '')).split(':decision')[0]


def family_of(row):
    ir = (row.get('task') or {}).get('program_ir') or {}
    return ir.get('family') or row.get('family') or row.get('task_family') or 'unknown'


def cmd_shard(a):
    out = Path(a.work) / a.name
    if (out / 'sharded.json').exists():
        print('already sharded', out); return
    out.mkdir(parents=True, exist_ok=True)
    size = sum(os.path.getsize(f) * (6 if f.endswith('.zst') else 1) for f in a.files)
    n = max(1, -(-size // (a.target_mb << 20)))
    handles = [open(out / f'shard-{i:04d}.jsonl', 'w', encoding='utf-8') for i in range(n)]
    rows = bad = 0
    for f in a.files:
        for line in open_lines(f):
            if not line.strip(): continue
            try:
                traj = trajectory_of(json.loads(line))
            except ValueError:
                bad += 1; continue
            i = int(hashlib.sha256(traj.encode()).hexdigest()[:8], 16) % n
            handles[i].write(line if line.endswith('\n') else line + '\n'); rows += 1
    for h in handles: h.close()
    json.dump({'source': a.name, 'files': a.files, 'shards': n, 'rows': rows, 'unparseable_lines': bad},
              open(out / 'sharded.json', 'w'), indent=1)
    print(json.dumps({'source': a.name, 'shards': n, 'rows': rows, 'unparseable_lines': bad}))


def target_text(row):
    t = row.get('target') or {}
    parts = [t.get('content') if isinstance(t.get('content'), str) else json.dumps(t.get('content') or '')]
    for call in t.get('tool_calls') or []:
        fn = call.get('function') or {}
        parts.append(fn.get('name', '') + ' ' + (fn.get('arguments') if isinstance(fn.get('arguments'), str) else json.dumps(fn.get('arguments'))))
    return '\n'.join(p for p in parts if p)


def convert_shard(shard: Path, tokenizer):
    from natlang_neuralese.train.trajectories import authenticated_recurrence_context_view
    stem = shard.with_suffix('')
    with tempfile.TemporaryDirectory(dir=shard.parent) as tmp:
        conv, pieces_path, summary = Path(tmp) / 'records.jsonl', Path(tmp) / 'pieces.jsonl', Path(tmp) / 'conversion.json'
        env = dict(os.environ, NODE_OPTIONS=os.environ.get('NODE_OPTIONS', '--max-old-space-size=6144'))
        proc = subprocess.run(['node', 'scripts/neuralese-convert-trajectories.mjs', '--admit-host-results', '--out', str(conv),
                               '--pieces', str(pieces_path), '--summary', str(summary), str(shard)],
                              cwd=TS_HOST, env=env, capture_output=True, text=True)
        omissions = []
        if proc.returncode != 0:
            # The whole shard failed conversion: register every row with the converter's error.
            reason = 'conversion_failed: ' + (proc.stderr.strip().splitlines() or ['?'])[-1][:300]
            for line in open(shard):
                if line.strip():
                    try: rid = json.loads(line).get('id')
                    except ValueError: rid = None
                    omissions.append({'id': rid, 'stage': 'convert', 'reason': reason})
            write_shard(stem, [], {}, [], omissions, {'converted': False, 'error': reason})
            return
        records = [json.loads(l) for l in open(conv) if l.strip()]
        pieces = {}
        for l in open(pieces_path):
            if l.strip():
                p = json.loads(l); pieces[p['name']] = p['text']
        conversion = json.load(open(summary))
    _, review = authenticated_recurrence_context_view(records, pieces)
    for e in review['excluded']:
        omissions.append({'id': e.get('id'), 'stage': 'authenticate', 'reason': e.get('reason')})
    by_id = {r['id']: r for r in records}
    edges = []
    for d in review['decisions']:
        for ref in d.get('references') or []:
            if ref.get('gradient_edge'):
                w = by_id.get(ref.get('writer_record_id'))
                edges.append({'writer_record_id': ref.get('writer_record_id'), 'reader_record_id': d['id'],
                              'write_name': ref.get('name'), 'body_sha256': ref.get('body_sha256'),
                              'kind': (ref.get('name') or '').split(':')[0], 'trajectory': trajectory_of(by_id[d['id']]),
                              'family': family_of(by_id[d['id']]), 'split': by_id[d['id']].get('split'),
                              'writer_split': (w or {}).get('split')})
    chains = {e['trajectory'] for e in edges}
    kept = [r for r in records if trajectory_of(r) in chains]
    used = set()

    def walk(v):
        if isinstance(v, dict):
            if v.get('type') == 'soft' and isinstance(v.get('name'), str): used.add(v['name'])
            for x in v.values(): walk(x)
        elif isinstance(v, list):
            for x in v: walk(x)
    for r in kept: walk(r.get('messages')); walk(r.get('target'))
    stats = {'converted': True, 'rows': len(records), 'kept_records': len(kept), 'edges': len(edges), 'chains': len(chains),
             'conversion_sites': {k: v for k, v in conversion.get('sites', {}).items() if 'child' in k or 'inline' in k or 'soft' in k or 'typed' in k},
             'by_family': {}}
    fam = collections.defaultdict(lambda: {'records': 0, 'edges': 0, 'chains': set(), 'supervised_tokens': 0, 'edge_kinds': collections.Counter()})
    for r in kept:
        f = fam[family_of(r)]; f['records'] += 1; f['chains'].add(trajectory_of(r))
        f['supervised_tokens'] += len(tokenizer(target_text(r), add_special_tokens=False)['input_ids'])
    for e in edges:
        fam[e['family']]['edges'] += 1; fam[e['family']]['edge_kinds'][e['kind']] += 1
    stats['by_family'] = {k: {**v, 'chains': len(v['chains']), 'edge_kinds': dict(v['edge_kinds'])} for k, v in fam.items()}
    write_shard(stem, kept, {n: pieces[n] for n in used if n in pieces}, edges, omissions, stats)


def write_shard(stem: Path, kept, pieces, edges, omissions, stats):
    staged = {}
    for suffix, rows in (('records', kept), ('edges', edges), ('omissions', omissions)):
        p = Path(f'{stem}.{suffix}.jsonl.tmp'); staged[p] = Path(f'{stem}.{suffix}.jsonl')
        with open(p, 'w', encoding='utf-8') as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    p = Path(f'{stem}.pieces.jsonl.tmp'); staged[p] = Path(f'{stem}.pieces.jsonl')
    with open(p, 'w', encoding='utf-8') as f:
        for n, t in sorted(pieces.items()): f.write(json.dumps({'name': n, 'text': t}, ensure_ascii=False) + '\n')
    for src, dst in staged.items(): src.rename(dst)
    json.dump(stats, open(f'{stem}.stats.json', 'w'), indent=1)  # written last: marks the shard done


def cmd_convert(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LFM_TOKENIZER)
    pattern = str(Path(a.work) / (a.source or '*') / 'shard-[0-9][0-9][0-9][0-9].jsonl')
    for shard in sorted(glob.glob(pattern)):
        stem = Path(shard).with_suffix('')
        if Path(f'{stem}.stats.json').exists(): continue
        convert_shard(Path(shard), tok)
        s = json.load(open(f'{stem}.stats.json'))
        print(json.dumps({'shard': shard, **{k: s.get(k) for k in ('rows', 'kept_records', 'edges', 'chains', 'error')}}), flush=True)


def content_key(row):
    """Exact-duplicate key: the record's canonical content without its id."""
    return canonical_json_sha256_hex({k: v for k, v in row.items() if k != 'id'})


def cmd_assemble(a):
    work = Path(a.work)
    out = Path(resolve_str('data_nvme', 'natlang-corpora', a.id))
    staged = out.with_name(out.name + '.building'); staged.mkdir(parents=True, exist_ok=True)
    held_groups = set()
    for rend in ('full', 'program-header', 'whole-trajectory'):
        for path in glob.glob(f'{a.held_union}/{rend}/*/text-data.jsonl')[:1]:
            for line in open(path):
                if '"test"' in line:
                    t = json.loads(line)
                    if t.get('split') == 'test': held_groups.update(t.get('source_groups') or [])
    pieces, piece_conflicts = {}, set()
    shard_stems = sorted(Path(p[:-len('.stats.json')]) for p in glob.glob(str(work / '*' / 'shard-*.stats.json')))
    for stem in shard_stems:
        for line in open(f'{stem}.pieces.jsonl'):
            p = json.loads(line)
            if p['name'] in pieces and pieces[p['name']] != p['text']: piece_conflicts.add(p['name'])
            pieces.setdefault(p['name'], p['text'])
    seen_ids, seen_content = {}, set()
    omissions = open(staged / 'omissions.jsonl', 'w')
    rec_out, edge_out = open(staged / 'records.jsonl', 'w'), open(staged / 'edges.jsonl', 'w')
    chains_out = open(staged / 'chains.jsonl', 'w')
    totals = collections.Counter()
    fam = collections.defaultdict(collections.Counter)
    fam_kinds = collections.defaultdict(collections.Counter)
    by_source = collections.defaultdict(collections.Counter)
    for stem in shard_stems:
        source = stem.parent.name
        for line in open(f'{stem}.omissions.jsonl'):
            o = json.loads(line); o['source'] = source; omissions.write(json.dumps(o) + '\n'); totals['omitted_' + o['stage']] += 1
        records = {}
        for line in open(f'{stem}.records.jsonl'):
            r = json.loads(line); records[r['id']] = r
        edges = [json.loads(l) for l in open(f'{stem}.edges.jsonl')]
        keep_ids, drop = set(), {}
        for rid, r in records.items():
            if rid in seen_ids:
                drop[rid] = 'duplicate_id_of_' + seen_ids[rid]; continue
            c = content_key(r)
            if c in seen_content:
                drop[rid] = 'exact_duplicate_content'; continue
            if any(n in piece_conflicts for n in _soft_names(r)):
                drop[rid] = 'piece_version_conflict'; continue
            keep_ids.add(rid)
        kept_edges = []
        for e in edges:
            w, rd = e['writer_record_id'], e['reader_record_id']
            if w not in keep_ids or rd not in keep_ids: totals['edges_dropped_endpoint'] += 1; continue
            if e['split'] != e['writer_split']: totals['edges_dropped_cross_split'] += 1; continue
            if e['split'] == 'train' and ((set(records[w].get('source_groups') or []) | set(records[rd].get('source_groups') or [])) & held_groups):
                totals['edges_dropped_train_shares_held_group'] += 1; continue
            kept_edges.append(e)
        chains = collections.defaultdict(list)
        for e in kept_edges: chains[e['trajectory']].append(e)
        for rid, why in drop.items():
            omissions.write(json.dumps({'id': rid, 'source': source, 'stage': 'assemble', 'reason': why}) + '\n')
            totals['omitted_assemble_' + why.split('_of_')[0]] += 1
        for traj, ces in chains.items():
            ids = [rid for rid, r in records.items() if rid in keep_ids and trajectory_of(r) == traj]
            for rid in ids:
                r = records[rid]; seen_ids[rid] = source; seen_content.add(content_key(r))
                rec_out.write(json.dumps(r, ensure_ascii=False) + '\n')
            for e in ces: edge_out.write(json.dumps({**e, 'source': source}, sort_keys=True) + '\n')
            f = family_of(records[ids[0]]) if ids else 'unknown'
            chains_out.write(json.dumps({'trajectory': traj, 'source': source, 'family': f, 'split': ces[0]['split'],
                                         'records': ids, 'edges': len(ces)}) + '\n')
            fam[f]['chains'] += 1; fam[f]['records'] += len(ids); fam[f]['edges'] += len(ces)
            for e in ces: fam_kinds[f][e['kind']] += 1
            by_source[source]['chains'] += 1; by_source[source]['records'] += len(ids); by_source[source]['edges'] += len(ces)
            totals['chains'] += 1; totals['records'] += len(ids); totals['edges'] += len(ces)
            totals['edges_' + ces[0]['split']] += len(ces)
    for h in (omissions, rec_out, edge_out, chains_out): h.close()
    with open(staged / 'pieces.jsonl', 'w') as f:
        for n, t in sorted(pieces.items()): f.write(json.dumps({'name': n, 'text': t}, ensure_ascii=False) + '\n')
    # Supervised tokens per family from the shard statistics (LFM2.5 tokenizer, target text).
    tokens = collections.Counter()
    for stem in shard_stems:
        for f, v in (json.load(open(f'{stem}.stats.json')).get('by_family') or {}).items(): tokens[f] += v.get('supervised_tokens', 0)
    receipt = {
        'schema': 'natlang.neuralese-e2e-cohort/2', 'id': a.id,
        'format': 'natlang.teacher_training_turn.native/1 records (whole trajectories of every chain) + pieces + edges + chains, consumed by train/trajectories.py --records/--pieces; tokenizer-agnostic',
        'selection': 'every gradient_edge reference of authenticated_recurrence_context_view over every converted native-turn source (conversion /15 with --admit-host-results: host-computed child returns are producers, kind result/host_result marked in conversion receipts); records = whole trajectories of chains',
        'exclusions': 'cross-split edges; train edges sharing a source group with a held document of the held union; duplicate ids (first source wins), exact duplicate content, piece version conflicts, conversion/authentication failures, all registered in omissions.jsonl',
        'held_union': a.held_union, 'held_groups': len(held_groups), 'piece_conflicts': sorted(piece_conflicts),
        'totals': totals, 'by_source': by_source,
        'by_family': {f: {**fam[f], 'edge_kinds': fam_kinds[f], 'supervised_tokens_lfm25_shard_kept': tokens[f]} for f in sorted(fam)},
        'sources': {d.name: json.load(open(d / 'sharded.json')) for d in sorted(work.iterdir()) if (d / 'sharded.json').exists()},
        'files': {n: {'sha256': sha256_file_hex(staged / n), 'bytes': os.path.getsize(staged / n)}
                  for n in ('records.jsonl', 'pieces.jsonl', 'edges.jsonl', 'chains.jsonl', 'omissions.jsonl')},
    }
    json.dump(receipt, open(staged / 'receipt.json', 'w'), indent=1, sort_keys=True, default=dict)
    if out.exists():
        raise SystemExit(f'{out} exists; corpora are immutable')
    staged.rename(out)
    print(json.dumps({k: receipt[k] for k in ('id', 'totals')}, indent=1, default=dict))


def _soft_names(row):
    out = set()

    def walk(v):
        if isinstance(v, dict):
            if v.get('type') == 'soft' and isinstance(v.get('name'), str): out.add(v['name'])
            for x in v.values(): walk(x)
        elif isinstance(v, list):
            for x in v: walk(x)
    walk(row.get('messages')); walk(row.get('target'))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('shard'); s.add_argument('name'); s.add_argument('files', nargs='+'); s.add_argument('--work', required=True)
    s.add_argument('--target-mb', type=int, default=150)
    c = sub.add_parser('convert'); c.add_argument('--work', required=True); c.add_argument('--source')
    m = sub.add_parser('assemble'); m.add_argument('--work', required=True); m.add_argument('--id', required=True)
    m.add_argument('--held-union', default=resolve_str('repo', 'runs/natlang-target-union-20261011-r3'))
    a = ap.parse_args()
    {'shard': cmd_shard, 'convert': cmd_convert, 'assemble': cmd_assemble}[a.cmd](a)


if __name__ == '__main__':
    main()
