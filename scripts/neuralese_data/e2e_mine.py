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
  e2e_mine.py shard SOURCE_NAME FILE [FILE...] --work DIR [--target-mb 150] [--pieces P [P...]]
      split a source by trajectory into shards (plain or .zst JSONL), so a trajectory's rows stay together. With
      --pieces the source is already converted (records + piece tables): conversion is skipped, edges are derived.
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
sys.path.insert(0, str(ROOT / 'scripts'))
from source_review import pending_unpinned_source_groups, registry_sha256  # noqa: E402
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex  # noqa: E402
from natlang_neuralese.common.paths import resolve_str  # noqa: E402
from natlang_neuralese.common.jsonio import open_jsonl_text  # noqa: E402

TS_HOST = Path(os.environ.get('E2E_MINE_TS_HOST', ROOT / 'ts-host'))
LFM_TOKENIZER = resolve_str('repo', 'models/hf/hub/models--LiquidAI--LFM2.5-350M/snapshots/9e6c6ccf47cd318696e137d381a7ded8fe4df09f')


def open_lines(path):
    with open_jsonl_text(path) as stream:
        yield from stream


def held_union_groups(work, union):
    """Bind held protection to the selected immutable files, including compressed views."""
    paths = []
    for rendering in ('full', 'program-header', 'whole-trajectory'):
        directory = Path(union) / rendering
        current = sorted([*directory.glob('*/text-data.jsonl'),
                          *directory.glob('*/text-data.jsonl.zst')])
        if not current:
            raise ValueError(f'held union has no {rendering} text view: {union}')
        paths.extend(current)
    bindings = [{'path': str(path.absolute()), 'sha256': sha256_file_hex(path)} for path in paths]
    cache = Path(work) / 'held-groups.json'
    if cache.exists():
        saved = json.loads(cache.read_text())
        if isinstance(saved, dict) and saved.get('bindings') == bindings:
            return set(saved['groups']), bindings
    groups = set()
    for path in paths:
        for line in open_lines(path):
            if '"test"' not in line:
                continue
            row = json.loads(line)
            if row.get('split') == 'test':
                groups.update(row.get('source_groups') or [])
    after = [{'path': str(path.absolute()), 'sha256': sha256_file_hex(path)} for path in paths]
    if after != bindings:
        raise ValueError('held union changed during protection snapshot')
    pending = cache.with_suffix('.pending')
    pending.write_text(json.dumps({'bindings': bindings, 'groups': sorted(groups)}) + '\n')
    pending.replace(cache)
    return groups, bindings


def read_lines(path):
    """Lines of a file, dropping its page cache afterwards (page cache counts against a ledger unit's budget)."""
    with open(path, encoding='utf-8') as f:
        yield from f
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def reviewed_trajectories(records, review_groups):
    """Whole train trajectories rejected by exact shared source-review groups."""
    blocked = collections.defaultdict(set)
    for record in records:
        if record.get('split') == 'train':
            matched = set(record.get('source_groups') or []) & review_groups
            if matched:
                blocked[trajectory_of(record)].update(matched)
    return blocked


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
    n = a.shards or max(1, -(-size // (a.target_mb << 20)))
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
    json.dump({'source': a.name, 'files': a.files, 'shards': n, 'rows': rows, 'unparseable_lines': bad,
               **({'preconverted_pieces': a.pieces} if a.pieces else {})},
              open(out / 'sharded.json', 'w'), indent=1)
    print(json.dumps({'source': a.name, 'shards': n, 'rows': rows, 'unparseable_lines': bad}))


def cmd_redo_failed(a):
    """Re-create the shards whose conversion or authentication failed (sharding is deterministic) for a new convert."""
    out = Path(a.work) / a.name
    meta = json.load(open(out / 'sharded.json'))
    failed = [Path(p[:-len('.stats.json')]) for p in glob.glob(str(out / 'shard-*.stats.json'))
              if json.load(open(p)).get('error')]
    if not failed:
        print('nothing to redo', a.name); return
    (out / 'sharded.json').rename(out / 'sharded.previous.json')
    cmd_shard(argparse.Namespace(name=a.name, files=meta['files'], work=a.work, target_mb=150, shards=meta['shards'],
                                 pieces=meta.get('preconverted_pieces')))
    redo = {str(stem) for stem in failed}
    for shard in glob.glob(str(out / 'shard-[0-9][0-9][0-9][0-9]*.jsonl')):
        if shard.count('.') > 1: continue
        stem = str(Path(shard).with_suffix(''))
        if stem in redo:
            for suffix in ('stats.json', 'records.jsonl', 'edges.jsonl', 'pieces.jsonl', 'omissions.jsonl'):
                Path(f'{stem}.{suffix}').unlink(missing_ok=True)
        else:
            os.remove(shard)
    (out / 'sharded.previous.json').unlink()
    print(json.dumps({'source': a.name, 'redo_shards': len(redo)}))


def target_text(row):
    t = row.get('target') or {}
    parts = [t.get('content') if isinstance(t.get('content'), str) else json.dumps(t.get('content') or '')]
    for call in t.get('tool_calls') or []:
        fn = call.get('function') or {}
        parts.append(fn.get('name', '') + ' ' + (fn.get('arguments') if isinstance(fn.get('arguments'), str) else json.dumps(fn.get('arguments'))))
    return '\n'.join(p for p in parts if p)


def convert_shard(shard: Path, tokenizer, preconverted_pieces=None):
    from natlang_neuralese.train.trajectories import authenticated_recurrence_context_view
    stem = shard.with_suffix('')
    omissions = []
    if preconverted_pieces:
        records = [json.loads(l) for l in open(shard) if l.strip()]
        pieces = {}
        for path in preconverted_pieces:
            for l in open(path):
                if l.strip():
                    p = json.loads(l); pieces.setdefault(p['name'], p['text'])
        conversion = {'sites': {}, 'preconverted': True}
        return derive(stem, records, pieces, conversion, omissions, tokenizer)
    with tempfile.TemporaryDirectory(dir=shard.parent) as tmp:
        conv, pieces_path, summary = Path(tmp) / 'records.jsonl', Path(tmp) / 'pieces.jsonl', Path(tmp) / 'conversion.json'
        env = dict(os.environ, NODE_OPTIONS=os.environ.get('NODE_OPTIONS', '--max-old-space-size=6144'))
        proc = subprocess.run(['node', 'scripts/neuralese-convert-trajectories.mjs', '--admit-host-results', '--out', str(conv),
                               '--pieces', str(pieces_path), '--summary', str(summary), str(shard)],
                              cwd=TS_HOST, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            # The whole shard failed conversion: register every row with the converter's error.
            lines = proc.stderr.strip().splitlines() or ['?']
            reason = 'conversion_failed: ' + next((l for l in lines if 'Error' in l), lines[-1])[:300]
            for line in open(shard):
                if line.strip():
                    try: rid = json.loads(line).get('id')
                    except ValueError: rid = None
                    omissions.append({'id': rid, 'stage': 'convert', 'reason': reason})
            write_shard(stem, [], {}, [], omissions, {'converted': False, 'error': reason})
            return
        records = [json.loads(l) for l in open(conv) if l.strip()]
        conversion = json.load(open(summary))
        unreadable = conversion.get('unreadable', 0)
        if unreadable or (not records and shard.stat().st_size):
            emitted_ids = {r.get('id') for r in records}
            for line in read_lines(shard):
                if not line.strip(): continue
                try: rid = json.loads(line).get('id')
                except (ValueError, AttributeError): rid = None
                if rid is None or rid not in emitted_ids:
                    omissions.append({'id': rid, 'stage': 'convert',
                                      'reason': 'conversion_unreadable_source_json',
                                      'source_line_sha256': hashlib.sha256(line.encode()).hexdigest()})
            if not records:
                write_shard(stem, [], {}, [], omissions,
                            {'converted': False, 'error': 'conversion_produced_no_records_from_nonempty_source',
                             'rows': 0, 'input_records': conversion.get('records', 0),
                             'unreadable_input_records': unreadable})
                return
        pieces = {}
        for l in open(pieces_path):
            if l.strip():
                p = json.loads(l); pieces[p['name']] = p['text']
    return derive(stem, records, pieces, conversion, omissions, tokenizer)


def derive(stem, records, pieces, conversion, omissions, tokenizer):
    from natlang_neuralese.train.trajectories import authenticated_recurrence_context_view
    # Producer->consumer edges never cross trajectories, and writer names (handover:<note sha>, result:...) are unique
    # only within one: authenticate each trajectory on its own, so one trajectory's failure is registered alone.
    groups = collections.defaultdict(list)
    for r in records: groups[trajectory_of(r)].append(r)
    review = {'decisions': [], 'excluded': []}
    failed = 0
    for traj, rows in groups.items():
        try:
            _, part = authenticated_recurrence_context_view(rows, pieces)
        except (ValueError, KeyError) as error:
            failed += 1
            for r in rows: omissions.append({'id': r.get('id'), 'stage': 'authenticate', 'reason': f'trajectory_failed: {error}'[:300]})
            continue
        review['decisions'] += part['decisions']; review['excluded'] += part['excluded']
    for e in review['excluded']:
        omissions.append({'id': e.get('id'), 'stage': 'authenticate', 'reason': e.get('reason')})
    by_id = {r['id']: r for r in records}
    edges, seen = [], set()
    for d in review['decisions']:
        for ref in d.get('references') or []:
            key = (ref.get('writer_record_id'), d['id'], ref.get('name'), ref.get('body_sha256'))
            if ref.get('gradient_edge') and key not in seen:
                seen.add(key)
                w = by_id.get(ref.get('writer_record_id'))
                edges.append({'writer_record_id': ref.get('writer_record_id'), 'reader_record_id': d['id'],
                              'write_name': ref.get('name'), 'body_sha256': ref.get('body_sha256'),
                              'kind': (ref.get('name') or '').split(':')[0], 'trajectory': trajectory_of(by_id[d['id']]),
                              'family': family_of(by_id[d['id']]), 'split': by_id[d['id']].get('split'),
                              'writer_split': (w or {}).get('split')})
    chains = {e['trajectory'] for e in edges}
    kept = [r for r in records if trajectory_of(r) in chains]
    without_edges = set(groups) - chains
    accounted_ids = {item.get('id') for item in omissions}
    for r in records:
        if trajectory_of(r) in without_edges and r.get('id') not in accounted_ids:
            omissions.append({'id': r.get('id'), 'stage': 'select_edge',
                              'trajectory': trajectory_of(r),
                              'reason': 'trajectory_has_no_authenticated_producer_consumer_edge'})
    used = set()

    def walk(v):
        if isinstance(v, dict):
            if v.get('type') == 'soft' and isinstance(v.get('name'), str): used.add(v['name'])
            for x in v.values(): walk(x)
        elif isinstance(v, list):
            for x in v: walk(x)
    for r in kept: walk(r.get('messages')); walk(r.get('target'))
    stats = {'converted': True, 'rows': len(records), 'kept_records': len(kept), 'edges': len(edges), 'chains': len(chains),
             'input_records': conversion.get('records', len(records)),
             'unreadable_input_records': conversion.get('unreadable', 0),
             'trajectories': len(groups), 'trajectories_failed_authentication': failed,
             'trajectories_without_edges': len(without_edges),
             'records_without_edges': sum(len(groups[t]) for t in without_edges),
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


def split_by_trajectory(shard: Path, max_mb: int):
    """Split an oversized shard into parts of whole trajectories (shard-NNNN-tKKK.jsonl), each at most max_mb unless a
    single trajectory is larger (then it is a part of its own, converted later in a unit with more memory)."""
    sizes, order = collections.Counter(), []
    for line in read_lines(shard):
        t = trajectory_of(json.loads(line))
        if t not in sizes: order.append(t)
        sizes[t] += len(line)
    part, used, assign = 0, 0, {}
    for t in sorted(order, key=lambda t: sizes[t]):
        if used and used + sizes[t] > max_mb << 20: part += 1; used = 0
        assign[t] = part; used += sizes[t]
    handles = {}
    for line in read_lines(shard):
        i = assign[trajectory_of(json.loads(line))]
        if i not in handles: handles[i] = open(shard.with_name(f'{shard.stem}-t{i:03d}.jsonl'), 'w', encoding='utf-8')
        handles[i].write(line)
    for h in handles.values(): h.close()
    os.remove(shard)
    print(json.dumps({'split': str(shard), 'parts': len(handles), 'trajectories': len(order)}), flush=True)


def cmd_convert(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LFM_TOKENIZER)
    import re
    shard_name = re.compile(r'^shard-\d{4}(-t\d{3})?\.jsonl$')
    shards = [p for p in glob.glob(str(Path(a.work) / (a.source or '*') / 'shard-*.jsonl')) if shard_name.match(Path(p).name)]
    for shard in sorted(shards):
        stem = Path(shard).with_suffix('')
        if Path(f'{stem}.stats.json').exists(): continue
        if a.max_mb and os.path.getsize(shard) > a.max_mb << 20:
            if not Path(shard).name.count('-t'):
                split_by_trajectory(Path(shard), a.max_mb)
                continue
            print(json.dumps({'shard': shard, 'deferred': f'over --max-mb {a.max_mb}: convert in a larger unit'}), flush=True)
            continue
        meta = json.load(open(Path(shard).parent / 'sharded.json'))
        convert_shard(Path(shard), tok, meta.get('preconverted_pieces'))
        os.remove(shard)  # the shard is an intermediate copy of its source; outputs and omissions are kept
        s = json.load(open(f'{stem}.stats.json'))
        print(json.dumps({'shard': shard, **{k: s.get(k) for k in ('rows', 'kept_records', 'edges', 'chains', 'error')}}), flush=True)


def content_key(row):
    """Exact-duplicate key: the record's canonical content without its id."""
    return canonical_json_sha256_hex({k: v for k, v in row.items() if k != 'id'})


def cmd_assemble(a):
    review_groups = pending_unpinned_source_groups()
    review_sha = registry_sha256()
    quality_groups, quality_bindings = set(), []
    for path in getattr(a, 'quarantine_reference', []):
        before = sha256_file_hex(path)
        for line in open_lines(path):
            quality_groups.update(json.loads(line)['source_groups'])
        if sha256_file_hex(path) != before:
            raise ValueError(f'quality quarantine changed during snapshot: {path}')
        quality_bindings.append({'path': str(path), 'sha256': before})
    work = Path(a.work)
    archived = sorted({*work.glob('*/archived-source.json'), *work.glob('archived-source.json')})
    if archived:
        raise ValueError('Restore archived mined sources and verify every recorded member before assembly: '
                         + ', '.join(str(path) for path in archived))
    out = Path(a.out_dir) / a.id if a.out_dir else Path(resolve_str('data_nvme', 'natlang-corpora', a.id))
    staged = out.with_name(out.name + '.building'); staged.mkdir(parents=True, exist_ok=True)
    held_groups, held_bindings = held_union_groups(work, a.held_union)
    pieces, piece_conflicts = {}, set()
    shard_stems = sorted(Path(p[:-len('.stats.json')]) for p in glob.glob(str(work / '*' / 'shard-*.stats.json')))
    for stem in shard_stems:
        for line in read_lines(f'{stem}.pieces.jsonl'):
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
        for line in read_lines(f'{stem}.omissions.jsonl'):
            o = json.loads(line); o['source'] = source; omissions.write(json.dumps(o) + '\n'); totals['omitted_' + o['stage']] += 1
        records = {}
        for line in read_lines(f'{stem}.records.jsonl'):
            r = json.loads(line); records[r['id']] = r
        edges = [json.loads(l) for l in read_lines(f'{stem}.edges.jsonl')]
        reviewed = reviewed_trajectories(records.values(), review_groups)
        quarantined = reviewed_trajectories(records.values(), quality_groups)
        keep_ids, drop = set(), {}
        for rid, r in records.items():
            if r.get('split') == 'train' and trajectory_of(r) in quarantined:
                drop[rid] = 'source_quality_quarantine_chain'; continue
            if r.get('split') == 'train' and trajectory_of(r) in reviewed:
                drop[rid] = 'source_review_pending_chain'; continue
            if rid in seen_ids:
                drop[rid] = 'duplicate_id_of_' + seen_ids[rid]; continue
            c = content_key(r)
            if c in seen_content:
                drop[rid] = 'exact_duplicate_content'; continue
            if any(n in piece_conflicts for n in _soft_names(r)):
                drop[rid] = 'piece_version_conflict'; continue
            keep_ids.add(rid)
        kept_edges, edge_keys = [], set()
        for e in edges:
            w, rd = e['writer_record_id'], e['reader_record_id']
            key = (w, rd, e['write_name'], e['body_sha256'])
            if key in edge_keys: totals['edges_duplicate_reference'] += 1; continue
            edge_keys.add(key)
            if w not in keep_ids or rd not in keep_ids: totals['edges_dropped_endpoint'] += 1; continue
            if e['split'] != e['writer_split']: totals['edges_dropped_cross_split'] += 1; continue
            if e['split'] == 'train' and ((set(records[w].get('source_groups') or []) | set(records[rd].get('source_groups') or [])) & held_groups):
                totals['edges_dropped_train_shares_held_group'] += 1; continue
            kept_edges.append(e)
        chains = collections.defaultdict(list)
        for e in kept_edges: chains[e['trajectory']].append(e)
        for rid, why in drop.items():
            groups = (quarantined if why == 'source_quality_quarantine_chain' else
                      reviewed if why == 'source_review_pending_chain' else None)
            detail = {'source_groups': sorted(groups[trajectory_of(records[rid])])} if groups is not None else {}
            omissions.write(json.dumps({'id': rid, 'source': source, 'stage': 'assemble', 'reason': why, **detail}) + '\n')
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
        for h in (rec_out, edge_out):  # written pages count against the unit's budget too: flush and drop them
            h.flush(); os.fdatasync(h.fileno()); os.posix_fadvise(h.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
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
        'held_union': a.held_union, 'held_groups': len(held_groups), 'held_union_bindings': held_bindings,
        'piece_conflicts': sorted(piece_conflicts),
        'source_review_registry_sha256': review_sha,
        'source_quality_quarantine_bindings': quality_bindings,
        'source_quality_quarantine_groups': len(quality_groups),
        'totals': totals, 'by_source': by_source,
        'by_family': {f: {**fam[f], 'edge_kinds': fam_kinds[f], 'supervised_tokens_lfm25_shard_kept': tokens[f]} for f in sorted(fam)},
        'sources': {d.name: {**json.load(open(d / 'sharded.json')), 'shards_done': len(glob.glob(str(d / 'shard-*.stats.json')))}
                    for d in sorted(work.iterdir()) if (d / 'sharded.json').exists()},
        'files': {n: {'sha256': sha256_file_hex(staged / n), 'bytes': os.path.getsize(staged / n)}
                  for n in ('records.jsonl', 'pieces.jsonl', 'edges.jsonl', 'chains.jsonl', 'omissions.jsonl')},
    }
    json.dump(receipt, open(staged / 'receipt.json', 'w'), indent=1, sort_keys=True, default=dict)
    if out.exists():
        raise SystemExit(f'{out} exists; corpora are immutable')
    staged.rename(out)
    print(json.dumps({k: receipt[k] for k in ('id', 'totals')}, indent=1, default=dict))


def cmd_review_filter(a):
    """A new immutable cohort; preserve complete good chains and held bytes."""
    import shutil
    src, out = Path(a.source), Path(a.out)
    if out.exists():
        raise SystemExit(f'{out} exists; corpora are immutable')
    staged = out.with_name(out.name + '.building')
    staged.mkdir(parents=True, exist_ok=True)
    chains = [json.loads(line) for line in read_lines(src / 'chains.jsonl')]
    membership = {}
    chain_index = {}
    for i, chain in enumerate(chains):
        key = (chain['source'], chain['trajectory'])
        if key in chain_index:
            raise ValueError(f'duplicate chain identity {key}')
        chain_index[key] = i
        for rid in chain['records']:
            if rid in membership:
                raise ValueError(f'duplicate chain membership {rid}')
            membership[rid] = i
    review_sha = registry_sha256()
    groups = pending_unpinned_source_groups()
    if registry_sha256() != review_sha:
        raise ValueError('source-review registry changed during snapshot')
    blocked, held_flags, seen = collections.defaultdict(set), [], set()
    for line in read_lines(src / 'records.jsonl'):
        row = json.loads(line); rid = row['id']
        if rid not in membership or rid in seen:
            raise ValueError(f'unknown or duplicate record {rid}')
        seen.add(rid); i = membership[rid]
        if row.get('split') != chains[i]['split']:
            raise ValueError(f'record/chain split mismatch {rid}')
        matched = set(row.get('source_groups') or []) & groups
        if matched:
            if row['split'] == 'train':
                blocked[i].update(matched)
            else:
                held_flags.append({'id': rid, 'source_groups': sorted(matched)})
    if seen != set(membership):
        raise ValueError('chain records missing from source cohort')
    kept_ids = {rid for rid, i in membership.items() if i not in blocked}
    with open(src / 'records.jsonl', 'rb') as source, open(staged / 'records.jsonl', 'wb') as target:
        for line in source:
            if json.loads(line)['id'] in kept_ids:
                target.write(line)
    edge_counts = collections.Counter()
    with open(src / 'edges.jsonl', 'rb') as source, open(staged / 'edges.jsonl', 'wb') as target:
        for line in source:
            edge = json.loads(line); i = chain_index[(edge['source'], edge['trajectory'])]
            w, r = edge['writer_record_id'], edge['reader_record_id']
            if membership.get(w) != i or membership.get(r) != i or edge['split'] != chains[i]['split'] or edge['writer_split'] != edge['split']:
                raise ValueError(f'edge is not closed within its split/chain: {w} -> {r}')
            edge_counts[i] += 1
            if i not in blocked:
                target.write(line)
    if any(edge_counts[i] != chain['edges'] for i, chain in enumerate(chains)):
        raise ValueError('source edge count differs from chain certificate')
    with open(staged / 'chains.jsonl', 'w') as stream:
        for i, chain in enumerate(chains):
            if i not in blocked:
                stream.write(json.dumps(chain) + '\n')
    with open(staged / 'omissions.jsonl', 'w') as stream:
        for i, matched in sorted(blocked.items()):
            for rid in chains[i]['records']:
                stream.write(json.dumps({'id': rid, 'trajectory': chains[i]['trajectory'], 'source': chains[i]['source'],
                    'stage': 'source_review', 'reason': 'source_review_pending_chain', 'source_groups': sorted(matched)}) + '\n')
    shutil.copyfile(src / 'pieces.jsonl', staged / 'pieces.jsonl')
    kept = [chain for i, chain in enumerate(chains) if i not in blocked]
    receipt = {'schema': 'natlang.neuralese-e2e-cohort/2', 'id': a.id,
        'selection': 'Existing admitted cohort, whole train chains removed by exact pending unpinned source-review groups; held chains unchanged',
        'source_review_registry_sha256': review_sha,
        'source': {'path': str(src), 'receipt_sha256': sha256_file_hex(src / 'receipt.json')},
        'totals': {'chains': len(kept), 'records': len(kept_ids), 'edges': sum(c['edges'] for c in kept),
                   'omitted_chains': len(blocked), 'omitted_records': len(membership) - len(kept_ids)},
        'by_split': {split: {'chains': sum(c['split'] == split for c in kept),
                    'records': sum(len(c['records']) for c in kept if c['split'] == split),
                    'edges': sum(c['edges'] for c in kept if c['split'] == split)} for split in ('train', 'test')},
        'held_source_review_flags': held_flags,
        'files': {name: {'sha256': sha256_file_hex(staged / name), 'bytes': (staged / name).stat().st_size}
                  for name in ('records.jsonl', 'edges.jsonl', 'chains.jsonl', 'pieces.jsonl', 'omissions.jsonl')}}
    (staged / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    staged.rename(out)
    print(json.dumps({'id': a.id, 'totals': receipt['totals']}), flush=True)


def cmd_slice(a):
    """A trainer-sized slice of an assembled corpus: the edge endpoints of whole chains (the v1 rule: both endpoints of
    every edge), every family represented, writer names unique.

    train/trajectories.py loads all records and authenticates them together (writer names must be unique across the
    file), so the full corpus is bound through slices. Chains are taken round-robin over families (so small families
    are kept whole before large ones fill the budget), densest chains (edges per endpoint record) first within a
    family; test chains up to --test-records endpoint records, train chains up to --train-records; a chain whose
    writer name already occurs is left for another slice.
    """
    from natlang_neuralese.train.trajectories import target_writes
    src = Path(a.source) if '/' in a.source else Path(resolve_str('data_nvme', 'natlang-corpora', a.source))
    chains = [json.loads(l) for l in read_lines(src / 'chains.jsonl')]
    ends = collections.defaultdict(set)
    index = {(c['source'], c['trajectory']): i for i, c in enumerate(chains)}
    for line in read_lines(src / 'edges.jsonl'):
        e = json.loads(line); i = index[(e['source'], e['trajectory'])]
        ends[i] |= {e['writer_record_id'], e['reader_record_id']}
    for i, c in enumerate(chains): c['records'] = [r for r in c['records'] if r in ends[i]]
    member = {rid: i for i, c in enumerate(chains) for rid in c['records']}
    names = collections.defaultdict(set); size = collections.Counter()
    for line in read_lines(src / 'records.jsonl'):
        r = json.loads(line); i = member.get(r['id'])
        if i is None: continue
        names[i].update(target_writes(r)); size[i] += len(line)
    rng = __import__('random').Random(a.seed)
    chosen, used = set(), set()
    for split, budget in (('test', a.test_records), ('train', a.train_records)):
        by_family = collections.defaultdict(list)
        for i, c in enumerate(chains):
            if c['split'] == split: by_family[c['family']].append(i)
        for v in by_family.values():
            rng.shuffle(v); v.sort(key=lambda i: chains[i]['edges'] / max(1, len(chains[i]['records'])))  # densest last: pop() takes it
        taken = 0
        while taken < budget and any(by_family.values()):
            for f in sorted(by_family):
                if not by_family[f] or taken >= budget: continue
                i = by_family[f].pop()
                if names[i] & used: continue
                chosen.add(i); used |= names[i]; taken += len(chains[i]['records'])
    out = Path(resolve_str('data_nvme', 'natlang-corpora', a.id)); staged = out.with_name(out.name + '.building')
    staged.mkdir(parents=True, exist_ok=True)
    ids = {rid for i in chosen for rid in chains[i]['records']}
    with open(staged / 'records.jsonl', 'w') as f:
        for line in read_lines(src / 'records.jsonl'):
            if json.loads(line)['id'] in ids: f.write(line)
    with open(staged / 'edges.jsonl', 'w') as f:
        for line in read_lines(src / 'edges.jsonl'):
            e = json.loads(line)
            if e['reader_record_id'] in ids and e['writer_record_id'] in ids: f.write(line)
    with open(staged / 'chains.jsonl', 'w') as f:
        for i in sorted(chosen): f.write(json.dumps(chains[i]) + '\n')
    import shutil; shutil.copyfile(src / 'pieces.jsonl', staged / 'pieces.jsonl')
    sel = [chains[i] for i in chosen]
    stats = {'chains': len(sel), 'records': len(ids), 'edges': sum(c['edges'] for c in sel),
             'by_split': {s_: {'chains': sum(c['split'] == s_ for c in sel), 'records': sum(len(c['records']) for c in sel if c['split'] == s_),
                               'edges': sum(c['edges'] for c in sel if c['split'] == s_)} for s_ in ('train', 'test')},
             'by_family': {f: {'chains': n} for f, n in collections.Counter(c['family'] for c in sel).items()}}
    receipt = {'schema': 'natlang.neuralese-e2e-cohort-slice/1', 'id': a.id, 'source': a.source,
               'source_receipt_sha256': sha256_file_hex(src / 'receipt.json'), 'source_path': str(src),
               'selection': cmd_slice.__doc__.strip(), 'seed': a.seed, 'train_records_budget': a.train_records,
               'test_records_budget': a.test_records, **stats,
               'files': {n: {'sha256': sha256_file_hex(staged / n), 'bytes': os.path.getsize(staged / n)}
                         for n in ('records.jsonl', 'pieces.jsonl', 'edges.jsonl', 'chains.jsonl')}}
    json.dump(receipt, open(staged / 'receipt.json', 'w'), indent=1, sort_keys=True)
    staged.rename(out)
    print(json.dumps({k: receipt[k] for k in ('id', 'chains', 'records', 'edges', 'by_split')}, indent=1))


def cmd_partition(a):
    """Partition a whole assembled corpus into trainer-sized slices that together hold every chain.

    Each part holds the edge endpoints of whole chains, at most --records records, writer names unique within the part
    (so each part re-authenticates as one file); chains are dealt round-robin over families so every part is
    family-mixed. Parts: <ID>/part-NNN/{records,edges,chains}.jsonl, one shared pieces.jsonl, receipt.json.
    """
    from natlang_neuralese.train.trajectories import target_writes
    src = Path(a.source) if '/' in a.source else Path(resolve_str('data_nvme', 'natlang-corpora', a.source))
    chains = [json.loads(l) for l in read_lines(src / 'chains.jsonl')]
    index = {(c['source'], c['trajectory']): i for i, c in enumerate(chains)}
    ends = collections.defaultdict(set)
    for line in read_lines(src / 'edges.jsonl'):
        e = json.loads(line); ends[index[(e['source'], e['trajectory'])]] |= {e['writer_record_id'], e['reader_record_id']}
    for i, c in enumerate(chains): c['records'] = [r for r in c['records'] if r in ends[i]]
    member = {rid: i for i, c in enumerate(chains) for rid in c['records']}
    names = collections.defaultdict(set)
    for line in read_lines(src / 'records.jsonl'):
        r = json.loads(line); i = member.get(r['id'])
        if i is not None: names[i].update(target_writes(r))
    by_family = collections.defaultdict(list)
    for i, c in enumerate(chains):
        if c['records']: by_family[c['family']].append(i)
    order = []
    queues = [collections.deque(v) for _, v in sorted(by_family.items())]
    while any(queues):
        for q in queues:
            if q: order.append(q.popleft())
    parts = []  # [records, used names, chain ids]
    part_of = {}
    for i in order:
        n = len(chains[i]['records'])
        for k, (count, used, ids) in enumerate(parts):
            if count + n <= a.records and not (names[i] & used):
                parts[k][0] += n; used |= names[i]; ids.append(i); part_of[i] = k; break
        else:
            parts.append([n, set(names[i]), [i]]); part_of[i] = len(parts) - 1
    out = Path(a.out_dir) / a.id if a.out_dir else Path(resolve_str('data_nvme', 'natlang-corpora', a.id))
    staged = out.with_name(out.name + '.building'); staged.mkdir(parents=True, exist_ok=True)
    rid_part = {rid: part_of[i] for rid, i in member.items()}
    dirs = [staged / f'part-{k:03d}' for k in range(len(parts))]
    for d in dirs: d.mkdir(exist_ok=True)
    rec = [open(d / 'records.jsonl', 'w') for d in dirs]
    for line in read_lines(src / 'records.jsonl'):
        k = rid_part.get(json.loads(line)['id'])
        if k is not None: rec[k].write(line)
    edg = [open(d / 'edges.jsonl', 'w') for d in dirs]
    for line in read_lines(src / 'edges.jsonl'):
        e = json.loads(line); edg[part_of[index[(e['source'], e['trajectory'])]]].write(line)
    for k, d in enumerate(dirs):
        with open(d / 'chains.jsonl', 'w') as f:
            for i in parts[k][2]: f.write(json.dumps(chains[i]) + '\n')
    for h in rec + edg: h.close()
    import shutil; shutil.copyfile(src / 'pieces.jsonl', staged / 'pieces.jsonl')
    info = []
    for k, d in enumerate(dirs):
        sel = [chains[i] for i in parts[k][2]]
        info.append({'part': d.name, 'chains': len(sel), 'records': parts[k][0], 'edges': sum(c['edges'] for c in sel),
                     'train_records': sum(len(c['records']) for c in sel if c['split'] == 'train'),
                     'records_sha256': sha256_file_hex(d / 'records.jsonl'), 'records_bytes': os.path.getsize(d / 'records.jsonl')})
    receipt = {'schema': 'natlang.neuralese-e2e-cohort-partition/1', 'id': a.id, 'source': str(src),
               'source_receipt_sha256': sha256_file_hex(src / 'receipt.json'), 'selection': cmd_partition.__doc__.strip(),
               'records_per_part_max': a.records, 'parts': info,
               'totals': {k: sum(p[k] for p in info) for k in ('chains', 'records', 'edges', 'train_records')},
               'by_family': {f: len(v) for f, v in sorted(by_family.items())},
               'pieces_sha256': sha256_file_hex(staged / 'pieces.jsonl')}
    json.dump(receipt, open(staged / 'receipt.json', 'w'), indent=1, sort_keys=True)
    staged.rename(out)
    print(json.dumps({'id': a.id, 'parts': len(info), **receipt['totals']}, indent=1))


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
    s.add_argument('--pieces', nargs='*')
    s.add_argument('--shards', type=int, help='shard count (compressed sources: their ratio varies, give it explicitly)')
    c = sub.add_parser('convert'); c.add_argument('--work', required=True); c.add_argument('--source')
    c.add_argument('--max-mb', type=int, default=0, help='split larger shards by trajectory; defer single parts above it')
    sl = sub.add_parser('slice'); sl.add_argument('--source', required=True, help='corpus id under data_nvme natlang-corpora, or a path'); sl.add_argument('--id', required=True)
    sl.add_argument('--train-records', type=int, default=12000); sl.add_argument('--test-records', type=int, default=1500)
    sl.add_argument('--seed', type=int, default=20261012)
    review = sub.add_parser('review-filter'); review.add_argument('--source', required=True)
    review.add_argument('--out', required=True); review.add_argument('--id', required=True)
    pa = sub.add_parser('partition'); pa.add_argument('--source', required=True); pa.add_argument('--id', required=True)
    pa.add_argument('--records', type=int, default=8000); pa.add_argument('--out-dir')
    r = sub.add_parser('redo-failed'); r.add_argument('name'); r.add_argument('--work', required=True)
    m = sub.add_parser('assemble'); m.add_argument('--work', required=True); m.add_argument('--id', required=True)
    m.add_argument('--out-dir', help='parent directory of the corpus (default: data_nvme natlang-corpora); bulk corpora go to the HDD')
    m.add_argument('--held-union', default=resolve_str('repo', 'runs/natlang-target-union-20261011-r3'))
    m.add_argument('--quarantine-reference', action='append', type=Path, default=[],
                   help='canonical source-quality group witnesses; exclude whole affected train chains')
    a = ap.parse_args()
    {'shard': cmd_shard, 'convert': cmd_convert, 'assemble': cmd_assemble, 'redo-failed': cmd_redo_failed, 'slice': cmd_slice, 'partition': cmd_partition, 'review-filter': cmd_review_filter}[a.cmd](a)


if __name__ == '__main__':
    main()
