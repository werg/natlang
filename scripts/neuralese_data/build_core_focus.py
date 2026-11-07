#!/usr/bin/env python3
"""Stratified *core* pending-focus set for production SFT (engineering, 2026-11-05).

The corrected mix has ~101k rows; production SFT trains a few hundred graphs per day. A pending focus
(`natlang.native-pending-focus/1`) promotes rows ahead of the ordinary schedule, so a small focus that touches
every target behaviour is trained first and yields a well-rounded checkpoint early.

Two facts shape the selection:
- `native_pending_focus.promote` promotes *whole packed batches*: every row sharing a batch with a focus row is
  trained at that point. Scattered focus rows therefore multiply the work (the 2,046-row diverse focus promoted
  ~7,100 graphs). The selection unit here is the packed batch itself, taken from the trainer's own
  `pending_exposure_batches` on the actual exposure and quotas, and the focus is the union of whole batches.
- Order inside the focus file is not consumed; the trainer keeps the schedule order of the promoted batches. The
  order a window will see is simulated with the trainer's promote/merge/interleave and reported.

Inputs are the existing audits (successor row mass, census2 flags/skeleton/DAG depth, row census) plus one decode
pass over the small natlang/communication/coding leaves for finish/blocked/decide/iterate/skill constructs on
*own-output* tokens (weight >= 0.5); the pass is cached.

Run under the frozen runtime that trains the lineage (PYTHONPATH of that runtime), through the memory ledger.
"""
from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import random
import re
import sys
import time
from pathlib import Path

from natlang_neuralese.common.hashing import sha256_file_hex as sha256_file
from natlang_neuralese.common.paths import resolve_str

CLASS_DIR = Path(resolve_str('data_hdd', 'natlang-lfm3-data-20261028/corrected-mix-diverse-1103'))
OPS = Path(resolve_str('data_nvme', 'lfm3-rl-dgx-20261029/ops/corrected-mix-diverse-1103'))
SCHEMA = 'natlang.native-pending-focus/1'
REPORT_SCHEMA = 'natlang.core-focus-report/1'
CODING_FAMILIES = {'coding-swe', 'coding-sessions', 'acquired-d13', 'agent-sessions'}
CAPPED_FAMILIES = {'port-record-wrapper': 'wrapper', 'native-communication-template': 'template-comm'}
SCAN_FAMILIES = {'natlang', 'natlang-inline', 'natlang-chains', 'native-communication', 'native-communication-rest27',
                 'coding-swe', 'coding-sessions'}
# fraction of the target row count per quota group (strata are assigned most-scarce first)
QUOTAS = collections.OrderedDict([
    ('semantic', 0.25), ('iterate', 0.12), ('reducer', 0.08), ('neuralese_d1', 0.10), ('neuralese_d2', 0.06),
    ('neuralese_other', 0.04), ('coding', 0.15), ('decide', 0.04), ('skills', 0.03), ('readout_guidance', 0.01),
    ('typed_finish', 0.02), ('blocked', 0.04), ('general', 0.06)])
GROUP = {'semantic': 'semantic delegation', 'iterate': 'structural iteration/reducers', 'reducer': 'structural iteration/reducers',
         'neuralese_d1': 'neuralese communication', 'neuralese_d2': 'neuralese communication',
         'neuralese_other': 'neuralese communication', 'coding': 'coding', 'decide': 'rare constructs',
         'skills': 'rare constructs', 'readout_guidance': 'rare constructs', 'typed_finish': 'rare constructs',
         'blocked': 'honest stops and general', 'general': 'honest stops and general'}
RX = {
    'finish_true': r'<parameter=finish>\s*true',
    'skill_path': r'skills?/[\w\-]+|read_skill|\.skill\b',
    'blocked': r'\bblocked\b',
    'status_stop': r'<parameter=status>\s*(?:blocked|failed)\s*</parameter>|<function=(?:blocked|failed)>',
    'cannot_phrase': r"(?:cannot|can't|unable to) (?:complete|be completed|proceed|determine|verify|find|continue|answer)",
    'missing': r'\bmissing\b',
    'decide': r'\bdecide\s*[(<]',
    'iterateOn': r'iterateOn',
    'neuralese': r'Neuralese<',
    'skill': r'\bskills?\b',
    'guidance': r'guidance',
    'readout': r'readout',
    'await_nl': r'await\s+nl',
    'folder': r'\bfolder\b|callable folder',
}


def zjsonl(path):
    import zstandard
    with open(path, 'rb') as raw:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding='utf8'):
            yield json.loads(line)


def rows_of(path):
    if str(path).endswith('.zst'):
        yield from zjsonl(path)
    else:
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def leaf_paths(manifest):
    result = {}

    def walk(sources):
        for item in sources:
            result[item['sha256']] = item['path']
            if str(item['path']).endswith('.json'):
                walk(json.loads(Path(item['path']).read_text())['sources'])
    walk(manifest['sources'])
    return result


def scan_features(entries, cohorts, family, paths, cache):
    """One decode pass over the small leaves: constructs on own-output tokens only (weight >= 0.5)."""
    if cache.is_file():
        return {r['id']: r for r in zjsonl(cache)}
    import zstandard
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(glob.glob(str(Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/*/tokenizer.json'))[0])
    rx = {k: re.compile(v) for k, v in RX.items()}
    wanted = {}
    for entry, cohort in zip(entries, cohorts):
        if family[entry['id']] in SCAN_FAMILIES:
            wanted.setdefault(cohort, set()).add(entry['id'])
    out = []
    t0 = time.time()
    for cohort, ids in wanted.items():
        for row in rows_of(paths[cohort]):
            if row['id'] not in ids:
                continue
            spans = []
            for node in row['nodes']:
                current = []
                for token, weight in zip(node['token_ids'], node['target_weights']):
                    if weight >= 0.5:
                        current.append(token)
                    elif current:
                        spans.append(current); current = []
                if current:
                    spans.append(current)
            text = '\n'.join(tok.decode(s, skip_special_tokens=False) for s in spans)
            feature = {k: len(r.findall(text)) for k, r in rx.items()}
            feature.update(id=row['id'], own_chars=len(text), nodes=len(row['nodes']), decisions=len(row.get('decisions') or []),
                           ext_states=len(row.get('external_states') or {}), tail=text[-300:])
            out.append(feature)
        print(json.dumps({'scan_rows': len(out), 'seconds': round(time.time() - t0)}), flush=True)
    tmp = cache.with_name(cache.name + '.tmp')
    with open(tmp, 'wb') as raw, zstandard.ZstdCompressor(level=6).stream_writer(raw) as stream:
        for feature in out:
            stream.write((json.dumps(feature, separators=(',', ':')) + '\n').encode())
    tmp.replace(cache)
    return {r['id']: r for r in out}


def load_rows(entries, cohorts, class_dir, scan):
    mass = {r['id']: r for r in zjsonl(class_dir / 'successor-row-mass.jsonl.zst')}
    census2 = {}
    for path in sorted(glob.glob(str(class_dir / 'census2/*.jsonl.zst'))):
        for r in zjsonl(path):
            census2[r['id']] = r
    census = {r['id']: r for r in zjsonl(class_dir / 'classification/row-census.jsonl.zst')}
    rows = []
    for index, (entry, cohort) in enumerate(zip(entries, cohorts)):
        ident = entry['id']
        c2, rc, sc = census2.get(ident) or {}, census.get(ident) or {}, scan.get(ident) or {}
        flags = {**(rc.get('flags') or {}), **(c2.get('flags') or {})}
        program = re.search(r'(?:^|:)teacher-program:([^:]+)', ident)
        family = mass[ident]['family']
        template = c2.get('skel') or (program.group(1) if program else None) or ((rc.get('sigs') or [None])[0]) \
            or (entry.get('groups') or [family])[0]
        rows.append({'index': index, 'id': ident, 'cohort': cohort, 'family': family, 'tokens': entry['forward_tokens'],
                     'communication': bool(entry.get('communication')), 'producer_consumer': bool(entry.get('producer_consumer')),
                     'external': bool(entry.get('external_state_consumption')), 'depth': c2.get('dag_depth') or 1,
                     'fan_in': c2.get('fan_in') or 0, 'nodes': c2.get('nodes') or rc.get('nodes') or sc.get('nodes') or 1,
                     'flags': flags, 'scan': sc, 'template': f'{family}|{template}', 'has_census': bool(c2 or rc),
                     'rc_group': rc.get('group')})
    return rows


def classify(row):
    """Construct tags from own-output evidence; the scan is authoritative where it exists."""
    tags = set()
    scan, flags, family = row['scan'], row['flags'], row['family']
    if family in CAPPED_FAMILIES:
        # The dataset-answer wrapper teaches no target behaviour: never selected. The big2 template edge may fill
        # depth-1 communication only up to its hard cap, after every genuine row.
        return {'capped:' + CAPPED_FAMILIES[family]} | ({'neuralese_d1'} if family == 'native-communication-template' else set())
    if row['communication'] and family != 'native-communication-template':
        tags.add('neuralese_d2' if row['depth'] >= 3 else 'neuralese_d1')
        if row['fan_in'] >= 2 or row['external']:
            tags.add('neuralese_other')
    if family in CODING_FAMILIES:
        tags.add('coding')
    if scan:
        if scan['iterateOn']:
            tags.add('iterate')
        if scan['decide']:
            tags.add('decide')
        if scan['await_nl'] or flags.get('per_item_nl'):
            tags.add('semantic')
        if scan['finish_true']:
            tags.add('typed_finish')
        if scan['status_stop']:
            tags.add('blocked')   # return_result(status blocked|failed) or the blocked/failed action: an honest stop
        if scan['decisions'] or scan['ext_states']:
            tags.add('readout_guidance')   # decision-readout graphs or stored external state
        if scan['skill_path'] and family not in CODING_FAMILIES:
            tags.add('skills')
    else:
        for key, tag in (('iterate', 'iterate'), ('decide', 'decide'), ('per_item_nl', 'semantic'), ('await_nl', 'semantic'),
                         ('skills', 'skills')):
            if flags.get(key):
                tags.add(tag)
    if flags.get('reducer') or flags.get('folder_apply'):
        tags.add('reducer')
    if flags.get('skills'):
        tags.add('skills')
    if not tags - {'neuralese_other'} and family in ('natlang', 'natlang-inline', 'natlang-chains', 'web-research',
                                                    'native-communication', 'native-communication-rest27'):
        tags.add('general')
    return tags


def schedule(entries, cohorts, exposure, required, global_step, batch_graphs, batch_tokens, focus_ids=()):
    """The trainer's own pending-batch order (promote, cohort-head merge, sustained interleave)."""
    from natlang_neuralese.train.native_state_sft import pending_exposure_batches
    from natlang_neuralese.train.native_pending_focus import promote
    from natlang_neuralese.train.native_diversity import cohort_head_merge, sustained_communication_interleave
    foundation = {e['id'] for e in entries if e.get('communication')}
    _, pending = pending_exposure_batches(entries, exposure, epochs=1.0, batch_graphs=batch_graphs, batch_tokens=batch_tokens,
                                          foundation_ids=foundation, foundation_communication_batches=1,
                                          minimum_visits=required, schedule_offset=global_step, cohort_ids=cohorts)
    if not focus_ids:
        return pending, pending
    base = pending
    pending = promote(pending, entries, focus_ids)
    pending = cohort_head_merge(entries, pending, cohorts, priority_ids=foundation, focus_ids=focus_ids,
                                batch_tokens=batch_tokens, batch_graphs=batch_graphs, schedule_offset=global_step)
    pending = sustained_communication_interleave(entries, pending, focus_ids=focus_ids, schedule_offset=global_step)
    return base, pending


def select(rows, units, target, seed, template_cap, cap_per_family, long_share):
    """Greedy stratified selection over whole packed batches; never duplicates a row."""
    rng = random.Random(seed)
    quota = {k: int(round(v * target)) for k, v in QUOTAS.items()}
    chosen_units, chosen_rows = [], set()
    template_count = collections.Counter()
    capped_count = collections.Counter()
    count = collections.Counter()            # rows by assigned stratum
    long_rows = [0]
    long_by = collections.Counter()   # 16k-32k rows already taken per requesting stratum
    long_budget = long_share * target
    # coding graphs are mostly long: reserve half of the 16k-32k allowance for them, the rest is shared by quota
    long_limit = {s: long_budget * (0.5 if s == 'coding' else 0.5 * q / (1.0 - QUOTAS['coding'])) for s, q in QUOTAS.items()}
    excluded = collections.Counter()
    tags = {r['index']: classify(r) for r in rows}

    def valid(unit, stratum=None):
        local = collections.Counter(); caps = collections.Counter(); longs = 0
        for i in unit:
            r = rows[i]
            if i in chosen_rows:
                return 'overlap'
            if r['tokens'] > 32768:
                return 'over 32k tokens'
            longs += r['tokens'] > 16384
            local[r['template']] += 1
            tg = tags[i]
            for t in tg:
                if t.startswith('capped:'):
                    caps[t] += 1
        if any(template_count[k] + v > template_cap for k, v in local.items()):
            return 'template cap'
        if any(capped_count[k] + v > cap_per_family for k, v in caps.items()):
            return 'wrapper/template-comm cap'
        if long_rows[0] + longs > long_budget or (stratum and long_by[stratum] + longs > long_limit[stratum]):
            return '16k-32k share'
        # rows that only add to strata which are already full dilute the mix: bounded per unit
        extra = sum(1 for i in unit if not any(count[t] < quota[t] for t in tags[i] if t in QUOTAS))
        plentiful = stratum is None or avail[stratum] >= 2 * quota[stratum]   # scarce strata take their neighbours too
        if plentiful and extra > max(1, 0.25 * len(unit)) and len(unit) > 1:
            return 'collateral rows of complete strata'
        return None

    def assign(unit, stratum):
        for i in unit:
            r = rows[i]
            template_count[r['template']] += 1
            for t in tags[i]:
                if t.startswith('capped:'):
                    capped_count[t] += 1
            long_rows[0] += r['tokens'] > 16384
            long_by[stratum] += r['tokens'] > 16384
            chosen_rows.add(i)
            own = [s for s in QUOTAS if s in tags[i]]
            # a row counts for the scarcest stratum with remaining room; else for the requested one
            pick = next((s for s in sorted(own, key=lambda s: avail[s]) if count[s] < quota[s]), stratum if stratum in own else (own[0] if own else 'other'))
            count[pick] += 1
        chosen_units.append(unit)

    avail = collections.Counter()   # candidate rows in units that pass the static length/cap checks
    for unit in units:
        if all(rows[i]['tokens'] <= 32768 and not any(t.startswith('capped:') for t in tags[i]) for i in unit):
            for i in unit:
                for t in tags[i]:
                    if t in QUOTAS:
                        avail[t] += 1
    order = sorted(QUOTAS, key=lambda s: avail[s])
    by_stratum = collections.defaultdict(list)
    for unit in units:
        present = set()
        for i in unit:
            present |= tags[i] & set(QUOTAS)
        for s in present:
            by_stratum[s].append(unit)
    ranked, pointer = {}, collections.Counter()
    for stratum in order:
        scored = []
        for unit in by_stratum[stratum]:
            capped = any(t.startswith('capped:') for i in unit for t in tags[i])
            purity = sum(1 for i in unit if stratum in tags[i]) / len(unit)
            long_unit = sum(rows[i]['tokens'] for i in unit) > 16384 * len(unit)
            scored.append((capped, -purity, long_unit, rng.random(), unit))
        scored.sort(key=lambda x: x[:4])
        ranked[stratum] = [x[4] for x in scored]
    passes = 0
    while True:
        for stratum in order:
            candidates = ranked[stratum]
            while pointer[stratum] < len(candidates) and count[stratum] < quota[stratum]:
                unit = candidates[pointer[stratum]]; pointer[stratum] += 1
                problem = valid(unit, stratum)
                if problem:
                    excluded[(stratum, problem)] += 1
                    continue
                assign(unit, stratum)
        # A stratum with no candidates left keeps what it has; its shortfall moves to the strata that still have
        # candidates, within its own quota group first (never by duplicating rows).
        moved = granted = 0
        for stratum in order:
            if pointer[stratum] >= len(ranked[stratum]) and count[stratum] < quota[stratum]:
                short = quota[stratum] - count[stratum]
                quota[stratum] = count[stratum]
                short = min(short, max(0, int(1.14 * target) - sum(count.values()) - granted))   # the file stays near the target size
                granted += short
                if short <= 0:
                    continue
                growable = [t for t in order if pointer[t] < len(ranked[t])]
                peers = [t for t in growable if GROUP[t] == GROUP[stratum]] or growable
                if not peers:
                    continue
                weight = sum(QUOTAS[t] for t in peers)
                for t in peers:
                    quota[t] += max(1, int(round(short * QUOTAS[t] / weight)))
                moved += short
        passes += 1
        if not moved or passes >= 8:
            break
    return chosen_units, chosen_rows, tags, quota, count, avail, excluded, template_count


def mixed_order(units, rows, tags):
    """Round-robin units across their dominant stratum for a mixed file order."""
    queues = collections.OrderedDict((s, collections.deque()) for s in QUOTAS)
    queues['other'] = collections.deque()
    for unit in units:
        votes = collections.Counter(s for i in unit for s in tags[i] if s in QUOTAS)
        queues[votes.most_common(1)[0][0] if votes else 'other'].append(unit)
    result = []
    while any(queues.values()):
        for q in queues.values():
            if q:
                result.append(q.popleft())
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train-sources', default=str(OPS / 'train-sources.json'))
    p.add_argument('--geometry', default=str(OPS / 'packed-geometry.json'))
    p.add_argument('--class-dir', default=str(CLASS_DIR))
    p.add_argument('--checkpoint', required=True, help='earned checkpoint whose exposure/quotas/active focus bind the schedule')
    p.add_argument('--scan-cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--target-rows', type=int, default=7000)
    p.add_argument('--seed', type=int, default=20261105)
    p.add_argument('--template-cap', type=int, default=2)
    p.add_argument('--capped-family-cap', type=int, default=40, help='rows of each of the port_record wrapper / big2 template families')
    p.add_argument('--long-share', type=float, default=0.10, help='share of rows allowed between 16k and 32k tokens')
    p.add_argument('--batch-graphs', type=int, default=64)
    p.add_argument('--batch-tokens', type=int, default=65536)
    p.add_argument('--reason', default='2026-11-05 core focus: stratified whole-batch selection replaces the scattered diverse focus')
    args = p.parse_args(argv)

    from natlang_neuralese.data.native_sources import NativeGraphSource
    from natlang_neuralese.train.native_cohort_schedule import source_cohort_ids
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    from natlang_neuralese.train.native_continuation import collect_sft_progress
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    identity = json.loads(Path(args.geometry).read_text())['identity']
    source = NativeGraphSource(Path(args.train_sources), identity['tokenizer'], required_split='train', role_ids=identity['roles'])
    entries, cohorts = source.entries, source_cohort_ids(source)
    manifest = json.loads(Path(args.train_sources).read_text())
    paths = leaf_paths(manifest)
    class_dir = Path(args.class_dir)
    family = {r['id']: r['family'] for r in zjsonl(class_dir / 'successor-row-mass.jsonl.zst')}
    scan = scan_features(entries, cohorts, family, paths, Path(args.scan_cache))
    rows = load_rows(entries, cohorts, class_dir, scan)

    training = CheckpointPolicy.load(Path(args.checkpoint))['training']
    frame = collect_sft_progress(training, borrow_frames=True)['native_joint_sft_v4']
    exposure = dict(frame.get('exposure_counts') or {})
    required = dict(frame.get('required_visits') or {})
    history = (frame.get('options') or {}).get('pending_focus_history') or []
    global_step = int(training['global_step'])
    active = history[-1] if history and history[-1].get('state') == 'active' else None

    base, _ = schedule(entries, cohorts, exposure, required, global_step, args.batch_graphs, args.batch_tokens)
    # a unit is a packed batch whose rows are all at their next pending visit (first promotion point)
    seen, units = set(), []
    for visit, indices, _ in base:
        if all(visit == exposure.get(entries[i]['id'], 0) and i not in seen for i in indices):
            units.append(tuple(indices))
        seen.update(indices)
    chosen, chosen_rows, tags, quota, count, avail, excluded, template_count = select(
        rows, units, args.target_rows, args.seed, args.template_cap, args.capped_family_cap, args.long_share)
    ordered = mixed_order(chosen, rows, tags)
    focus_rows = [i for unit in ordered for i in unit]
    focus_ids = {entries[i]['id'] for i in focus_rows}

    # Simulate the trainer's order of the promoted batches and the realized graph count.
    _, final = schedule(entries, cohorts, exposure, required, global_step, args.batch_graphs, args.batch_tokens, focus_ids)
    promoted = []
    seen_ids = set()
    for visit, indices, _ in final:
        present = {entries[i]['id'] for i in indices} & focus_ids
        if present - seen_ids:
            promoted.append(indices); seen_ids.update(present)
    promoted_rows = [i for b in promoted for i in b]
    report_windows = []
    per_window = 8 * 16
    for w in range(0, len(promoted), per_window):
        cnt = collections.Counter()
        for b in promoted[w:w + per_window]:
            for i in b:
                for t in tags[i] & set(QUOTAS):
                    cnt[GROUP[t]] += 1
        report_windows.append({'batches': len(promoted[w:w + per_window]), 'rows': sum(len(b) for b in promoted[w:w + per_window]),
                               'groups': dict(cnt)})

    robustness = {}
    if active:
        # The window that adopts this focus starts later: rows the active focus earns meanwhile leave the pending
        # lists and shift batch boundaries. Re-derive the promotion after 1..3 windows of that progress.
        from natlang_neuralese.train.native_pending_focus import dependencies as focus_dependencies
        old_doc, _ = focus_dependencies(active['request'])
        position = {e['id']: i for i, e in enumerate(entries)}
        old_ids = {r['id'] for r in old_doc['rows'] if r['id'] in position and r['id'] not in set(active.get('consumed') or [])
                   and exposure.get(r['id'], 0) == (active.get('initial_credits') or {}).get(r['id'], 0)}
        _, old_schedule = schedule(entries, cohorts, exposure, required, global_step, args.batch_graphs, args.batch_tokens, old_ids)
        advanced = dict(exposure)
        for windows in (1, 2, 3):
            for _, indices, _ in old_schedule[(windows - 1) * 128: windows * 128]:
                for i in indices:
                    advanced[entries[i]['id']] = advanced.get(entries[i]['id'], 0) + 1
            _, later = schedule(entries, cohorts, advanced, required, global_step + 8 * windows, args.batch_graphs, args.batch_tokens, focus_ids)
            seen_later, count_later = set(), 0
            for _, indices, _ in later:
                present = {entries[i]['id'] for i in indices} & focus_ids
                if present - seen_later:
                    count_later += len(indices); seen_later.update(present)
            robustness[f'after_{windows}_windows_of_old_focus'] = {'promoted_graphs': count_later,
                'focus_rows_still_pending': len(focus_ids & {e['id'] for e in entries if advanced.get(e['id'], 0) < required.get(e['id'], 1)})}
    focus_doc = {'schema': SCHEMA, 'rows': []}
    report_path = out / 'report.json'
    tokens = [rows[i]['tokens'] for i in focus_rows]
    hist_edges = [2048, 4096, 8192, 16384, 24576, 32768]
    histogram = collections.OrderedDict((f'<= {e}', sum(1 for t in tokens if (t <= e and (not hist_edges.index(e) or t > hist_edges[hist_edges.index(e) - 1])))) for e in hist_edges)
    stratum_rows = collections.Counter(s for i in focus_rows for s in (tags[i] & set(QUOTAS)))
    report = {
        'schema': REPORT_SCHEMA, 'generated_at_global_step': global_step, 'seed': args.seed, 'target_rows': args.target_rows,
        'source': {'path': args.train_sources, 'sha256': source.file_sha256, 'rows': len(entries)},
        'checkpoint': {'path': args.checkpoint, 'earned_visit_rows': sum(1 for v in exposure.values() if v),
                       'global_step': global_step},
        'active_focus': active and {'request': active['request'], 'consumed': len(active.get('consumed') or [])},
        'focus_rows': len(focus_rows), 'focus_batches': len(ordered),
        'simulated_promoted_batches': len(promoted), 'simulated_promoted_graphs': len(promoted_rows),
        'target_quota': {k: int(round(v * args.target_rows)) for k, v in QUOTAS.items()}, 'final_quota': quota, 'assigned_count': dict(count), 'membership_by_stratum': dict(stratum_rows),
        'stratum_availability_in_eligible_units': dict(avail),
        'quota_groups': {g: sum(count[s] for s in QUOTAS if GROUP[s] == g) for g in dict.fromkeys(GROUP.values())},
        'by_family': dict(collections.Counter(rows[i]['family'] for i in focus_rows)),
        'length': {'tokens_total': sum(tokens), 'mean': sum(tokens) / max(1, len(tokens)), 'max': max(tokens or [0]),
                   'share_over_16k': sum(1 for t in tokens if t > 16384) / max(1, len(tokens)), 'histogram': histogram},
        'template_repetition': {'distinct_templates': len(template_count), 'max_rows_per_template': max(template_count.values() or [0]),
                                'rows_in_templates_over_cap': sum(v for v in template_count.values() if v > args.template_cap),
                                'capped_wrapper_rows': sum(1 for i in focus_rows if rows[i]['family'] == 'port-record-wrapper'),
                                'capped_template_comm_rows': sum(1 for i in focus_rows if rows[i]['family'] == 'native-communication-template')},
        'excluded_units': {f'{s}: {why}': n for (s, why), n in sorted(excluded.items())},
        'eligible_units': len(units), 'window_mix_simulated': report_windows, 'robustness_to_adoption_delay': robustness,
        'scope': 'TRAIN rows of the corrected source only; protected/held data never enters the source. No row is duplicated.'}
    report_path.write_text(json.dumps(report, indent=1, sort_keys=True))
    ev = [{'path': str(report_path), 'sha256': sha256_file(report_path)}]
    for i in focus_rows:
        focus_doc['rows'].append({'id': entries[i]['id'], 'binding': source.bindings[entries[i]['id']],
                                  'source_sha256': cohorts[i], 'split': 'train', 'evidence': ev})
    if active:
        focus_doc['supersedes'] = {'request_sha256': active['request']['sha256'], 'reason': args.reason}
    focus_path = out / 'focus.json'
    focus_path.write_text(json.dumps(focus_doc, indent=1, sort_keys=True))
    # The same loader/resolver the trainer applies (native_pending_focus), including the supersession declaration.
    from natlang_neuralese.train.native_pending_focus import dependencies as load_focus, resolve as resolve_focus, supersede_declared_focus
    loaded, _ = load_focus({'path': str(focus_path), 'sha256': sha256_file(focus_path)})
    options = frame.get('options') or {}
    active_ids, retired = resolve_focus(loaded, entries, source.bindings, cohorts, options.get('withdrawal_history') or [])
    if len(active_ids) != len(focus_rows) or retired:
        raise SystemExit('focus does not resolve against the source it was built from')
    supersede_declared_focus([dict(h) for h in history], loaded, request_sha256=sha256_file(focus_path), global_step=global_step)
    print(json.dumps({'focus': str(focus_path), 'sha256': sha256_file(focus_path), 'rows': len(focus_rows),
                      'promoted_graphs_simulated': len(promoted_rows), 'groups': report['quota_groups']}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
