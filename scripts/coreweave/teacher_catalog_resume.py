"""Continue a task catalog by source identity instead of freezing its registry.

The supplied historical policy is authenticated against the catalog's inputs;
its discovery and extraction functions remain the sole policy implementation.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import sqlite3
import subprocess
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
from natlang_neuralese.common.jsonio import iter_jsonl_rows, write_json_durable, write_jsonl_text
from scripts.coreweave.teacher_catalog_storage import encode_body, decode_body
from scripts.coreweave.teacher_generation_guardian import binding
from scripts.coreweave.teacher_reference_append import read, immutable, eligible_program_sql

_STATIC = ('scripts/source_review.py', 'scripts/neuralese_data/decision_holds.py',
           'training/source-reviews/datasets.json', 'training/source-reviews/family-holds.json',
           'scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py',
           'scripts/neuralese_data/splits.py',
           'training/neuralese/natlang_neuralese/common/hashing.py',
           'training/neuralese/natlang_neuralese/common/jsonio.py')


def load_policy(reference):
    read_bytes = Path(reference['path']).read_bytes()
    import hashlib
    if len(read_bytes) != reference['bytes'] or hashlib.sha256(read_bytes).hexdigest() != reference['sha256']:
        raise ValueError('historical catalog policy changed')
    spec = importlib.util.spec_from_file_location('bound_catalog_policy', reference['path'])
    module = importlib.util.module_from_spec(spec)
    # Compile the bytes just verified, avoiding a second mutable source read.
    exec(compile(read_bytes, reference['path'], 'exec'), module.__dict__)
    return module


def discover_sources(repo, registry, initial):
    """Keep historical ordering, but derive membership from current includes."""
    entries = {}; corpora = []
    for corpus in registry['corpora']:
        if not corpus.get('training_admission'):
            continue
        root = Path(repo) / corpus['path']; matched = set(); patterns = []
        for pattern in corpus.get('include', []):
            if 'turns' not in pattern or not ('.jsonl' in pattern or '**' in pattern):
                continue
            patterns.append(pattern)
            matched.update(str(p.absolute()) for p in root.glob(pattern)
                           if p.is_file() and p.name.endswith(('.jsonl', '.jsonl.zst')))
        if patterns:
            corpora.append(dict(corpus_id=corpus['id'], root=str(root), include=patterns,
                                root_exists=root.exists(), matched_files=len(matched)))
        for path in matched:
            entry = entries.setdefault(path, dict(path=path, corpora=[], priority=1))
            entry['corpora'].append(corpus['id'])
    for path, prior in initial['source_files'].items():
        if path in entries:
            entries[path].update(priority=0, expected_sha256=prior['sha256'])
    return sorted(entries.values(), key=lambda f:(f['priority'],f['path'])), corpora


def changed_files(db, old_groups, new_groups, *, general=False):
    """Select original prefixes needing a policy audit without losing cursors."""
    if general:
        return {row[0] for row in db.execute('SELECT path FROM files WHERE cursor>0')}
    changed = set().union(*(old_groups[k] ^ new_groups[k] for k in old_groups))
    if not changed:
        return set()
    db.execute('CREATE TEMP TABLE changed_groups(name TEXT PRIMARY KEY)')
    db.executemany('INSERT INTO changed_groups VALUES(?)', ((g,) for g in sorted(changed)))
    try:
        return {row[0] for row in db.execute('''SELECT DISTINCT o.file FROM occurrences o,
            json_each(o.groups_json) g JOIN changed_groups c ON c.name=g.value''')}
    finally:
        db.execute('DROP TABLE changed_groups')


def store_row(db, policy, path, line, row, protection, holds):
    program = (row.get('task') or {}).get('program_ir')
    if not isinstance(program, dict):
        db.execute('INSERT OR REPLACE INTO omissions VALUES(?,?,?)', (path, line, 'no_program_ir'))
        db.execute('DELETE FROM occurrences WHERE file=? AND line=?', (path, line))
        return
    sha = digest(program)
    reason = policy.reason_for(row, program, protection, holds)
    row_sha = digest(row)
    previous = db.execute('SELECT row_sha,program_sha FROM occurrences WHERE file=? AND line=?',
                          (path, line)).fetchone()
    if previous and previous != (row_sha, sha):
        raise ValueError('committed source row identity changed')
    family = (program.get('curriculum') or {}).get('family') or program.get('family')
    if not previous:
        db.execute('INSERT OR IGNORE INTO programs VALUES(?,?,?,?)',
                   (sha, program.get('id'), encode_body(policy.encode(program)), family))
    groups = sorted(set(row.get('source_groups') or []) | set(program.get('source_groups') or []))
    db.execute('INSERT OR REPLACE INTO occurrences VALUES(?,?,?,?,?,?)',
               (path, line, row_sha, sha, reason, policy.encode(groups)))
    db.execute('DELETE FROM omissions WHERE file=? AND line=?', (path, line))


def scan_file(db, policy, entry, protection, holds, *, audit_cursor=None, audit_save=None):
    path = Path(entry['path']); key = str(path)
    before = binding(path)
    expected = entry.get('expected_sha256')
    if expected and expected != before['sha256']:
        raise ValueError('declared source bytes changed: ' + key)
    previous = db.execute('SELECT sha,bytes,cursor,complete FROM files WHERE path=?', (key,)).fetchone()
    if previous and previous[0] is not None and tuple(previous[:2]) != (before['sha256'], before['bytes']):
        raise ValueError('committed source identity changed: ' + key)
    auditing = audit_cursor is not None
    if not auditing and previous and previous[3]:
        return
    limit = previous[2] if auditing and previous else None
    cursor = audit_cursor if auditing else (previous[2] if previous else 0)
    if not auditing:
        db.execute('INSERT OR REPLACE INTO files VALUES(?,?,?,?,?,?)',
                   (key, before['sha256'], before['bytes'], cursor, 0, None)); db.commit()
    last = cursor
    for line, (_, row) in enumerate(iter_jsonl_rows(path), 1):
        if limit is not None and line > limit:
            break
        if line <= cursor:
            continue
        last = line
        store_row(db, policy, key, line, row, protection, holds)
        if line % 1000 == 0:
            policy.floor(path.parent)
            if not auditing:
                db.execute('UPDATE files SET cursor=? WHERE path=?', (line, key))
            db.commit()
            # Commit first: an interrupted sidecar write only replays idempotent
            # row replacements; it cannot claim work not present in SQLite.
            if auditing:
                audit_save(line)
    if binding(path) != before:
        db.rollback()
        raise ValueError('source changed during scan: ' + key)
    if auditing:
        if last != limit:
            raise ValueError('source ended before its committed prefix: ' + key)
    else:
        db.execute('UPDATE files SET cursor=?,complete=1,reason=NULL WHERE path=?', (last, key))
    db.commit()
    if auditing:
        audit_save(last)


def resume(out, parent_inputs, parent_inventory, protection_path, *, legacy_static_baseline=None,
           bootstrap_review=None):
    out = Path(out)
    parent = read(binding(Path(parent_inputs)))
    population = read(binding(Path(parent_inventory)))
    if population['input_sha256'] != digest(parent):
        raise ValueError('historical source population authority differs')
    policy = load_policy(parent['mechanism'])
    policy.PROTECTION = Path(protection_path)
    protection, holds, current = policy.inputs()
    registry = read(current['registry'])
    current_files, corpora = discover_sources(policy.REPO, registry, read(current['initial_selection']))
    static = [binding(policy.REPO / p) for p in _STATIC]
    latest = out / 'adaptive-catalog-state.json'
    if latest.exists():
        prior_state = json.loads(latest.read_text())
        prior = read(prior_state['authority'])
        old_groups = {k: set(v) for k, v in prior['groups'].items()}
        general = (prior['static_guards'] != static or prior['decision_holds'] != current['decision_holds']
                   or prior['current']['source_reviews_sha256'] != current['source_reviews_sha256']
                   or prior['current']['mechanism'] != current['mechanism'])
    else:
        old_protection = policy.current_protection(Path(parent['protection']['path']), verify_current=False)
        old_groups = {k: old_protection[k] for k in ('held', 'quality', 'pending')}
        general = True
        if legacy_static_baseline is not None and bootstrap_review is not None:
            # Explicit bootstrap from an independently reviewed, unchanged Git
            # policy baseline. This is recorded, never invented as old metadata.
            for relative, item in zip(_STATIC, static):
                raw = subprocess.check_output(['git', '-C', str(policy.REPO), 'show',
                                               legacy_static_baseline + ':' + relative])
                import hashlib
                if hashlib.sha256(raw).hexdigest() != item['sha256']:
                    raise ValueError('legacy static guard baseline differs: ' + relative)
            review = read(binding(Path(bootstrap_review)))
            expected = dict(parent_inputs=binding(Path(parent_inputs)), policy=parent['mechanism'],
                            static_guards=static, baseline_commit=legacy_static_baseline,
                            source_reviews_sha256=current['source_reviews_sha256'],
                            decision_holds=current['decision_holds'])
            if review.get('reviewed') is not True or review.get('basis') != expected or not review.get('reviewer'):
                raise ValueError('legacy static reuse requires exact independent review evidence')
            general = (parent['decision_holds'] != current['decision_holds']
                       or parent['source_reviews_sha256'] != current['source_reviews_sha256'])
    groups = {k: protection[k] for k in ('held', 'quality', 'pending')}
    authority = dict(current=current, static_guards=static,
                     decision_holds=current['decision_holds'], groups={k: sorted(v) for k,v in groups.items()},
                     source_population_sha256=digest(dict(files=current_files, corpora=corpora)),
                     scanner=binding(Path(__file__)), parent_inputs=binding(Path(parent_inputs)),
                     parent_inventory=binding(Path(parent_inventory)), legacy_static_baseline=legacy_static_baseline,
                     bootstrap_review=binding(Path(bootstrap_review)) if bootstrap_review else None)
    key = digest(authority)
    authority_ref = immutable(out / ('adaptive-authority-' + key + '.json'), authority)
    inventory_ref = immutable(out / ('adaptive-population-' + key + '.json'),
                              dict(files=current_files, corpora=corpora, authority=authority_ref))
    audit_file = out / ('adaptive-audit-' + key + '.json')
    audit = json.loads(audit_file.read_text()) if audit_file.exists() else {}
    with (out / 'adaptive-catalog.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Policy selection above is read-only. Exclude stale selection after
        # another scanner completes while this caller is acquiring the lease.
        if latest.exists() != ('prior_state' in locals()) or (latest.exists() and json.loads(latest.read_text()) != prior_state):
            raise ValueError('catalog authority advanced while acquiring lease; retry continuation')
        db = sqlite3.connect(out / 'catalog.sqlite')
        try:
            db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
            affected = changed_files(db, old_groups, groups, general=general)
            by_path = {f['path']: f for f in current_files}
            inflight = out / 'adaptive-policy-inflight.json'
            previous_intent = json.loads(inflight.read_text()) if inflight.exists() else None
            if previous_intent:
                affected.update(previous_intent['affected_files'])
                if previous_intent['authority'] != authority_ref:
                    # A→B interrupted→A must audit every prefix B may have
                    # rewritten, even though the last completed authority is A.
                    for path in previous_intent['affected_files']:
                        audit.pop(path, None)
            new_or_incomplete = {p for p in by_path if not db.execute(
                'SELECT 1 FROM files WHERE path=? AND complete=1', (p,)).fetchone()}
            touched = affected | new_or_incomplete
            write_json_durable(inflight, dict(authority=authority_ref, affected_files=sorted(touched)))
            for path in sorted(affected & by_path.keys()):
                previous = db.execute('SELECT cursor FROM files WHERE path=?', (path,)).fetchone()
                if not previous or previous[0] == 0:
                    continue
                def saved(cursor, path=path):
                    audit[path] = cursor; write_json_durable(audit_file, audit)
                scan_file(db, policy, by_path[path], protection, holds,
                          audit_cursor=audit.get(path, 0), audit_save=saved)
            for entry in current_files:
                policy.floor(out)
                if not Path(entry['path']).is_file():
                    continue  # Explicit omission recorded below, never complete.
                scan_file(db, policy, entry, protection, holds)
                write_json_durable(out / 'adaptive-scan-progress.json', dict(last_file=entry['path'], authority=authority_ref))
            # A registry append during scanning does not discard any source
            # cursor. The next invocation discovers its new files.
            stable = (policy.inputs()[2] == current
                      and [binding(policy.REPO / p) for p in _STATIC] == static)
            result = dict(authority=authority_ref, population=inventory_ref,
                          authority_stable=stable, training_admission=False,
                          missing_sources=[f['path'] for f in current_files if not Path(f['path']).is_file()],
                          complete_files=db.execute('SELECT count(*) FROM files WHERE complete=1').fetchone()[0])
            write_json_durable(latest, result)
            deferred = touched - by_path.keys()
            if deferred:
                write_json_durable(inflight, dict(authority=authority_ref, affected_files=sorted(deferred)))
            else:
                inflight.unlink()
            return result
        finally:
            db.close()


def export_candidates(out, state_reference):
    """Publish a compressed candidate inventory using only the current sources."""
    out = Path(out)
    state = read(state_reference)
    if not state['authority_stable'] or state['missing_sources']:
        raise ValueError('candidate publication needs a stable complete source population')
    authority = read(state['authority']); population = read(state['population'])
    policy = load_policy(authority['current']['mechanism'])
    policy.PROTECTION = Path(authority['current']['protection']['path'])
    def current_authority():
        return (policy.inputs()[2] == authority['current']
                and [binding(policy.REPO / p) for p in _STATIC] == authority['static_guards'])
    if not current_authority():
        raise ValueError('candidate publication authority changed; continue the existing catalog')
    key = state['authority']['sha256']
    proof_path = out / ('adaptive-candidates-' + key + '-proof.json')
    target = out / ('adaptive-candidates-' + key + '.jsonl.zst')
    partial = target.with_name(target.stem + '.pending.zst')
    with (out / 'adaptive-catalog.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(binding(out / 'adaptive-catalog-state.json')) != state:
            raise ValueError('candidate state was superseded; continue the current catalog')
        inflight = out / 'adaptive-policy-inflight.json'
        active_paths = {f['path'] for f in population['files']}
        if inflight.exists() and active_paths & set(json.loads(inflight.read_text())['affected_files']):
            raise ValueError('candidate population has an incomplete policy audit')
        db = sqlite3.connect(out / 'catalog.sqlite')
        try:
            db.execute('CREATE TEMP TABLE active_sources(path TEXT PRIMARY KEY)')
            db.executemany('INSERT INTO active_sources VALUES(?)', ((f['path'],) for f in population['files']))
            missing = db.execute('''SELECT a.path FROM active_sources a LEFT JOIN files f ON f.path=a.path
                WHERE f.complete IS NULL OR f.complete!=1''').fetchall()
            if missing:
                raise ValueError('candidate population has incomplete sources')
            def verify_sources():
                for path, sha, size in db.execute('''SELECT f.path,f.sha,f.bytes FROM files f
                        JOIN active_sources a ON a.path=f.path'''):
                    actual = binding(Path(path))
                    if actual['sha256'] != sha or actual['bytes'] != size:
                        raise ValueError('candidate source bytes changed: ' + path)
            verify_sources()
            if proof_path.exists():
                proof = read(binding(proof_path))
                if binding(Path(proof['cases_binding']['path'])) != proof['cases_binding']:
                    raise ValueError('published candidate artifact changed')
                return binding(proof_path)
            # Only current sources grant eligibility/provenance. Historical
            # protected occurrences and identity conflicts remain barriers:
            # withdrawing a source does not turn its held content into training.
            query = 'SELECT p.sha,p.body FROM programs p WHERE ' + eligible_program_sql() + ' ORDER BY p.rowid'
            count = 0
            with write_jsonl_text(partial) as stream:
                for sha, body in db.execute(query):
                    stream.write(decode_body(body, sha) + '\n'); count += 1
                    if count % 100 == 0:
                        policy.floor(out)
            verify_sources()
            if not current_authority():
                raise ValueError('publication authority changed; completed scan remains usable')
            import os
            with partial.open('rb') as stream:
                os.fsync(stream.fileno())
            try:
                os.link(partial, target)
            except FileExistsError:
                previous, actual = binding(target), binding(partial)
                if any(previous[k] != actual[k] for k in ('sha256', 'bytes')):
                    raise ValueError('existing immutable candidate artifact differs')
            partial.unlink()
            fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            summary = dict(schema='natlang.strong-teacher-native-program-catalog/1',
                           inputs=authority['current'], source_population=state['population'],
                           cases=count, training_admission=False, generator_calls=0,
                           source_population_files=len(population['files']))
            summary_ref = immutable(out / ('adaptive-summary-' + key + '.json'), summary)
            return immutable(proof_path, dict(cases=count, cases_binding=binding(target),
                catalog_summary=summary_ref, inputs=authority['current'], source_population=state['population'],
                scope='Candidate extraction only; reference execution and fresh publication protection required',
                training_admission=False, generator_calls=0))
        finally:
            db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--parent-inputs', type=Path, required=True)
    parser.add_argument('--parent-inventory', type=Path, required=True)
    parser.add_argument('--protection', type=Path, required=True)
    parser.add_argument('--legacy-static-baseline')
    parser.add_argument('--bootstrap-review', type=Path)
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    result = resume(args.out, args.parent_inputs, args.parent_inventory, args.protection,
                    legacy_static_baseline=args.legacy_static_baseline, bootstrap_review=args.bootstrap_review)
    if args.export:
        result['candidates'] = export_candidates(args.out, binding(args.out / 'adaptive-catalog-state.json'))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
