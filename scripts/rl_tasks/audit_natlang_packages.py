#!/usr/bin/env python3
"""Build a structural and task diversity manifest for authored natlang packages.

This is an audit, not a quality score. It measures executable source shape,
normalized syntax, callable topology, prompt scaffold reuse, authored task
labels, fixtures, and cross-package near duplicates. Missing evidence is
reported as unknown. It never converts author labels into semantic coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

TEXT_SUFFIXES = {'.nl', '.ts'}
TOKEN = re.compile(r"[A-Za-z_$][\w$]*|\d+(?:\.\d+)?|(?:===|!==|=>|<=|>=|==|!=|&&|\|\||\?\.|\.\.\.)|[^\s]")
WORD = re.compile(r"[a-z][a-z0-9]+")
NL_SUFFIX = '.nl'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_package(root: Path) -> tuple[dict[str, str], dict]:
    if not root.is_dir():
        raise ValueError(f'not a package directory: {root}')
    files: dict[str, str] = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'symlinks are not allowed in package snapshots: {path}')
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES and path.name != 'natlang.json':
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith('../') or '\\' in rel:
            raise ValueError(f'invalid package-relative path: {rel}')
        files[rel] = path.read_text(encoding='utf-8')
    if not files or not any(path.endswith(NL_SUFFIX) for path in files):
        raise ValueError(f'no .nl sources found under {root}')
    metadata_path = root.parent / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    if not isinstance(metadata, dict):
        raise ValueError(f'{metadata_path} must contain a JSON object')
    for rel in metadata.get('package_assets', []):
        if not isinstance(rel, str) or not rel or rel.startswith('/') or '\\' in rel or any(part in {'.', '..'} for part in rel.split('/')):
            raise ValueError(f'unsafe metadata.package_assets entry: {rel!r}')
        if any(part.lower() in {'worlds', 'tests', 'test', 'fixtures', 'held', 'confirmation', 'gold', 'expected'}
               for part in Path(rel).parts):
            raise ValueError(f'worlds, tests and expected answers cannot be packaged: {rel}')
        asset = root / rel
        if asset.is_symlink() or not asset.is_file():
            raise ValueError(f'metadata.package_assets entry is not a regular file: {rel}')
        files[rel] = asset.read_text(encoding='utf-8')
    return files, metadata


def _frontmatter(source: str) -> tuple[dict[str, str], str]:
    match = re.match(r'^---\n([\s\S]*?)\n---\s*\n?', source)
    fields: dict[str, str] = {}
    if not match:
        return fields, source
    key = None
    for line in match.group(1).splitlines():
        m = re.match(r'^([A-Za-z_][\w-]*):\s?(.*)$', line)
        if m:
            key = m.group(1)
            fields[key] = m.group(2)
        elif key:
            fields[key] += '\n' + line
    return fields, source[match.end():]


def _norm_tokens(text: str, *, prose: bool = False, symbols: set[str] | None = None) -> list[str]:
    # Comments and literal payloads often contain copied domain examples. Keep
    # their syntax marker while excluding payload text from the architecture view.
    text = re.sub(r'/\*[\s\S]*?\*/|//[^\n]*|#[^\n]*', ' ', text)
    out = []
    for token in TOKEN.findall(text):
        if token[0] in "'\"`":
            out.append('STR')
        elif token[0].isdigit():
            out.append('NUM')
        elif token[0].isalpha() or token[0] in '_$':
            low = token.lower()
            if prose:
                out.append('SYMBOL' if symbols and token in symbols else 'TXT:' + low)
            else:
                out.append(low if low in {'export', 'import', 'type', 'interface', 'function', 'async', 'await', 'return', 'const', 'let', 'if', 'else', 'for', 'of', 'new', 'throw', 'try', 'catch', 'uses', 'args', 'returns', 'kind'} else 'ID')
        else:
            out.append(token)
    return out


def _shingles(tokens: list[str], width: int = 5) -> set[tuple[str, ...]]:
    if not tokens:
        return set()
    if len(tokens) < width:
        return {tuple(tokens)}
    return {tuple(tokens[i:i + width]) for i in range(len(tokens) - width + 1)}


def _similarity(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _package_record(root: Path) -> tuple[dict, dict]:
    files, metadata = _read_package(root)
    nl = {path: text for path, text in files.items() if path.endswith('.nl')}
    ts = {path: text for path, text in files.items() if path.endswith('.ts')}
    parents: Counter[str] = Counter()
    uses_fanout = []
    max_depth = 0
    symbols = {part.rsplit('.', 1)[0] for path in files for part in [Path(path).name]}
    for path in sorted(set(nl) | set(ts)):
        parent = path.rsplit('/', 1)[0] if '/' in path else ''
        parents[parent] += 1
        max_depth = max(max_depth, path.count('/'))
    for path, source in nl.items():
        fm, body = _frontmatter(source)
        uses = fm.get('uses', '')
        # `uses` is declared visibility, not an observed execution/call graph.
        uses_fanout.append(len([x for x in re.split(r'[,\n\[\]\s]+', uses) if x and x not in {'uses'}]))
        symbols.update(re.findall(r'\b[A-Za-z_$][\w$]*\b', fm.get('args', '') + ' ' + fm.get('returns', '') + ' ' + uses))
    topology = {
        'nl_functions': len(nl), 'typescript_modules': len(ts),
        'root_functions': sum('/' not in path for path in nl),
        'max_callable_depth': max_depth,
        'callable_folder_degree_multiset': sorted(parents.values()),
        'declared_uses_fanout_multiset': sorted(uses_fanout),
        'uses_measurement': 'static declared visibility only; not an observed execution graph',
    }
    syntax = []
    prompt = []
    for path, body in sorted(files.items()):
        if path.endswith('.nl'):
            fm, prose = _frontmatter(body)
            signature = {k: _norm_tokens(v) for k, v in fm.items() if k in {'args', 'returns', 'uses', 'kind'}}
            syntax.extend(['NL', json.dumps(signature, sort_keys=True), *_norm_tokens(prose)])
            prompt.extend(_norm_tokens(prose, prose=True, symbols=symbols))
        elif path.endswith('.ts'):
            syntax.extend(['TS', *_norm_tokens(body)])
    syntax_fingerprint = sha(' '.join(syntax).encode())
    prompt_fingerprint = sha(' '.join(prompt).encode())
    source_bytes = json.dumps(files, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    source_sha = sha(source_bytes)
    normalized_tokens = syntax
    prompt_tokens = prompt
    task_rows = []
    worlds = root.parent / 'worlds'
    if worlds.is_dir():
        for path in sorted(worlds.glob('*.jsonl')):
            for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f'{path}:{line_no}: fixture row must be an object')
                task_rows.append(row)
    exact = sum(1 for row in task_rows if row.get('oracle', 'exact') == 'exact')
    judged = sum(1 for row in task_rows if isinstance(row.get('oracle'), dict) and row['oracle'].get('level') == 'judged')
    target_hashes = [sha(json.dumps(row.get('expected'), sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())
                     for row in task_rows if 'expected' in row]
    target_counts = Counter(target_hashes)
    task_tokens = []
    for row in task_rows:
        if isinstance(row.get('task'), str):
            task_tokens.extend(_norm_tokens(row['task'], prose=True, symbols=symbols))
    actual_task_families = Counter(row.get('task_family') for row in task_rows if isinstance(row.get('task_family'), str))
    actual_operations = Counter(row.get('operation') for row in task_rows if isinstance(row.get('operation'), str))
    actual_domains = Counter(row.get('domain') for row in task_rows if isinstance(row.get('domain'), str))
    actual_failure_modes = Counter(row.get('failure_mode') for row in task_rows if isinstance(row.get('failure_mode'), str))
    record = {
        'program_id': metadata.get('program_id') or root.parent.name,
        'package_path': str(root.resolve()),
        'source_sha256': source_sha,
        'files': len(files), 'bytes': sum(len(x.encode()) for x in files.values()),
        'topology': topology,
        'normalized_syntax_sha256': syntax_fingerprint,
        'normalized_prompt_sha256': prompt_fingerprint,
        'domain': metadata.get('domain'),
        'user_goal': metadata.get('user_goal'),
        'decision_types': metadata.get('decision_types'),
        'failure_modes': metadata.get('failure_modes'),
        'task_families': metadata.get('task_families'),
        'world_rows': len(task_rows), 'exact_world_rows': exact, 'judged_world_rows': judged,
        'unique_task_prompts': len({sha(str(row.get('task', '')).encode()) for row in task_rows if row.get('task')}),
        'unique_expected_targets': len(target_counts),
        'repeated_target_groups': sum(1 for count in target_counts.values() if count > 1),
        'max_target_repetition': max(target_counts.values(), default=0),
        'target_repetition_share': round(max(target_counts.values(), default=0) / max(1, len(target_hashes)), 5),
        'world_operation_counts': dict(sorted(actual_operations.items())),
        'world_task_family_counts': dict(sorted(actual_task_families.items())),
        'world_domain_counts': dict(sorted(actual_domains.items())),
        'world_failure_mode_counts': dict(sorted(actual_failure_modes.items())),
        'semantic_coverage': 'unknown' if not exact else 'exact-fixture rows present; independent fixture review and runtime qualification still required',
        'unknown_fields': [key for key, value in {
            'domain': metadata.get('domain'), 'user_goal': metadata.get('user_goal'),
            'decision_types': metadata.get('decision_types'), 'failure_modes': metadata.get('failure_modes'),
            'task_families': metadata.get('task_families'), 'exact_world_rows': exact,
        }.items() if not value],
        '_syntax': _shingles(normalized_tokens), '_prompt': _shingles(prompt_tokens),
        '_task_prompts': _shingles(task_tokens), '_targets': set(target_hashes),
    }
    return record, {'files': files, 'metadata': metadata, 'world_rows': task_rows}


def audit(package_roots: list[Path], *, near_threshold: float = .88, max_near_partners: int = 1) -> dict:
    if not package_roots:
        raise ValueError('provide at least one package root')
    records, internals = [], []
    seen_ids = set()
    for root in package_roots:
        record, internal = _package_record(root.resolve(strict=True))
        if record['program_id'] in seen_ids:
            raise ValueError(f'duplicate program_id: {record["program_id"]}')
        seen_ids.add(record['program_id'])
        records.append(record)
        internals.append(internal)
    near = []
    partners: Counter[int] = Counter()
    for i, left in enumerate(records):
        for j in range(i + 1, len(records)):
            right = records[j]
            syntax = _similarity(left['_syntax'], right['_syntax'])
            prompt = _similarity(left['_prompt'], right['_prompt'])
            task_prompt = _similarity(left['_task_prompts'], right['_task_prompts'])
            target_overlap = len(left['_targets'] & right['_targets']) / max(1, len(left['_targets'] | right['_targets']))
            if syntax >= near_threshold or prompt >= near_threshold or task_prompt >= near_threshold or target_overlap >= near_threshold:
                near.append({'left': left['program_id'], 'right': right['program_id'],
                             'normalized_syntax_jaccard': round(syntax, 5),
                             'prompt_scaffold_jaccard': round(prompt, 5),
                             'world_task_prompt_jaccard': round(task_prompt, 5),
                             'expected_target_overlap': round(target_overlap, 5),
                             'reason': 'near-duplicate normalized source or prompt scaffold; review shared boilerplate and target duplication'})
                partners[i] += 1
                partners[j] += 1
    for record in records:
        record.pop('_syntax', None)
        record.pop('_prompt', None)
        record.pop('_task_prompts', None)
        record.pop('_targets', None)
    domains = Counter({})
    task_families = Counter({})
    failure_modes = Counter({})
    operations = Counter({})
    for record in records:
        domains.update(record['world_domain_counts'] or ({record['domain']: record['world_rows']} if record['domain'] else {}))
        task_families.update(record['world_task_family_counts'])
        failure_modes.update(record['world_failure_mode_counts'])
        operations.update(record['world_operation_counts'])
    identical_sources = Counter(r['source_sha256'] for r in records)
    cap_violations = [r['program_id'] for r in records if partners[records.index(r)] > max_near_partners]
    return {
        'schema': 'natlang.codebase_diversity_manifest/1',
        'method': {'normalized_syntax': 'comments and identifier/literal names normalized; callable topology measured separately',
                   'near_duplicate_similarity': 'heuristic Jaccard over five-token source and lexical prompt shingles; not semantic proof',
                   'near_duplicate_threshold': near_threshold,
                   'max_near_duplicate_partners': max_near_partners,
                   'semantic_claim_policy': 'labels are metadata; semantic coverage is unknown without independently reviewed fixtures and exact runtime evidence'},
        'summary': {'packages': len(records), 'exact_duplicate_source_hash_groups': sum(1 for n in identical_sources.values() if n > 1),
                    'near_duplicate_pairs': len(near), 'near_duplicate_cap_violations': len(cap_violations),
                    'domains_observed': len(domains), 'task_families_observed': len(task_families),
                    'failure_modes_observed': len(failure_modes), 'operations_observed': len(operations),
                    'packages_with_unknown_semantic_coverage': sum(r['semantic_coverage'] == 'unknown' for r in records)},
        'counts': {'domain': dict(sorted(domains.items())), 'task_family': dict(sorted(task_families.items())),
                   'operation': dict(sorted(operations.items())),
                   'failure_mode': dict(sorted(failure_modes.items()))},
        'near_duplicates': near,
        'near_duplicate_cap_violations': cap_violations,
        'packages': records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, action='append', required=True,
                        help='package/ directory; repeat for cross-package comparison')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--near-threshold', type=float, default=.88)
    parser.add_argument('--max-near-partners', type=int, default=1)
    parser.add_argument('--allow-cap-violations', action='store_true')
    args = parser.parse_args()
    report = audit(args.package, near_threshold=args.near_threshold,
                   max_near_partners=args.max_near_partners)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({'output': str(args.output), **report['summary']}, sort_keys=True))
    if report['near_duplicate_cap_violations'] and not args.allow_cap_violations:
        raise SystemExit('near-duplicate partner cap exceeded; inspect manifest before intake')


if __name__ == '__main__':
    main()
