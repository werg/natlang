#!/usr/bin/env python3
"""Invented programs (`natlang.invented-program/1`) -> crisp requirement checks, `natlang.program/2` cases, qualification.

The program inventor (`applications/program-inventor/`, plans/NOVEL_PROGRAMS_20261124.md) builds one complete natlang
program per new idea, with two or three literal input instances, exact checkers and controls
(`applications/program-inventor/BUILD_BRIEF.md`). This module is the crisp mechanism around it:

  invented.py check SUBMISSION            static crisp requirements (requirements.json, check "crisp"), metrics
  invented.py package SUBMISSION OUT      one `natlang.program/2` case per instance (the collector's case format)
  invented.py qualify SUBMISSION [--runtime TS_HOST]
      check + `natlang check --json` (F1) + package + scripts/rl_tasks/verify.mjs (reference replay F3, wrong answers
      F4, keyword baselines S3) + scripts/rl_tasks/ablate.mjs (judgment ablations S2); writes
      SUBMISSION/qualification/{cases.jsonl,report.json}. Run through the memory ledger on the DGX.

It never invents gold, wrong answers or references: everything it judges with comes from the submission. Each idea is
its own family (`inv_<slug>`) and world (`invented:<slug>`): all instances of a program share one split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ.get('NATLANG_REPO_ROOT') or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(REPO))
from scripts.rl_tasks.common import CURRICULUM_VERSION, slug, text_digest  # noqa: E402

SCHEMA = 'natlang.invented-program/1'
GENERATOR = 'natlang.invented-programs/1'
ORACLES = frozenset({'exact', 'normalized', 'verifier'})
VERIFIER_ROOT = 'training/semantic_fp_tasks/verifiers/'
SUBMISSION_ENTRIES = frozenset({'program', 'submission.json', 'NOTES.md', 'qualification'})
PROGRAM_EXTRA = frozenset({'natlang.json', 'package.json', 'tsconfig.json', 'examples.json', '.natlang', 'node_modules'})
STD = ('semanticMap', 'semanticFilter', 'semanticPartition', 'semanticReduce', 'semanticMatch', 'semanticRank',
       'semanticMerge', 'semanticRepair', 'fixedPoint', 'proposeAndVerify', 'branchAndSelect', 'escalate',
       'withContext', 'compileRule')
# Semantic-FP constructs found statically (D2). Each pattern must appear in program source, not in comments only.
CONSTRUCT_PATTERNS = {
    'decide': r'\bdecide\s*\(|^readout:\s*(decision|instant)\s*$',
    'std-combinator': r'\b(' + '|'.join(STD) + r')\s*\(',
    'higher-order': r'\.partial\s*\(|^\s+\w+\??:\s*["\']?\(.*\)\s*=>|=>\s*Promise<|returns:\s*["\']?\(',
    'iterate-until': r'\biterateOn\s*\([\s\S]*?\.until\s*\(',
    'structural-recursion': r'^(recursive|measure):',
    'ensures': r'^ensures:',
    'refinement': r'\bIs<|\bUntrusted<|\buntrusted\s*\(',
    'neuralese': r'\bNeuralese<',
    'folder-transaction': r'\.(propose|accept|apply)\s*\(',
    'pluggable': r'\bpluggable\s*\(',
    'inline-nl': r'\bnl\s*(<[^`]*>)?\s*`',
}
INLINE_NL = re.compile(r'\bnl\s*(?:<[^`]*>)?\s*`([^`]*)`')
TS_EXPORT = re.compile(r'^export\s+(?:default\s+)?(?:async\s+)?function\b', re.M)


def load(submission: Path) -> tuple[dict, dict[str, str]]:
    data = json.loads((submission / 'submission.json').read_text())
    program = submission / 'program'
    files = {}
    for path in sorted(program.rglob('*')):
        rel = path.relative_to(program).as_posix()
        if path.is_file() and rel.split('/')[0] not in PROGRAM_EXTRA and not rel.startswith('.'):
            files[rel] = path.read_text()
    return data, files


def frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith('---'):
        return '', text
    _, head, body = text.split('---', 2)
    return head, body


def strip_comments(text: str) -> str:
    return re.sub(r'//[^\n]*|/\*[\s\S]*?\*/', '', text)


def words(value) -> int:
    if isinstance(value, str):
        return len(value.split())
    if isinstance(value, dict):
        return sum(words(v) for v in value.values())
    if isinstance(value, list):
        return sum(words(v) for v in value)
    return 0


def longest_list(value) -> int:
    if isinstance(value, list):
        return max([len(value), *(longest_list(v) for v in value)])
    if isinstance(value, dict):
        return max([0, *(longest_list(v) for v in value.values())])
    return 0


def metrics(data: dict, files: dict[str, str]) -> dict:
    root = data['root']
    stem = root[:-3]
    nl_files = [p for p in files if p.endswith('.nl')]
    ts_files = [p for p in files if p.endswith('.ts') and p != 'types.ts' and p.startswith(stem + '/')]
    inline = sorted({m.group(1).strip()[:80] for p in ts_files for m in INLINE_NL.finditer(files[p])})
    exported = sum(len(TS_EXPORT.findall(files[p])) for p in ts_files)
    ts_lines = sum(1 for p in ts_files for line in strip_comments(files[p]).splitlines()
                   if line.strip() and not line.strip().startswith(('import ', 'export type', 'type ')))
    # Call depth: root (1) -> an item of its callable folder (2) -> a judgment that item calls: a .nl in the item's own
    # folder, a sibling .nl a TypeScript helper imports, an inline nl in it, or a std combinator it runs (3).
    depth = 1 + bool([p for p in files if p.startswith(stem + '/')])
    for path in files:
        if not path.startswith(stem + '/') or path.count('/') != 1:
            continue
        own = path.rsplit('.', 1)[0] + '/'
        text = files[path]
        if any(p.startswith(own) for p in files) or (path.endswith('.ts') and (
                re.search(r"from\s+'\./[^']+\.nl'", text) or INLINE_NL.search(text)
                or re.search(r'\b(' + '|'.join(STD) + r')\s*\(', text))):
            depth = 3
    for path in files:
        if path.startswith(stem + '/') and path.count('/') >= 2:
            depth = max(depth, path.count('/') + 1)
    source = '\n'.join(strip_comments(text) if not path.endswith('.nl') else text for path, text in files.items())
    constructs = sorted(name for name, pattern in CONSTRUCT_PATTERNS.items() if re.search(pattern, source, re.M))
    judgment_files = [p for p in nl_files if p != root]
    instances = data.get('instances', [])
    return {'nl_functions': len(nl_files), 'judgment_files': len(judgment_files), 'inline_nl': len(inline),
            'ts_helpers': len(ts_files), 'ts_exported_functions': exported, 'ts_logic_lines': ts_lines,
            'functions': len(nl_files) + exported, 'call_depth': depth, 'constructs': constructs,
            'instances': [{'name': i.get('name'), 'words': words(i.get('inputs')), 'longest_list': longest_list(i.get('inputs'))}
                          for i in instances]}


def static_problems(submission: Path, data: dict, files: dict[str, str], m: dict) -> list[str]:
    problems = []
    if data.get('schema') != SCHEMA:
        problems.append(f'schema must be {SCHEMA}')
    root = data.get('root', '')
    if root not in files or '/' in root or not root.endswith('.nl'):
        return problems + ['root must name a top-level .nl file of program/']
    stem = root[:-3]
    for path in files:
        if path not in (root, 'types.ts') and not path.startswith(stem + '/'):
            problems.append(f'program/{path} must live in the root callable folder {stem}/ (or be types.ts)')
    # N2: no generator anywhere in the submission
    for entry in submission.iterdir():
        if entry.name not in SUBMISSION_ENTRIES and not entry.name.startswith('.'):
            problems.append(f'N2: {entry.name} is not part of a submission (program/, submission.json, NOTES.md); '
                            'instances are literal data, not generated')
    instances = data.get('instances') or []
    if not 2 <= len(instances) <= 3:
        problems.append('F2: two or three instances')
    canonical = lambda value: json.dumps(value, sort_keys=True)  # noqa: E731
    helpers = {Path(p).stem for p in files if p.startswith(stem + '/') and p.count('/') == 1}
    judgments = data.get('judgments') or []
    names = {j.get('name') for j in judgments}
    for index, inst in enumerate(instances):
        label = f'instance {inst.get("name", index)}'
        if inst.get('oracle') not in ORACLES:
            problems.append(f'F2: {label}: oracle must be exact, normalized or verifier')
        if inst.get('oracle') == 'verifier':
            path = (inst.get('verifier') or {}).get('path', '')
            if not path.startswith(VERIFIER_ROOT) or not (REPO / path).is_file():
                problems.append(f'F2: {label}: verifier must be an existing module under {VERIFIER_ROOT}')
        if 'expected' not in inst:
            problems.append(f'F2: {label}: expected is missing')
        reference = inst.get('reference') or {}
        root_actions = reference.get('root') or []
        code = ' '.join(str(a[1].get('code', '')) for a in root_actions if isinstance(a, list) and a and a[0] == 'eval')
        if not code or not any(re.search(rf'\b{re.escape(h)}\s*\(', code) for h in helpers):
            problems.append(f'F3: {label}: the reference root eval must call the program\'s own helpers')
        if not reference.get('children'):
            problems.append(f'F3: {label}: the reference needs scripted children for the judgment calls')
        wrong = inst.get('wrong_answers') or []
        if len(wrong) < 3 or not all(isinstance(w, dict) and 'value' in w and w.get('mistake') for w in wrong):
            problems.append(f'F4: {label}: at least three wrong answers, each {{"value", "mistake"}}')
        elif any(canonical(w['value']) == canonical(inst.get('expected')) for w in wrong):
            problems.append(f'F4: {label}: a wrong answer equals expected')
        ablated = {a.get('judgment') for a in inst.get('ablations') or [] if a.get('children')}
        if names - ablated:
            problems.append(f'S2: {label}: no ablation for judgment(s) {", ".join(sorted(n for n in names - ablated if n))}')
        baseline = inst.get('keyword_baseline')
        if not isinstance(baseline, str) or 'return' not in baseline:
            problems.append(f'S3: {label}: keyword_baseline is a root eval script ending in return')
        elif INLINE_NL.search(baseline) or re.search(r'\bdecide\s*\(', baseline):
            problems.append(f'S3: {label}: the keyword baseline makes a natural-language call')
        if m['instances'][index]['longest_list'] < 8 and m['instances'][index]['words'] < 1200:
            problems.append(f'C2: {label}: {m["instances"][index]["longest_list"]} items, {m["instances"][index]["words"]} '
                            'words; at least eight items or 1,200 words')
    if len(judgments) < 3 or m['judgment_files'] + m['inline_nl'] < 3:
        problems.append('S1: at least three distinct judgments (judgment .nl files or inline nl calls), listed in judgments')
    if m['functions'] < 5:
        problems.append(f'C1: {m["functions"]} functions; at least five (.nl files and exported TypeScript functions)')
    if m['call_depth'] < 3:
        problems.append(f'C1: call depth {m["call_depth"]}; at least three (root, helper, judgment)')
    if m['ts_logic_lines'] < 40:
        problems.append(f'D1: {m["ts_logic_lines"]} lines of TypeScript logic in the callable folder; at least 40')
    found = [c for c in m['constructs'] if c != 'inline-nl']
    if len(found) < 3:
        problems.append(f'D2: semantic-FP constructs found: {", ".join(found) or "none"}; at least three')
    head, body = frontmatter(files[root])
    returns = re.search(r'^returns:\s*["\']?([A-Za-z_]\w*)["\']?\s*$', head, re.M)
    types = files.get('types.ts', '')
    if not returns or not re.search(rf'/\*\*[\s\S]*?\*/\s*export\s+type\s+{returns.group(1)}\b', types):
        problems.append('D4: the root returns a named type exported from types.ts with a doc comment')
    if re.search(r'output contract', body, re.I):
        problems.append('D4: the output format is declared in types.ts, not in an instruction paragraph')
    return problems


def world(data: dict) -> str:
    return 'invented:' + data['slug']


def split_of(data: dict) -> str:
    bucket = int(text_digest('invented-20261124|' + world(data))[:8], 16) % 100
    return 'train' if bucket < 70 else ('dev' if bucket < 85 else 'confirmation')


def verifier_binding(inst: dict) -> dict | None:
    if inst.get('oracle') != 'verifier':
        return None
    spec = inst['verifier']
    return {'path': spec['path'], 'sha256': hashlib.sha256((REPO / spec['path']).read_bytes()).hexdigest(),
            'oracle': spec['oracle']}


def package(data: dict, files: dict[str, str]) -> list[dict]:
    family = 'inv_' + re.sub(r'[^a-z0-9]+', '_', data['slug'].lower()).strip('_')
    root = data['root']
    root_body = frontmatter(files[root])[1]
    author = data.get('author') or {}
    cases = []
    for inst in data['instances']:
        verifier = verifier_binding(inst)
        aliases = inst.get('alias_answers') or []
        oracle = 'exact' if verifier else ({'level': inst['oracle'], 'alternates': aliases} if aliases else inst['oracle'])
        payload = json.dumps({'files': files, 'instance': inst}, sort_keys=True)
        case_id = f'inv:{slug(data["slug"], 48)}:{slug(inst["name"], 24)}:{text_digest(payload)[:8]}'
        constructs = sorted(set(data.get('constructs') or []))
        cases.append({
            'version': 'natlang.program/2', 'id': case_id, 'kind': 'lambda_source', 'family': family,
            'source': 'invented:' + str(author.get('builder', 'unknown')), 'split': split_of(data),
            'source_ids': [f'invented:{data["slug"]}:{inst["name"]}'], 'source_groups': [world(data)],
            'source_revisions': [GENERATOR], 'license': 'invented',
            'gold_sources': ['invented-verifier-oracle' if verifier else 'invented-exact-oracle'],
            'generation': {'generator': GENERATOR, 'teacher_generation': True, 'author': author,
                           'method': 'a new program idea (checked against the atlas of existing ideas), its program, '
                                     'literal instances, exact checker and controls written by frontier models; '
                                     'qualified by reference replay through the program, wrong answers, judgment '
                                     'ablations and keyword baselines; no sampled trajectory'},
            'curriculum': {
                'version': CURRICULUM_VERSION, 'family': family, 'family_version': 1, 'shape': data['slug'],
                'variant': 'base', 'pair_group': None, 'split_group': world(data), 'slice': 'nested_scoped',
                'domain': 'other', 'mode': 'single_call', 'inline': 'optional', 'named': 'required',
                'evidence': {'world': [], 'retrieved': [], 'background': []}, 'assumptions': [], 'decisive': [],
                'plausible_actions': [], 'minimum_sequence': [],
                'reference': {'root': inst['reference']['root'], 'children': inst['reference']['children']},
                'rl': {'level': int(inst.get('level', 3)), 'family': family,
                       'basis': 'invented-verifier-oracle-reference-verified' if verifier
                       else 'invented-exact-oracle-reference-verified'},
                'external': {'source': 'invented', 'upstream': 'invented:' + str(author.get('inventor', 'unknown')),
                             'upstream_id': f'{data["slug"]}:{inst["name"]}', 'upstream_split': 'train-carve',
                             'unit': world(data), 'shape': data['slug'],
                             'instruction_sha256': text_digest(root_body)[:16], 'instruction_id': text_digest(root_body)[:12],
                             'neuralese_room': '', 'framing': 'invented', 'constructs': constructs,
                             'subject_domain': 'other', 'oracle_depth': 'answer',
                             'checks': {'wrong_answers': [w['value'] for w in inst['wrong_answers']],
                                        'alias_answers': aliases,
                                        'keyword_baseline_code': [inst['keyword_baseline']],
                                        'ablations': inst.get('ablations') or []}}},
            'semantics': {'root': root, 'files': files, 'inputs': inst['inputs'], 'expected': inst['expected'],
                          'oracle': oracle, **({'trusted_exact_verifier': verifier} if verifier else {})},
        })
    return cases


def node() -> str:
    for candidate in sorted(Path.home().glob('.local/share/natlang-dev/node-v24*/bin/node'), reverse=True):
        return str(candidate)
    return 'node'


def natlang_check(program: Path) -> list[str]:
    natlang = os.environ.get('NATLANG_BIN') or str(Path.home() / '.local/bin/natlang')
    run = subprocess.run([natlang, 'check', '--json'], cwd=program, capture_output=True, text=True, timeout=900)
    try:
        report = json.loads(run.stdout[run.stdout.index('{'):])
    except ValueError:
        return [f'F1: natlang check gave no JSON (exit {run.returncode}): {(run.stderr or run.stdout)[-600:]}']
    diagnostics = report.get('diagnostics') or []
    errors = [d for d in diagnostics if d.get('severity', 'error') == 'error']
    out = [f'F1: {d.get("file", "")}:{d.get("line", "")} {d.get("code", "")} {d.get("message", "")}'[:400] for d in errors]
    if run.returncode != 0 and not out:
        out.append(f'F1: natlang check exited {run.returncode}: {json.dumps(report)[:400]}')
    return out


def qualify(submission: Path, runtime: Path) -> dict:
    data, files = load(submission)
    m = metrics(data, files)
    problems = static_problems(submission, data, files, m)
    out = submission / 'qualification'
    out.mkdir(exist_ok=True)
    report: dict = {'slug': data.get('slug'), 'metrics': m}
    problems += natlang_check(submission / 'program')
    try:
        cases = package(data, files)
    except (KeyError, TypeError, ValueError) as error:
        problems.append(f'package: {error}')
        cases = []
    (out / 'cases.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
    if cases:
        verified = out / 'verified'
        run = subprocess.run([node(), str(REPO / 'scripts/rl_tasks/verify.mjs'), str(runtime), str(out / 'cases.jsonl'),
                              str(verified), '2'], capture_output=True, text=True, timeout=3600)
        try:
            verification = json.loads((verified / 'verification.json').read_text())
        except (OSError, ValueError):
            verification = {'error': (run.stderr or run.stdout)[-1500:]}
            problems.append('verify.mjs did not finish: ' + verification['error'][-300:])
        report['verification'] = {key: verification.get(key) for key in ('summary', 'failures', 'error')}
        for row in verification.get('results', []):
            name = row.get('id', '?')
            if not row.get('reference_ok'):
                problems.append(f'F3: {name}: reference replay not accepted: {json.dumps(row.get("problems"))[:400]}')
            for wrong in row.get('wrong', []):
                if wrong.get('accepted') is not False:
                    problems.append(f'F4: {name}: wrong answer not rejected: {json.dumps(wrong)[:300]}')
            for baseline in row.get('keyword_baseline', []):
                if baseline.get('accepted') is not False:
                    problems.append(f'S3: {name}: keyword baseline not rejected: {json.dumps(baseline)[:300]}')
        run = subprocess.run([node(), str(REPO / 'scripts/rl_tasks/ablate.mjs'), str(runtime), str(out / 'cases.jsonl'),
                              str(out / 'ablations.json')], capture_output=True, text=True, timeout=3600)
        try:
            ablations = json.loads((out / 'ablations.json').read_text())
        except (OSError, ValueError):
            ablations = {'error': (run.stderr or run.stdout)[-1500:]}
            problems.append('ablate.mjs did not finish: ' + ablations['error'][-300:])
        report['ablations'] = ablations
        for row in ablations.get('results', []):
            if row.get('rejected') is not True:
                problems.append(f'S2: {row["id"]}: ablating {row["judgment"]} did not change the answer to a rejected one'
                                f' ({row.get("error") or json.dumps(row.get("value"))[:200]})')
    report['problems'] = problems
    report['ok'] = not problems
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check')
    c.add_argument('submission', type=Path)
    p = sub.add_parser('package')
    p.add_argument('submission', type=Path)
    p.add_argument('out', type=Path)
    q = sub.add_parser('qualify')
    q.add_argument('submission', type=Path)
    q.add_argument('--runtime', type=Path, default=REPO / 'ts-host')
    args = parser.parse_args(argv)
    submission = args.submission.resolve()
    if args.command == 'check':
        data, files = load(submission)
        m = metrics(data, files)
        problems = static_problems(submission, data, files, m)
        print(json.dumps({'ok': not problems, 'problems': problems, 'metrics': m}, indent=1))
        return 0 if not problems else 1
    if args.command == 'package':
        data, files = load(submission)
        cases = package(data, files)
        args.out.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
        print(json.dumps({'cases': len(cases)}))
        return 0
    report = qualify(submission, args.runtime.resolve())
    print(json.dumps({'ok': report['ok'], 'problems': report['problems'][:40]}, indent=1))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
