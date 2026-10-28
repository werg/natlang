#!/usr/bin/env python3
"""Re-render registered native-turn corpora for another student tokenizer, from each render's own receipt.

One mechanism for adding a student to the existing renders (LFM3-1.7B, 2026-10-28; plans/neuralese/LFM3_PORT_20261028.md
part B "full conversion plan"): every registered render of ``render_native_turn_corpus.sh`` (direct) or
``render_parts.py`` (parts) is rerun with its receipt's exact arguments (source corpus id, sources, exclude-programs,
reservations, protected index, held text, pieces, renderings, full sample) for ``--student`` only, into a new directory.
Nothing about selection changes: the renderer still checks registry admission, reservations, protected overlap and
held source groups itself. The new render is then verified against the registered render of the reference student
(default lfm25): the same documents (ids, splits, source groups, source record ids, in order) per view, every token
sequence decodes to its text, every supervised target parses back to the source decision's tool calls, and no train
document shares a source group with a held one. A difference is reported, never silently accepted.

  student_rerender.py plan   --student lfm3 --queue Q.json [--out-root runs/lfm3-renders] [--id ID ...] [--tier-external 3]
  student_rerender.py launch --queue Q.json [--max-running N] [--cpus C] [--tier T]   (idempotent; run it repeatedly)
  student_rerender.py status --queue Q.json
  student_rerender.py verify --queue Q.json --id ID     (writes OUT/student-verification.json; run under the ledger)
  student_rerender.py entry  --queue Q.json --id ID     (prints the registry entry of a verified render)

The queue file is the resumable worklist: one item per registered render, with its reconstructed command, tier, and
a status that `launch` advances (pending -> running -> rendered -> verified/failed). Items whose sources are absent on
this machine stay `missing-source` and are retried by later `launch` calls (Pop-owned corpora being recovered).
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
sys.path.insert(0, str(REPO / 'scripts' / 'neuralese_data'))
REGISTRY = REPO / 'training/neuralese_corpora.json'
WRAPPER = 'scripts/neuralese_data/render_native_turn_corpus.sh'
PARTS = 'scripts/neuralese_data/render_parts.py'
LEDGER = REPO / 'scripts/memory_ledger.py'
VENV_PY = REPO / '.venv-neuralese/bin/python'
PYTHONPATH = f'{REPO}/training/neuralese:{REPO}'


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
    tmp.replace(path)


class Queue:
    """The queue file under an exclusive lock (launch and verify units update it concurrently)."""
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.lock = open(self.path.with_name(self.path.name + '.lock'), 'w')
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        self.value = load(self.path) if self.path.exists() else {'items': {}}
        return self.value

    def __exit__(self, kind, *_):
        if kind is None:
            save(self.path, self.value)
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()


def rel(path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(REPO)) if path.is_absolute() else str(path)
    except ValueError:
        return str(path)


def short_unit(prefix: str, ident: str) -> str:
    digest = hashlib.sha1(ident.encode()).hexdigest()[:8]
    return f'{prefix}-{re.sub(r"[^a-z0-9]+", "-", ident.lower())[:40].strip("-")}-{digest}'


def direct_command(entry: dict, receipt: dict, out: Path, student: str) -> tuple[list[str] | None, str | None]:
    """The wrapper arguments of a direct render, from its receipt; (None, reason) when they are not recorded."""
    selection = receipt.get('selection') or {}
    if 'approved_through_include_set' in selection or 'not_in_include_set' in selection:
        return None, 'include-set render: its --include-ids file is not recorded in the receipt'
    build = entry.get('build') if isinstance(entry.get('build'), dict) else {}
    sources = list((receipt.get('sources') or build.get('sources') or {}).keys())
    if not sources:
        return None, 'receipt names no sources'
    args = ['--corpus-id', receipt['corpus_id'], '--source', *sources]
    excludes = [item['path'] if isinstance(item, dict) else item for item in receipt.get('exclude_programs') or []]
    if excludes:
        args += ['--exclude-programs', *excludes]
    if isinstance(receipt.get('reserved'), dict):
        args += ['--reserved', receipt['reserved']['path']]
    if isinstance(receipt.get('protected'), dict):
        args += ['--protected', receipt['protected']['path']]
    if receipt.get('held_text'):
        args += ['--held-text', *receipt['held_text'].keys()]
    pieces = [item['path'] for item in receipt.get('explicit_piece_bindings') or []]
    if pieces:
        args += ['--pieces', *pieces]
    if 'omitted:held_split_kept_out' in selection:
        args.append('--train-only')
    renderings = receipt.get('renderings')
    if isinstance(renderings, list):
        args += ['--renderings', *renderings]
    fraction = (receipt.get('full_sample') or {}).get('fraction')
    if fraction is not None:
        args += ['--full-sample', str(fraction)]
    return [WRAPPER, str(out), *args, '--compress-text'], None


def parts_command(entry: dict, path: Path, out: Path, student: str, cpus: int) -> tuple[list[str] | None, str | None]:
    parts = load(path / 'parts.json')
    first = path / 'parts' / 'p00' / 'receipt.json'
    if not first.exists():
        return None, 'parts render without a part receipt'
    receipt = load(first)
    sys.path.insert(0, str(REPO / 'scripts' / 'neuralese_data'))
    from render_parts import HELD_TEXT, PROTECTED
    args = [str(VENV_PY), PARTS, 'render', '--corpus-id', receipt['corpus_id'], '--out', str(out),
            '--source', *parts['sources'], '--students', student, '--compress-all', '--cpus', str(cpus),
            '--unit-prefix', short_unit('lfm3-data-p', entry['id'])]
    if isinstance(receipt.get('protected'), dict) and receipt['protected']['path'] not in (PROTECTED, str(REPO / PROTECTED)):
        args += ['--protected', receipt['protected']['path']]
    defaults = {str(REPO / p) for p in HELD_TEXT} | set(HELD_TEXT)
    for held in (receipt.get('held_text') or {}):
        if held not in defaults:
            args += ['--held-text', held]
    for item in receipt.get('exclude_programs') or []:
        args += ['--exclude-programs', item['path'] if isinstance(item, dict) else item]
    if parts.get('part_decisions') not in (None, 12000):
        return None, f'non-default part size {parts.get("part_decisions")} (render_parts has no option for it)'
    return args, None


def plan(args):
    registry = load(REGISTRY)['corpora']
    registered = {c['id'] for c in registry}
    wanted = set(args.id or [])
    with Queue(args.queue) as queue:
        queue.update(student=args.student, reference=args.reference, out_root=args.out_root)
        items = queue.setdefault('items', {})
        for entry in registry:
            renders = entry.get('renders')
            if not isinstance(renders, dict) or (wanted and entry['id'] not in wanted):
                continue
            if not any(isinstance(v, dict) and isinstance(v.get(args.reference), dict) for v in renders.values()):
                continue
            new_id = f'{entry["id"]}-{args.student}'
            if entry.get('training_admission') is not True:
                continue
            if new_id in registered or entry['id'].endswith('-' + args.student):
                continue
            if entry['id'] in items and items[entry['id']].get('status') not in (None, 'blocked', 'missing-source'):
                continue
            path = REPO / entry['path']
            out = REPO / args.out_root / entry['id']
            item = {'id': entry['id'], 'new_id': new_id, 'path': entry['path'], 'out': rel(out),
                    'tier': args.tier_external if entry['id'].startswith('external-expansion') else 1,
                    'reference_tokens': sum(
                        sum(s.get('tokens', 0) for s in (v[args.reference].get('by_split') or {}).values()
                            if isinstance(s, dict))
                        for v in renders.values() if isinstance(v, dict) and isinstance(v.get(args.reference), dict)),
                    'views': sorted(k for k, v in renders.items()
                                    if isinstance(v, dict) and isinstance(v.get(args.reference), dict))}
            if (path / 'parts.json').exists():
                command, reason = parts_command(entry, path, out, args.student, args.cpus)
                item['kind'] = 'parts'
            elif (path / 'receipt.json').exists() and (entry.get('build') or {}).get('wrapper') == WRAPPER:
                receipt = load(path / 'receipt.json')
                command, reason = direct_command(entry, receipt, out, args.student)
                item['kind'] = 'direct'
            else:
                command, reason = None, ('union or derived render: rebuild from its members\' renders'
                                         if (path / 'receipt.json').exists() else 'no render receipt on this machine')
                item['kind'] = 'other'
            item['command'] = command
            if command is None:
                item.update(status='blocked', reason=reason)
            else:
                missing = [s for s in sources_of(item) if not (REPO / s).exists()]
                item.update(status='missing-source' if missing else 'pending', missing=missing[:5])
            items[entry['id']] = item
        counts = collections.Counter(i['status'] for i in items.values())
    print(json.dumps(dict(counts)))


def sources_of(item):
    command = item['command'] or []
    if '--source' not in command:
        return []
    out, take = [], False
    for word in command[command.index('--source') + 1:]:
        if word.startswith('--'):
            break
        out.append(word)
    return out


def unit_state(unit: str) -> str:
    result = subprocess.run(['systemctl', '--user', 'is-active', unit + '.service'], capture_output=True, text=True)
    state = result.stdout.strip()
    if state in ('active', 'activating', 'deactivating', 'reloading'):
        return 'active'
    try:
        status = json.loads(subprocess.run(['python3', str(LEDGER), 'status'], capture_output=True, text=True).stdout)
        if any(w.get('unit', '').startswith(unit) for w in status.get('waiting', [])):
            return 'active'
    except ValueError:
        pass
    return state


def launch(args):
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    with Queue(args.queue) as queue:
        items = queue['items']
        for item in items.values():  # refresh
            if item['status'] in ('missing-source',) and not [s for s in sources_of(item) if not (REPO / s).exists()]:
                item['status'] = 'pending'
            if item['status'] == 'running':
                done = (REPO / item['out'] / 'receipt.json').exists()
                if unit_state(item['unit']) != 'active':
                    item['status'] = 'rendered' if done else 'failed'
            if item['status'] == 'verifying' and unit_state(item['verify_unit']) != 'active':
                verification = REPO / item['out'] / 'student-verification.json'
                item['status'] = ('verified' if verification.exists() and load(verification).get('passed')
                                  else 'verify-failed')
        running = [i for i in items.values() if i['status'] in ('running', 'verifying')]
        slots = args.max_running - len(running)
        for item in items.values():  # verify rendered items first (cheap, and they unblock registration)
            if slots <= 0:
                break
            if item['status'] == 'rendered':
                unit = short_unit('lfm3-data-v', item['id']) + f'-{int(time.time()) % 100000}'
                cmd = ['python3', str(LEDGER), 'run', '--unit', unit, '--budget-gb', str(args.verify_budget_gb),
                       '--priority', 'low', '--resumable', '--wait', '86400', '--detach', '--workdir', str(REPO),
                       '--', 'bash', '-c',
                       f'PYTHONPATH={PYTHONPATH} nice -n 10 {VENV_PY} {Path(__file__).resolve()} verify '
                       f'--queue {Path(args.queue).resolve()} --id {item["id"]} > {logs}/{unit}.log 2>&1']
                subprocess.run(cmd, check=True, capture_output=True)
                item.update(status='verifying', verify_unit=unit, verify_log=f'{logs}/{unit}.log')
                slots -= 1
        order = sorted((i for i in items.values() if i['status'] == 'pending' and i['tier'] <= args.tier),
                       key=lambda i: (i['tier'], i['reference_tokens']))
        for item in order:
            if slots <= 0:
                break
            unit = short_unit('lfm3-data-r', item['id']) + f'-{int(time.time()) % 100000}'
            log = f'{logs}/{unit}.log'
            if item['kind'] == 'direct':
                out = REPO / item['out']
                container = (f'render-{out.name}-' + hashlib.sha1(str(out).encode()).hexdigest()[:10])
                # Memory scales with the corpus (the renderer keeps a working copy of its selection).
                budget = 6 if item['reference_tokens'] < 30e6 else 12 if item['reference_tokens'] < 200e6 else args.budget_gb
                inner = (f'STUDENTS={queue["student"]} CPUS={args.cpus} MEM={budget - 1}g nice -n 10 '
                         + ' '.join(shell(w) for w in item['command'][:2]) + ' '
                         + ' '.join(shell(w) for w in item['command'][2:]) + f' --workers {args.cpus}')
                cmd = ['python3', str(LEDGER), 'run', '--unit', unit, '--budget-gb', str(budget),
                       '--priority', 'low', '--resumable', '--wait', '86400', '--detach', '--workdir', str(REPO),
                       '--container', container, '--stop-seconds', '90', '--', 'bash', '-c', f'{inner} > {log} 2>&1']
            else:  # parts coordinator: small, it launches one ledger unit per part itself
                inner = f'PYTHONPATH={PYTHONPATH} ' + ' '.join(shell(w) for w in item['command'])
                cmd = ['python3', str(LEDGER), 'run', '--unit', unit, '--budget-gb', '2', '--priority', 'low',
                       '--resumable', '--wait', '86400', '--detach', '--workdir', str(REPO), '--',
                       'bash', '-c', f'{inner} > {log} 2>&1']
            subprocess.run(cmd, check=True, capture_output=True)
            item.update(status='running', unit=unit, log=log, launched=time.time())
            slots -= 1
        counts = collections.Counter(i['status'] for i in items.values())
    print(json.dumps(dict(counts)))


def shell(word: str) -> str:
    import shlex
    return shlex.quote(word)


def status(args):
    queue = load(args.queue)
    by = collections.defaultdict(list)
    for item in queue['items'].values():
        by[item['status']].append(item)
    for state, items in sorted(by.items()):
        tokens = sum(i['reference_tokens'] for i in items)
        print(f'{state}: {len(items)} corpora, {tokens / 1e6:.1f} M reference tokens')
        if args.verbose:
            for i in sorted(items, key=lambda i: i['reference_tokens']):
                print(f'   T{i["tier"]} {i["reference_tokens"] / 1e6:8.1f}M {i["id"]} {i.get("reason") or ""}')


# ---------------------------------------------------------------------------------------------------------- verify

def view_file(root: Path, view: str, tok: str) -> Path | None:
    for name in ('text-data.jsonl.zst', 'text-data.jsonl'):
        if (root / view / tok / name).exists():
            return root / view / tok / name
    return None


def rows(path: Path):
    from natlang_neuralese.common.jsonio import iter_jsonl_rows
    for _, row in iter_jsonl_rows(path):
        yield row


SOURCE_SHA_SUFFIX = re.compile(r':source-sha256:[0-9a-f]{64}$')


def identity(row):
    # Whole-trajectory ids gained a ':source-sha256:<chain digest>' suffix after the 2026-10-1x renders; the chain is
    # still identified by its source record ids, compared exactly.
    return (SOURCE_SHA_SUFFIX.sub('', row['id']), row['split'], tuple(row.get('source_groups') or ()),
            tuple(row.get('source_record_ids') or ()))


def source_targets(paths: list[Path], wanted: set[str]) -> dict:
    """{decision id: ([(name, arguments)], tools)} for the wanted decisions of the source turn files."""
    from natlang_neuralese.common.jsonio import iter_jsonl_rows
    out = {}
    for path in paths:
        for _, row in iter_jsonl_rows(path):
            if row.get('id') in wanted:
                calls = []
                for call in (row.get('target') or {}).get('tool_calls') or []:
                    fn = call.get('function', call)
                    arguments = fn.get('arguments')
                    calls.append((fn.get('name'), json.loads(arguments) if isinstance(arguments, str) else arguments))
                out[row['id']] = (calls, row.get('tools'))
    return out


def verify(args):
    from transformers import AutoTokenizer
    from natlang_neuralese.serve.chat import build_message
    from student_tokenizers import resolve
    queue = load(args.queue)
    item = queue['items'][args.id]
    student, reference = queue['student'], queue['reference']
    tok = AutoTokenizer.from_pretrained(str(resolve(student)['path']), local_files_only=True, trust_remote_code=False)
    new, old = REPO / item['out'], REPO / item['path']
    report = {'schema': 'natlang.student-rerender-verification/1', 'id': item['id'], 'student': student,
              'reference': reference, 'views': {}, 'problems': []}
    new_receipt, old_receipt = load(new / 'receipt.json'), load(old / 'receipt.json')
    if item['kind'] == 'direct':
        if new_receipt.get('sources') != old_receipt.get('sources'):
            report['problems'].append('source identities differ from the reference render receipt')
    wanted_targets = set()
    view_rows = {}
    for view in item['views']:
        ref_path, new_path = view_file(old, view, reference), view_file(new, view, student)
        if ref_path is None:
            continue
        if new_path is None:
            report['problems'].append(f'{view}: no {student} rendering')
            continue
        ref_ids = [identity(r) for r in rows(ref_path)]
        stats = collections.Counter()
        new_ids, groups = [], {'train': set(), 'test': set()}
        for row in rows(new_path):
            new_ids.append(identity(row))
            groups.setdefault(row['split'], set()).update(row.get('source_groups') or [])
            stats['tokens'] += len(row['token_ids'])
            stats[f'tokens:{row["split"]}'] += len(row['token_ids'])
            if tok.decode(row['token_ids'], skip_special_tokens=False) != row['text']:
                stats['decode_mismatch'] += 1
            if '<|tool_call_start|>' in row['text']:
                stats['pythonic_call_marker'] += 1
            start = row.get('supervised_suffix_start')
            if start is not None and len(row.get('source_record_ids') or []) == 1:
                reply = tok.decode(row['token_ids'][start:], skip_special_tokens=False)
                view_rows.setdefault(row['source_record_ids'][0], []).append((view, reply))
                wanted_targets.add(row['source_record_ids'][0])
        ref_set, new_set = set(ref_ids), set(new_ids)
        stats.update(documents=len(new_ids), reference_documents=len(ref_ids),
                     same_order=int(ref_ids == new_ids), only_reference=len(ref_set - new_set),
                     only_student=len(new_set - ref_set),
                     train_test_group_overlap=len(groups['train'] & groups.get('test', set())))
        ref_by_id = {i[0]: i for i in ref_ids}
        stats['split_or_group_changed'] = sum(1 for i in new_ids if i[0] in ref_by_id and ref_by_id[i[0]] != i)
        report['views'][view] = dict(stats)
        if stats['only_reference'] or stats['only_student']:
            report['views'][view]['only_reference_sample'] = sorted(i[0] for i in ref_set - new_set)[:5]
            report['views'][view]['only_student_sample'] = sorted(i[0] for i in new_set - ref_set)[:5]
        for key in ('decode_mismatch', 'pythonic_call_marker', 'train_test_group_overlap', 'split_or_group_changed'):
            if stats[key]:
                report['problems'].append(f'{view}: {key} {stats[key]}')
    # Targets: every supervised suffix parses back to its source decision's calls.
    sources = [REPO / s for s in (new_receipt.get('sources') or {})] if item['kind'] == 'direct' else []
    if sources:
        targets = source_targets(sources, wanted_targets)
        parse = collections.Counter()
        for decision, replies in view_rows.items():
            want, tools = targets.get(decision, (None, None))
            for view, reply in replies:
                if want is None:
                    parse[f'{view}:source_missing'] += 1
                    continue
                try:  # the serving parser, typed by the decision's own tool declarations
                    got = [(c['function']['name'], json.loads(c['function']['arguments']))
                           for c in build_message(reply, [], tools=tools).get('tool_calls') or []]
                except Exception as error:  # an unparsable target is a finding, not a crash
                    got = [('<error>', str(error)[:200])]
                if got == want:
                    parse[f'{view}:exact'] += 1
                elif [n for n, _ in got] == [n for n, _ in want] and all(
                        json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) or str(a) == str(b)
                        for (_, a), (_, b) in zip(got, want)):
                    parse[f'{view}:equal_modulo_untyped_values'] += 1
                else:
                    parse[f'{view}:differ'] += 1
                    if parse[f'{view}:differ'] <= 3:
                        report.setdefault('target_differences', []).append(
                            {'view': view, 'decision': decision, 'want': json.dumps(want)[:400],
                             'got': json.dumps(got)[:400]})
        report['target_parse'] = dict(parse)
        if any(k.endswith(':differ') or k.endswith(':source_missing') for k in parse):
            report['problems'].append('supervised targets that do not parse back to their source calls')
    else:
        report['target_parse'] = 'not checked (parts render: sources are streamed part inputs)'
    report['passed'] = not report['problems']
    # The same document set per view (row order follows the shard count and is not training-relevant).
    report['identical_documents'] = all(not v.get('only_reference') and not v.get('only_student')
                                        and not v.get('split_or_group_changed') for v in report['views'].values())
    save(new / 'student-verification.json', report)
    print(json.dumps({'id': args.id, 'passed': report['passed'], 'problems': report['problems'][:10]}))


# ----------------------------------------------------------------------------------------------------------- entry

def entry(args):
    queue = load(args.queue)
    item = queue['items'][args.id]
    student = queue['student']
    print(json.dumps(registry_entry(item, student), indent=1))


def registry_entry(item: dict, student: str) -> dict:
    from natlang_neuralese.common.hashing import sha256_file_hex
    registry = {c['id']: c for c in load(REGISTRY)['corpora']}
    source = registry[item['id']]
    out = REPO / item['out']
    receipt = load(out / 'receipt.json')
    verification = load(out / 'student-verification.json')
    renders = {}
    for key, info in receipt['renders'].items():
        view, tok = key.split('/')
        if tok != student:
            continue
        renders.setdefault(view, {})[tok] = {k: v for k, v in info.items() if k in (
            'documents', 'by_split', 'max_document_tokens', 'text_data_file', 'text_data_sha256',
            'decoded_text_data_sha256', 'decoded_text_data_bytes')}
        if 'text_data_sha256' not in renders[view][tok]:
            path = view_file(out, view, tok)
            renders[view][tok].update(text_data_file=path.name, text_data_sha256=sha256_file_hex(path))
    keep = ('source_class', 'split', 'license', 'license_provenance', 'training_admission_reason')
    new = {'id': item['new_id'], 'owner': 'dgx', 'path': item['out'],
           'include': ['receipt.json', 'student-verification.json', 'render-policy-binding.json', 'parts.json',
                       'merge.json', '*/*/text-data.jsonl', '*/*/text-data.jsonl.zst', '*/*/omissions.jsonl',
                       'selection-omissions.jsonl', 'render-omissions/*/*', 'parts/*/receipt.json',
                       'parts/*/render-policy-binding.json', 'parts/*/*.jsonl', 'parts/*/*/*/omissions.jsonl',
                       'parts/*/*/*/omissions.jsonl.zst', 'parts/*/render-omissions/*/omissions.jsonl'],
           **{k: source[k] for k in keep if k in source},
           'admission': (f'{student} render of the registered render {item["id"]} (scripts/neuralese_data/'
                         f'student_rerender.py: its receipt\'s exact selection arguments, {student} tokenizer only); '
                         'inherits that render\'s admission, splits and held decisions; no new example admission'),
           'training_admission': source.get('training_admission'),
           'derived_from': [item['id'], *(source.get('derived_from') or [])],
           'renders': renders,
           'student_rerender': {'reference_render': item['id'], 'student': student,
                                'identical_documents': verification['identical_documents'],
                                'views': {v: {k: s.get(k) for k in ('documents', 'reference_documents', 'same_order',
                                                                    'only_reference', 'only_student')}
                                          for v, s in verification['views'].items()},
                                'target_parse': verification.get('target_parse')}}
    if receipt.get('render_policy_binding_sha256'):
        new['render_policy_binding_sha256'] = receipt['render_policy_binding_sha256']
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--queue', required=True)
    p.add_argument('--student', required=True)
    p.add_argument('--reference', default='lfm25')
    p.add_argument('--out-root', default='runs/lfm3-renders')
    p.add_argument('--id', action='append')
    p.add_argument('--tier-external', type=int, default=3)
    p.add_argument('--cpus', type=int, default=4)
    p = sub.add_parser('launch')
    p.add_argument('--queue', required=True)
    p.add_argument('--max-running', type=int, default=4)
    p.add_argument('--cpus', type=int, default=4)
    p.add_argument('--budget-gb', type=int, default=20, help='ledger budget of the largest direct renders')
    p.add_argument('--verify-budget-gb', type=int, default=6)
    p.add_argument('--tier', type=int, default=1)
    p.add_argument('--logs', required=True)
    p = sub.add_parser('status')
    p.add_argument('--queue', required=True)
    p.add_argument('-v', '--verbose', action='store_true')
    p = sub.add_parser('verify')
    p.add_argument('--queue', required=True)
    p.add_argument('--id', required=True)
    p = sub.add_parser('entry')
    p.add_argument('--queue', required=True)
    p.add_argument('--id', required=True)
    args = ap.parse_args()
    {'plan': plan, 'launch': launch, 'status': status, 'verify': verify, 'entry': entry}[args.cmd](args)


if __name__ == '__main__':
    main()
