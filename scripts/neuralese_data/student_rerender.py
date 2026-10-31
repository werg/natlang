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
    if selection.get('approved_by_registry_admission'):
        # Rows held for a reason the registry entry admits; without the reason the student render drops them all
        # (gemini-clinc LFM3 renders: 0 of 20,140 rows, 2026-10-28). Older receipts did not record the reason.
        reason = receipt.get('registry_admits_reason') or row_hold_reason(sources)
        if not reason:
            return None, 'registry-admitted rows but no single recorded or source hold reason'
        args += ['--registry-admits-reason', reason]
    renderings = receipt.get('renderings')
    if isinstance(renderings, list):
        args += ['--renderings', *renderings]
    fraction = (receipt.get('full_sample') or {}).get('fraction')
    if fraction is not None:
        args += ['--full-sample', str(fraction)]
    return [WRAPPER, str(out), *args, '--compress-text'], None


def row_hold_reason(sources: list[str]) -> str | None:
    """The one training-admission hold reason of the unapproved rows of SOURCES, or None if absent or ambiguous."""
    from natlang_neuralese.common.jsonio import iter_jsonl_rows
    reasons = set()
    for source in sources:
        for _, row in iter_jsonl_rows(REPO / source):
            admission = row.get('training_admission')
            if isinstance(admission, dict) and admission.get('approved') is False and admission.get('reason'):
                reasons.add(str(admission['reason']))
    return reasons.pop() if len(reasons) == 1 else None


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


def late_receipt_status(item: dict, queue: dict) -> str | None:
    """Promote terminal failures only when late artifacts bind to this render.

    Verification remains authoritative for retained rows and explicit renderer
    omissions; this does not change its admission or parity policy.
    """
    out, reference = REPO / item['out'], REPO / item['path']
    try:
        report = load(out / 'student-verification.json')
        if (report.get('id') != item['id'] or report.get('student') != queue['student']
                or report.get('reference') != queue['reference'] or report.get('passed') is not True):
            return None
        from natlang_neuralese.common.hashing import sha256_file_hex
        if report.get('render_receipt_sha256') != sha256_file_hex(out / 'receipt.json'):
            return None
        if report.get('reference_receipt_sha256') != sha256_file_hex(reference / 'receipt.json'):
            return None
        if not report.get('identical_documents'):
            missing = []
            for view, stats in report.get('views', {}).items():
                n = stats.get('only_reference', 0)
                sample = stats.get('only_reference_sample', [])
                if stats.get('only_student', 0) or stats.get('split_or_group_changed', 0) or len(sample) != n:
                    return None
                missing.extend((view, ident) for ident in sample)
            omissions = []
            for path in (out / 'render-omissions').rglob('*'):
                if path.is_file():
                    if path.stat().st_size > 1024 * 1024:
                        return None
                    omissions.extend(json.loads(line) for line in path.read_text().splitlines())
            explained = [(o.get('rendering'), o.get('id')) for o in omissions
                         if o.get('reason') == 'native_chat_render_failed'
                         and 'tool-call-unrepresentable' in o.get('detail', '')]
            if not missing or sorted(missing) != sorted(explained):
                return None
            render_receipt = load(out / 'receipt.json')
            for view in {v for v, _ in missing}:
                count = (render_receipt.get('render_omissions') or {}).get(
                    f'{queue["student"]}/{view}/native_chat_render_failed')
                if count != sum(v == view for v, _ in missing):
                    return None
        if item['kind'] == 'direct':
            original, rendered = load(reference / 'receipt.json'), load(out / 'receipt.json')
            if not original.get('sources') or original.get('sources') != rendered.get('sources'):
                return None
            if original.get('selection') != rendered.get('selection'):
                return None
            for view in item['views']:
                for base, tok, receipt in ((reference, queue['reference'], original),
                                           (out, queue['student'], rendered)):
                    data = view_file(base, view, tok)
                    binding = (receipt.get('renders') or {}).get(f'{view}/{tok}') or {}
                    if data is None or sha256_file_hex(data) != binding.get('text_data_sha256'):
                        return None
        if item['status'] == 'verify-failed':
            return 'verified'
        if item['status'] != 'graph-failed' or not graph_views(item):
            return None
        from student_tokenizers import model_id
        for view in graph_views(item):
            receipt = load(out / 'native-graphs' / view / 'receipt.json')
            source = serving_view_file(item, view, queue['student'])
            if (source is None or receipt.get('model_id') != model_id(queue['student'])
                    or Path(receipt['source']['path']).resolve() != source.resolve()
                    or sha256_file_hex(source) != receipt['source']['sha256']):
                return None
            outputs = receipt.get('outputs') or []
            if not outputs:
                return None
            target = (out / 'native-graphs' / view).resolve()
            for entry in outputs:
                path = Path(entry['path']).resolve()
                if not path.is_relative_to(target) or sha256_file_hex(path) != entry['sha256']:
                    return None
        return 'graphed'
    except (OSError, ValueError, KeyError, TypeError):
        return None


def launch(args):
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    with Queue(args.queue) as queue:
        items = queue['items']
        for item in items.values():  # refresh
            if item['status'] in ('missing-source',) and not [s for s in sources_of(item) if not (REPO / s).exists()]:
                item['status'] = 'pending'
            if item['status'] == 'failed' and (REPO / item['out'] / 'receipt.json').exists():
                item['status'] = 'rendered'  # its receipt landed after the unit was last seen
            if item['status'] in ('verify-failed', 'graph-failed'):
                recovered = late_receipt_status(item, queue)
                if recovered:
                    item['status'] = recovered
            if item['status'] == 'running':
                done = (REPO / item['out'] / 'receipt.json').exists()
                if unit_state(item['unit']) != 'active':
                    item['status'] = 'rendered' if done else 'failed'
            if item['status'] == 'verifying' and unit_state(item['verify_unit']) != 'active':
                verification = REPO / item['out'] / 'student-verification.json'
                item['status'] = ('verified' if verification.exists() and load(verification).get('passed')
                                  else 'verify-failed')
            if item['status'] == 'graphing' and unit_state(item['graph_unit']) != 'active':
                item['status'] = 'graphed' if graphs_done(item) else 'graph-failed'
        # Two pools: renders, and the lighter verify/graph steps (which must not starve new renders).
        post = args.max_post - sum(1 for i in items.values() if i['status'] in ('verifying', 'graphing'))
        slots = post
        for item in items.values():  # verify rendered items first (cheap, and they unblock registration)
            if slots <= 0:
                break
            if item['status'] == 'rendered':
                unit = short_unit('lfm3-data-v', item['id']) + f'-{int(time.time()) % 100000}'
                cmd = ['python3', str(LEDGER), 'run', '--unit', unit, '--budget-gb', str(args.verify_budget_gb),
                       '--priority', args.priority, '--resumable', '--wait', '86400', '--detach', '--workdir', str(REPO),
                       '--', 'bash', '-c',
                       f'PYTHONPATH={PYTHONPATH} nice -n 10 {VENV_PY} {Path(__file__).resolve()} verify '
                       f'--queue {Path(args.queue).resolve()} --id {item["id"]} > {logs}/{unit}.log 2>&1']
                subprocess.run(cmd, check=True, capture_output=True)
                item.update(status='verifying', verify_unit=unit, verify_log=f'{logs}/{unit}.log')
                slots -= 1
            elif item['status'] == 'verify-failed' and not item.get('subset_unit') and subset_eligible(
                    load(REPO / item['out'] / 'student-verification.json'), item):
                unit = short_unit('lfm3-data-x', item['id']) + f'-{int(time.time()) % 100000}'
                cmd = ['python3', str(LEDGER), 'run', '--unit', unit, '--budget-gb', str(args.verify_budget_gb),
                       '--priority', args.priority, '--resumable', '--wait', '86400', '--detach', '--workdir', str(REPO),
                       '--', 'bash', '-c',
                       f'PYTHONPATH={PYTHONPATH} nice -n 10 {VENV_PY} {Path(__file__).resolve()} subset '
                       f'--queue {Path(args.queue).resolve()} --id {item["id"]} > {logs}/{unit}.log 2>&1']
                subprocess.run(cmd, check=True, capture_output=True)
                item.update(status='verifying', verify_unit=unit, subset_unit=unit, verify_log=f'{logs}/{unit}.log')
                slots -= 1
            elif item['status'] == 'verified' and graph_views(item):
                unit = short_unit('lfm3-data-g', item['id']) + f'-{int(time.time()) % 100000}'
                script = graph_script(item, queue['student'])
                cmd = ['python3', str(LEDGER), 'run', '--unit', unit, '--budget-gb', str(args.verify_budget_gb),
                       '--priority', args.priority, '--resumable', '--wait', '86400', '--detach', '--workdir', str(REPO),
                       '--', 'bash', '-c', f'nice -n 10 bash {script} > {logs}/{unit}.log 2>&1']
                subprocess.run(cmd, check=True, capture_output=True)
                item.update(status='graphing', graph_unit=unit, graph_log=f'{logs}/{unit}.log')
                slots -= 1
        slots = args.max_running - sum(1 for i in items.values() if i['status'] == 'running')
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
                budget = 4 if item['reference_tokens'] < 5e6 else 6 if item['reference_tokens'] < 30e6 else 12 if item['reference_tokens'] < 200e6 else args.budget_gb
                inner = (f'STUDENTS={queue["student"]} CPUS={args.cpus} MEM={budget - 1}g nice -n 10 '
                         + ' '.join(shell(w) for w in item['command'][:2]) + ' '
                         + ' '.join(shell(w) for w in item['command'][2:]) + f' --workers {args.cpus}')
                cmd = ['python3', str(LEDGER), 'run', '--unit', unit, '--budget-gb', str(budget),
                       '--priority', args.priority, '--resumable', '--wait', '86400', '--detach', '--workdir', str(REPO),
                       '--container', container, '--stop-seconds', '90', '--', 'bash', '-c', f'{inner} > {log} 2>&1']
            else:  # parts coordinator: small, it launches one ledger unit per part itself
                inner = f'PYTHONPATH={PYTHONPATH} ' + ' '.join(shell(w) for w in item['command'])
                cmd = ['python3', str(LEDGER), 'run', '--unit', unit, '--budget-gb', '2', '--priority', args.priority,
                       '--resumable', '--wait', '86400', '--detach', '--workdir', str(REPO), '--',
                       'bash', '-c', f'{inner} > {log} 2>&1']
            subprocess.run(cmd, check=True, capture_output=True)
            item.update(status='running', unit=unit, log=log, launched=time.time())
            slots -= 1
        counts = collections.Counter(i['status'] for i in items.values())
    print(json.dumps(dict(counts)))


GRAPH_VIEWS = ('program-header', 'whole-trajectory')  # the serving renderings; `full` is the warm-up sample only


def graph_views(item: dict) -> list[str]:
    return [view for view in GRAPH_VIEWS if view in item['views']]


def graphs_done(item: dict) -> bool:
    return all((REPO / item['out'] / 'native-graphs' / view / 'receipt.json').exists() for view in graph_views(item))


def graph_script(item: dict, student: str) -> Path:
    """Native one-node graph views (build_native_graph_text_corpus.py) of a verified render's serving views;
    resumable per view (a finished view keeps its receipt)."""
    from student_tokenizers import model_id, resolve
    out = REPO / item['out']
    lines = ['set -euo pipefail', f'cd {REPO}', f'export PYTHONPATH={PYTHONPATH}']
    for view in graph_views(item):
        source = serving_view_file(item, view, student)
        target = out / 'native-graphs' / view
        lines.append(f'if [ ! -f {target}/receipt.json ]; then rm -rf {target}; '
                     f'{VENV_PY} scripts/neuralese_data/build_native_graph_text_corpus.py --source {source} '
                     f'--source-sha256 $(sha256sum {source} | cut -d" " -f1) --base-tokenizer {resolve(student)["path"]} '
                     f'--model-id {model_id(student)} --output {target} | tail -c 2000; fi')
    script = out / 'native-graphs.sh'
    script.write_text('\n'.join(lines) + '\n')
    return script


def shell(word: str) -> str:
    import shlex
    return shlex.quote(word)


def status(args):
    queue = load(args.queue)
    by = collections.defaultdict(list)
    for item in queue['items'].values():
        by[item['status']].append(item)
    for state, items in sorted(by.items()):
        tokens = sum(i.get('reference_tokens') or 0 for i in items)
        print(f'{state}: {len(items)} corpora, {tokens / 1e6:.1f} M reference tokens')
        if args.verbose:
            for i in sorted(items, key=lambda i: i.get('reference_tokens') or 0):
                print(f'   T{i["tier"]} {(i.get("reference_tokens") or 0) / 1e6:8.1f}M {i["id"]} {i.get("reason") or ""}')


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
    # A student-only render (no registered reference render: item path None, e.g. a corpus that only had graph
    # views for the older students) gets the reference-free checks: decode, call markup, train/held group overlap
    # and target parse-back. Its selection is the renderer's own (registry admission, reservations, held groups).
    if item.get('path') is None:
        reference = None
    new = REPO / item['out']
    old = REPO / item['path'] if reference else None
    report = {'schema': 'natlang.student-rerender-verification/1', 'id': item['id'], 'student': student,
              'reference': reference, 'views': {}, 'problems': []}
    new_receipt = load(new / 'receipt.json')
    old_receipt = load(old / 'receipt.json') if old else {}
    from natlang_neuralese.common.hashing import sha256_file_hex
    report['render_receipt_sha256'] = sha256_file_hex(new / 'receipt.json')
    if old:
        report['reference_receipt_sha256'] = sha256_file_hex(old / 'receipt.json')
    if item['kind'] == 'direct' and old:
        if new_receipt.get('sources') != old_receipt.get('sources'):
            report['problems'].append('source identities differ from the reference render receipt')
    wanted_targets = set()
    view_rows = {}
    for view in item['views']:
        ref_path = view_file(old, view, reference) if old else None
        new_path = view_file(new, view, student)
        if ref_path is None and old:
            continue
        if new_path is None:
            report['problems'].append(f'{view}: no {student} rendering')
            continue
        ref_ids = [identity(r) for r in rows(ref_path)] if ref_path else None
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
        if ref_ids is None:  # reference-free
            stats.update(documents=len(new_ids),
                         train_test_group_overlap=len(groups['train'] & groups.get('test', set())))
            report['views'][view] = dict(stats)
            for key in ('decode_mismatch', 'pythonic_call_marker', 'train_test_group_overlap'):
                if stats[key]:
                    report['problems'].append(f'{view}: {key} {stats[key]}')
            continue
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
    if subset_eligible(report, item):
        report['exact_subset'] = write_exact_subset(item, queue, report)
        report['problems'] = []
    report['passed'] = not report['problems']
    # The same document set per view (row order follows the shard count and is not training-relevant).
    report['identical_documents'] = all(not v.get('only_reference') and not v.get('only_student')
                                        and not v.get('split_or_group_changed') for v in report['views'].values())
    save(new / 'student-verification.json', report)
    print(json.dumps({'id': args.id, 'passed': report['passed'], 'problems': report['problems'][:10]}))


# ----------------------------------------------------------------------------------------------------------- entry

# ---------------------------------------------------------------------------------------------- per-document subset
# A render whose only verification problems are documents whose split or source groups changed against the reference
# (typically held protection that widened since the reference render) is used per document instead of being rejected
# whole: the exact subset keeps every document whose identity (id, split, source groups, source record ids) equals a
# reference document, and drops train documents that share a source group with any held document of either render.
# The dropped ids are recorded. Mixture building still applies the complete current protection union.

SUBSET_DIR = 'exact-subset'
SUBSET_PROBLEM = re.compile(r'^[a-z-]+: split_or_group_changed \d+$')


def subset_eligible(report: dict, item: dict) -> bool:
    """Only split/group changes, and every checked supervised target parses back to its source call."""
    if not report.get('problems') or not all(SUBSET_PROBLEM.match(p) for p in report['problems']):
        return False
    parse = report.get('target_parse')
    if isinstance(parse, dict):
        return not any(k.endswith(':differ') or k.endswith(':source_missing') for k in parse)
    # Parts renders stream their part inputs; their verification never checks targets (as for passed parts renders).
    return item['kind'] == 'parts' and isinstance(parse, str) and parse.startswith('not checked (parts render')


def subset_file(out: Path, view: str, student: str) -> Path:
    return out / SUBSET_DIR / view / student / 'text-data.jsonl.zst'


def write_exact_subset(item: dict, queue: dict, report: dict) -> dict:
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.common.jsonio import iter_jsonl_rows, write_jsonl_text
    student, reference = queue['student'], queue['reference']
    new, old = REPO / item['out'], REPO / item['path']
    views = [v for v in item['views'] if view_file(old, v, reference) and view_file(new, v, student)]
    held, refs = set(), {}
    for view in views:
        refs[view] = set()
        for _, row in iter_jsonl_rows(view_file(old, view, reference)):
            refs[view].add(identity(row))
            if row['split'] != 'train':
                held.update(row.get('source_groups') or ())
        for _, row in iter_jsonl_rows(view_file(new, view, student), split='test'):
            held.update(row.get('source_groups') or ())
    result = {}
    for view in views:
        target = subset_file(new, view, student)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name('.tmp-' + target.name)
        counts, excluded, groups = collections.Counter(), [], {'train': set(), 'test': set()}
        with write_jsonl_text(tmp) as stream:
            for line, row in iter_jsonl_rows(view_file(new, view, student)):
                if identity(row) not in refs[view]:
                    counts['excluded_nonidentical'] += 1
                    excluded.append(row['id'])
                elif row['split'] == 'train' and set(row.get('source_groups') or ()) & held:
                    counts['excluded_held_group'] += 1
                    excluded.append(row['id'])
                else:
                    stream.write(line if line.endswith('\n') else line + '\n')
                    counts[row['split']] += 1
                    groups.setdefault(row['split'], set()).update(row.get('source_groups') or ())
        if groups['train'] & groups['test']:
            tmp.unlink()
            raise ValueError(f'{item["id"]} {view}: exact subset still has train/held group overlap')
        tmp.replace(target)
        result[view] = {'file': str(target.relative_to(new)), 'sha256': sha256_file_hex(target),
                        'documents': counts['train'] + counts['test'], 'counts': dict(counts),
                        'excluded_ids': excluded}
    return {'schema': 'natlang.exact-rerender-subset/2', 'resolved_problems': list(report['problems']),
            'policy': 'keep documents identical to the reference (id, split, source groups, source record ids); drop '
                      'train documents sharing a source group with any held document of either render',
            'views': result}


def subset(args):
    """Write the exact subset of a verify-failed render whose report is subset-eligible; marks the report passed."""
    queue = load(args.queue)
    item = queue['items'][args.id]
    path = REPO / item['out'] / 'student-verification.json'
    report = load(path)
    if not subset_eligible(report, item):
        raise SystemExit(f'{args.id}: verification problems are not limited to split/group changes')
    report['exact_subset'] = write_exact_subset(item, queue, report)
    report['problems'] = []
    report['passed'] = True
    save(path, report)
    print(json.dumps({'id': args.id, 'exact_subset': {v: s['counts'] for v, s in report['exact_subset']['views'].items()}}))


def serving_view_file(item: dict, view: str, student: str) -> Path | None:
    """The file a render's graphs and registration use: its exact subset when one was written, else the render."""
    out = REPO / item['out']
    path = subset_file(out, view, student)
    return path if path.exists() else view_file(out, view, student)


def add(args):
    """Queue a student-only render (no registered reference render) of an admitted source corpus."""
    registry = {c['id']: c for c in load(REGISTRY)['corpora']}
    source = registry[args.source_corpus]
    if not source.get('training_admission'):
        raise SystemExit(f'{args.source_corpus} is not training-admitted')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    with Queue(args.queue) as queue:
        if args.id in queue['items'] and not args.replace:
            raise SystemExit(f'{args.id} is already queued')
        queue['items'][args.id] = {
            'id': args.id, 'new_id': f'{args.id}-{queue["student"]}', 'kind': args.kind, 'status': 'pending',
            'tier': args.tier, 'out': rel(args.out), 'path': None, 'source_corpus': args.source_corpus,
            'views': args.views, 'reference_tokens': args.reference_tokens, 'missing': [], 'command': command,
            'note': args.note or 'student-only render of an admitted corpus (reference-free verification)'}
    print(json.dumps({'queued': args.id}))


def entry(args):
    queue = load(args.queue)
    item = queue['items'][args.id]
    student = queue['student']
    print(json.dumps(registry_entry(item, student), indent=1))


def registry_entry(item: dict, student: str) -> dict:
    from natlang_neuralese.common.hashing import sha256_file_hex
    registry = {c['id']: c for c in load(REGISTRY)['corpora']}
    source = registry[item.get('source_corpus') or item['id']]
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
        subset_view = (verification.get('exact_subset') or {}).get('views', {}).get(view)
        if subset_view:  # training uses the exact subset (per-document verification), not the whole render
            renders[view][tok]['exact_subset'] = {k: subset_view[k] for k in ('file', 'sha256', 'documents', 'counts')}
    keep = ('source_class', 'split', 'license', 'license_provenance', 'training_admission_reason')
    new = {'id': item['new_id'], 'owner': 'dgx', 'path': item['out'],
           'include': ['receipt.json', 'student-verification.json', 'render-policy-binding.json', 'parts.json',
                       'merge.json', '*/*/text-data.jsonl', '*/*/text-data.jsonl.zst', '*/*/omissions.jsonl',
                       'selection-omissions.jsonl', 'render-omissions/*/*', 'parts/*/receipt.json',
                       'parts/*/render-policy-binding.json', 'parts/*/*.jsonl', 'parts/*/*/*/omissions.jsonl',
                       'parts/*/*/*/omissions.jsonl.zst', 'parts/*/render-omissions/*/omissions.jsonl',
                       'native-graphs.sh', 'native-graphs/*/receipt.json', 'exact-subset/*/*/text-data.jsonl.zst', 'native-graphs/*/*/native-graphs.jsonl.zst'],
           **{k: source[k] for k in keep if k in source},
           'admission': (f'{student} render of the registered corpus {item.get("source_corpus") or item["id"]} (scripts/neuralese_data/'
                         f'student_rerender.py: its receipt\'s exact selection arguments, {student} tokenizer only); '
                         'inherits that render\'s admission, splits and held decisions; no new example admission'),
           'training_admission': source.get('training_admission'),
           'derived_from': [item.get('source_corpus') or item['id'], *(source.get('derived_from') or [])],
           'renders': renders,
           'student_rerender': {'reference_render': item['id'] if item.get('path') else None, 'student': student,
                                'identical_documents': verification['identical_documents'],
                                'views': {v: {k: s.get(k) for k in ('documents', 'reference_documents', 'same_order',
                                                                    'only_reference', 'only_student')}
                                          for v, s in verification['views'].items()},
                                'target_parse': verification.get('target_parse'),
                                **({'exact_subset': {'policy': verification['exact_subset']['policy'],
                                                     'resolved_problems': verification['exact_subset']['resolved_problems'],
                                                     'excluded': {v: len(x['excluded_ids']) for v, x in
                                                                  verification['exact_subset']['views'].items()}}}
                                   if verification.get('exact_subset') else {})}}
    graphs = {}
    for view in graph_views(item):
        path = out / 'native-graphs' / view / 'receipt.json'
        if path.exists():
            graph_receipt = load(path)
            graphs[view] = {'train': next(o['rows'] for o in graph_receipt['outputs'] if '/train/' in o['path']),
                            'test': next(o['rows'] for o in graph_receipt['outputs'] if '/test/' in o['path']),
                            'outputs': [{'path': rel(o['path']), 'rows': o['rows'], 'bytes': o['bytes'],
                                         'sha256': o['sha256']} for o in graph_receipt['outputs']],
                            'expanded_tokenizer_fingerprint': graph_receipt['expanded_tokenizer_fingerprint'],
                            'role_token_ids': graph_receipt['role_token_ids']}
            model = graph_receipt['model_id']
    if graphs:
        new['native_graph_views'] = {student: {'model_id': model,
                                               'views': {v: g['train'] + g['test'] for v, g in graphs.items()},
                                               'detail': graphs}}
    if receipt.get('render_policy_binding_sha256'):
        new['render_policy_binding_sha256'] = receipt['render_policy_binding_sha256']
    return new


def register_graphed(args):
    """Register every graphed (verified) render not yet registered, in one registration commit."""
    import register_corpora
    queue = load(args.queue)
    items, skipped = [], {}
    for i in queue['items'].values():
        if i['status'] != 'graphed':
            continue
        # A graphed item without a passed reference verification (e.g. an LFM3-only render queued by hand) is
        # reported, not registered; it must not stop the registration of every other render.
        path = REPO / i['out'] / 'student-verification.json'
        if not path.exists():
            skipped[i['id']] = 'no student-verification.json'
        elif not load(path).get('passed'):
            skipped[i['id']] = 'verification not passed'
        else:
            items.append(i)
    if skipped:
        print(json.dumps({'skipped': skipped}), flush=True)
    if args.limit:
        items = items[:args.limit]
    if not items:
        print(json.dumps({'registered': []}))
        return
    entries = [registry_entry(item, queue['student']) for item in items]
    done = register_corpora.register(entries, f'Register {len(entries)} {queue["student"]} renders with native graph '
                                     f'views (student_rerender.py)')
    with Queue(args.queue) as live:
        for item in items:
            live['items'][item['id']]['status'] = 'registered'
    print(json.dumps({'registered': done}))


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
    p.add_argument('--max-running', type=int, default=4, help='concurrent renders')
    p.add_argument('--max-post', type=int, default=4, help='concurrent verify/graph steps')
    p.add_argument('--priority', default='high', choices=('low', 'medium', 'high'),
                   help='ledger priority (2026-10-28: data conversion is the LFM3 critical path: high)')
    p.add_argument('--cpus', type=int, default=4)
    p.add_argument('--budget-gb', type=int, default=20, help='ledger budget of the largest direct renders')
    p.add_argument('--verify-budget-gb', type=int, default=3)
    p.add_argument('--tier', type=int, default=1)
    p.add_argument('--logs', required=True)
    p = sub.add_parser('status')
    p.add_argument('--queue', required=True)
    p.add_argument('-v', '--verbose', action='store_true')
    p = sub.add_parser('verify')
    p.add_argument('--queue', required=True)
    p.add_argument('--id', required=True)
    p = sub.add_parser('add')
    p.add_argument('--queue', required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--source-corpus', required=True, help='the training-admitted registry corpus rendered')
    p.add_argument('--out', required=True)
    p.add_argument('--kind', choices=('direct', 'parts'), required=True)
    p.add_argument('--views', nargs='+', default=['full', 'program-header', 'whole-trajectory'])
    p.add_argument('--tier', type=int, default=1)
    p.add_argument('--reference-tokens', type=int, default=0, help='size estimate for ordering and budgets')
    p.add_argument('--note')
    p.add_argument('--replace', action='store_true')
    p.add_argument('command', nargs=argparse.REMAINDER, help='-- the render command')
    p = sub.add_parser('subset')
    p.add_argument('--queue', required=True)
    p.add_argument('--id', required=True)
    p = sub.add_parser('entry')
    p.add_argument('--queue', required=True)
    p.add_argument('--id', required=True)
    p = sub.add_parser('register')
    p.add_argument('--queue', required=True)
    p.add_argument('--limit', type=int)
    args = ap.parse_args()
    {'plan': plan, 'launch': launch, 'status': status, 'verify': verify, 'entry': entry, 'subset': subset, 'add': add,
     'register': register_graphed}[args.cmd](args)


if __name__ == '__main__':
    main()
