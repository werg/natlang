#!/usr/bin/env python3
"""Student-tokenizer twin of a registered verified-teacher native-graph package (LFM3-1.7B, 2026-10-28).

A registered verified-positive teacher package (``build_verified_teacher_trajectory_graphs.py`` output: one ordinary
one-node graph per admitted teacher action) exists for the student views it was converted for (LFM2.5, Mellum2.1).
Rerunning that converter for a new student needs the original sandbox/relocation custody; the admission work it did
is already recorded per graph (``conversion.teacher_source``: exact source row id and digest, turn index, decision
admission, mask policy). This tool re-renders exactly the reference view's graphs from the package's own verified
source trajectories with the SAME rendering functions (``render_teacher_packet``, ``reviewed_teacher_context_guard``,
``text_row_graph``, ``provision``) for another declared line backbone. Graph ids, splits, source groups, source
record ids and teacher provenance are those of the reference graph; the reference view's deduplication and decision
omissions are inherited (the row set is the reference's). A turn that cannot render for the student (e.g. an XML
tool call that is not representable) is an explicit omission with its reason, never a silent drop.

Each output is verified in the same pass: token decode equals text, the supervised suffix parses back
(``serve.chat.build_message`` with the turn's own tool declarations) to the teacher's raw calls, ids/splits/groups
equal the reference rows, and train/test source groups are disjoint.

  teacher_graph_student_twin.py convert --package-id ID [--package-id ...] --model-key lfm3 --out-root DIR
  teacher_graph_student_twin.py plan --model-key lfm3 --out-root DIR --queue Q.json
  teacher_graph_student_twin.py run-queue --queue Q.json --shard K --shards N     (resumable; skips finished)
  teacher_graph_student_twin.py collect --queue Q.json --outputs OUT.json        (outputs + cross-package duplicates)
  teacher_graph_student_twin.py entries --queue Q.json --out ENTRIES.json        (registry entries of verified twins)
"""
from __future__ import annotations

import argparse
import collections
import copy
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))

SCHEMA = 'natlang.verified-teacher-native-graph-student-twin/1'
REGISTRY = ROOT / 'training/neuralese_corpora.json'


def _load(path):
    return json.loads(Path(path).read_text())


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True) + '\n')
    with tmp.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def is_teacher_package(entry: dict) -> bool:
    return ('verified-positive-native-graphs' in entry['id']
            or entry['id'].startswith('luna-program-training-worlds-native-graphs'))


def package_root(entry: dict) -> Path:
    path = Path(entry['path'])
    return path if path.is_absolute() else ROOT / path


def student_tokenizer(model_key: str):
    """The pinned student tokenizer, history-bound and provisioned exactly as the converter does."""
    from transformers import AutoTokenizer
    from natlang_neuralese.common import line_backbones
    from natlang_neuralese.serve.chat import bind_history_reasoning
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import provision, tokenizer_fingerprint
    from scripts.neuralese_data.student_tokenizers import resolve
    directory = resolve(model_key)['path']
    tokenizer = AutoTokenizer.from_pretrained(str(directory), trust_remote_code=False, local_files_only=True)
    bind_history_reasoning(tokenizer, policy=None, require_declared=True, allow_transition=False)
    contract = line_backbones.tokenizer_contract(model_key)
    source = tokenizer_fingerprint(tokenizer)
    if source != contract['source_fingerprint']:
        raise ValueError(f'{model_key} source tokenizer fingerprint {source} differs from its declaration')
    role_ids = provision(tokenizer)
    expanded = tokenizer_fingerprint(tokenizer)
    if expanded != contract['expanded_fingerprint']:
        raise ValueError(f'{model_key} expanded tokenizer fingerprint {expanded} differs from its declaration')
    return tokenizer, {'directory': str(directory), 'source_fingerprint': source, 'expanded_fingerprint': expanded,
                       'role_token_ids': list(role_ids), 'model_id': line_backbones.model_id(model_key),
                       'history_reasoning': tokenizer.natlang_history_reasoning}


def source_trajectories(root: Path, receipt: dict) -> tuple[Path, dict]:
    """The package's verified source trajectories, bound to the reference receipt's source digest."""
    from natlang_neuralese.common.hashing import sha256_file_hex
    want = receipt['source']['sha256']
    for candidate in (root / 'evidence' / 'native-trajectories.jsonl', Path(receipt['source']['path'])):
        if candidate.is_file() and sha256_file_hex(candidate) == want:
            rows = {}
            with candidate.open('rb') as stream:
                for raw in stream:
                    if raw.strip():
                        row = json.loads(raw)
                        rows[row['id']] = (hashlib.sha256(raw.rstrip(b'\r\n')).hexdigest(), row)
            return candidate, rows
    raise FileNotFoundError(f'no source trajectories with sha256 {want} for {root}')


def reference_graphs(path: Path):
    from natlang_neuralese.common.jsonio import iter_jsonl_rows
    if not path.exists():
        return
    for _, graph in iter_jsonl_rows(path):
        yield graph


def _calls(raw_calls):
    out = []
    for call in raw_calls or []:
        fn = call.get('function', call)
        arguments = fn.get('arguments')
        out.append((fn.get('name'), json.loads(arguments) if isinstance(arguments, str) else arguments))
    return out


def program_header_parity(tokenizer, context, target, tools, turn, text, token_ids, boundary, checks):
    """None when the program-header row equals strip_row(full row), else the omission reason; header telemetry."""
    sys.path.insert(0, str(ROOT / 'scripts' / 'neuralese_data'))
    from system_tool_sections import strip_row
    from strip_generic_system_prompt import split_system
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import render_teacher_packet
    parts = split_system(text)
    header = len(parts[1]) if parts else 0
    checks['header_chars_total'] += header
    checks['header_chars_max'] = max(checks['header_chars_max'], header)
    checks['header_over_2000_chars'] += header > 2000
    packet, omission = render_teacher_packet(tokenizer, context, target, tools, turn, prompt_rendering='full')
    if omission is not None:
        return 'program-header parity: full rendering unavailable: ' + str(omission.get('reason'))
    (full_text, full_ids, full_boundary), _ = packet
    full = {'text': full_text, 'token_ids': full_ids, 'supervised_suffix_start': full_boundary}
    im_start, im_end = tokenizer.convert_tokens_to_ids('<|im_start|>'), tokenizer.convert_tokens_to_ids('<|im_end|>')
    stripped, reason, _info = strip_row(full, tokenizer, im_start, im_end)
    if stripped is None:
        if reason == 'no leading system turn' and (full_text, full_ids) == (text, token_ids):
            checks['parity_equal_no_system_turn'] += 1
            return None
        return 'program-header parity: strip_failed: ' + str(reason)
    if (stripped['text'], stripped['token_ids'], stripped['supervised_suffix_start']) == (text, token_ids, boundary):
        checks['parity_equal'] += 1
        return None
    if stripped['text'] == text and (tokenizer.decode(stripped['token_ids'][stripped['supervised_suffix_start']:])
                                     == tokenizer.decode(token_ids[boundary:])):
        checks['parity_equal_text_serving_token_boundaries'] += 1  # serving tokenization kept, as the renderer does
        return None
    return 'program-header parity: renderer_and_strip_script_differ'


def view_dir(model_key: str, prompt_rendering: str = 'full') -> str:
    """Output view directory: ``<model_key>`` (full system prompt) or ``<model_key>-program-header`` (serving mode)."""
    return model_key if prompt_rendering == 'full' else f'{model_key}-{prompt_rendering}'


def convert(package_id: str, model_key: str, out_root: Path, *, reference_key: str = 'lfm25',
            tokenizer_state=None, prompt_rendering: str = 'full') -> dict:
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.data.native_graph_conversion import text_row_graph
    from natlang_neuralese.serve.chat import build_message
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import (
        GraphWriter, render_teacher_packet, reviewed_teacher_context_guard)
    entries = {e['id']: e for e in _load(REGISTRY)['corpora']}
    entry = entries[package_id]
    root = package_root(entry)
    out = out_root / package_id / view_dir(model_key, prompt_rendering)
    if (out / 'receipt.json').exists():
        return _load(out / 'receipt.json')
    reference_receipt_path = root / reference_key / 'receipt.json'
    reference = _load(reference_receipt_path)
    if reference.get('schema') != 'natlang.verified-teacher-native-graph-build/1':
        raise ValueError(f'{package_id}: reference view is not a verified-teacher native graph build')
    tokenizer, identity = tokenizer_state or student_tokenizer(model_key)
    source_path, trajectories = source_trajectories(root, reference)
    failed_policy = reference.get('failed_assistant_history_policy', 'preserve')
    writers = {split: GraphWriter(out / split / 'native-graphs.jsonl.zst') for split in ('train', 'test')}
    omissions, groups = [], {'train': set(), 'test': set()}
    checks = collections.Counter()
    reference_outputs = {}
    try:
        for split in ('train', 'test'):
            ref_path = root / reference_key / split / 'native-graphs.jsonl.zst'
            if ref_path.exists():
                reference_outputs[split] = {'path': str(ref_path), 'sha256': sha256_file_hex(ref_path)}
            for ref in reference_graphs(ref_path):
                checks[f'reference_rows:{split}'] += 1
                checks['reference_tokens'] += sum(len(n['token_ids']) for n in ref['nodes'])
                conversion = ref['conversion']
                source = conversion['teacher_source']
                row_sha, row = trajectories.get(source['source_row_id'], (None, None))
                if row is None or row_sha != source['source_row_sha256']:
                    raise ValueError(f'{package_id}: source row {source["source_row_id"]} missing or changed')
                index = source['teacher_turn_index']
                turn = row['trajectory'][index]
                if (turn.get('raw_response_sha256') != source.get('raw_response_sha256')
                        or turn.get('request_sha256') != source.get('request_sha256')):
                    raise ValueError(f'{package_id}: turn {index} of {row["id"]} differs from its reference binding')
                if 'decision_admission' in source:
                    decision = source['decision_admission'] or {}
                    if decision.get('approved') is not True:
                        raise ValueError(f'{package_id}: reference graph without an approved decision: {ref["id"]}')
                else:
                    # Builds before per-turn decision records (2026-10-15 v1-v12 era) admitted every exact-success
                    # action of their passed receipt; the reference row itself is that admission. No parent hold.
                    decision = {}
                    checks['reference_rows_without_decision_record'] += 1
                context, assistant = turn.get('context'), turn.get('assistant')
                raw_calls = (turn.get('model_response') or {}).get('raw_calls')
                tools = turn.get('tools_offered')
                target = {'role': 'assistant', 'content': assistant.get('content', ''), 'tool_calls': raw_calls}
                if isinstance(assistant.get('reasoning'), str) and assistant['reasoning']:
                    target['reasoning_content'] = assistant['reasoning']

                def omit(reason, **extra):
                    omissions.append({'id': ref['id'], 'split': split, 'source_row_id': row['id'],
                                      'teacher_turn_index': index, 'reason': reason, **extra})
                try:
                    packet, omission = render_teacher_packet(tokenizer, context, target, tools, turn,
                                                             prompt_rendering=prompt_rendering)
                except ValueError as error:  # e.g. RequestError('tool-call-unrepresentable') under XML calls
                    omit(f'{model_key} rendering refused: {error}'[:300])
                    continue
                if omission is not None:
                    omit(omission.get('reason'), detail=omission)
                    continue
                (text, token_ids, boundary), _emulation = packet
                if prompt_rendering == 'program-header':
                    # Same parity as render_native_turn_corpus.py: the serving renderer's program-header row must
                    # equal the strip script's derivation from the full row; a mismatch is an omission.
                    reason = program_header_parity(tokenizer, context, target, tools, turn, text, token_ids,
                                                   boundary, checks)
                    if reason is not None:
                        omit(reason)
                        continue
                mask_history = bool(decision.get('parent_quality_hold') or decision.get('guided_parent_review'))
                guard = None
                if mask_history or failed_policy == 'mask-runtime-failures':
                    try:
                        guard = reviewed_teacher_context_guard(turn, context, target, tools)
                    except ValueError as error:
                        if str(error) != 'ambiguous embedded chat boundary in reviewed context':
                            raise
                        omit(str(error))
                        continue
                identifier = conversion['source_row_id']
                row_data = {'id': identifier, 'split': split, 'source_groups': ref['source_groups'],
                            'tokenizer_sha256': identity['source_fingerprint'], 'text': text,
                            'token_ids': token_ids, 'supervised_suffix_start': boundary,
                            'source_twin_id': identifier, 'source_record_ids': conversion['source_record_ids']}
                try:
                    graph = text_row_graph(row_data, source_tokenizer_fingerprint=identity['source_fingerprint'],
                        tokenizer=tokenizer, expanded_tokenizer_fingerprint=identity['expanded_fingerprint'],
                        context_weight=conversion.get('context_weight', .05), mask_assistant_history=mask_history,
                        mask_failed_assistant_history=failed_policy == 'mask-runtime-failures',
                        assistant_history_guard=guard)
                except ValueError as error:
                    omit(f'{model_key} graph conversion refused: {error}'[:300])
                    continue
                if graph['id'] != ref['id'] or graph['split'] != ref['split'] or graph['source_groups'] != ref['source_groups']:
                    raise ValueError(f'{package_id}: twin identity differs from reference graph {ref["id"]}')
                graph['conversion']['teacher_source'] = copy.deepcopy(source)
                graph['conversion']['student_twin'] = {
                    'schema': SCHEMA, 'reference_model_key': reference_key, 'reference_graph_id': ref['id'],
                    'reference_output_sha256': reference_outputs[split]['sha256'],
                    'reference_tokenizer_fingerprint': ref['tokenizer_fingerprint'],
                    'source_trajectories_sha256': reference['source']['sha256']}
                # Verification in the same pass.
                if tokenizer.decode(token_ids, skip_special_tokens=False) != text:
                    checks['decode_mismatch'] += 1
                reply = tokenizer.decode(token_ids[boundary:], skip_special_tokens=False)
                try:
                    got = [(c['function']['name'], json.loads(c['function']['arguments']))
                           for c in build_message(reply, [], tools=tools).get('tool_calls') or []]
                except Exception as error:
                    got = [('<error>', str(error)[:200])]
                want = _calls(raw_calls)
                if got == want:
                    checks['target_calls_exact'] += 1
                elif [n for n, _ in got] == [n for n, _ in want] and all(
                        json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) or str(a) == str(b)
                        for (_, a), (_, b) in zip(got, want)):
                    checks['target_calls_equal_modulo_untyped_values'] += 1
                else:
                    checks['target_calls_differ'] += 1
                    if checks['target_calls_differ'] <= 3:
                        omissions.append({'id': ref['id'], 'check': 'target_calls_differ (kept; see verification)',
                                          'want': json.dumps(want)[:300], 'got': json.dumps(got)[:300]})
                groups[split].update(graph['source_groups'])
                checks['tokens'] += len(token_ids)
                writers[split].write(graph)
        outputs = {split: writers[split].finish() for split in ('train', 'test')}
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise
    overlap = sorted(groups['train'] & groups['test'])
    problems = []
    for key in ('decode_mismatch', 'target_calls_differ'):
        if checks[key]:
            problems.append(f'{key}: {checks[key]}')
    if overlap:
        problems.append(f'train/test source groups overlap: {overlap[:5]}')
    real_omissions = [o for o in omissions if 'reason' in o]
    verification = {'passed': not problems, 'problems': problems, 'checks': dict(checks),
                    'emitted_rows': sum(o['rows'] for o in outputs.values()),
                    'reference_rows': checks['reference_rows:train'] + checks['reference_rows:test'],
                    'omitted_rows': len(real_omissions),
                    'rows_accounted': (sum(o['rows'] for o in outputs.values()) + len(real_omissions)
                                       == checks['reference_rows:train'] + checks['reference_rows:test']),
                    'train_test_group_overlap': len(overlap)}
    if not verification['rows_accounted']:
        verification['passed'] = False
        problems.append('emitted + omitted rows differ from the reference rows')
    receipt = {'schema': SCHEMA, 'package_id': package_id, 'model_key': model_key, 'model_id': identity['model_id'],
               'prompt_rendering': prompt_rendering,
               **({'prompt_piece_table': {'path': 'training/prompt-pieces/pieces.json',
                                          'sha256': sha256_file_hex(ROOT / 'training/prompt-pieces/pieces.json')},
                   'program_header_parity': 'render_native_turn_corpus rule: serving program-header row equals '
                                            'system_tool_sections.strip_row(full row) (text and target; serving token '
                                            'boundaries kept), else an explicit omission'}
                  if prompt_rendering == 'program-header' else {}),
               'tokenizer': {k: identity[k] for k in ('source_fingerprint', 'expanded_fingerprint',
                                                      'role_token_ids', 'history_reasoning')},
               'reference': {'model_key': reference_key, 'model_id': reference.get('model_id'),
                             'receipt': {'path': str(reference_receipt_path),
                                         'sha256': sha256_file_hex(reference_receipt_path)},
                             'outputs': reference_outputs},
               'source': {'path': str(source_path), 'sha256': reference['source']['sha256'],
                          'rows': len(trajectories)},
               'failed_assistant_history_policy': failed_policy,
               'tool': {'path': str(Path(__file__).resolve().relative_to(ROOT)),
                        'sha256': sha256_file_hex(Path(__file__).resolve())},
               'omissions': omissions, 'outputs': outputs, 'verification': verification,
               'conversion': ('re-render of the reference view\'s admitted teacher actions from the exact source '
                              'trajectories for the student tokenizer with the verified converter\'s rendering and '
                              'graph functions; reference deduplication and decision omissions inherited')}
    _save(out / 'receipt.json', receipt)
    return receipt


# ------------------------------------------------------------------------------------------------- queue / batch

class Locked:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.lock = open(self.path.with_name(self.path.name + '.lock'), 'w')
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        self.value = _load(self.path) if self.path.exists() else {}
        return self.value

    def __exit__(self, kind, *_):
        if kind is None:
            _save(self.path, self.value)
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()


def plan(args):
    entries = [e for e in _load(REGISTRY)['corpora'] if is_teacher_package(e)]
    with Locked(args.queue) as queue:
        queue.update(model_key=args.model_key, reference_key=args.reference_key, out_root=str(args.out_root),
                     prompt_rendering=args.prompt_rendering)
        items = queue.setdefault('items', {})
        for entry in entries:
            if entry.get('training_admission') is not True or entry['id'].endswith('-' + args.model_key):
                continue
            root = package_root(entry)
            receipt = root / args.reference_key / 'receipt.json'
            item = items.setdefault(entry['id'], {'id': entry['id']})
            if item.get('status') in ('converted', 'failed'):
                continue
            if not receipt.exists():
                item.update(status='no-reference-view', reason=f'no {args.reference_key} receipt on this machine')
                continue
            ref = _load(receipt)
            source = Path(ref['source']['path'])
            evidence = root / 'evidence' / 'native-trajectories.jsonl'
            item.update(status='pending' if (evidence.exists() or source.exists()) else 'missing-source',
                        reference_rows=ref['outputs']['train']['rows'] + ref['outputs'].get('test', {}).get('rows', 0))
    print(json.dumps(collections.Counter(i['status'] for i in _load(args.queue)['items'].values())))


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _claim(queue_path, shard: int, shards: int):
    """Claim the next pending package (this shard's own first, then any): every running unit drains the queue."""
    with Locked(queue_path) as locked:
        items = locked['items']
        for item in items.values():  # a claim of a stopped/preempted process returns to pending
            if item.get('status') == 'claimed' and not _alive(item.get('claim_pid')):
                item['status'] = 'pending'
        # Smallest packages first: a preempted unit restarts its current package, so many short packages finish
        # between preemptions and the few very large ones come last.
        ids = sorted(items, key=lambda i: (items[i].get('reference_rows') or 0, i))
        order = [i for k, i in enumerate(ids) if k % shards == shard] + ids
        for package_id in order:
            if items[package_id].get('status') == 'pending':
                items[package_id].update(status='claimed', claim_pid=os.getpid(), claimed_at=time.time())
                return package_id
    return None


def run_queue(args):
    queue = _load(args.queue)
    out_root = Path(queue['out_root'])
    state = student_tokenizer(queue['model_key'])
    while (package_id := _claim(args.queue, args.shard, args.shards)) is not None:
        started = time.time()
        try:
            receipt = convert(package_id, queue['model_key'], out_root, reference_key=queue['reference_key'],
                              tokenizer_state=state, prompt_rendering=queue.get('prompt_rendering', 'full'))
            update = {'status': 'converted' if receipt['verification']['passed'] else 'verify-failed',
                      'rows': receipt['verification']['emitted_rows'],
                      'omitted': receipt['verification']['omitted_rows'], 'seconds': round(time.time() - started)}
        except FileNotFoundError as error:
            update = {'status': 'missing-source', 'reason': str(error)[:300]}
        except Exception as error:  # recorded per package; the unit continues
            update = {'status': 'failed', 'reason': f'{type(error).__name__}: {error}'[:500]}
        with Locked(args.queue) as locked:
            locked['items'][package_id].update(update)
            locked['items'][package_id].pop('claim_pid', None)
        print(json.dumps({'package': package_id, **update}), flush=True)


def collect(args):
    """Outputs of every converted twin, with cross-package exact-content duplicates (registry order: first kept)."""
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import graph_content_sha256
    from natlang_neuralese.common.jsonio import iter_jsonl_rows
    queue = _load(args.queue)
    out_root = Path(queue['out_root'])
    order = [e['id'] for e in _load(REGISTRY)['corpora'] if e['id'] in queue['items']]
    prod = {s['path'] for s in _load(args.production_sources)['sources']} if args.production_sources else set()
    seen, files, totals = {}, [], collections.Counter()
    for package_id in order:
        item = queue['items'][package_id]
        if item.get('status') != 'converted':
            continue
        receipt = _load(out_root / package_id / view_dir(queue['model_key'], queue.get('prompt_rendering', 'full'))
                        / 'receipt.json')
        root = package_root({'path': _registry_path(package_id)})
        in_production = any(str(root) in p for p in prod)
        for split, output in receipt['outputs'].items():
            if not output['rows']:
                continue
            duplicates = []
            for _, graph in iter_jsonl_rows(Path(output['path'])):
                content = graph_content_sha256(graph)
                if content in seen:
                    duplicates.append({'id': graph['id'], 'represented_by': seen[content]})
                else:
                    seen[content] = graph['id']
            totals[f'rows:{split}'] += output['rows']
            totals[f'tokens:{split}'] += output['tokens']
            totals['cross_package_duplicates'] += len(duplicates)
            files.append({'package_id': package_id,
                          'registry_id': f'{package_id}-{view_dir(queue["model_key"], queue.get("prompt_rendering", "full"))}',
                          'prompt_rendering': queue.get('prompt_rendering', 'full'),
                          'split': split, 'path': output['path'], 'sha256': output['sha256'],
                          'rows': output['rows'], 'tokens': output['tokens'],
                          'reference_tokens': receipt['verification']['checks'].get('reference_tokens'),
                          'in_lfm25_production_sources': in_production,
                          'cross_package_duplicate_ids': [d['id'] for d in duplicates]})
    _save(args.outputs, {'schema': 'natlang.teacher-graph-student-twin-outputs/1', 'model_key': queue['model_key'],
                         'prompt_rendering': queue.get('prompt_rendering', 'full'),
                         'model_id': files and _load(out_root / files[0]['package_id'] /
                                                     view_dir(queue['model_key'], queue.get('prompt_rendering', 'full'))
                                                     / 'receipt.json')['model_id'],
                         'totals': dict(totals), 'files': files,
                         'duplicates_note': 'exact graph content (tokens + weights) already emitted by an earlier '
                                            'package in registry order; exclude those ids to avoid double exposure'})
    print(json.dumps(dict(totals)))


_PATHS = None


def _registry_path(package_id):
    global _PATHS
    if _PATHS is None:
        _PATHS = {e['id']: e['path'] for e in _load(REGISTRY)['corpora']}
    return _PATHS[package_id]


def entries(args):
    queue = _load(args.queue)
    out_root = Path(queue['out_root'])
    registry = {e['id']: e for e in _load(REGISTRY)['corpora']}
    result = []
    for package_id, item in sorted(queue['items'].items()):
        view = view_dir(queue['model_key'], queue.get('prompt_rendering', 'full'))
        new_id = f'{package_id}-{view}'
        if item.get('status') != 'converted' or new_id in registry:
            continue
        source = registry[package_id]
        receipt = _load(out_root / package_id / view_dir(queue['model_key'], queue.get('prompt_rendering', 'full'))
                        / 'receipt.json')
        directory = out_root / package_id
        rel = str(directory.relative_to(ROOT)) if directory.is_relative_to(ROOT) else str(directory)
        keep = ('source_class', 'split', 'license', 'license_provenance', 'training_admission_reason')
        mk = view
        result.append({
            'id': new_id, 'owner': 'dgx', 'path': rel,
            'include': [f'{mk}/receipt.json', f'{mk}/train/native-graphs.jsonl.zst', f'{mk}/test/native-graphs.jsonl.zst'],
            **{k: source[k] for k in keep if k in source},
            'prompt_rendering': queue.get('prompt_rendering', 'full'),
            'admission': (f'{queue["model_key"]} ({queue.get("prompt_rendering", "full")} rendering) student twin of {package_id} (scripts/neuralese_data/teacher_graph_student_twin.py): '
                          f'the reference {queue["reference_key"]} view\'s admitted teacher actions re-rendered from '
                          'the exact source trajectories; inherits its admission, splits, deduplication and decision '
                          f'omissions; {receipt["verification"]["omitted_rows"]} explicit render omissions'),
            'training_admission': source.get('training_admission'),
            'derived_from': [package_id],
            'source_binding': {'model_id': receipt['model_id'], 'model_key': queue['model_key'],
                               'tokenizer': {k: receipt['tokenizer'][k] for k in
                                             ('source_fingerprint', 'expanded_fingerprint', 'role_token_ids')},
                               'source_trajectories_sha256': receipt['source']['sha256'],
                               'reference_receipt_sha256': receipt['reference']['receipt']['sha256'],
                               'rows': receipt['verification']['emitted_rows'],
                               'omitted_rows': receipt['verification']['omitted_rows'],
                               'verification_passed': receipt['verification']['passed']},
            'note': 'Registration establishes availability only; learner exposure is recorded by each continuation.'})
    _save(args.out, result)
    print(len(result), 'entries')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('convert')
    p.add_argument('--package-id', action='append', required=True)
    p.add_argument('--model-key', required=True)
    p.add_argument('--reference-key', default='lfm25')
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--prompt-rendering', choices=('full', 'program-header'), default='program-header')
    p = sub.add_parser('plan')
    p.add_argument('--model-key', required=True)
    p.add_argument('--reference-key', default='lfm25')
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--prompt-rendering', choices=('full', 'program-header'), default='program-header',
                   help='program-header is the serving mode (policy serving_mode); full keeps the generic system prompt')
    p = sub.add_parser('run-queue')
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p = sub.add_parser('collect')
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--outputs', type=Path, required=True)
    p.add_argument('--production-sources', type=Path)
    p = sub.add_parser('entries')
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == 'convert':
        state = student_tokenizer(args.model_key)
        for package_id in args.package_id:
            receipt = convert(package_id, args.model_key, args.out_root.resolve(), reference_key=args.reference_key,
                              tokenizer_state=state, prompt_rendering=args.prompt_rendering)
            print(json.dumps({'package': package_id, 'verification': receipt['verification']}))
    else:
        {'plan': plan, 'run-queue': run_queue, 'collect': collect, 'entries': entries}[args.cmd](args)


if __name__ == '__main__':
    main()
