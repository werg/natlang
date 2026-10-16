#!/usr/bin/env python3
"""Render a large zstd-compressed native-turn corpus in parts with the shared renderer, within a bounded disk peak.

One implementation for every generated corpus (static curriculum batches, port-record call corpora, ...): the shared
renderer (render_native_turn_corpus.py) reads plain JSONL and keeps a working copy of its selection, so a whole corpus
at once needs ~2x its decompressed size (~0.4 MB a decision): 70 GB for one 94k-decision static batch (2026-10-12).
Here the turn stream (the given .turns.jsonl.zst files, in order) is cut into parts of about PART_DECISIONS decisions
at program boundaries (a program's decisions stay in one part), each part is decompressed only while it renders, and
the part receipts are merged into one receipt for the corpus.

  OUT/parts.json            {"cuts": [0, n1, ..., total], "part_decisions": N}  (decision indexes of the stream)
  OUT/parts/pKK/            the shared renderer's output for part KK (receipt.json, program-header/, whole-trajectory/,
                            full/ = the 5% warm-up sample, zstd)
  OUT/receipt.json          merged receipt (documents and per-split counts summed over parts)

After the parts, merge_parts() joins them into OUT/RENDERING/TOK/text-data.jsonl (full: .jsonl.zst) and re-applies the
renderer's corpus-level rules ACROSS parts: held groups/texts of every part, exact duplicates, twin alignment (Pop
2026-10-12). The union builder (build_native_turn_union.py) enforces the same across corpora. Resumable: finished
parts are skipped.

  render_parts.py render --corpus-id ID --out OUT --source F.turns.jsonl.zst [F ...] [--register ENTRY.json]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex
# io-quiet markers (checkpoint writes in progress); loaded by path so no package import is needed.
_spec = importlib.util.spec_from_file_location(
    'io_quiet', REPO / 'training/neuralese/natlang_neuralese/common/io_quiet.py')
io_quiet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(io_quiet)
PART_DECISIONS = 12000
PROTECTED = 'data/neuralese/corpora/protected-bgkit-20261003/bgkit-benchmarks.protected.json'
HELD_TEXT = [
    'runs/neuralese-foundation-text-input-4962-root-adopted-union-20261009-v1/text-data.jsonl',
    'data/neuralese/corpora/harness-bench-swe-rebench-openhands-pi-text-lfm25-350m-20261010-v2/text.jsonl',
]
DISK_MIN_GB = int(os.environ.get('NATLANG_DISK_MIN_GB', '100'))
PROGRAM = re.compile(rb'"program_ir_id":"((?:[^"\\]|\\.)*)"')


def load(path: Path):
    return json.loads(path.read_text())


def save(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
    tmp.replace(path)


def file_identity(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {'path': str(path), 'resolved_path': str(resolved), 'bytes': resolved.stat().st_size,
            'sha256': sha256_file_hex(resolved)}


def render_binding(sources: list[Path], corpus_id: str, cpus: int, budget_gb: int, docker_gb: int,
                   part_decisions: int, compress_all: bool, protected: Path | None, reserved: Path | None,
                   held_text: list[Path] | None, decision_exclude_ids: list[Path] | None,
                   exclude_programs: list[Path] | None, target_exclude_ids: list[Path] | None = None) -> dict:
    """Identity of every semantic input/control used by a resumable render output."""
    effective_protected = protected or REPO / PROTECTED
    effective_reserved = reserved or REPO / 'training/source-split-reservations.json'
    effective_held = [REPO / p for p in HELD_TEXT] + (held_text or [])
    controls = [{'kind': kind, **file_identity(path)} for kind, paths in (
        ('protected', [effective_protected]), ('reserved', [effective_reserved]),
        ('held-text', effective_held), ('decision-exclude-ids', decision_exclude_ids or []),
        ('target-exclude-ids', target_exclude_ids or []),
        ('exclude-programs', exclude_programs or [])) for path in paths]
    policy_paths = ([effective_protected, effective_reserved] + effective_held +
                    (decision_exclude_ids or []) + (exclude_programs or []) + (target_exclude_ids or []))
    controls.extend({'kind': 'control-provenance-proof', **file_identity(proof)}
                    for source in policy_paths
                    if (proof := source.with_suffix('.proof.json')).is_file())
    code = [REPO / 'scripts/neuralese_data/render_parts.py',
            REPO / 'scripts/neuralese_data/render_native_turn_corpus.py',
            REPO / 'scripts/neuralese_data/render_native_turn_corpus.sh']
    return {'schema': 'natlang.render-parts-policy-binding/1', 'corpus_id': corpus_id,
            'sources': [file_identity(path) for path in sources], 'controls': controls,
            'renderer_code': [file_identity(path) for path in code],
            'options': {'part_decisions': part_decisions, 'compress_all': compress_all,
                        'full_sample': 0.05, 'shards': 32, 'program_header_as_full': False,
                        'cpus': cpus, 'budget_gb': budget_gb, 'docker_gb': docker_gb}}


def validate_binding(path: Path, expected: dict, *, legacy_description: str) -> str:
    digest = hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if path.exists():
        actual = load(path)
        if actual.get('sha256') != digest or actual.get('binding') != expected:
            raise RuntimeError(f'{legacy_description} exists with a different render policy/input identity: {path}')
        return digest
    if (path.parent / 'receipt.json').exists():
        raise RuntimeError(f'{legacy_description} has no policy binding; use a fresh output or explicitly re-filter it at union: {path.parent}')
    save(path, {'schema': 'natlang.render-parts-policy-binding-file/1', 'sha256': digest, 'binding': expected})
    return digest


def wait_for_disk(path: Path) -> None:
    """Pause while the NVMe has less than DISK_MIN_GB free or a trainer is writing a checkpoint (io-quiet marker);
    same rule as scripts/disk_guard.sh."""
    io_quiet.wait_while_quiet()
    said = False
    while shutil.disk_usage(path).free < DISK_MIN_GB * 2**30:
        if not said:
            print(json.dumps({'disk_guard': 'pausing', 'free_gb': shutil.disk_usage(path).free // 2**30}), flush=True)
            said = True
        time.sleep(120)


def stream(sources: list[Path]):
    """The decisions of all sources in order, one raw line each."""
    for source in sources:
        # Storage relocation preserves logical source paths with symlinks. zstd
        # refuses symlink arguments by default; resolve only its read argument,
        # keeping the logical path in manifests and errors.
        proc = subprocess.Popen(['zstd', '-dcq', '--long=27', str(source.resolve(strict=True))], stdout=subprocess.PIPE)
        yield from proc.stdout
        if proc.wait():
            raise RuntimeError(f'zstd failed on {source}')


def program_of(line: bytes) -> bytes:
    found = PROGRAM.search(line, 0, 4096) or PROGRAM.search(line)
    if not found:
        raise ValueError('turn without source_ref.program_ir_id')
    return found.group(1)


def plan(sources: list[Path], out: Path, part_decisions: int = PART_DECISIONS) -> list[int]:
    path = out / 'parts.json'
    if path.exists():
        return load(path)['cuts']
    out.mkdir(parents=True, exist_ok=True)
    cuts, n, last = [0], 0, None
    for line in stream(sources):
        program = program_of(line)
        if n - cuts[-1] >= part_decisions and program != last:
            cuts.append(n)
        last, n = program, n + 1
    save(path, {'cuts': cuts + [n], 'part_decisions': part_decisions,
                'sources': [str(s) for s in sources]})
    return load(path)['cuts']


def stream_parts(sources: list[Path], out: Path, part_decisions: int, on_part) -> list[int]:
    """Partition and consume the ordered source stream in one pass.

    Each closed part is atomically spooled as zstd JSONL and handed to
    ``on_part`` immediately, so production starts before the corpus scan ends.
    Existing spools are retained for the normal resumable render_part path;
    completed receipts are skipped while cuts are still recomputed from source.
    """
    out.mkdir(parents=True, exist_ok=True)
    cuts, n, last, k = [0], 0, None, 0
    target = None
    compressor = None
    raw = None

    def begin(index: int):
        nonlocal target, compressor, raw
        target = out / f'.p{index:02d}.turns.jsonl.zst'
        if (out / 'parts' / f'p{index:02d}' / 'receipt.json').exists() or target.exists():
            return
        tmp = target.with_name(target.name + '.tmp')
        raw = open(tmp, 'wb')
        compressor = subprocess.Popen(['zstd', '-q', '-3', '--long=27', '-T2', '-c'],
                                       stdin=subprocess.PIPE, stdout=raw)

    def finish(stop: int):
        nonlocal compressor, raw, target, k
        if target is not None and compressor is not None:
            compressor.stdin.close()
            code = compressor.wait()
            raw.close()
            if code:
                target.with_name(target.name + '.tmp').unlink(missing_ok=True)
                raise RuntimeError(f'zstd failed while spooling part {k}')
            target.with_name(target.name + '.tmp').replace(target)
            compressor = raw = None
        if stop > cuts[-1]:
            cuts.append(stop)
        start = cuts[-2] if len(cuts) > 1 else 0
        if stop > start:
            # render_part validates the binding and cheaply returns on a receipt;
            # still collect its receipt so corpus aggregation includes every part.
            on_part(k, start, stop)
            if (out / 'parts' / f'p{k:02d}' / 'receipt.json').exists():
                (out / f'.p{k:02d}.turns.jsonl.zst').unlink(missing_ok=True)
        compressor = raw = target = None
        k += 1

    begin(k)
    try:
        for line in stream(sources):
            program = program_of(line)
            if n - cuts[-1] >= part_decisions and program != last:
                finish(n)
                begin(k)
            if compressor is not None:
                compressor.stdin.write(line)
            last, n = program, n + 1
        if n > cuts[-1]:
            finish(n)
        elif compressor is not None:
            compressor.stdin.close()
            compressor.wait()
            raw.close()
            target.with_name(target.name + '.tmp').unlink(missing_ok=True)
            compressor = raw = target = None
    except BaseException:
        if compressor is not None:
            try:
                compressor.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            if compressor.poll() is None:
                compressor.terminate()
            compressor.wait()
        if raw is not None and not raw.closed:
            raw.close()
        if target is not None:
            target.with_name(target.name + '.tmp').unlink(missing_ok=True)
        raise
    return cuts


def render_part(sources: list[Path], out: Path, k: int, start: int, stop: int, corpus_id: str, unit_prefix: str,
                cpus: int, budget_gb: int, docker_gb: int, *, compress_all: bool = False,
                protected: Path | None = None, reserved: Path | None = None,
                held_text: list[Path] | None = None, decision_exclude_ids: list[Path] | None = None,
                exclude_programs: list[Path] | None = None, policy_binding_sha256: str | None = None,
                target_exclude_ids: list[Path] | None = None) -> None:
    part = out / 'parts' / f'p{k:02d}'
    if (part / 'receipt.json').exists():
        if policy_binding_sha256 is not None:
            validate_binding(part / 'render-policy-binding.json', {'sha256': policy_binding_sha256},
                             legacy_description='completed render part')
        return
    wait_for_disk(out)
    part.mkdir(parents=True, exist_ok=True)
    if policy_binding_sha256 is not None:
        validate_binding(part / 'render-policy-binding.json', {'sha256': policy_binding_sha256},
                         legacy_description='render part')
    # A part unit left from an interrupted run (preempted and requeued by the ledger) is waited for, not duplicated.
    prefix = f'{unit_prefix}-p{k:02d}-'
    # Its decompressed input is kept until the part has a receipt (a requeued unit reads it when relaunched), and a
    # complete one is reused after a restart of this process.
    plain = out / (f'.p{k:02d}.turns.jsonl.zst' if compress_all else f'.p{k:02d}.turns.jsonl')
    if not plain.exists():
        from natlang_neuralese.common.jsonio import write_jsonl_text
        tmp = plain.with_name(plain.stem + '.tmp' + plain.suffix)
        with write_jsonl_text(tmp) as f:
            for i, line in enumerate(stream(sources)):
                if i >= stop:
                    break
                if i >= start:
                    f.write(line.decode('utf-8'))
        tmp.rename(plain)
    while pending_units(prefix) and not (part / 'receipt.json').exists():
        time.sleep(60)
    if (part / 'receipt.json').exists():
        plain.unlink(missing_ok=True)
        return
    try:
        # A preempted part unit ends cleanly and waits in the ledger queue to be relaunched (it resumes its finished
        # shards): keep waiting, and keep its decompressed input, until it finishes; relaunch one that failed.
        for attempt in range(5):
            unit = f'{prefix}{int(time.time()) % 100000}'
            container = f'render-{part.name}-{hashlib.sha1(str(part.resolve()).encode()).hexdigest()[:10]}'
            protected = protected or REPO / PROTECTED
            reserved = reserved or REPO / 'training/source-split-reservations.json'
            # Caller-supplied held groups supplement the established shared references; a new global index must not
            # silently replace the baseline held sources.
            held_text = [REPO / p for p in HELD_TEXT] + (held_text or [])
            decision_exclude_ids = decision_exclude_ids or []
            exclude_programs = exclude_programs or []
            cmd = ['python3', str(REPO / 'scripts/memory_ledger.py'), 'run', '--unit', unit, '--budget-gb',
                   str(budget_gb), '--priority', 'low', '--resumable', '--wait', '86400', '--workdir', str(REPO),
                   '--container', container, '--stop-seconds', '90', '--',
                   'env', f'MEM={docker_gb}g', f'CPUS={cpus}', 'nice', '-n', '10',
                   'scripts/neuralese_data/render_native_turn_corpus.sh', str(part), '--corpus-id', corpus_id,
                   '--source', str(plain), '--workers', str(cpus), '--shards', '32', '--full-sample', '0.05',
                   '--protected', str(protected), '--reserved', str(reserved),
                   '--held-text', *[str(p) for p in held_text]]
            if decision_exclude_ids:
                cmd.extend(['--decision-exclude-ids', *[str(p) for p in decision_exclude_ids]])
            if target_exclude_ids:
                cmd.extend(['--target-exclude-ids', *[str(p) for p in target_exclude_ids]])
            if exclude_programs:
                cmd.extend(['--exclude-programs', *[str(p) for p in exclude_programs]])
            if compress_all:
                cmd.append('--compress-text')
            if subprocess.run(cmd).returncode == 0:
                time.sleep(30)
                while pending_units(unit):
                    time.sleep(60)
            if (part / 'receipt.json').exists():
                break
            print(json.dumps({'part': k, 'attempt': attempt, 'unit': unit, 'result': 'no receipt; relaunching'}),
                  flush=True)
            time.sleep(300)
        else:
            raise RuntimeError(f'render of {corpus_id} part {k} failed 5 times; see journalctl --user -u {prefix}*')
    except Exception:
        plain.unlink(missing_ok=True)  # a real failure; an interruption (TERM) keeps the input for the requeued unit
        raise
    plain.unlink(missing_ok=True)
    # The full rendering is the 5% warm-up sample (--full-sample, shared rule); it is still stored compressed (~50x).
    for path in sorted((part / 'full').glob('*/*.jsonl')):
        subprocess.run(['zstd', '-q', '-3', '--long=27', '-T4', '--rm', '-f', str(path)], check=True)


def pending_units(prefix: str) -> list[str]:
    listed = subprocess.run(['systemctl', '--user', 'list-units', '--state=active,activating', '--plain', '--no-legend',
                             f'{prefix}*'], capture_output=True, text=True).stdout.split()
    try:
        status = json.loads(subprocess.run(['python3', str(REPO / 'scripts/memory_ledger.py'), 'status'],
                                           capture_output=True, text=True).stdout)
        queued = [w['unit'] for w in status.get('waiting', [])]
    except (ValueError, KeyError, TypeError):
        queued = []
    return [u for u in listed + queued if u.startswith(prefix)]


RENDERINGS = ('full', 'program-header', 'whole-trajectory')


def part_files(out: Path, rendering: str, tok: str) -> list[Path]:
    files = []
    for part in sorted((out / 'parts').glob('p*')):
        for name in ('text-data.jsonl', 'text-data.jsonl.zst'):
            if (part / rendering / tok / name).exists():
                files.append(part / rendering / tok / name)
    return files


def compress_part_views(part: Path) -> None:
    """Compress completed part text before accumulating the next part."""
    for path in sorted(part.glob('*/*/text-data.jsonl')):
        subprocess.run(['zstd', '-q', '-3', '--long=27', '-T2', '--rm', '-f', str(path)], check=True)


def read_rows(path: Path):
    if path.suffix == '.zst':
        proc = subprocess.Popen(['zstd', '-dcq', '--long=27', str(path)], stdout=subprocess.PIPE)
        lines = proc.stdout
    else:
        proc, lines = None, open(path, 'rb')
    for line in lines:
        if line.strip():
            yield line, json.loads(line)
    if proc and proc.wait():
        raise RuntimeError(f'zstd failed on {path}')


def text_sha(row: dict) -> str:
    import hashlib
    return hashlib.sha256(row['text'].encode()).hexdigest()


def merge_parts(out: Path, *, compress_all: bool = False) -> dict:
    """Join the parts into one member file per rendering and tokenizer (OUT/RENDERING/TOK/text-data.jsonl, the layout
    the target-set manifest reads), restoring the renderer's corpus-level rules across parts (Pop 2026-10-12: part
    renders must keep global split decontamination): held stays held; a train document sharing a source group with a
    held document of ANY part, or equal in text to one, is dropped; exact duplicates (split, text) across parts are kept
    once; twin alignment: a document dropped for one tokenizer is dropped for all. Part text files are removed after
    the merge (their receipts and omissions stay). The union builder applies the same rules across corpora."""
    toks = sorted({p.parent.name for p in (out / 'parts').glob('p*/*/*/text-data.jsonl*')})
    held_groups, held_texts = set(), set()
    for rendering in RENDERINGS:
        for tok in toks:
            for path in part_files(out, rendering, tok):
                for _, row in read_rows(path):
                    if row['split'] != 'train':
                        held_groups.update(row.get('source_groups') or [])
                        held_texts.add(text_sha(row))
    counts: dict = {}
    for rendering in RENDERINGS:
        drop: dict[str, str] = {}
        for tok in toks:
            seen = set()
            for path in part_files(out, rendering, tok):
                for _, row in read_rows(path):
                    key = (row['split'], text_sha(row), tuple(row.get('context_only_assistant_turns', [])))
                    if row['split'] == 'train' and set(row.get('source_groups') or []) & held_groups:
                        drop.setdefault(row['id'], 'train_document_shares_held_source_group_across_parts')
                    elif row['split'] == 'train' and key[1] in held_texts:
                        drop.setdefault(row['id'], 'train_copy_of_held_document_across_parts')
                    elif key in seen:
                        drop.setdefault(row['id'], 'exact_duplicate_document_across_parts')
                    seen.add(key)
        for tok in toks:
            files = part_files(out, rendering, tok)
            if not files:
                continue
            dst = out / rendering / tok
            dst.mkdir(parents=True, exist_ok=True)
            compress = compress_all or rendering == 'full'
            target = dst / ('text-data.jsonl.zst' if compress else 'text-data.jsonl')
            tmp = target.with_name(target.name + '.tmp')
            decoded_digest, decoded_bytes = hashlib.sha256(), 0
            with open(tmp, 'wb') as raw:
                zp = subprocess.Popen(['zstd', '-q', '-3', '--long=27', '-T2', '-c'], stdin=subprocess.PIPE,
                                      stdout=raw) if compress else None
                sink = zp.stdin if zp else raw
                kept = by_split = 0
                splits: dict = {}
                for path in files:
                    for line, row in read_rows(path):
                        if row['id'] in drop:
                            continue
                        data = line if line.endswith(b'\n') else line + b'\n'
                        sink.write(data)
                        decoded_digest.update(data)
                        decoded_bytes += len(data)
                        splits[row['split']] = splits.get(row['split'], 0) + 1
                        kept += 1
                if zp:
                    zp.stdin.close()
                    if zp.wait():
                        raise RuntimeError('zstd failed')
            tmp.replace(target)
            with open(dst / 'omissions.jsonl', 'w') as f:
                for doc, reason in sorted(drop.items()):
                    f.write(json.dumps({'id': doc, 'reason': reason}) + '\n')
            counts[f'{rendering}/{tok}'] = {'documents': kept, 'documents_by_split': splits,
                                            'text_data_file': target.name,
                                            'text_data_sha256': sha256_file_hex(target),
                                            **({'decoded_text_data_sha256': decoded_digest.hexdigest(),
                                                'decoded_text_data_bytes': decoded_bytes} if compress else {}),
                                            'merge_omissions': dict(sorted(
                                                {r: list(drop.values()).count(r) for r in set(drop.values())}.items()))}
            for path in files:
                path.unlink()
    return {'held_groups': len(held_groups), 'held_texts': len(held_texts), 'merged': counts}


def merge_receipts(receipts: list[dict], cuts: list[int], merged: dict | None = None) -> dict:
    """One corpus-level receipt over its parts: per-split counts summed over parts (before the cross-part merge),
    documents after the merge, maxima kept."""
    renders: dict = {}
    for receipt in receipts:
        for key, info in receipt['renders'].items():
            agg = renders.setdefault(key, {'documents': 0, 'by_split': {}, 'max_document_tokens': 0})
            agg['documents'] += info.get('documents', 0)
            agg['max_document_tokens'] = max(agg['max_document_tokens'], info.get('max_document_tokens', 0))
            for split, counts in info.get('by_split', {}).items():
                into = agg['by_split'].setdefault(split, {})
                for field, value in counts.items():
                    into[field] = into.get(field, 0) + value
    for key, info in ((merged or {}).get('merged') or {}).items():
        if key in renders:
            renders[key]['documents_before_merge'] = renders[key]['documents']
            renders[key]['documents'] = info['documents']
            renders[key]['merge_omissions'] = info['merge_omissions']
            for field in ('text_data_file', 'text_data_sha256', 'decoded_text_data_sha256',
                          'decoded_text_data_bytes'):
                if field in info:
                    renders[key][field] = info[field]
    return {'schema': 'natlang.render-parts/2', 'parts': len(receipts), 'cuts': cuts, 'renders': renders,
            'cross_part_merge': {k: v for k, v in (merged or {}).items() if k != 'merged'},
            'note': 'by_split token counts are summed over parts before the cross-part merge; documents are after it'}


def render(sources: list[Path], out: Path, corpus_id: str, unit_prefix: str, cpus: int = 6, budget_gb: int = 6,
           docker_gb: int = 8, part_decisions: int = PART_DECISIONS, *, compress_all: bool = False,
           protected: Path | None = None, reserved: Path | None = None,
           held_text: list[Path] | None = None, decision_exclude_ids: list[Path] | None = None,
           exclude_programs: list[Path] | None = None, stream_source_parts: bool = False,
           target_exclude_ids: list[Path] | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    custom_policy_requested = any(value is not None for value in
                                   (protected, reserved, held_text, decision_exclude_ids, exclude_programs, target_exclude_ids))
    binding_path = out / 'render-policy-binding.json'
    has_unbound_output = (not binding_path.exists() and
                          ((out / 'receipt.json').exists() or (out / 'parts.json').exists() or
                           any((out / 'parts').glob('p*/receipt.json'))))
    if has_unbound_output:
        if custom_policy_requested:
            raise RuntimeError('legacy render output has no binding for requested protection controls; use a fresh '
                               'output or let the union re-filter and label their historical coverage')
        if (out / 'receipt.json').exists():
            receipt = load(out / 'receipt.json')
            if receipt.get('render_policy_binding_sha256'):
                raise RuntimeError('receipt claims a render binding but its binding file is missing')
            receipt['legacy_policy_unbound'] = True
            receipt['legacy_policy_note'] = 'historical default inputs; current protections must be rechecked at union'
            return receipt
        legacy_resume = True
    else:
        legacy_resume = False
    if legacy_resume:
        binding_sha256 = None
    else:
        binding = render_binding(sources, corpus_id, cpus, budget_gb, docker_gb, part_decisions, compress_all,
                                 protected, reserved, held_text, decision_exclude_ids, exclude_programs, target_exclude_ids)
        binding_sha256 = validate_binding(binding_path, binding, legacy_description='render output')
    if (out / 'receipt.json').exists():
        receipt = load(out / 'receipt.json')
        if binding_sha256 is not None and receipt.get('render_policy_binding_sha256') != binding_sha256:
            raise RuntimeError(f'render receipt lacks matching policy binding: {out / "receipt.json"}')
        return receipt
    receipts = []
    def consume_part(k: int, start: int, stop: int) -> None:
        render_part(sources, out, k, start, stop, corpus_id, unit_prefix, cpus, budget_gb, docker_gb,
                    compress_all=compress_all, protected=protected, reserved=reserved,
                    held_text=held_text, decision_exclude_ids=decision_exclude_ids,
                    exclude_programs=exclude_programs, policy_binding_sha256=binding_sha256,
                    target_exclude_ids=target_exclude_ids)
        if compress_all:
            compress_part_views(out / 'parts' / f'p{k:02d}')
        receipts.append(load(out / 'parts' / f'p{k:02d}' / 'receipt.json'))
    if stream_source_parts:
        if not compress_all:
            raise ValueError('--stream-source-parts requires --compress-all')
        cuts = stream_parts(sources, out, part_decisions, consume_part)
        save(out / 'parts.json', {'cuts': cuts, 'part_decisions': part_decisions,
                                  'sources': [str(s) for s in sources]})
    else:
        cuts = plan(sources, out, part_decisions)
        for k, (start, stop) in enumerate(zip(cuts, cuts[1:])):
            consume_part(k, start, stop)
    merged = load(out / 'merge.json') if (out / 'merge.json').exists() else None
    if merged is None:
        merged = merge_parts(out, compress_all=compress_all)
        save(out / 'merge.json', merged)
    receipt = merge_receipts(receipts, cuts, merged)
    if binding_sha256 is None:
        receipt['legacy_policy_unbound'] = True
        receipt['legacy_policy_note'] = 'historical default inputs; current protections must be rechecked at union'
    else:
        completed_binding = render_binding(sources, corpus_id, cpus, budget_gb, docker_gb, part_decisions,
                                           compress_all, protected, reserved, held_text,
                                           decision_exclude_ids, exclude_programs, target_exclude_ids)
        completed_sha = hashlib.sha256(json.dumps(completed_binding, sort_keys=True,
                                                   separators=(',', ':')).encode()).hexdigest()
        if completed_sha != binding_sha256:
            raise RuntimeError('render source, control, option, or renderer identity changed during production')
        receipt['render_policy_binding_sha256'] = binding_sha256
    save(out / 'receipt.json', receipt)
    return receipt


INCLUDE = ['receipt.json', 'parts.json', 'merge.json', 'render-policy-binding.json',
           'parts/*/render-policy-binding.json', '*/*/text-data.jsonl', '*/*/text-data.jsonl.zst',
           '*/*/omissions.jsonl', 'parts/*/receipt.json', 'parts/*/*.jsonl', 'parts/*/*/*/omissions.jsonl',
           'parts/*/*/*/omissions.jsonl.zst', 'parts/*/render-omissions/*/omissions.jsonl']


def render_summary(receipt: dict) -> dict:
    """The registry's `renders` block: per rendering and tokenizer, documents, per-split counts and the maximum."""
    out: dict = {}
    for key, info in receipt['renders'].items():
        mode, tok = key.split('/')
        out.setdefault(mode, {})[tok] = {k: v for k, v in info.items()
                                         if k in ('documents', 'by_split', 'max_document_tokens', 'text_data_file',
                                                  'text_data_sha256', 'decoded_text_data_sha256',
                                                  'decoded_text_data_bytes')}
    return out


def render_entry(source_id: str, rel_path: str, receipt: dict, **extra) -> dict:
    """Registry entry of a parts render of the admitted corpus SOURCE_ID (stored at the repo-relative REL_PATH)."""
    entry = {
        'id': f'{source_id}-render', 'owner': 'dgx', 'path': rel_path, 'include': INCLUDE,
        'source_class': 'rendered-natlang-native-teacher-turn-text',
        'split': 'per document, inherited from the source rows (validation/dev rendered as test); protected bgkit '
                 'benchmark overlaps and explicit test reservations dropped; train documents sharing a source group '
                 'with held text (4962 union, harness bench) dropped',
        'admission': f'renders of {source_id} in parts (scripts/neuralese_data/render_parts.py): program-header and '
                     'whole-trajectory for every program, full only for the 5% warm-up sample (zstd); LFM2.5 and '
                     'Mellum2.1 twins; no new example admission',
        'training_admission': True,
        'training_admission_reason': 'owner 2026-10-11: maximize data; renders of a training-admitted corpus',
        'derived_from': [source_id], 'license': 'facts from the source rows', 'renders': render_summary(receipt),
        'render_policy_binding_sha256': receipt.get('render_policy_binding_sha256'),
        'legacy_policy_unbound': receipt.get('legacy_policy_unbound', False),
        'legacy_policy_note': receipt.get('legacy_policy_note'),
    }
    entry.update(extra)
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('render')
    p.add_argument('--corpus-id', required=True, help='the admitted source corpus id')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', type=Path, nargs='+', required=True, help='.turns.jsonl.zst files, in order')
    p.add_argument('--unit-prefix', help='ledger unit name prefix of the part renders (default render-<corpus>)')
    p.add_argument('--cpus', type=int, default=6)
    p.add_argument('--budget-gb', type=int, default=6)
    p.add_argument('--docker-gb', type=int, default=8)
    p.add_argument('--compress-all', action='store_true',
                   help='compress completed parts and every merged view; decoded rows and selection are unchanged')
    p.add_argument('--stream-source-parts', action='store_true',
                   help='partition and render each compressed part during one source scan (requires --compress-all)')
    p.add_argument('--register', type=Path, metavar='REL_PATH',
                   help='register the render (repo-relative path of OUT) through register_corpora.py')
    p.add_argument('--protected', type=Path, help='protected benchmark index (default: existing shared index)')
    p.add_argument('--reserved', type=Path, help='test reservation index (default: existing shared index)')
    p.add_argument('--held-text', type=Path, action='append',
                   help='additional rendered held-text member supplying source groups; repeatable (shared held corpora always apply)')
    p.add_argument('--decision-exclude-ids', type=Path, action='append',
                   help='JSONL {id} dedup exclusions; decisions still complete whole trajectories; repeatable')
    p.add_argument('--target-exclude-ids', type=Path, action='append',
                   help='JSONL {id} quality target holds, context retained with zero native target loss; repeatable')
    p.add_argument('--exclude-programs', type=Path, action='append',
                   help='reviewed JSONL {id, reason_code?} whole-program exclusions; repeatable')
    p.add_argument('--link', action='store_true',
                   help='with --register: create the REL_PATH symlink to OUT in the shared checkout')
    args = ap.parse_args()
    receipt = render(args.source, args.out, args.corpus_id, args.unit_prefix or f'render-{args.corpus_id}'[:60],
                     args.cpus, args.budget_gb, args.docker_gb, compress_all=args.compress_all,
                     protected=args.protected, reserved=args.reserved, held_text=args.held_text,
                     decision_exclude_ids=args.decision_exclude_ids, exclude_programs=args.exclude_programs,
                     stream_source_parts=args.stream_source_parts, target_exclude_ids=args.target_exclude_ids)
    print(json.dumps({'rendered': args.corpus_id, 'parts': receipt['parts']}), flush=True)
    if args.register:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import register_corpora
        entry = render_entry(args.corpus_id, str(args.register), receipt)
        links = {str(args.register): str(args.out.resolve())} if args.link else None
        register_corpora.register([entry], f'Register {entry["id"]} (parts render)', links=links)
        print(json.dumps({'registered': entry['id']}), flush=True)


if __name__ == '__main__':
    main()
