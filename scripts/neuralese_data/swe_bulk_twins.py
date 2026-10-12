#!/usr/bin/env python3
"""Text twins of the bulk harness-bench records (training/neuralese/scripts/harness_bench_bulk.sh), per tokenizer.

Produces program-header and whole-trajectory twins for each tokenizer. Whole-trajectory documents: one document per maximal decision chain of a trajectory, every
assistant turn supervised (tool output at the shared feedback weight), instead of one document per decision, which
repeats the whole context for each of a trajectory's ~8 targets (~4.8B Mellum tokens for ~1% supervised). Each
finished records shard (a set of whole repositories, so its split is the whole corpus's) is rendered by the shared
renderer (scripts/neuralese_data/render_native_turn_corpus.py --renderings whole-trajectory, both tokenizers in one
pass) into TWIN_ROOT/render/NN. Each finished shard's four text files are compressed at once (zstd, verified by the
receipt's text hash, then the plain file is removed: one copy on disk). Once every shard is rendered, each twin is the
concatenation of the shards' zstd frames in shard order, TWIN_ROOT/<name>/text.jsonl.zst (a valid zstd stream of the
concatenated text; the receipt binds both the plain text hash and the file hash), and each shard's frame is removed
once the assembled file is verified. Program-header text compresses ~48x (it repeats each trajectory's context per
decision); the union builder streams .zst members. Resumable: a shard with a receipt is done; an unfinished one
resumes from the renderer's own work dir; an assembled twin with a matching receipt is kept.

  swe_bulk_twins.py --bulk BULK_OUT --twin-root DIR --tokenizer lfm25=SNAPSHOT --tokenizer mellum21=SNAPSHOT [--parallel 2]
     [--wait]   (keep polling until the records run has finished every shard)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / '.venv-neuralese/bin/python'
# The records' system prompt is the pi harness's own prompt, not natlang's generic runtime prose: keep it whole.
HEADER = ['--program-header-as-full']
RENDERINGS = ('program-header', 'whole-trajectory')


def twin_path(root: Path, name: str, rendering: str) -> Path:
    # Keep the established whole-trajectory paths; header data has its own directory.
    return root / name if rendering == 'whole-trajectory' else root / rendering / name


def verify_shard_receipt(shard: Path, out: Path, tokenizers: dict) -> dict:
    receipt = json.loads((out / 'receipt.json').read_text())
    expected = {f'{mode}/{name}' for mode in RENDERINGS for name in tokenizers}
    if not expected.issubset(receipt.get('renders', {})):
        raise ValueError(f'{out}: finished render lacks requested twins; use a fresh twin root')
    source_hashes = {entry['sha256'] for entry in receipt.get('sources', {}).values()}
    if file_sha(shard / 'records.jsonl') not in source_hashes:
        raise ValueError(f'{out}: source records changed; use a fresh twin root')
    piece_hashes = {entry['sha256'] for entry in receipt.get('explicit_piece_bindings', [])}
    if file_sha(shard / 'pieces.jsonl') not in piece_hashes:
        raise ValueError(f'{out}: source pieces changed; use a fresh twin root')
    return receipt


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


ZSTD = ['zstd', '-q', '--long=27']


def zst_plain_sha(paths: list[Path], hasher=None) -> tuple[str, int]:
    """sha256 and byte count of the decompressed concatenation of zstd files (streamed, nothing written)."""
    h = hasher or hashlib.sha256()
    total = 0
    for path in paths:
        proc = subprocess.Popen([*ZSTD, '-dc', str(path)], stdout=subprocess.PIPE)
        for chunk in iter(lambda: proc.stdout.read(1 << 22), b''):
            h.update(chunk)
            total += len(chunk)
        proc.stdout.close()
        if proc.wait() != 0:
            raise RuntimeError(f'zstd failed on {path}')
    return h.hexdigest(), total


def compact_shard(out: Path, receipt: dict, tokenizers: dict) -> None:
    """Replace a finished shard's plain text files by verified zstd copies (one copy on disk)."""
    for rendering in RENDERINGS:
        for name in tokenizers:
            plain = out / rendering / name / 'text-data.jsonl'
            packed = plain.with_name('text-data.jsonl.zst')
            want = receipt['renders'][f'{rendering}/{name}']['text_data_sha256']
            if not plain.exists():
                continue  # compacted earlier; absent altogether once assembled (assemble() checks every frame)
            if file_sha(plain) != want:
                raise ValueError(f'{plain}: rendered text hash mismatch')
            tmp = packed.with_name(packed.name + '.tmp')
            subprocess.run([*ZSTD, '-f', '-T2', str(plain), '-o', str(tmp)], check=True)
            if zst_plain_sha([tmp])[0] != want:
                raise ValueError(f'{tmp}: compressed copy does not decompress to the rendered text')
            tmp.rename(packed)
            plain.unlink()


def render(shard: Path, tokenizers: dict, root: Path, workers: int) -> str:
    out = root / 'render' / shard.name
    if (out / 'receipt.json').exists():
        compact_shard(out, verify_shard_receipt(shard, out, tokenizers), tokenizers)
        return f'{shard.name} kept'
    out.parent.mkdir(parents=True, exist_ok=True)
    # A shard's twins take ~3.4 GB per 1.15 GB of records (both renderings, both
    # tokenizers), ~110 GB for the whole bulk run: pause while the NVMe is short, like the other generators
    # (coordinator 2026-10-12: disk_guard.sh, 100 GB).
    subprocess.call([str(ROOT / 'scripts/disk_guard.sh'), str(root)])
    log = root / 'render' / f'{shard.name}.log'
    cmd = ['nice', '-n', '10', str(PY), str(ROOT / 'scripts/neuralese_data/render_native_turn_corpus.py'),
           '--corpus-id', f'swe-rebench-bulk-{shard.name}', '--source', str(shard / 'records.jsonl'),
           '--pieces', str(shard / 'pieces.jsonl'), '--out-dir', str(out), '--renderings', *RENDERINGS,
           '--full-sample', '0', '--workers', str(workers), '--shards', '16', *HEADER]
    for name, path in tokenizers.items():
        cmd += ['--tokenizer', f'{name}={path}']
    with open(log, 'a') as handle:
        code = subprocess.call(cmd, stdout=handle, stderr=subprocess.STDOUT)
    if code == 0 and (out / 'receipt.json').exists():
        compact_shard(out, verify_shard_receipt(shard, out, tokenizers), tokenizers)
    return f'{shard.name} exit {code}'


def assemble(root: Path, name: str, rendering: str, shards: list[Path], receipts: dict) -> None:
    """Concatenate the shards' verified zstd frames into <twin>/text.jsonl.zst, then remove the shards' frames."""
    target = twin_path(root, name, rendering)
    packed = target / 'text.jsonl.zst'
    if (target / 'receipt.json').exists():
        summary = json.loads((target / 'receipt.json').read_text())
        if summary.get('rendering') != rendering or summary.get('text_zst_sha256') != file_sha(packed):
            raise ValueError(f'{target}: assembled twin does not match its receipt')
        return
    totals: dict = {}
    parts, frames = [], []
    target.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        part = root / 'render' / shard.name
        receipt = receipts[shard.name]
        render = receipt['renders'][f'{rendering}/{name}']
        frame = part / rendering / name / 'text-data.jsonl.zst'
        if zst_plain_sha([frame])[0] != render['text_data_sha256']:
            raise ValueError(f'{frame}: compressed text does not match the shard receipt')
        frames.append(frame)
        for split, counts in render['by_split'].items():
            into = totals.setdefault(split, {})
            for key, value in counts.items():
                into[key] = into.get(key, 0) + value
        totals.setdefault('max_document_tokens', 0)
        totals['max_document_tokens'] = max(totals['max_document_tokens'], render['max_document_tokens'])
        keep = target / 'shards' / shard.name
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy(part / 'receipt.json', keep / 'receipt.json')
        shutil.copy(part / rendering / name / 'omissions.jsonl', keep / 'omissions.jsonl')
        shutil.copy(part / 'render-omissions' / name / 'omissions.jsonl', keep / 'render-omissions.jsonl')
        parts.append({'shard': shard.name, 'receipt_sha256': file_sha(part / 'receipt.json'),
                      'text_data_sha256': render['text_data_sha256'], 'tokens': render['by_split'],
                      'merge_omissions': render['merge_omissions'], 'selection': receipt['selection']})
    # zstd frames concatenate: the file decompresses to the shards' text in shard order.
    tmp = target / 'text.jsonl.zst.tmp'
    with open(tmp, 'wb') as dst:
        for frame in frames:
            with open(frame, 'rb') as src:
                shutil.copyfileobj(src, dst, 1 << 24)
    expected = hashlib.sha256()
    for frame in frames:  # the plain text of the concatenation, from the verified frames
        zst_plain_sha([frame], expected)
    plain_sha, plain_bytes = zst_plain_sha([tmp])
    if plain_sha != expected.hexdigest():
        raise ValueError(f'{tmp}: assembled file does not decompress to the shards\' text')
    tmp.rename(packed)
    first = receipts[shards[0].name]
    summary = {'schema': 'natlang.harness-bench-bulk-twin/3',
               'builder': 'scripts/neuralese_data/render_native_turn_corpus.py --renderings program-header whole-trajectory per shard',
               'rendering': rendering, 'renderer_code': first['renderer_code'],
               'tokenizer': first['tokenizers'][name], 'whole_trajectory': first['whole_trajectory'],
               'text_file': 'text.jsonl.zst', 'text_jsonl_sha256': plain_sha, 'text_jsonl_bytes': plain_bytes,
               'text_zst_sha256': file_sha(packed), 'tokens': totals, 'shards': parts,
               'concatenation': 'shard order 00..NN of zstd frames; each shard is a set of whole repositories'}
    (target / 'receipt.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    for frame in frames:  # the assembled twin is now the one copy
        frame.unlink()
    print(json.dumps({f'{rendering}/{name}': totals}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bulk', type=Path, required=True)
    parser.add_argument('--twin-root', type=Path, required=True)
    parser.add_argument('--tokenizer', action='append', required=True, help='NAME=SNAPSHOT')
    parser.add_argument('--parallel', type=int, default=2, help='shards rendered at once')
    parser.add_argument('--workers', type=int, default=3, help='renderer worker processes per shard')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    tokenizers = dict(item.split('=', 1) for item in args.tokenizer)
    shards = sorted(p for p in (args.bulk / 'shards').iterdir() if p.is_dir())
    if not shards:
        parser.error('bulk source has no records shards')
    while True:
        ready = [s for s in shards if (s / 'summary.json').exists()]
        for shard in ready:
            summary = json.loads((shard / 'summary.json').read_text())
            if summary.get('target_selection', {}).get('final_clean_assistant_retained') is not True:
                raise ValueError(f'{shard}: rebuild records from the existing replay with final assistant coverage')
        with ThreadPoolExecutor(args.parallel) as pool:
            for line in pool.map(lambda shard: render(shard, tokenizers, args.twin_root, args.workers), ready):
                print(line, flush=True)
        if len(ready) == len(shards):
            failed = [s.name for s in shards if not (args.twin_root / 'render' / s.name / 'receipt.json').exists()]
            if failed:
                print(json.dumps({'failed': failed}), flush=True)
                return 1
            # Bind source shards once before assembling four twins, rather than rehashing
            # every multi-GB records/piece file for each rendering and tokenizer.
            receipts = {s.name: verify_shard_receipt(s, args.twin_root / 'render' / s.name, tokenizers)
                        for s in shards}
            for name in tokenizers:
                for rendering in RENDERINGS:
                    subprocess.call([str(ROOT / 'scripts/disk_guard.sh'), str(args.twin_root)])
                    assemble(args.twin_root, name, rendering, shards, receipts)
            return 0
        if not args.wait:
            return 0
        time.sleep(300)


if __name__ == '__main__':
    sys.exit(main())
