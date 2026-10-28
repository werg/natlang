#!/usr/bin/env python3
"""Find registered training-side sources that contain rows of given task families (oracle correction passes).

The file list is bounded by the registry: the manifest files of every training/neuralese_corpora.json entry that is not
explicitly held, evaluation-only or evidence-only. Each file is streamed in 32 MiB blocks (``.zst`` decompressed in a
stream) and the clean page cache of what was read is dropped, so the job's charged memory stays well under 1 GiB.
Progress is appended per chunk of files to ``OUT/progress.txt``; rerunning resumes. ``OUT/matches.txt`` lists
``path:count`` for files mentioning a family name, and ``OUT/by-corpus.json`` groups them by registry corpus with its
admission text. A mention is a candidate, not a classified row: inspect matching rows' ``curriculum.family`` and
``semantics`` (oracle form) before correcting them.

  scripts/memory_ledger.py run --unit family-scan-<id> --budget-gb 2 --priority low --no-resumable -- \\
    .venv-neuralese/bin/python scripts/neuralese_data/scan_task_family_rows.py --out DIR edit_stream contract_diagnosis
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOT_TRAINING = re.compile(r'held|evaluation-only|no-training|not-training|not training|evidence-only|model-artifact|'
                          r'never training|generation-input|scaffolding only|review-only|supplemental', re.I)
BLOCK = 32 << 20
CHUNK = 50


def candidate_corpora(root: Path, data_root: Path) -> list[dict]:
    registry = json.loads((root / 'training/neuralese_corpora.json').read_text(encoding='utf-8'))
    out = []
    for corpus in registry['corpora']:
        admission = corpus.get('admission')
        if corpus.get('training_admission') is False:
            continue
        if isinstance(admission, str) and NOT_TRAINING.search(admission) and not re.search(r'\badmitted\b|trained', admission, re.I):
            continue
        manifest = root / 'training/corpus-manifests' / f"{corpus['id']}.json"
        if not manifest.exists():
            continue
        data = json.loads(manifest.read_text(encoding='utf-8'))
        base = data.get('path') or corpus.get('path') or ''
        files = [str(data_root / base / item['path']) for item in data.get('files', [])
                 if isinstance(item, dict) and str(item.get('path', '')).endswith(('.jsonl', '.json', '.jsonl.zst'))]
        out.append({'id': corpus['id'], 'admission': json.dumps(admission)[:200], 'files': files})
    return out


def count(path: str, patterns: tuple[bytes, ...]) -> int:
    """Occurrences of the patterns, or -1 when the file cannot be opened (reported, never counted as clean)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return -1
    total, tail = 0, b''
    try:
        raw = os.fdopen(fd, 'rb', buffering=0, closefd=False)
        stream = raw
        if path.endswith('.zst'):
            import zstandard
            stream = zstandard.ZstdDecompressor().stream_reader(raw)
        while block := stream.read(BLOCK):
            data = tail + block
            total += sum(data.count(p) for p in patterns) - sum(tail.count(p) for p in patterns)
            tail = data[-64:]
            os.posix_fadvise(fd, 0, raw.tell(), os.POSIX_FADV_DONTNEED)
    except Exception:  # unreadable or truncated file: report what was counted
        pass
    finally:
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
        os.close(fd)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--data-root', type=Path, default=ROOT,
                        help='checkout whose runs/ and data/ hold the registered files (default: this checkout)')
    parser.add_argument('--corpora-file', type=Path, help='scan only the corpus ids listed (one per line or a JSON list)')
    parser.add_argument('--exclude', default=None, help='regular expression of file paths to skip (for example rendered views)')
    parser.add_argument('families', nargs='+')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    corpora = candidate_corpora(ROOT, args.data_root.resolve())
    if args.corpora_file:
        text = args.corpora_file.read_text(encoding='utf-8')
        wanted = set(json.loads(text)) if text.lstrip().startswith('[') else {line.strip() for line in text.splitlines() if line.strip()}
        corpora = [c for c in corpora if c['id'] in wanted]
    skip = re.compile(args.exclude) if args.exclude else None
    files = sorted({f for corpus in corpora for f in corpus['files'] if not (skip and skip.search(f))})
    chunks = [files[i:i + CHUNK] for i in range(0, len(files), CHUNK)]
    progress = args.out / 'progress.txt'
    done = {int(line.split('\t', 1)[0]) for line in progress.read_text().splitlines() if line} if progress.exists() else set()
    patterns = tuple(f.encode() for f in args.families)

    def run(index: int) -> tuple[int, list[str]]:
        return index, [f'{p}:{n}' for p in chunks[index] if (n := count(p, patterns))]

    with open(progress, 'a', encoding='utf-8') as out, ThreadPoolExecutor(args.threads) as pool:
        for future in as_completed([pool.submit(run, i) for i in range(len(chunks)) if i not in done]):
            index, lines = future.result()
            out.write(f'{index}\t' + '\x1f'.join(lines) + '\n')
            out.flush()
    entries = {m for line in progress.read_text().splitlines() for m in line.partition('\t')[2].split('\x1f') if m}
    missing = sorted(m.rpartition(':')[0] for m in entries if m.endswith(':-1'))
    matches = sorted(m for m in entries if not m.endswith(':-1'))
    (args.out / 'missing.txt').write_text(''.join(p + '\n' for p in missing), encoding='utf-8')
    (args.out / 'matches.txt').write_text('\n'.join(matches) + '\n', encoding='utf-8')
    hits = {m.rpartition(':')[0]: int(m.rpartition(':')[2]) for m in matches}
    grouped = [{'corpus': c['id'], 'admission': c['admission'], 'files': {f: hits[f] for f in c['files'] if f in hits}}
               for c in corpora if any(f in hits for f in c['files'])]
    (args.out / 'by-corpus.json').write_text(json.dumps({'families': args.families, 'files_scanned': len(files),
                                                         'unreadable_files': len(missing),
                                                         'corpora': grouped}, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'files': len(files), 'unreadable_files': len(missing), 'matching_files': len(matches), 'corpora': len(grouped)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
