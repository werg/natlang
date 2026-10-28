#!/usr/bin/env python3
"""Check rendered native documents' system turns against the CURRENT ts-host prompt pieces.

ts-host builds the system prompt from its piece registry (``ts-host/src/native/system-prompts.ts`` ``promptPieces()``);
the serving template then renders it. This check takes a sample of rendered rows (``text-data.jsonl``) and, for each
document, splits its system turn's prose (before the template's tool section) into paragraphs and looks each one up
verbatim among the paragraphs of the current pieces (``node`` imports ``ts-host/dist``):

- ``program-header`` rows: every kept header paragraph must be a verbatim current paragraph (exact parity of the part
  a model trained past the warm-up is served with); the header must also equal what
  ``prompt_rendering.program_header_paragraphs`` keeps of the same text.
- ``program-header`` rows: a header paragraph that is a current GENERIC piece is counted
  (``header_documents_with_generic_paragraphs``): ts-host drops it at inference, so it is train/inference skew.
- ``full`` rows: reports the share of documents whose whole system prose is current ts-host text, and the drifted
  paragraphs (recorded prompts of older runtime versions), most frequent first.

Rows may be .jsonl or .jsonl.zst. Usage: check_prompt_against_ts_host.py --rows FILE [--rows FILE ...] [--sample N] [--out JSON]
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.data.prompt_rendering import default_pieces, render_system_prompt  # noqa: E402

SYSTEM_OPEN, TURN_END = '<|im_start|>system\n', '<|im_end|>'
TOOL_SECTION_MARKERS = ('List of tools: ', '# Tools\n')


def ts_host_paragraphs() -> tuple[set, str]:
    # The piece registry (the text-channel emulation declaration was removed from ts-host on 2026-10-23).
    script = ("import('./dist/native/system-prompts.js')"
              ".then(m => process.stdout.write(JSON.stringify(m.promptPieces())))")
    pieces = json.loads(subprocess.check_output(['node', '-e', script], cwd=ROOT / 'ts-host'))
    paragraphs, generic = set(), set()
    for piece in pieces:
        for p in piece['text'].split('\n\n'):
            if p.strip():
                paragraphs.add(p.strip())
                paragraphs.update(q.strip() for q in p.split('\n') if q.strip())
                if piece.get('scope') == 'generic':
                    generic.add(p.strip())
    commit = subprocess.check_output(['git', 'log', '-1', '--format=%h', '--', 'ts-host/src/native'], cwd=ROOT,
                                     text=True).strip()
    return paragraphs, generic, commit


def system_prose(text: str):
    start = text.find(SYSTEM_OPEN)
    if start < 0 or text[:start] not in ('', '<|startoftext|>'):
        return None
    end = text.find(TURN_END, start)
    content = text[start + len(SYSTEM_OPEN):end]
    cut = min((content.find(m) for m in TOOL_SECTION_MARKERS if m in content), default=len(content))
    return content[:cut]


def check(path: Path, sample: int, current: set, generic: set, rng: random.Random):
    if path.suffix == '.zst':
        lines = subprocess.run(['zstd', '-dcq', str(path)], check=True, capture_output=True, text=True).stdout.splitlines()
    else:
        with open(path) as f:
            lines = f.readlines()
    rows = [json.loads(line) for line in (rng.sample(lines, sample) if len(lines) > sample else lines)]
    stats = Counter()
    drift = Counter()
    generic_kept = Counter()
    for row in rows:
        mode = row.get('prompt_rendering') or ('program-header' if row.get('system_prompt_mode') else 'full')
        prose = system_prose(row['text'])
        stats['documents'] += 1
        if prose is None:
            stats['no_system_turn'] += 1
            continue
        paragraphs = [p for p in prose.split('\n\n') if p.strip()]
        missing = [p for p in paragraphs if p.strip() not in current
                   and not all(q.strip() in current for q in p.split('\n') if q.strip())]
        if mode == 'program-header':
            stats['header_documents'] += 1
            stats['header_equals_scope_table'] += (render_system_prompt(prose, 'program-header', default_pieces())
                                                   == prose.strip())
            # A generic runtime paragraph left in a header: ts-host drops it at inference, so training sees text the
            # served model never does (a stale training/prompt-pieces/pieces.json does this).
            kept_generic = [p for p in paragraphs if p.strip() in generic]
            stats['header_documents_with_generic_paragraphs'] += bool(kept_generic)
            for p in kept_generic:
                generic_kept[p[:160]] += 1
        stats['all_paragraphs_current'] += not missing
        stats['paragraphs'] += len(paragraphs)
        stats['paragraphs_not_current'] += len(missing)
        for p in missing:
            drift[p[:160]] += 1
    return {'rows': str(path), **stats, 'top_drift': drift.most_common(8),
            'generic_in_header': generic_kept.most_common(8)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rows', type=Path, action='append', required=True)
    p.add_argument('--sample', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path)
    a = p.parse_args(argv)
    current, generic, commit = ts_host_paragraphs()
    rng = random.Random(a.seed)
    report = {'schema': 'natlang.prompt-ts-host-parity/1', 'ts_host_native_commit': commit,
              'current_paragraphs': len(current), 'sample': a.sample,
              'results': [check(path, a.sample, current, generic, rng) for path in a.rows]}
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if a.out:
        a.out.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
