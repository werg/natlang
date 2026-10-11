#!/usr/bin/env python3
"""Daily Clef-flash labelling of the new source pools (DGX share of the shared Workers AI free allocation).

Plumbing around the existing Clef pipeline (ts-host/scripts/inline-curriculum/clef-programs.mjs, clef-label.mjs,
replay-demonstrations.mjs); the programs, their questions and the confidence gating live there. One batch per UTC day:

  run [--day YYYYMMDD] [--budget N] [--count N]
      1. skeletons for the day's seed over NEW_POOLS (CLINC150 skill routing, emotion wellbeing triage, CoEdIT edit
         review, CUAD clause risk); train split only, by the shared content partition (source-split.mjs);
      2. Clef-flash answers up to the day's budget minus what this batch already spent (resumable: answered items are
         skipped, so an interrupted run continues);
      3. assemble the confidence-gated cases, replay them into native teacher-training turns (no model calls);
      4. write summary.json and registry-entry.json (the batch's registry entry, ready to merge);
      5. queue the shared render (render_native_turn_corpus.sh, both tokenizers) as a low-priority ledger unit.
      6. register the batch and every finished render of earlier batches (register_corpora.py: entries and manifests
         committed and pushed from a private worktree off origin/main; no agent commits by hand). A render finishes
         after the run, so its registration happens on the next timer run (06:10 or the next day's), or with
  register [--day YYYYMMDD]
      register that day's batch and every finished, unregistered render now.

Budget: the free allocation (~1.22M Clef-flash input tokens/day, reset 00:00 UTC) is shared with Pop; the DGX share is
800k/day (plans/neuralese/CLEF_DECISION_DISTILLATION.md, memory clef-quota-shared). Credentials: DGX's own wrangler
login (clef-label refreshes it); never Pop's.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register_corpora import register as register_entries, registered_ids  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TS = ROOT / 'ts-host'
CURRICULUM = TS / 'scripts/inline-curriculum'
CORPORA = ROOT / 'data/neuralese/corpora'
EPOCH = dt.date(2026, 10, 11)
SEED_BASE = 8100  # seeds 8100+day; 71/72 (clef-new), 6145 (dataset expansion), 102/900 (evaluation) stay unused
DGX_SHARE = 800_000
# systemd units see /usr/bin/node (v18), which lacks the runtime's crypto.randomUUID path: use the interactive node.
NODE = os.environ.get('NATLANG_NODE') or shutil.which('node', path=f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}") or 'node'


VERSION = os.environ.get('CLEF_POOLS_VERSION', 'v1')  # a rebuilt batch (fixed template, same answers) is a new id


def corpus_id(day: str) -> str:
    return f'clef-pools-{day}-{VERSION}'


def batch_dir(day: str) -> Path:
    return CORPORA / corpus_id(day)


def seed_for(day: str) -> int:
    return SEED_BASE + (dt.datetime.strptime(day, '%Y%m%d').date() - EPOCH).days


def node(*args: str) -> str:
    # One retry: a concurrent ts-host dist rebuild by another agent can fail a run transiently.
    for attempt in range(2):
        out = subprocess.run([NODE, *args], cwd=TS, capture_output=True, text=True)
        sys.stderr.write(out.stderr)
        if out.returncode == 0:
            return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ''
        sys.stderr.write(f'node {args[0]} exited {out.returncode} (attempt {attempt + 1})\n')
    raise SystemExit(f'node {args[0]} failed twice')


def spent(answers: Path) -> int:
    if not answers.exists():
        return 0
    return sum(json.loads(line).get('usage', {}).get('input_tokens', 0) for line in answers.open() if line.strip())


def count_lines(path: Path) -> int:
    return sum(1 for _ in path.open()) if path.exists() else 0


def registered(day: str) -> bool:
    return corpus_id(day) in registered_ids()


def run(day: str, budget: int, count: int) -> None:
    if registered(day):  # a published manifest is immutable: never rewrite a registered batch
        print(json.dumps({'day': day, 'skipped': 'already registered'}))
        register(None)  # renders of earlier batches that finished since
        return
    out = batch_dir(day)
    out.mkdir(parents=True, exist_ok=True)
    seed = seed_for(day)
    sk, ans = out / f'clef-pools-{day}.skeletons.jsonl', out / f'clef-pools-{day}.answers.jsonl'
    ir, rows, turns = (out / f'clef-pools-{day}.{kind}.jsonl' for kind in ('ir', 'rows', 'turns'))
    if not sk.exists():
        node(str(CURRICULUM / 'clef-programs.mjs'), 'skeletons', str(seed), str(count), str(sk), 'new')
    remaining = budget - spent(ans)
    label = None
    if remaining > 2_000:
        label = json.loads(node(str(CURRICULUM / 'clef-label.mjs'), str(sk), str(ans), '--budget', str(remaining),
                                '--concurrency', '4') or '{}')
    assembled = json.loads(node(str(CURRICULUM / 'clef-programs.mjs'), 'assemble', str(sk), str(ans), str(ir)))
    replay = json.loads(node(str(CURRICULUM / 'replay-demonstrations.mjs'), str(ir), str(rows), str(turns)))
    families: dict[str, int] = {}
    for line in ir.open():
        family = json.loads(line)['curriculum']['family']
        families[family] = families.get(family, 0) + 1
    summary = {'day': day, 'seed': seed, 'skeletons': count_lines(sk), 'answered_items': count_lines(ans),
               'input_tokens': spent(ans), 'budget': budget, 'last_label_run': label, 'assembled': assembled,
               'replay': replay, 'families': families,
               'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    entry = {
        'id': corpus_id(day), 'owner': 'dgx', 'path': f'data/neuralese/corpora/{corpus_id(day)}',
        'include': ['*.jsonl', 'summary.json'],
        'admission': 'reference-demonstrations-verified-replay; clef-flash-distilled child labels with per-item confidence '
                     'gating; action-notes reasoning (not trained); not teacher-sampled',
        'training_admission': True,
        'training_admission_reason': "owner 2026-10-11 maximize data: admit unless a concrete defect (broken format, leak "
                                     "into held/eval, protected eval, verified wrong labels); 'pending review' is recorded "
                                     "as review status, not a reason to exclude; splits and source-disjointness kept "
                                     "(plans/neuralese/DATA_INVENTORY_20261011.md)",
        'review_status': f"verified replay {replay.get('admitted')}/{replay.get('cases')} cases; Clef-flash labels unreviewed",
        'required_conversion': 'render (render_native_turn_corpus.sh, both tokenizers); root rationales in a teacher window',
        'split': 'train only: items drawn from the train side of the shared content partition (source-split.mjs, '
                 'training/source-split-reservations.json); generated case ids carry the seed',
        'source_class': 'public-dataset-items-with-clef-flash-decisions',
        'license': 'per source: CLINC150 CC-BY-3.0, dair-ai/emotion (dataset card), CoEdIT Apache-2.0, CUAD v1 CC-BY-4.0; '
                   'Clef-flash outputs (Cloudflare Workers AI terms). Provenance facts only (owner rule 2026-10-10)',
        'build': {'driver': 'scripts/neuralese_data/clef_pools_daily.py', 'seed': seed,
                  'templates': 'clef-programs.mjs NEW_POOLS (skill_routing, wellbeing_triage, edit_review, clause_risk)',
                  'model': '@cf/cloudflare/clef-flash', 'input_tokens': summary['input_tokens']},
        'note': f"Clef-flash-distilled inline-lambda programs over new source pools, day {day}: {replay.get('admitted')} "
                f"cases, {replay.get('decisions')} decisions ({families}). Files: *.skeletons (Clef requests), *.answers "
                "(Clef outputs with usage), *.ir (cases), *.rows (trajectories), *.turns (native training turns).",
        'consumable_by_direct_stage': False,
    }
    (out / 'registry-entry.json').write_text(json.dumps(entry, indent=1) + '\n')
    print(json.dumps(summary))
    render(day)
    register(day)


HELD_TEXT = ['runs/neuralese-foundation-text-input-4962-root-adopted-union-20261009-v1/text-data.jsonl',
             'data/neuralese/corpora/harness-bench-swe-rebench-openhands-pi-text-lfm25-350m-20261010-v2/text.jsonl']


def render(day: str) -> None:
    """Queue the shared three-rendering twin build (both tokenizers) as a low-priority resumable ledger unit."""
    out = render_dir(corpus_id(day))
    if (out / 'receipt.json').exists():
        return
    turns = batch_dir(day) / f'clef-pools-{day}.turns.jsonl'
    subprocess.run([sys.executable, str(ROOT / 'scripts/memory_ledger.py'), 'run', '--unit', f'render-turns-clef-pools-{day}',
                    '--budget-gb', '5', '--priority', 'low', '--resumable', '--wait', '43200', '--',
                    'env', 'MEM=5g', 'CPUS=2', 'scripts/neuralese_data/render_native_turn_corpus.sh', str(out),
                    '--corpus-id', corpus_id(day), '--source', str(turns.relative_to(ROOT)), '--workers', '2',
                    '--shards', '8', '--held-text', *HELD_TEXT], cwd=ROOT, check=False)


def render_dir(cid: str) -> Path:
    return ROOT / f'runs/natlang-native-turns-{cid}-render'


def register(day: str | None = None) -> None:
    """Register DAY's batch (if built) and every finished render whose batch is registered, through the worktree
    helper. Idempotent: registered ids are skipped (a published manifest is immutable)."""
    known = registered_ids()
    if day and (batch_dir(day) / 'registry-entry.json').exists() and corpus_id(day) not in known:
        entry = json.loads((batch_dir(day) / 'registry-entry.json').read_text())
        register_entries([entry], f"Register Clef pools batch {entry['id']}")
        known.add(entry['id'])
    sys.path.insert(0, str(ROOT / 'scripts/neuralese_data'))
    import register_native_turn_render as rntr
    batches = sorted(json.loads(f.read_text())['id'] for f in CORPORA.glob('clef-pools-*/registry-entry.json'))
    for cid in batches:
        if any(other.rsplit('-', 1)[0] == cid.rsplit('-', 1)[0] and other > cid for other in batches):
            continue  # superseded by a rebuilt version of the same day
        out, rid = render_dir(cid), f'natlang-native-turns-{cid}-render'
        if cid not in known or rid in known or not (out / 'receipt.json').exists():
            continue
        a = rntr.parse(['--id', rid, '--render-dir', str(out), '--source-corpus', cid])
        register_entries([rntr.make_entry(a)], f'Register {rid}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['run', 'register'])
    parser.add_argument('--day', default=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d'))
    parser.add_argument('--budget', type=int, default=DGX_SHARE)
    parser.add_argument('--count', type=int, default=700, help='skeletons per day (more than the budget answers)')
    args = parser.parse_args()
    run(args.day, args.budget, args.count) if args.action == 'run' else register(args.day)


if __name__ == '__main__':
    main()
