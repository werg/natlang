#!/usr/bin/env python3
"""Continuous static inline-curriculum generation: fresh seeds -> verified cases -> replayed native turns -> renders.

Owner goal 2026-10-11: bring in as much data as we can. The inline-curriculum families
(ts-host/scripts/inline-curriculum/families.mjs) build verified cases from exact oracles with zero model calls; this
driver runs them on fresh seeds at scale and feeds the shared renderer. It is plumbing (job scheduling, files,
dedup, resumability); every decision about the data itself stays in the existing builders:

  build.mjs                 (seed, family, index range) -> verified natlang.program/2 cases (--no-hints, split train;
                            source families keep their upstream split; failures dropped and counted)
  replay-demonstrations.mjs cases -> collector rows + natlang.teacher_training_turn.native/1 decisions (stock action
                            notes as untrained reasoning: teacher_reasoning_trained false; rationales come later in
                            teacher windows through rationales.mjs)
  render_native_turn_corpus.py (shared renderer) -> full / program-header / whole-trajectory for LFM2.5 and Mellum2.1

Layout under ROOT (NVMe; turns are ~180 KB per decision uncompressed, so everything is stored zstd --long=27, ~250x):
  config.json                          fixed per campaign: seeds, shapes, families, exclusions with reasons
  shards/sSEED/FAMILY.{ir,rows,turns}.jsonl.zst + FAMILY.done.json
  batches/bNN/{cases.ir,rows,turns}.jsonl.zst + batch.json      deduplicated by case id (existing corpora and earlier
                                                                batches), registered as one immutable corpus each
  renders/bNN/                                                  shared-renderer output (both tokenizers)

Source families sample their dataset by index, so every seed gets its own index range (start = offset * STRIDE);
generated families are new problems per seed. Resumable: finished shards, batches and renders are skipped.

  static_curriculum_campaign.py init  --root ROOT [--seed-start 7001] [--shapes 4] [--seeds-per-batch 10]
  static_curriculum_campaign.py run   --root ROOT [--workers 8] [--batches N]        generate + consolidate
  static_curriculum_campaign.py render-loop --root ROOT [--render-cpus 6]            render batches as they land
  static_curriculum_campaign.py rationalize-loop --root ROOT                         teacher rationales in GPU gaps -> rNN
  static_curriculum_campaign.py status --root ROOT
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TS = REPO / 'ts-host'
# The ts-host scripts need Node >= 22 (util.parseArgs allowNegative); user units see only the system Node 18.
NODE = os.environ.get('NATLANG_NODE') or next((str(p) for p in (Path.home() / '.local/bin/node',) if p.exists()), 'node')
STRIDE = 4  # largest family weight: a family indexes round(shapes * weight) items from start
# Families that cannot run here today, with the reason (recorded in config.json and the registry note).
EXCLUDED = {
    'commaqa_question': 'CommaQA not in the dataset cache (acquire.mjs --source commaqa)',
    'alfworld_task': 'vendor/alfworld-venv missing on DGX',
    'scienceworld_task': 'ScienceWorld runtime missing on DGX (all references fail to spawn)',
    'anli_batch': 'source_review_pending (current source-conflict rule holds these sources)',
    'folio_batch': 'source_review_pending (current source-conflict rule holds these sources)',
    'folio_entailment': 'source_review_pending (current source-conflict rule holds these sources)',
    'textworld_iterate': 'reference actions diverge on some games (iteration diverged); reported as a bug candidate',
}
# Case-id sources of corpora this campaign must not repeat.
EXISTING_IR = [
    'data/neuralese/corpora/static-lambda-rationalized-20261006-v2/lambda-v3-s61.ir.jsonl',
    'data/neuralese/corpora/static-lambda-rationalized-20261006-v2/extras-s61.ir.jsonl',
    'runs/bgkit-knowledge-rationalized-20261007-v1/bgkit-knowledge.ir.jsonl',
    'data/neuralese/corpora/clef-new-rationalized-20261006-v2/clef-new-20261006.ir.jsonl',
]
HELD_TEXT = [
    'runs/neuralese-foundation-text-input-4962-root-adopted-union-20261009-v1/text-data.jsonl',
    'data/neuralese/corpora/harness-bench-swe-rebench-openhands-pi-text-lfm25-350m-20261010-v2/text.jsonl',
]
PROTECTED = 'data/neuralese/corpora/protected-bgkit-20261003/bgkit-benchmarks.protected.json'


def load(path: Path):
    return json.loads(path.read_text())


def save(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
    tmp.replace(path)


def families() -> list[str]:
    out = subprocess.run([NODE, '-e', "import('./scripts/inline-curriculum/families.mjs').then(m => console.log("
                          "JSON.stringify(Object.entries(m.FAMILIES).filter(([k, v]) => (v.track ?? 'interpreter')"
                          " !== 'authoring').map(([k]) => k))))"], cwd=TS, check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def init(args) -> None:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'config.json').exists():
        sys.exit(f'{root}/config.json exists; a campaign config is fixed')
    names = [name for name in families() if name not in EXCLUDED]
    save(root / 'config.json', {'schema': 'natlang.static-curriculum-campaign/1', 'seed_start': args.seed_start,
                                'shapes': args.shapes, 'seeds_per_batch': args.seeds_per_batch, 'stride': STRIDE,
                                'families': names, 'excluded': EXCLUDED, 'split': 'train', 'hints': False,
                                'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print(f'{len(names)} families, {len(EXCLUDED)} excluded')


def sh(cmd: list[str], log: Path, cwd=TS) -> None:
    cmd = [NODE if cmd[0] == 'node' else cmd[0]] + cmd[1:]
    with open(log, 'ab') as f:
        subprocess.run(['nice', '-n', '15'] + cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=True)


def shard(root: Path, cfg: dict, seed: int, family: str) -> dict:
    out = root / 'shards' / f's{seed}'
    done = out / f'{family}.done.json'
    if done.exists():
        return load(done)
    out.mkdir(parents=True, exist_ok=True)
    work = out / f'.{family}.work'
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    log = work / 'log.txt'
    started = time.time()
    start = (seed - cfg['seed_start']) * cfg['shapes'] * cfg['stride']
    ir, rows, turns = work / 'ir.jsonl', work / 'rows.jsonl', work / 'turns.jsonl'
    stats = {'seed': seed, 'family': family, 'start': start}
    try:
        sh(['node', 'scripts/inline-curriculum/build.mjs', '--seed', str(seed), '--shapes', str(cfg['shapes']),
            '--start', str(start), '--no-hints', '--allow-failures', '--split', 'train', '--families', family,
            '--out', str(ir)], log)
    except subprocess.CalledProcessError:
        stats['error'] = log.read_text()[-2000:]
    report = load(Path(f'{ir}.report.json')) if Path(f'{ir}.report.json').exists() else {}
    stats.update(cases=report.get('cases', 0), verified=report.get('verified', 0),
                 failures=[f['problems'][0][:300] for f in report.get('failures', [])][:5])
    if ir.exists() and ir.stat().st_size > 1:
        sh(['node', 'scripts/inline-curriculum/replay-demonstrations.mjs', str(ir), str(rows), str(turns)], log)
        replay = json.loads(log.read_text().strip().splitlines()[-1])
        stats.update(admitted=replay['admitted'], decisions=replay['decisions'], skipped=replay['skipped'])
        for path, name in ((ir, 'ir'), (rows, 'rows'), (turns, 'turns')):
            subprocess.run(['zstd', '-q', '-3', '--long=27', '--rm', '-f', str(path), '-o', str(out / f'{family}.{name}.jsonl.zst')],
                           check=True)
    else:
        stats.update(admitted=0, decisions=0)
    stats['seconds'] = round(time.time() - started, 1)
    save(done, stats)
    shutil.rmtree(work, ignore_errors=True)
    return stats


def zlines(path: Path):
    proc = subprocess.Popen(['zstd', '-dcq', '--long=27', str(path)], stdout=subprocess.PIPE)
    yield from proc.stdout
    if proc.wait():
        raise RuntimeError(f'zstd failed on {path}')


class ZWriter:
    def __init__(self, path: Path):
        self.f = open(path, 'wb')
        self.p = subprocess.Popen(['zstd', '-q', '-3', '--long=27', '-T2', '-c'], stdin=subprocess.PIPE, stdout=self.f)

    def write(self, line: bytes) -> None:
        self.p.stdin.write(line)

    def close(self) -> None:
        self.p.stdin.close()
        if self.p.wait():
            raise RuntimeError('zstd failed')
        self.f.close()


def seen_ids(root: Path, before: int) -> set[str]:
    ids: set[str] = set()
    for rel in EXISTING_IR:
        path = REPO / rel
        if path.exists():
            with open(path, 'rb') as f:
                for line in f:
                    ids.add(json.loads(line)['id'])
    for b in range(before):
        idx = root / 'batches' / f'b{b:02d}' / 'case-ids.txt'
        ids.update(idx.read_text().split())
    return ids


def consolidate(root: Path, cfg: dict, b: int) -> dict:
    out = root / 'batches' / f'b{b:02d}'
    if (out / 'batch.json').exists():
        return load(out / 'batch.json')
    tmp = root / 'batches' / f'.b{b:02d}.building'
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    seen = seen_ids(root, b)
    seeds = range(cfg['seed_start'] + b * cfg['seeds_per_batch'], cfg['seed_start'] + (b + 1) * cfg['seeds_per_batch'])
    writers = {name: ZWriter(tmp / f'{name}.jsonl.zst') for name in ('cases.ir', 'rows', 'turns')}
    stats = {'batch': b, 'seeds': [seeds.start, seeds.stop - 1], 'cases': 0, 'duplicate_cases': 0, 'rows': 0,
             'decisions': 0, 'families': {}, 'build_failures': 0}
    kept: set[str] = set()
    for seed in seeds:
        for family in cfg['families']:
            done = load(root / 'shards' / f's{seed}' / f'{family}.done.json')
            stats['build_failures'] += done.get('cases', 0) - done.get('verified', 0)
            base = root / 'shards' / f's{seed}' / family
            if not Path(f'{base}.ir.jsonl.zst').exists():
                continue
            fam = stats['families'].setdefault(family, {'cases': 0, 'decisions': 0})
            for line in zlines(Path(f'{base}.ir.jsonl.zst')):
                cid = json.loads(line)['id']
                if cid in seen:
                    stats['duplicate_cases'] += 1
                    continue
                seen.add(cid)
                kept.add(cid)
                writers['cases.ir'].write(line)
                stats['cases'] += 1
                fam['cases'] += 1
            for line in zlines(Path(f'{base}.rows.jsonl.zst')):
                if json.loads(line)['task']['program_ir']['id'] in kept:
                    writers['rows'].write(line)
                    stats['rows'] += 1
            for line in zlines(Path(f'{base}.turns.jsonl.zst')):
                if json.loads(line)['source_ref']['program_ir_id'] in kept:
                    writers['turns'].write(line)
                    stats['decisions'] += 1
                    fam['decisions'] += 1
    for w in writers.values():
        w.close()
    (tmp / 'case-ids.txt').write_text('\n'.join(sorted(kept)) + '\n')
    save(tmp / 'batch.json', stats)
    tmp.rename(out)
    return stats


def render(root: Path, name: str, cpus: int, budget_gb: int, docker_gb: int) -> None:
    """Render batches/NAME (bNN: action-note reasoning; rNN: the same cases with teacher rationales) to renders/NAME."""
    out = root / 'renders' / name
    if (out / 'receipt.json').exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    # A render unit left running by an interrupted loop (preempted, restarted) is waited for, not duplicated.
    listed = subprocess.run(['systemctl', '--user', 'list-units', '--state=active,activating', '--plain', '--no-legend',
                             f'static-render-{name}-*'], capture_output=True, text=True).stdout.split()
    for running in [u for u in listed if u.startswith(f'static-render-{name}-')]:
        while subprocess.run(['systemctl', '--user', 'is-active', '-q', running]).returncode == 0:
            time.sleep(30)
    if (out / 'receipt.json').exists():
        return
    plain = root / 'batches' / f'.{name}.turns.jsonl'
    subprocess.run(['zstd', '-dq', '--long=27', '-f', str(root / 'batches' / name / 'turns.jsonl.zst'), '-o', str(plain)],
                   check=True)
    unit = f'static-render-{name}-{int(time.time()) % 100000}'
    corpus_id = f"{root.name}-{name}"
    cmd = ['python3', str(REPO / 'scripts/memory_ledger.py'), 'run', '--unit', unit, '--budget-gb', str(budget_gb),
           '--priority', 'low', '--no-resumable',
           '--wait', '86400', '--workdir', str(REPO), '--', 'env', f'MEM={docker_gb}g', f'CPUS={cpus}',
           'nice', '-n', '10', 'scripts/neuralese_data/render_native_turn_corpus.sh', str(out), '--corpus-id', corpus_id,
           '--source', str(plain), '--workers', str(cpus), '--shards', '64', '--protected', str(REPO / PROTECTED),
           '--reserved', str(REPO / 'training/source-split-reservations.json'),
           '--held-text', *[str(REPO / p) for p in HELD_TEXT]]
    subprocess.run(cmd, check=True)
    # The ledger unit runs asynchronously: wait for it to finish.
    while subprocess.run(['systemctl', '--user', 'is-active', '-q', f'{unit}.service']).returncode == 0:
        time.sleep(30)
    plain.unlink(missing_ok=True)
    if not (out / 'receipt.json').exists():
        raise RuntimeError(f'render of {name} failed; see journalctl --user -u {unit}')
    # Disk: the full rendering (warm-up share, generic system prompt) is ~70% of the bytes and the warm-up already has
    # the union; keep it, compressed (~50x), so it is restorable without a re-render. Trainers read the other two.
    for path in sorted((out / 'full').glob('*/*.jsonl')):
        subprocess.run(['zstd', '-q', '-3', '--long=27', '-T4', '--rm', '-f', str(path)], check=True)


def teacher_serving() -> bool:
    return subprocess.run(['curl', '-sf', '-m', '5', 'http://127.0.0.1:8082/v1/models'], capture_output=True).returncode == 0


def ensure_teacher_queued() -> None:
    """Queue the local teacher for a memory gap: low priority and resumable, so training (high) preempts it and the
    ledger relaunches it when memory frees again. Never service class here: that would hold memory training needs."""
    active = subprocess.run(['systemctl', '--user', 'is-active', '-q', 'natlang-qwen36-teacher.service']).returncode == 0
    if active or teacher_serving():
        return
    subprocess.run(['systemctl', '--user', 'reset-failed', 'natlang-qwen36-teacher.service'], capture_output=True)
    # --wait blocks until admitted; run it detached so this loop keeps preparing prompts meanwhile.
    subprocess.Popen(['python3', str(REPO / 'scripts/memory_ledger.py'), 'run', '--unit', 'natlang-qwen36-teacher',
                      '--budget-gb', '68', '--priority', 'low', '--resumable', '--wait', '604800', '--workdir', str(REPO),
                      '--', 'docker', 'start', '-a', 'natlang-qwen36-nvfp4-server'], start_new_session=True,
                     stdout=open(root_log(), 'ab'), stderr=subprocess.STDOUT)


def root_log() -> Path:
    path = Path.home() / '.local/state/natlang/static-curriculum-teacher-queue.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, 'rb') as f:
        out = set()
        for line in f:
            try:  # a write cut by a preemption leaves at most one partial last line
                out.add(json.loads(line)['key'])
            except ValueError:
                pass
        return out


def rationalize(root: Path, b: int, concurrency: int) -> bool:
    """Root-turn rationales from the local teacher for batch bNN (rationales.mjs; owner policy: root turns only),
    then the replay with rationales as batch rNN. Returns False while the teacher is not serving."""
    name, rname = f'b{b:02d}', f'r{b:02d}'
    if (root / 'batches' / rname / 'batch.json').exists():
        return True
    work = root / 'rationales' / name
    work.mkdir(parents=True, exist_ok=True)
    cases, prompts, rationales = work / 'cases.ir.jsonl', work / 'prompts.jsonl', work / 'rationales.jsonl'
    if not cases.exists():
        subprocess.run(['zstd', '-dq', '--long=27', '-f', str(root / 'batches' / name / 'cases.ir.jsonl.zst'), '-o',
                        str(cases)], check=True)
    if not (work / 'prompts.done').exists():
        sh(['node', 'scripts/inline-curriculum/rationales.mjs', 'collect', str(cases), str(prompts)], work / 'log.txt')
        (work / 'prompts.done').write_text('')
    wanted = keys(prompts)
    while not wanted <= keys(rationales):
        if not teacher_serving():
            ensure_teacher_queued()
            return False
        sh(['node', 'scripts/inline-curriculum/rationales.mjs', 'generate', str(prompts), str(rationales),
            '--concurrency', str(concurrency)], work / 'log.txt')
    tmp = root / 'batches' / f'.{rname}.building'
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    rows, turns = tmp / 'rows.jsonl', tmp / 'turns.jsonl'
    sh(['node', 'scripts/inline-curriculum/replay-demonstrations.mjs', str(cases), str(rows), str(turns),
        '--rationales', str(rationales)], work / 'log.txt')
    replay = json.loads((work / 'log.txt').read_text().strip().splitlines()[-1])
    for path in (rows, turns):
        subprocess.run(['zstd', '-q', '-3', '--long=27', '-T4', '--rm', '-f', str(path)], check=True)
    shutil.copy(root / 'batches' / name / 'cases.ir.jsonl.zst', tmp / 'cases.ir.jsonl.zst')
    shutil.copy(root / 'batches' / name / 'case-ids.txt', tmp / 'case-ids.txt')
    save(tmp / 'batch.json', {'batch': rname, 'rationalizes': name, 'replay': replay,
                              'rationales': len(keys(rationales)), 'synthetic_reasoning': 'rationalized-actions/1'})
    tmp.rename(root / 'batches' / rname)
    cases.unlink()
    return True


def rationalize_loop(args) -> None:
    """Teacher rationales for every consolidated batch, in GPU gaps (the teacher is queued at low priority)."""
    root = Path(args.root)
    b = 0
    while args.batches is None or b < args.batches:
        if not (root / 'batches' / f'b{b:02d}' / 'batch.json').exists() or not rationalize(root, b, args.concurrency):
            time.sleep(300)
            continue
        print(json.dumps({'batch': b, 'rationalized': True}), flush=True)
        b += 1


def run(args) -> None:
    root = Path(args.root)
    cfg = load(root / 'config.json')
    b = 0
    while args.batches is None or b < args.batches:
        seeds = range(cfg['seed_start'] + b * cfg['seeds_per_batch'], cfg['seed_start'] + (b + 1) * cfg['seeds_per_batch'])
        jobs = [(seed, family) for seed in seeds for family in cfg['families']]
        started, decisions = time.time(), 0
        errors = []
        with cf.ThreadPoolExecutor(args.workers) as pool:
            for stats in pool.map(lambda job: shard(root, cfg, *job), jobs):
                decisions += stats.get('decisions', 0)
                if 'error' in stats:
                    errors.append(f"s{stats['seed']}/{stats['family']}")
        # A systematic failure (toolchain, missing cache) must not consolidate an empty or skewed batch.
        if len(errors) > len(jobs) // 10:
            sys.exit(f'{len(errors)} of {len(jobs)} shards errored (e.g. {errors[:3]}); fix and delete their done files')
        stats = consolidate(root, cfg, b)
        print(json.dumps({'batch': b, 'cases': stats['cases'], 'decisions': stats['decisions'],
                          'duplicates': stats['duplicate_cases'], 'generate_seconds': round(time.time() - started)}),
              flush=True)
        free = shutil.disk_usage(root).free / 2**30
        if free < args.min_free_gb:
            sys.exit(f'stopping: {free:.0f} GiB free < {args.min_free_gb}')
        b += 1


def render_loop(args) -> None:
    """Render every batch (bNN as it is consolidated, rNN when its rationales land); runs beside `run`."""
    root = Path(args.root)
    while True:
        names = sorted(p.parent.name for p in (root / 'batches').glob('[br]*/batch.json'))
        todo = [n for n in names if not (root / 'renders' / n / 'receipt.json').exists()]
        if not todo:
            time.sleep(120)
            continue
        name = todo[0]
        for attempt in range(4):
            try:
                render(root, name, args.render_cpus, args.render_budget_gb, args.render_docker_gb)
                break
            except (RuntimeError, subprocess.CalledProcessError) as error:  # stopped for a higher priority, say
                print(json.dumps({'batch': name, 'render_attempt': attempt, 'error': str(error)}), flush=True)
                if attempt == 3:
                    raise
                time.sleep(600)
        print(json.dumps({'batch': name, 'rendered': True}), flush=True)


def status(args) -> None:
    root = Path(args.root)
    out = {'batches': [], 'renders': []}
    for p in sorted((root / 'batches').glob('[br]*/batch.json')):
        s = load(p)
        out['batches'].append({k: s.get(k) for k in ('batch', 'seeds', 'cases', 'decisions', 'duplicate_cases', 'replay')})
    out['renders'] = sorted(p.parent.name for p in (root / 'renders').glob('*/receipt.json'))
    out['shards_done'] = len(list((root / 'shards').glob('s*/*.done.json')))
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('init')
    p.add_argument('--root', required=True)
    p.add_argument('--seed-start', type=int, default=7001)
    p.add_argument('--shapes', type=int, default=4)
    p.add_argument('--seeds-per-batch', type=int, default=10)
    p = sub.add_parser('run')
    p.add_argument('--root', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--batches', type=int)
    p.add_argument('--min-free-gb', type=int, default=60)
    p = sub.add_parser('rationalize-loop')
    p.add_argument('--root', required=True)
    p.add_argument('--batches', type=int)
    p.add_argument('--concurrency', type=int, default=48)
    p = sub.add_parser('render-loop')
    p.add_argument('--root', required=True)
    p.add_argument('--render-cpus', type=int, default=6)
    # Ledger claim ~1.3x the measured peak of the render container (~0.7-2 GB); docker's own cap is separate.
    p.add_argument('--render-budget-gb', type=int, default=3)
    p.add_argument('--render-docker-gb', type=int, default=8)
    p = sub.add_parser('status')
    p.add_argument('--root', required=True)
    args = ap.parse_args()
    {'init': init, 'run': run, 'render-loop': render_loop, 'rationalize-loop': rationalize_loop, 'status': status}[args.cmd](args)


if __name__ == '__main__':
    main()
