"""Tiny fixture for `run-lfm3-lane.sh --dry-run`: the whole lane on a tiny native LFM checkpoint, CPU, minutes.

    dry_run_fixture.py build --out DIR             tiny checkpoint, supply programs with persisted grouped splits
    dry_run_fixture.py tune --program P --checkpoint CK --out RESULT
                                                   stand-in for `natlang tune P --rounds` + `--evaluate development`

The stand-in runs the real `tune.build` fit (the trainer `natlang tune` calls) on fixture token rows, one per
privileged training reference of the program (so supply custody still decides which roots give rows), records a
one-round journal marked as a dry run and writes the lane's tune result through the real `lane tune-result`. Natlang
collection itself (privileged self-teaching, custody, reuse) is covered by `ts-host/test/tune-selfteach.test.mjs`;
the fixture model cannot execute natlang programs. Nothing here is a measurement: dry-run development numbers are 0/0.
It imports the repository's test fixtures (tests/neuralese), so it runs from a checkout only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'tests' / 'neuralese'))
DOMAINS = ['access-request', 'refund-decision', 'dock-assignment']


def build(out):
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.tune import supply
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'model' / 'native.pt').exists():
        (out / 'model').mkdir(exist_ok=True)
        engine = engine_fixture(out / 'model', 'lfm')
        engine.store.close()
    programs = out / 'programs'
    if not (programs / 'programs-prepared.json').exists():
        supply.write_programs(programs, domains=DOMAINS, variants=1, worlds=6, prefix='dry')
        # The real lane's splits were persisted by the ts-host tuner (`currentSplit`); here: two of four worlds train.
        for item in json.loads((programs / 'programs-prepared.json').read_text())['programs']:
            directory = programs / item['program']
            groups = sorted({json.loads(line)['group'] for line in (directory / 'tune.examples.jsonl').read_text().splitlines()})
            (directory / '.natlang' / 'tune').mkdir(parents=True, exist_ok=True)
            (directory / '.natlang' / 'tune' / 'splits.json').write_text(json.dumps({'schema': 'natlang.tune-split/v1', 'components': {
                group: ('train', 'train', 'development', 'confirmation')[index % 4] for index, group in enumerate(groups)}}))
    fixture = {'checkpoint': str(out / 'model' / 'native.pt'), 'programs': str(programs)}
    (out / 'fixture.json').write_text(json.dumps(fixture, indent=1))
    print(json.dumps({'command': 'fixture', **fixture}))


def tune(program, checkpoint, result):
    import torch
    from test_tune_trainer import manifest_for, rows_for
    from natlang_neuralese.serve import load_native_engine
    from natlang_neuralese.tune.build import main as fit
    from natlang_neuralese.tune.learned.lane import tune_result
    program = Path(program).resolve()
    references = [line for line in (program / 'tune.privileged.jsonl').read_text().splitlines() if line.strip()]
    if not references:
        raise SystemExit(f'{program}: no privileged training references (run the supply stage first)')
    engine = load_native_engine(str(checkpoint), device='cpu', dtype=torch.float32)
    try:
        manifest = {**manifest_for(engine), 'program': {'id': program.name, 'buildHash': 'dry-run'}}
        half = (len(references) + 1) // 2
        rows = rows_for(engine, 'bx', half) + rows_for(engine, 'by', len(references) - half, offset=2)
    finally:
        engine.store.close()
    key = hashlib.sha256(json.dumps([program.name, str(checkpoint), len(rows)]).encode()).hexdigest()[:24]
    job = program / '.natlang' / 'tune' / 'jobs' / f'dry-{key}'
    job.mkdir(parents=True, exist_ok=True)
    (job / 'train.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (job / 'plan.json').write_text(json.dumps({
        'schema': 'natlang.tune-plan/v1', 'manifest': manifest, 'rows': 'train.jsonl', 'softStates': False, 'exports': [],
        'student': {'name': 'dry-run-fixture', 'checkpoint': str(checkpoint)}, 'label': 'round-0-sft',
        'recipe': {'steps': 2, 'batch_rows': 4, 'prompt_weight': 0, 'ranks': {'tree': 4, 'folder': 2, 'lambda': 2}}}, indent=1))
    if fit(['--plan', str(job / 'plan.json'), '--out', str(job / 'out')]) != 0:
        raise SystemExit(f'{program}: the fixture fit failed')
    journal = program / '.natlang' / 'tune' / 'rounds' / 'dry-run' / 'journal.json'
    journal.parent.mkdir(parents=True, exist_ok=True)
    unmeasured = {'success': 0, 'roots': 0}
    journal.write_text(json.dumps({'schema': 'natlang.tune-rounds/v1', 'schedule': ['sft'], 'stopped': None, 'dryRun': True,
                                   'rounds': [{'index': 0, 'kind': 'sft', 'parent': None, 'status': 'evaluated', 'release': 'dry-run',
                                               'build': str(job / 'out'), 'development': unmeasured, 'improved': True,
                                               'reason': 'dry run: fixture fit, not measured'}],
                                   'best': {'round': 0, 'release': 'dry-run', 'build': str(job / 'out'), 'success': 0}}, indent=1))
    tune_result(program, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    one = commands.add_parser('build')
    one.add_argument('--out', required=True)
    two = commands.add_parser('tune')
    two.add_argument('--program', required=True)
    two.add_argument('--checkpoint', required=True)
    two.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.command == 'build':
        build(args.out)
    else:
        tune(args.program, args.checkpoint, args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
