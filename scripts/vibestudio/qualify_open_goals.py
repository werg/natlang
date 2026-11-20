#!/usr/bin/env python3
"""Run the exact model-free Vibestudio references in a pinned open-goal catalog."""
import argparse
import re
import signal
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path

from audit_open_goal_references import read_catalog


def qualify_group(host, workspace, baseline, cases, model, destination, test_timeout_ms):
    group = f'{workspace}@{baseline}' if baseline else workspace
    output = destination / group
    output.mkdir()
    results = output / 'results'
    instance = f'vibe-open-reference-{uuid.uuid4().hex[:12]}'
    command = ['pnpm', '--silent', 'system-test', '--instance', instance, '--workspace', workspace]
    if baseline:
        command += ['--workspace-baseline', baseline]

    def invoke(args, log, *, check=False):
        with (output / log).open('wb') as stream:
            return subprocess.run([*command, *args], cwd=host, stdout=stream, stderr=subprocess.STDOUT,
                                  check=check).returncode

    print(f'{group} running {len(cases)} references in {instance}', flush=True)
    failures = 0
    try:
        invoke(['doctor', '--approve-startup', '--model', model], 'doctor.log', check=True)
        args = ['run', *cases, '--json', '--concurrency', '1', '--model', model, '--out-dir', str(results)]
        if test_timeout_ms is not None:
            args += ['--test-timeout-ms', str(test_timeout_ms)]
        failures += bool(invoke(args, 'run.log'))
        runs = [folder for folder in results.glob('st_*') if folder.is_dir()]
        if len(runs) != 1:
            raise ValueError(f'Expected one managed run in {results}, found {len(runs)}')
        run_id = runs[0].name
        for case in cases:
            status = invoke(['trajectory', run_id, case, '--full', '--json', '--out-dir', str(results)],
                            f'{case}.log')
            archive = results / run_id / f'trajectory-{case}-full.json.gz'
            failures += bool(status) or not archive.is_file()
    except (subprocess.CalledProcessError, ValueError) as error:
        print(f'{group} {error}. Evidence is in {output}', file=sys.stderr, flush=True)
        failures += 1
    finally:
        failures += bool(invoke(['stop'], 'stop.log'))

    print(f'{group} evidence in {output} ({failures} failed commands or missing artifacts)', flush=True)
    return failures == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--family', action='append', default=[])
    parser.add_argument('--case', action='append', default=[])
    parser.add_argument('--model', default='openai-codex:gpt-6-luna')
    parser.add_argument('--test-timeout-ms', type=int)
    args = parser.parse_args()
    if args.test_timeout_ms is not None and args.test_timeout_ms <= 0:
        parser.error('--test-timeout-ms must be positive')

    source, catalog = read_catalog(args.catalog)
    chosen = {case: row for case, row in catalog.items()
              if (not args.case or case in args.case) and (not args.family or row['family'] in args.family)}
    if not chosen or set(args.case) - chosen.keys() or set(args.family) - {row['family'] for row in chosen.values()}:
        parser.error('Every requested case and family must match the pinned catalog')
    groups = defaultdict(list)
    for case, row in chosen.items():
        workspace, baseline = row['workspace'], row.get('workspaceBaseline', '')
        if not re.fullmatch(r'[a-z][a-z-]*', workspace) or (baseline and not re.fullmatch(r'[a-f0-9]{40}', baseline)):
            parser.error(f'Invalid managed workspace or baseline for {case}')
        if not re.fullmatch(r'open-goal-[a-z0-9-]+', case):
            parser.error(f'Invalid goal identifier {case}')
        groups[workspace, baseline].append('open-goal-oracle-' + case.removeprefix('open-goal-'))

    destination = args.out.resolve()
    destination.mkdir(parents=True)
    (destination / 'catalog.jsonl').write_bytes(source)
    all_passed = True
    for (workspace, baseline), cases in groups.items():
        passed = qualify_group(args.host.resolve(), workspace, baseline, cases, args.model, destination,
                               args.test_timeout_ms)
        all_passed &= passed
    return 0 if all_passed else 1


if __name__ == '__main__':
    def terminate(_signal, _frame):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(main())
