#!/usr/bin/env python3
"""Run the exact model-free Vibestudio references in a pinned open-goal catalog."""
import argparse
import fcntl
import json
import re
import signal
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path

from audit_open_goal_references import audit, managed_prerequisites, read_catalog


def terminal_startup_failure(log):
    for line in reversed(log.read_text(errors='replace').splitlines()):
        try:
            result = json.loads(line)
        except ValueError:
            continue
        if isinstance(result, dict) and result.get('error'):
            return result.get('recoverable') is False
    return False


def qualify_group(host, workspace, baseline, cases, model, destination, test_timeout_ms, batch=None, devices=(),
                  retain_trajectories=False):
    group = f'{workspace}@{baseline}' if baseline else workspace
    if devices:
        group += '@' + '+'.join(devices)
    if batch is not None:
        group = f'{group}-{batch:03d}'
    output = destination / group
    output.mkdir()
    results = output / 'results'
    instance = f'vibe-open-reference-{uuid.uuid4().hex[:12]}'
    command = ['pnpm', '--silent', 'system-test', '--instance', instance, '--workspace', workspace]
    if baseline:
        command += ['--workspace-baseline', baseline]
    for device in devices:
        command += ['--device', device]

    def invoke(args, log, *, check=False):
        with (output / log).open('wb') as stream:
            return subprocess.run([*command, *args], cwd=host, stdout=stream, stderr=subprocess.STDOUT,
                                  check=check).returncode

    print(f'{group} running {len(cases)} references in {instance}', flush=True)
    failures = 0
    unavailable = False
    try:
        try:
            invoke(['doctor', '--approve-startup', '--model', model], 'doctor.log', check=True)
        except subprocess.CalledProcessError:
            unavailable = terminal_startup_failure(output / 'doctor.log')
            raise
        args = ['run', *cases, '--json', '--concurrency', '1', '--model', model, '--out-dir', str(results)]
        if test_timeout_ms is not None:
            args += ['--test-timeout-ms', str(test_timeout_ms)]
        if retain_trajectories:
            args.append('--retain-trajectories')
        failures += bool(invoke(args, 'run.log'))
        runs = [folder for folder in results.glob('st_*') if folder.is_dir()]
        if len(runs) != 1:
            raise ValueError(f'Expected one managed run in {results}, found {len(runs)}')
        run_id = runs[0].name
        for case in cases:
            artifact_name = f'trajectory-{case}-full.json.gz'
            archive = results / run_id / artifact_name
            if archive.is_file() or (archive.parent / 'failure-evidence' / artifact_name).is_file():
                continue
            status = invoke(['trajectory', run_id, case, '--full', '--json', '--out-dir', str(results)],
                            f'{case}.log')
            failures += bool(status) or not archive.is_file()
    except (subprocess.CalledProcessError, ValueError) as error:
        print(f'{group} {error}. Evidence is in {output}', file=sys.stderr, flush=True)
        failures += 1
    finally:
        failures += bool(invoke(['stop'], 'stop.log'))

    print(f'{group} evidence in {output} ({failures} failed commands or missing artifacts)', flush=True)
    return failures == 0, unavailable


def run_batches(args, chosen, destination):
    roots = [*args.reuse_references, *([destination] if args.resume else [])]
    if roots:
        previous = audit(args.catalog, roots, set(args.family))['proofs']
        pending = {case: row for case, row in chosen.items() if case not in previous}
        print(f'Reusing {len(chosen) - len(pending)} verified references, {len(pending)} goals pending', flush=True)
        chosen = pending
    if not chosen:
        return 0

    groups = defaultdict(list)
    for case, row in chosen.items():
        devices = tuple(sorted(set(row.get('devices', []))))
        groups[row['workspace'], row.get('workspaceBaseline', ''), devices].append(
            'open-goal-oracle-' + case.removeprefix('open-goal-'))
    if args.resume:
        destination = destination / f'attempt-{uuid.uuid4().hex[:12]}'
        destination.mkdir()
    all_passed = True
    host = args.host.resolve()
    for (workspace, baseline, devices), cases in groups.items():
        size = args.cases_per_instance or len(cases)
        for index, offset in enumerate(range(0, len(cases), size), start=1):
            if args.drain_file and args.drain_file.exists():
                print(f'Drain requested at {args.drain_file}. Completed batches are available for --resume.',
                      flush=True)
                return 0 if all_passed else 1
            passed, unavailable = qualify_group(host, workspace, baseline, cases[offset:offset + size],
                                                args.model, destination, args.test_timeout_ms,
                                                batch=index if len(cases) > size else None, devices=devices,
                                                retain_trajectories=args.retain_trajectories)
            all_passed &= passed
            if unavailable:
                print(f'{workspace} startup is unrecoverable for this baseline. See the doctor log.',
                      file=sys.stderr, flush=True)
                break
    return 0 if all_passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true',
                        help='Reuse accepted references in this exact output catalog and retain separate attempts')
    parser.add_argument('--drain-file', type=Path,
                        help='If this file appears, finish the current batch, stop its instance, and leave later batches for --resume')
    parser.add_argument('--family', action='append', default=[])
    parser.add_argument('--case', action='append', default=[])
    parser.add_argument('--local-only', action='store_true',
                        help='Select goals requiring only a local managed workspace')
    parser.add_argument('--reuse-references', action='append', type=Path, default=[],
                        metavar='RESULT_DIR',
                        help='Skip goals already backed by an accepted managed reference in this directory')
    parser.add_argument('--model', default='openai-codex:gpt-6-luna')
    parser.add_argument('--cases-per-instance', type=int, default=12,
                        help='Recycle the managed workspace after this many cases (0 keeps one per workspace)')
    parser.add_argument('--retain-trajectories', action='store_true',
                        help='Use a compatible Host pin to save full trajectories during each test run')
    parser.add_argument('--test-timeout-ms', type=int)
    args = parser.parse_args()
    if args.test_timeout_ms is not None and args.test_timeout_ms <= 0:
        parser.error('--test-timeout-ms must be positive')
    if args.cases_per_instance < 0:
        parser.error('--cases-per-instance cannot be negative')
    if args.drain_file and not args.resume:
        parser.error('--drain-file requires --resume')

    source, catalog = read_catalog(args.catalog)
    chosen = {case: row for case, row in catalog.items()
              if (not args.case or case in args.case) and (not args.family or row['family'] in args.family)}
    if not chosen or set(args.case) - chosen.keys() or set(args.family) - {row['family'] for row in chosen.values()}:
        parser.error('Every requested case and family must match the pinned catalog')
    for case, row in chosen.items():
        workspace, baseline = row['workspace'], row.get('workspaceBaseline', '')
        if not re.fullmatch(r'[a-z][a-z-]*', workspace) or (baseline and not re.fullmatch(r'[a-f0-9]{40}', baseline)):
            parser.error(f'Invalid managed workspace or baseline for {case}')
        if not re.fullmatch(r'open-goal-[a-z0-9-]+', case):
            parser.error(f'Invalid goal identifier {case}')
        devices = row.get('devices', [])
        if not isinstance(devices, list) or any(device not in ('desktop', 'terminal') for device in devices):
            parser.error(f'Invalid managed device driver for {case}')
    if args.local_only:
        external = {case: requirements for case, row in chosen.items()
                    if (requirements := managed_prerequisites(row))}
        requested = sorted(set(args.case) & external.keys())
        if requested:
            summary = ', '.join(f'{case} ({", ".join(external[case])})' for case in requested)
            parser.error(f'Explicit cases require more than a local managed workspace: {summary}')
        if external:
            print(f'Selected local workspaces only, deferring {len(external)} goals with prerequisites',
                  flush=True)
        chosen = {case: row for case, row in chosen.items() if case not in external}
        if not chosen:
            parser.error('No selected goals can run with only the local managed workspace')
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=args.resume)
    with (destination / '.reference-run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f'Another reference run owns {destination}')
        pinned_catalog = destination / 'catalog.jsonl'
        if pinned_catalog.exists():
            if pinned_catalog.read_bytes() != source:
                parser.error('The resume directory belongs to a different pinned catalog')
        else:
            temporary = destination / '.catalog.jsonl.pending'
            temporary.write_bytes(source)
            temporary.replace(pinned_catalog)
        return run_batches(args, chosen, destination)


if __name__ == '__main__':
    def terminate(_signal, _frame):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(main())
