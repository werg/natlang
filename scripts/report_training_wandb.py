#!/usr/bin/env python3
"""Report saved training metrics without loading a model or blocking training."""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import signal
import time
from pathlib import Path


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


SUPPORT_SCALAR_FIELDS = (
    'rows', 'class_count', 'source_id_count', 'source_id_links',
    'source_group_count', 'source_group_links', 'modality_count',
)


def read_audited_support_sidecar(spec):
    """Read an explicitly hash-pinned aggregate-only support receipt.

    The sidecar contains no row identities or labels. Only the listed counts are
    returned for W&B; source/class names and per-row records are never logged.
    """
    path = Path(spec['path'])
    expected = spec['sha256']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f'audited-support sidecar hash mismatch: {path}')
    value = json.loads(path.read_text())
    if value.get('schema') != 'natlang.periodic_heldout_support_aggregate/1':
        raise ValueError('unsupported audited-support sidecar schema')
    if value.get('status') != 'audited' or not isinstance(value.get('identity_sha256'), str):
        raise ValueError('audited-support sidecar has no audited identity')
    if not isinstance(value.get('step'), int) or value['step'] < 1:
        raise ValueError('audited-support sidecar has no valid checkpoint step')
    identity = value.get('identity')
    if not isinstance(identity, dict) or identity.get('step') != value['step']:
        raise ValueError('audited-support sidecar identity does not bind its checkpoint step')
    identity_sha = hashlib.sha256(json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if identity_sha != value['identity_sha256']:
        raise ValueError('audited-support sidecar identity hash mismatch')
    if (spec.get('step') != value['step'] or
            spec.get('identity_sha256') != value['identity_sha256']):
        raise ValueError('audited-support sidecar differs from its pinned plan identity')
    counts = value.get('counts')
    if not isinstance(counts, dict) or set(counts) != set(SUPPORT_SCALAR_FIELDS):
        raise ValueError('audited-support sidecar must contain exactly the approved count fields')
    if any(not isinstance(counts[key], int) or counts[key] < 0 for key in SUPPORT_SCALAR_FIELDS):
        raise ValueError('audited-support counts must be nonnegative integers')
    return digest, value['identity_sha256'], value['step'], counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    plan = read_json(args.plan)
    root = Path(plan['report_directory'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'reporter.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import wandb
        stop = False

        def stop_now(*_):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGTERM, stop_now)
        signal.signal(signal.SIGINT, stop_now)
        saved = read_json(root / 'reporter-state.json') or {'last_training_step': 0, 'evaluation_keys': []}
        run = wandb.init(entity=plan['entity'], project=plan['project'], id=plan['run_id'],
                         name=plan['name'], resume='allow', dir=str(root), mode='online',
                         config=plan['public_config'],
                         settings=wandb.Settings(disable_git=True, disable_code=True,
                                                 x_disable_stats=True, console='off'))
        run.define_metric('optimizer_step')
        run.define_metric('train/*', step_metric='optimizer_step')
        run.define_metric('eval/*', step_metric='optimizer_step')
        run.define_metric('eval/audited_support/*', step_metric='optimizer_step')
        # Local SDK queuing is not a remote acknowledgement. On restart replay any
        # locally queued suffix absent from the server summary, keeping the same run.
        remote_summary = wandb.Api().run(f"{plan['entity']}/{plan['project']}/{plan['run_id']}").summary
        confirmed_step = remote_summary.get('reported_through_optimizer_step', 0)
        saved['last_training_step'] = min(saved['last_training_step'], confirmed_step)
        if not confirmed_step:
            saved['baseline_logged'] = False
            saved['evaluation_keys'] = []
            saved['audited_support_keys'] = []
        else:
            saved['evaluation_keys'] = remote_summary.get('reported_evaluation_keys', [])
            saved['audited_support_keys'] = remote_summary.get('reported_audited_support_keys', [])
        baseline_logged = saved.get('baseline_logged', False)
        seen = set(saved['evaluation_keys'])
        try:
            while not stop:
                current = read_json(args.plan)
                if not current or any(current.get(k) != plan.get(k) for k in
                                      ('entity', 'project', 'run_id', 'report_directory',
                                       'audited_support_sidecars')):
                    raise ValueError('Reporter identity changed')
                steps, checkpoints = {}, []
                for directory in current['training_outputs']:
                    directory = Path(directory)
                    telemetry = read_json(directory / 'throughput.json')
                    if telemetry:
                        for row in telemetry.get('steps', []):
                            steps[row['step']] = row
                    checkpoint = read_json(directory / 'checkpoint/state.json')
                    if checkpoint:
                        checkpoints.append(checkpoint)
                if not baseline_logged:
                    baseline = next((s.get('heldout_before') for s in checkpoints
                                     if finite(s.get('heldout_before'))), None)
                    if baseline is not None:
                        run.log({'optimizer_step': 0, 'eval/baseline_example_mean_loss': baseline})
                        baseline_logged = True
                for step, row in sorted(steps.items()):
                    if step <= saved['last_training_step']:
                        continue
                    metrics = {'optimizer_step': step}
                    for source, target in [('loss', 'loss'), ('seconds', 'step_seconds'),
                                           ('prepare_seconds', 'prepare_seconds'), ('tokens', 'tokens'),
                                           ('completion_tokens', 'completion_tokens'),
                                           ('examples', 'examples'), ('padded_tokens', 'padded_tokens')]:
                        if finite(row.get(source)):
                            metrics['train/' + target] = row[source]
                    if row.get('seconds', 0) > 0:
                        metrics['train/tokens_per_second'] = row.get('tokens', 0) / row['seconds']
                    run.log(metrics)
                    saved['last_training_step'] = step
                # Evaluation producers append immutable completion records; partial evaluations
                # are never published as a completed score. Config lists their exact files.
                for filename in current.get('evaluation_files', []):
                    try:
                        lines = Path(filename).read_text().splitlines()
                    except FileNotFoundError:
                        continue
                    for line in lines:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        key = hashlib.sha256(line.encode()).hexdigest()
                        if key in seen or row.get('status') != 'completed':
                            continue
                        step = row.get('step')
                        metrics = row.get('metrics', {})
                        if not isinstance(step, int) or not isinstance(metrics, dict):
                            continue
                        values = {'eval/' + k: v for k, v in metrics.items() if finite(v)}
                        if values:
                            run.log({'optimizer_step': step, **values})
                            seen.add(key)
                support_seen = set(saved.get('audited_support_keys', []))
                for spec in current.get('audited_support_sidecars', []):
                    digest, identity, support_step, counts = read_audited_support_sidecar(spec)
                    key = hashlib.sha256(f'{digest}\0{identity}'.encode()).hexdigest()
                    if key in support_seen:
                        continue
                    values = {f'eval/audited_support/{name}': counts[name]
                              for name in SUPPORT_SCALAR_FIELDS}
                    values['optimizer_step'] = support_step
                    run.log(values)
                    support_seen.add(key)
                saved.update(baseline_logged=baseline_logged, evaluation_keys=sorted(seen),
                             audited_support_keys=sorted(support_seen),
                             url=run.url, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             delivery='queued in online W&B SDK; server receipt verified separately')
                run.summary['reported_through_optimizer_step'] = saved['last_training_step']
                run.summary['reported_evaluation_keys'] = sorted(seen)
                run.summary['reported_audited_support_keys'] = sorted(support_seen)
                atomic_json(root / 'reporter-state.json', saved)
                print(json.dumps({'url': run.url, 'training_step': saved['last_training_step'],
                                  'evaluations': len(seen)}), flush=True)
                for _ in range(current.get('poll_seconds', 15)):
                    if stop:
                        break
                    time.sleep(1)
        finally:
            run.finish()


if __name__ == '__main__':
    main()
