"""Resolved/unresolved labels for bgkit `compaction_v2` SWE-Zero trajectories, by actually running the tests.

SWE-Zero trajectories are execution free: no test result exists upstream. This tool applies each trajectory's
`model_patch` in the instance's published environment image and runs the instance's FAIL_TO_PASS / PASS_TO_PASS tests
with the standard SWE-bench evaluation script, then grades the log with the harness's own parser and report code.
Nothing is inferred from patch similarity: an instance without an environment image or test list gets no label.

The harness is a library (the SWE-rebench fork of SWE-bench for `rebench`, the SWE-Gym fork for `swegym`). Docker is
not required: the published instance images are x86_64 and run as CoreWeave CPU sandbox images, where this tool
repeats `swebench.harness.eval.run_instance` step for step (git apply, `git diff`, `/eval.sh`, grade).

  prepare  jobs.jsonl from the compaction_v2 trajectory ids, the instance spec source and the image check
  run      execute jobs in sandboxes (resumable; one test_output file per job)
  grade    labels.jsonl (trajectory_id -> resolved with log hash and harness version) and a summary
  Controls: `--controls N` adds N gold-patch jobs (must resolve) and N no-patch jobs (must not).

Run `prepare` and `grade` under the harness venv; `run` needs `cwsandbox` too (install it in the same venv).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from natlang_neuralese.common.paths import resolve as data_path  # noqa: E402

SWE_ZERO = Path.home() / 'bgkit-data-nvme' / 'capability_packaging' / 'trajectories_ext' / 'swe_zero' / 'data'
TASKS = Path.home() / 'bgkit-data-nvme' / 'bgkit2' / 'tasks' / 'compaction_v2.parquet'
REBENCH_DIR = data_path('data_hdd', 'bgkit-data', 'swe_rebench')
OUT = data_path('data_hdd', 'natlang-compaction-operators-20261103', 'swezero-eval')
DATASETS = {'rebench': 'nebius/SWE-rebench', 'swegym': 'SWE-Gym/SWE-Gym-Raw'}
SCHEMA = 'natlang.swezero-eval/1'
TEST_TIMEOUT = 1800          # the harness default
MAX_OUTPUT = 30 << 20        # sandbox read_file cap is 32 MiB


# ---------------------------------------------------------------------------------------------------------------
# targets and specs
# ---------------------------------------------------------------------------------------------------------------

def compaction_trajectory_ids(tasks: Path = TASKS) -> set:
    import pyarrow.parquet as pq
    ids = set()
    for batch in pq.ParquetFile(tasks).iter_batches(batch_size=5000, columns=['meta']):
        ids.update(json.loads(m)['trajectory_id'] for m in batch.column('meta').to_pylist())
    return ids


def swe_zero_targets(dataset: str, ids: set, root: Path = SWE_ZERO) -> list:
    """SWE-Zero rows of one source dataset whose trajectory id is indexed by compaction_v2."""
    import pyarrow.parquet as pq
    out = []
    for path in sorted(root.glob('*.parquet')):
        table = pq.read_table(path, columns=['trajectory_id', 'instance_id', 'dataset', 'model_patch'])
        for row in table.to_pylist():
            if row['dataset'] == dataset and row['trajectory_id'] in ids:
                out.append(row)
    return out


def image_exists(repository: str) -> bool:
    """Registry HEAD of `latest` with an anonymous pull token (a missing repository is a 401/404)."""
    url = f'https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repository}:pull'
    token = json.load(urllib.request.urlopen(url, timeout=30))['token']
    accept = ('application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, '
              'application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json')
    request = urllib.request.Request(f'https://registry-1.docker.io/v2/{repository}/manifests/latest', method='HEAD',
                                     headers={'Authorization': f'Bearer {token}', 'Accept': accept})
    for attempt in range(5):
        try:
            return urllib.request.urlopen(request, timeout=30).status == 200
        except urllib.error.HTTPError as error:
            if error.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if error.code in (401, 403, 404):
                return False
            raise
    raise RuntimeError(f'registry rate limit persisted for {repository}')


def rebench_image(instance_id: str) -> str:
    return 'swerebench/sweb.eval.x86_64.' + instance_id.lower().replace('__', '_1776_')


def swegym_image(instance_id: str) -> str:
    return 'xingyaoww/sweb.eval.x86_64.' + instance_id.lower().replace('__', '_s_')


def rebench_instances(wanted: set) -> dict:
    import pyarrow.parquet as pq
    found = {}
    for path in sorted(REBENCH_DIR.glob('*.parquet')):
        for row in pq.read_table(path).to_pylist():
            if row['instance_id'] in wanted:
                found.setdefault(row['instance_id'], row)
    return found


def swegym_instances(wanted: set, path: Path) -> dict:
    import pyarrow.parquet as pq
    return {row['instance_id']: row for row in pq.read_table(path).to_pylist() if row['instance_id'] in wanted}


def harness_version() -> dict:
    import subprocess
    import swebench
    root = Path(swebench.__file__).resolve().parents[1]
    try:
        commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout
        remote = subprocess.run(['git', '-C', str(root), 'remote', 'get-url', 'origin'], capture_output=True,
                                text=True).stdout
    except OSError:
        commit = remote = ''
    return {'package': 'swebench', 'path': str(root), 'git_commit': commit.strip(), 'git_remote': remote.strip()}


def make_spec(kind: str, instance: dict, image: str, built: bool = False):
    """The harness TestSpec of one instance: the eval script, test lists and log parser come from the harness itself."""
    row = dict(instance)
    if kind == 'swegym':    # the SWE-Gym fork's TestSpec has no image field: the instance image is its default one
        from swebench.harness.test_spec import make_test_spec
        return make_test_spec(row)
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.utils import _clean_install_config
    from swebench.harness.test_spec.test_spec import _make_remote_eval_compatible
    row.pop('image_name', None)
    spec = make_test_spec(_clean_install_config(row))
    # The published image may hold the package installed at another path (borg: /borg, sanic: missing), so the eval
    # script keeps its own `install` step as in a local image build; only the conda activation is the remote form.
    if not built:       # an environment built here from the harness Dockerfiles keeps the harness's own activation
        spec.eval_script_list = _make_remote_eval_compatible(spec.eval_script_list, None)
    return spec


def apply_commands(kind: str) -> list:
    """The harness's own patch application attempts, in order."""
    if kind == 'swegym':    # swebench/harness/run_evaluation.py of the SWE-Gym fork
        return ['git apply --allow-empty -v', 'patch --batch --fuzz=5 -p1 -i']
    from swebench.harness.eval import GIT_APPLY_CMDS
    return list(GIT_APPLY_CMDS)


def grade_report(kind: str, spec, instance_id: str, patch: str, log_path: str) -> dict:
    prediction = {'instance_id': instance_id, 'model_patch': patch, 'model_name_or_path': 'swe-zero'}
    from swebench.harness.grading import get_eval_report
    if kind == 'swegym':    # the fork reads the repository from the log's parent directory name
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / instance_id.lower()   # the fork's constants are lower case
            target.mkdir()
            shutil.copyfile(log_path, target / 'test_output.txt')
            return get_eval_report(spec, prediction, str(target / 'test_output.txt'), True)[instance_id]
    return get_eval_report(test_spec=spec, prediction=prediction, test_log_path=log_path,
                           include_tests_status=True)[instance_id]


# ---------------------------------------------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------------------------------------------

def prepare(kind: str, out: Path, swegym_parquet: Path | None, controls: int, seed: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ids = compaction_trajectory_ids()
    targets = swe_zero_targets(DATASETS[kind], ids)
    wanted = {t['instance_id'] for t in targets}
    instances = rebench_instances(wanted) if kind == 'rebench' else swegym_instances(wanted, swegym_parquet)
    images = {}
    for instance_id in sorted(instances):
        name = (instances[instance_id].get('image_name') if kind == 'rebench' else None) or (
            rebench_image(instance_id) if kind == 'rebench' else swegym_image(instance_id))
        images[instance_id] = name if image_exists(name) else None
    jobs, labels_without = [], []
    for target in targets:
        instance_id = target['instance_id']
        if instance_id not in instances:
            labels_without.append({**_ids(target), 'status': 'no-test-spec',
                                   'reason': 'instance has no FAIL_TO_PASS/PASS_TO_PASS list in the spec source'})
        elif not images[instance_id]:
            labels_without.append({**_ids(target), 'status': 'no-environment',
                                   'reason': 'no published instance image for the instance'})
        else:
            jobs.append({**_ids(target), 'kind': 'trajectory', 'image': images[instance_id],
                         'patch': target['model_patch']})
    rng = random.Random(seed)
    chosen = rng.sample(sorted({j['instance_id'] for j in jobs}), min(controls, len({j['instance_id'] for j in jobs})))
    for instance_id in chosen:
        gold = instances[instance_id]['patch']
        jobs.append({'trajectory_id': f'control:gold:{instance_id}', 'instance_id': instance_id, 'kind': 'control-gold',
                     'dataset': DATASETS[kind], 'image': images[instance_id], 'patch': gold})
        jobs.append({'trajectory_id': f'control:base:{instance_id}', 'instance_id': instance_id, 'kind': 'control-base',
                     'dataset': DATASETS[kind], 'image': images[instance_id], 'patch': None})
    for job in jobs:   # the spec is built now so a harness error surfaces before any sandbox starts
        spec = make_spec(kind, instances[job['instance_id']], job['image'])
        job['eval_script'] = spec.eval_script
        job['fail_to_pass'], job['pass_to_pass'] = len(spec.FAIL_TO_PASS), len(spec.PASS_TO_PASS)
    meta = {'schema': SCHEMA, 'kind': kind, 'harness': harness_version(), 'targets': len(targets), 'jobs': len(jobs),
            'without_label': len(labels_without), 'controls': len(chosen)}
    (out / f'{kind}.jobs.jsonl').write_text(''.join(json.dumps(j) + '\n' for j in jobs))
    (out / f'{kind}.unlabelled.jsonl').write_text(''.join(json.dumps(j) + '\n' for j in labels_without))
    (out / f'{kind}.prepare.json').write_text(json.dumps(meta, indent=1))
    return meta


def _ids(target: dict) -> dict:
    return {'trajectory_id': target['trajectory_id'], 'instance_id': target['instance_id'],
            'dataset': target['dataset']}


def add_gold(kind: str, out: Path, swegym_parquet: Path | None) -> int:
    """One gold-patch validation job per instance that has trajectory jobs. An environment whose gold patch does not
    resolve cannot label a trajectory unresolved (its tests may not run at all), so such instances stay unlabelled."""
    path = out / f'{kind}.jobs.jsonl'
    jobs = [json.loads(line) for line in path.read_text().splitlines()]
    have = {j['instance_id'] for j in jobs if j['kind'] == 'control-gold'}
    first = {}
    for job in jobs:
        if job['kind'] == 'trajectory':
            first.setdefault(job['instance_id'], job)
    instances = rebench_instances(set(first)) if kind == 'rebench' else swegym_instances(set(first), swegym_parquet)
    added = []
    for instance_id, job in sorted(first.items()):
        if instance_id not in have:
            added.append({'trajectory_id': f'control:gold:{instance_id}', 'instance_id': instance_id,
                          'kind': 'control-gold', 'dataset': job['dataset'], 'image': job['image'],
                          'patch': instances[instance_id]['patch'], 'eval_script': job['eval_script'],
                          'fail_to_pass': job['fail_to_pass'], 'pass_to_pass': job['pass_to_pass']})
    path.write_text(''.join(json.dumps(j) + '\n' for j in jobs + added))
    return len(added)


def add_build(kind: str, out: Path) -> dict:
    """Jobs for instances without a published image: the harness's own base/env/instance Dockerfiles and setup scripts,
    executed in a sandbox (`build_in_sandbox`). Only rebench specs are supported (the SWE-Gym fork builds are not)."""
    assert kind == 'rebench'
    unlabelled = [json.loads(line) for line in (out / f'{kind}.unlabelled.jsonl').read_text().splitlines()]
    pending = [u for u in unlabelled if u['status'] == 'no-environment']
    keep = [u for u in unlabelled if u['status'] != 'no-environment']
    instances = rebench_instances({u['instance_id'] for u in pending})
    targets = {t['trajectory_id']: t for t in swe_zero_targets(DATASETS[kind], {u['trajectory_id'] for u in pending})}
    jobs = []
    for entry in pending:
        instance = instances[entry['instance_id']]
        spec = make_spec(kind, instance, '', built=True)
        base_from = [l for l in spec.base_dockerfile.splitlines() if l.startswith('FROM')][0].split()[-1]
        common = {'instance_id': entry['instance_id'], 'dataset': entry['dataset'], 'image': base_from, 'build': {
            'stages': [{'name': 'base', 'dockerfile': spec.base_dockerfile, 'files': {}},
                       {'name': 'env', 'dockerfile': spec.env_dockerfile,
                        'files': {'./setup_env.sh': spec.setup_env_script}},
                       {'name': 'instance', 'dockerfile': spec.instance_dockerfile,
                        'files': {'./setup_repo.sh': spec.install_repo_script}}]},
                  'eval_script': spec.eval_script, 'fail_to_pass': len(spec.FAIL_TO_PASS),
                  'pass_to_pass': len(spec.PASS_TO_PASS)}
        jobs.append({**common, 'trajectory_id': f'control:gold:{entry["instance_id"]}', 'kind': 'control-gold',
                     'patch': instance['patch']})
        jobs.append({**common, 'trajectory_id': entry['trajectory_id'], 'kind': 'trajectory',
                     'patch': targets[entry['trajectory_id']]['model_patch']})
    path = out / f'{kind}.jobs.jsonl'
    have = {json.loads(line)['trajectory_id'] for line in path.read_text().splitlines()}
    with open(path, 'a') as stream:
        for job in jobs:
            if job['trajectory_id'] not in have:
                stream.write(json.dumps(job) + '\n')
    (out / f'{kind}.unlabelled.jsonl').write_text(''.join(json.dumps(u) + '\n' for u in keep))
    return {'build_instances': len(pending), 'jobs_added': len(jobs)}


def dockerfile_steps(text: str) -> list:
    """(directive, argument) pairs of a harness Dockerfile; only the directives those files use are accepted."""
    lines, current = [], ''
    for raw in text.splitlines():
        line = raw.rstrip()
        if not current and (not line.strip() or line.lstrip().startswith('#')):
            continue
        current += line[:-1] + ' ' if line.endswith('\\') else line
        if not line.endswith('\\'):
            lines.append(current.strip())
            current = ''
    steps = []
    for line in lines:
        directive, _, argument = line.partition(' ')
        if directive not in ('FROM', 'ARG', 'ENV', 'RUN', 'COPY', 'WORKDIR'):
            raise ValueError(f'unsupported Dockerfile directive {directive}')
        steps.append((directive, argument.strip()))
    return steps


def build_in_sandbox(sandbox, build: dict, log: dict, timeout: float = 3300) -> bool:
    """Run the harness's Dockerfile stages in the sandbox as `docker build` would. False on a failed step."""
    shell = ['/bin/sh', '-c']
    state = {'workdir': '/'}
    sandbox.write_file('/etc/swezero-env.sh', b'').result()
    for stage in build['stages']:
        for directive, argument in dockerfile_steps(stage['dockerfile']):
            if directive == 'FROM':
                continue
            if directive in ('ARG', 'ENV'):
                if '=' not in argument:
                    continue            # a bare ARG declares nothing
                name, value = argument.split('=', 1)
                command = f"echo 'export {name}=\"{value}\"' >> /etc/swezero-env.sh"
            elif directive == 'WORKDIR':
                state['workdir'] = argument
                command = f'mkdir -p {argument}'
            elif directive == 'COPY':
                source, destination = argument.split()
                sandbox.exec(shell + [f'mkdir -p {destination}'], timeout_seconds=60).result()
                sandbox.write_file(destination.rstrip('/') + '/' + Path(source).name,
                                   stage['files'][source].encode()).result()
                continue
            else:
                command = argument
            result = sandbox.exec(['/bin/sh', '-c', f'. /etc/swezero-env.sh; cd {state["workdir"]}; {command}'],
                                  timeout_seconds=timeout).result()
            if result.returncode != 0:
                log['build_failed'] = {'stage': stage['name'], 'step': f'{directive} {argument}'[:300],
                                       'returncode': result.returncode,
                                       'output': (_text(result.stdout)[-1500:] + _text(result.stderr)[-1500:])}
                return False
    return True


# ---------------------------------------------------------------------------------------------------------------
# run (sandboxes)
# ---------------------------------------------------------------------------------------------------------------

SHAPES = [{'cpu': '2', 'memory': '8Gi'}, {'cpu': '2', 'memory': '6Gi'}, {'cpu': '1', 'memory': '4Gi'}]


def _text(value) -> str:
    return value.decode('utf-8', 'replace') if isinstance(value, bytes) else (value or '')


def run_job(job: dict, kind: str, out: Path, account: str) -> list:
    """One instance in one sandbox, mirroring swebench.harness.eval.run_instance; a built instance (job['multi'])
    is built once and its gold and trajectory patches are evaluated one after another on a reset checkout."""
    from cwsandbox import Sandbox
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'coreweave'))
    import accounts as cwaccounts
    evals = job.get('multi') or [job]
    base = {'account': account, 'image': job['image']}
    started, sandbox, entries = time.time(), None, []
    try:
        auth, last = cwaccounts.auth_headers(account), None
        for shape in SHAPES:
            try:
                lifetime = TEST_TIMEOUT * len(evals) + 1200 + (6000 if job.get('build') else 0)
                sandbox = Sandbox.run('sleep', str(lifetime - 60), container_image=job['image'], auth=auth,
                                      resources=shape, max_lifetime_seconds=lifetime, tags=['swezero-eval', kind])
                base['shape'] = shape
                break
            except Exception as error:   # noqa: BLE001 - SandboxResourceExhausted backs off to a smaller shape
                last = error
                if 'ResourceExhausted' not in type(error).__name__:
                    raise
        if sandbox is None:
            raise last
        base['sandbox_id'] = sandbox.sandbox_id

        def sh(command: str, timeout: float = 300):
            return sandbox.exec(['bash', '-c', command], cwd='/testbed', timeout_seconds=timeout).result()

        if job.get('build'):
            build_log = {}
            if not build_in_sandbox(sandbox, job['build'], build_log):
                return [{**base, 'trajectory_id': e['trajectory_id'], 'status': 'build-failed', **build_log,
                         'seconds': round(time.time() - started, 1)} for e in evals]
        for index, item in enumerate(evals):
            log = {**base, 'trajectory_id': item['trajectory_id']}
            entries.append(log)
            if index:
                sh('git reset --hard -q && git clean -fdq')
            applied = None
            if item['patch'] is not None:
                sandbox.write_file('/tmp/patch.diff', item['patch'].encode()).result()
                for command in apply_commands(kind):
                    result = sh(f'{command} /tmp/patch.diff')
                    if result.returncode == 0:
                        applied = command
                        break
                log['patch_applied'] = applied
                if applied is None:
                    log['apply_output'] = _text(result.stdout)[-2000:] + _text(result.stderr)[-2000:]
                    log['status'] = 'patch-apply-failed'
                    continue
            sandbox.write_file('/eval.sh', item['eval_script'].encode()).result()
            result = sh('/bin/bash /eval.sh > /tmp/test_output.txt 2>&1; echo $?', TEST_TIMEOUT)
            log['eval_exit'] = _text(result.stdout).strip()[-20:]
            size = int(_text(sh('stat -c %s /tmp/test_output.txt').stdout).strip() or 0)
            if size > MAX_OUTPUT:
                log['status'] = 'harness-failure'
                log['reason'] = f'test output {size} bytes exceeds {MAX_OUTPUT}'
                continue
            output = sandbox.read_file('/tmp/test_output.txt').result()
            (out / 'logs' / f'{item["trajectory_id"]}.txt').write_bytes(output)
            log['status'] = 'ran'
            log['test_log_sha256'] = hashlib.sha256(output).hexdigest()
            log['test_log_bytes'] = len(output)
        return entries
    except Exception as error:   # noqa: BLE001 - recorded as a harness failure, retried on the next run
        text = f'{type(error).__name__}: {error}'
        status = 'timeout' if 'imeout' in text or 'DEADLINE' in text else 'harness-failure'
        done = {e['trajectory_id'] for e in entries if e.get('status')}
        entries = [e for e in entries if e.get('status')]
        return entries + [{**base, 'trajectory_id': e['trajectory_id'], 'status': status, 'reason': text[:1000]}
                          for e in evals if e['trajectory_id'] not in done]
    finally:
        for entry in entries:
            entry['seconds'] = round(time.time() - started, 1)
        if sandbox is not None:
            try:
                sandbox.stop().result()
            except Exception:   # noqa: BLE001 - lifetime expiry reclaims it
                pass


def run(kind: str, out: Path, workers: int, accounts: list, limit: int | None, builds_only: bool = False) -> dict:
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(line) for line in (out / f'{kind}.jobs.jsonl').read_text().splitlines()]
    results = out / f'{kind}.runs.jsonl'
    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            entry = json.loads(line)
            if entry['status'] in ('ran', 'patch-apply-failed', 'build-failed'):
                done.add(entry['trajectory_id'])
    todo = [j for j in jobs if j['trajectory_id'] not in done]
    grouped = {}
    for job in [j for j in todo if j.get('build')]:     # one build per instance serves its gold and trajectory runs
        grouped.setdefault(job['instance_id'], []).append(job)
    todo = [j for j in todo if not j.get('build') and not builds_only] + [
        {**sorted(group, key=lambda j: j['kind'] != 'control-gold')[0], 'multi': sorted(
            group, key=lambda j: j['kind'] != 'control-gold')} for group in grouped.values()]
    # controls first: a broken harness shows before the bulk is spent
    todo.sort(key=lambda j: j['kind'] == 'trajectory')
    if limit:
        todo = todo[:limit]
    lock, counter = threading.Lock(), iter(range(10 ** 9))

    def work(job):
        account = accounts[next(counter) % len(accounts)]
        entries = run_job(job, kind, out, account)
        with lock, open(results, 'a') as stream:
            for entry in entries:
                stream.write(json.dumps(entry) + '\n')
            stream.flush()
        for entry in entries:
            print(json.dumps({k: entry.get(k) for k in ('trajectory_id', 'status', 'seconds', 'reason')}), flush=True)
        return [entry['status'] for entry in entries]

    with ThreadPoolExecutor(workers) as pool:
        statuses = [status for group in pool.map(work, todo) for status in group]
    summary = {'requested': len(statuses), 'already_done': len(done)}
    for status in set(statuses):
        summary[status] = statuses.count(status)
    return summary


# ---------------------------------------------------------------------------------------------------------------
# grade
# ---------------------------------------------------------------------------------------------------------------

def grade(kind: str, out: Path, swegym_parquet: Path | None) -> dict:
    meta = json.loads((out / f'{kind}.prepare.json').read_text())
    jobs = {}
    for line in (out / f'{kind}.jobs.jsonl').read_text().splitlines():
        job = json.loads(line)
        jobs[job['trajectory_id']] = job
    runs = {}
    for line in (out / f'{kind}.runs.jsonl').read_text().splitlines():
        entry = json.loads(line)
        runs[entry['trajectory_id']] = entry       # the last attempt wins
    wanted = {j['instance_id'] for j in jobs.values()}
    instances = rebench_instances(wanted) if kind == 'rebench' else swegym_instances(wanted, swegym_parquet)
    labels, controls = [], {'control-gold': [0, 0], 'control-base': [0, 0]}
    gold = {}
    for tid, run_entry in runs.items():
        if tid.startswith('control:gold:') and run_entry['status'] in ('ran', 'patch-apply-failed'):
            gold[jobs[tid]['instance_id']] = None      # filled with the graded result below
    for tid, job in sorted(jobs.items()):
        run_entry = runs.get(tid)
        record = {'trajectory_id': tid, 'instance_id': job['instance_id'], 'dataset': job['dataset'],
                  'image': job['image'], 'resolved': None, 'status': 'not-run',
                  'harness': {'name': 'swe-rebench-fork' if kind == 'rebench' else 'swe-gym-fork',
                              **{k: meta['harness'][k] for k in ('git_commit', 'git_remote')}},
                  'fail_to_pass': job['fail_to_pass'], 'pass_to_pass': job['pass_to_pass']}
        if run_entry is None:
            pass
        elif run_entry['status'] == 'patch-apply-failed':
            record.update(resolved=False, status='patch-apply-failed')
        elif run_entry['status'] != 'ran':
            record.update(status=run_entry['status'], reason=run_entry.get('reason'))
        else:
            spec = make_spec(kind, instances[job['instance_id']], job['image'])
            report = grade_report(kind, spec, job['instance_id'], job['patch'] or '', str(out / 'logs' / f'{tid}.txt'))
            record.update(resolved=bool(report['resolved']), status='resolved' if report['resolved'] else 'unresolved',
                          test_log_sha256=run_entry['test_log_sha256'],
                          patch_successfully_applied=report['patch_successfully_applied'])
            tests = report.get('tests_status') or {}
            record['tests'] = {key: {k: len(v) for k, v in value.items()} for key, value in tests.items()
                               if isinstance(value, dict)}
        record['provenance'] = {'tool': 'scripts/neuralese_data/swezero_eval.py', 'schema': SCHEMA,
                                'executed': 'cwsandbox CPU sandbox from the published instance image',
                                'sandbox_id': (run_entry or {}).get('sandbox_id')}
        if job['kind'] == 'control-gold':
            gold[job['instance_id']] = record['resolved']
        if job['kind'] in controls and record['resolved'] is not None:
            controls[job['kind']][0] += int(record['resolved'])
            controls[job['kind']][1] += 1
        if job['kind'] == 'trajectory':
            labels.append(record)
        else:
            record['control'] = job['kind']
            labels.append(record)
    for record in labels:
        if 'control' in record or record['resolved'] is None or record['instance_id'] not in gold:
            continue
        if gold[record['instance_id']] is not True:
            record.update(resolved=None, status='env-unvalidated',
                          reason='the gold patch does not resolve this instance in its environment')
    (out / f'{kind}.labels.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in labels))
    unlabelled = [json.loads(line) for line in (out / f'{kind}.unlabelled.jsonl').read_text().splitlines()]
    count = {}
    for record in labels:
        if 'control' not in record:
            count[record['status']] = count.get(record['status'], 0) + 1
    summary = {'kind': kind, 'dataset': DATASETS[kind], 'targets': meta['targets'], 'evaluated': sum(
                   v for k, v in count.items() if k in ('resolved', 'unresolved', 'patch-apply-failed')),
               'by_status': count, 'unlabelled': len(unlabelled),
               'unlabelled_by_status': {s: sum(1 for u in unlabelled if u['status'] == s)
                                        for s in {u['status'] for u in unlabelled}},
               'controls_resolved_of_run': {k: v for k, v in controls.items()}}
    (out / f'{kind}.summary.json').write_text(json.dumps(summary, indent=1))
    return summary


def merge_labels(out: Path) -> Path:
    """One labels file for the converter: trajectory_id -> resolved, from every harness kind (controls excluded)."""
    path = out / 'labels.jsonl'
    rows = []
    for kind in DATASETS:
        file = out / f'{kind}.labels.jsonl'
        if file.exists():
            rows += [json.loads(line) for line in file.read_text().splitlines()]
        file = out / f'{kind}.unlabelled.jsonl'
        if file.exists():
            rows += [{**json.loads(line), 'resolved': None} for line in file.read_text().splitlines()]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows if 'control' not in r))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'grade', 'add-gold', 'add-build'):
        p = sub.add_parser(name)
        p.add_argument('--kind', choices=sorted(DATASETS), required=True)
        p.add_argument('--out', default=str(OUT))
        p.add_argument('--swegym-parquet', default=None, help='SWE-Gym/SWE-Gym train parquet (validated instances)')
        if name == 'prepare':
            p.add_argument('--controls', type=int, default=6)
            p.add_argument('--seed', type=int, default=20261103)
        if name == 'run':
            p.add_argument('--workers', type=int, default=8)
            p.add_argument('--accounts', default='cw1,cw2,cw3,cw4')
            p.add_argument('--limit', type=int, default=None)
            p.add_argument('--builds-only', action='store_true', help='only instances built from the harness Dockerfiles')
    sub.add_parser('merge').add_argument('--out', default=str(OUT))
    args = parser.parse_args(argv)
    out = Path(args.out)
    gym = Path(args.swegym_parquet) if getattr(args, 'swegym_parquet', None) else None
    if args.command == 'prepare':
        print(json.dumps(prepare(args.kind, out, gym, args.controls, args.seed)))
    elif args.command == 'run':
        print(json.dumps(run(args.kind, out, args.workers, args.accounts.split(','), args.limit, args.builds_only)))
    elif args.command == 'add-build':
        print(json.dumps(add_build(args.kind, out)))
    elif args.command == 'add-gold':
        print(add_gold(args.kind, out, gym))
    elif args.command == 'grade':
        print(json.dumps(grade(args.kind, out, gym), indent=1))
    else:
        print(merge_labels(out))


if __name__ == '__main__':
    main()
