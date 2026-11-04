#!/usr/bin/env python3
"""LFM3 lineages on CoreWeave GPU sandboxes, one lineage per sandbox, the DGX as hub
(plans/neuralese/LFM3_RL_CLOUD_20261031.md; sandbox side: lfm3_rl_lineage.sh and rl_cloud_sandbox.py).

Run with the CoreWeave client venv (cwctl.VENV). Credentials never leave the
DGX: the sandbox gets per-object signed URLs (capabilities) for the private W&B storage (werg/natlang).

  rl_cloud.py plan RUN.json                    every lineage's launch plan (accounts, resources, start boundary,
                                               base-chain availability); allocates and uploads nothing
  rl_cloud.py seed RUN.json LINEAGE            the lineage's seed export, uploaded once from the DGX (content-addressed).
                                               A joint-SFT lineage reads its named DGX controller's newest earned
                                               boundary now: window 0's sandbox control, the exact input closure (blobs
                                               already in a readable store are not uploaded again) and the parent as an
                                               exact ZIP delta over a declared full base when smaller
  rl_cloud.py bundle RUN.json                  the run's input bundle: code snapshot, wheelhouse, base model (HF URLs);
                                               for RL lineages also the custody packets, task source and (only when an
                                               actor uses it) the SGLang fork source + x86_64 lock
  rl_cloud.py launch RUN.json LINEAGE [--account A] [--dry-run]
                                               one GPU sandbox from the newest DGX-verified export of LINEAGE
  rl_cloud.py export RUN.json [--once] [--interval S]
                                               the exporter: uploads each lineage's newest staged export (sandbox ->
                                               W&B by signed PUT), then GETs it to the DGX hub (ROOT/<lineage>,
                                               verified), the resume point of the next launch
  rl_cloud.py prune RUN.json [--execute]       retire superseded blobs (the exporter does this every pass)
  rl_cloud.py status RUN.json
  rl_cloud.py deliver-parent RUN.json LINEAGE  the parent of a sandbox launched with --parent-later
  rl_cloud.py stop RUN.json LINEAGE

RUN.json (schema natlang.rl-cloud-run/1):
  {"id", "root" (the lineage root, the same absolute path in sandbox and on DGX), "job" (template), "hf" (base model
   REPO@REV=LOCAL), "rounds" (per sandbox: RL rounds or SFT windows; 0 = until stopped),
   "blob_stores" (roots of other runs' blob stores to read, e.g. the private recovery store),
   "checkpoint_bases" ([{"sha256", "bytes", "path" (DGX copy, to build deltas against), "encoding": "raw" |
                         "zip-delta" with "base", "receipt" {sha256, bytes}, "payload" {sha256, bytes}}]),
   "runtime_archive" ({"runtime", "archive", "sha256", "bytes", "closure_sha256"}: the frozen runtime SFT windows run),
   "lineages": {NAME: joint SFT: {"controller" (the DGX lineage's current window control), "window" (optional
                         refresh_after_updates), "prepared_control" (window 0 is that exact DGX-prepared, never-started
                         control, e.g. a corrected-mix boundary), "rebuild_window" (seed from an incomplete window's parent, forking
                         from the DGX window), "config" (overrides), "resources", "account", "allow_missing",
                         "service" (true: also expose the DiLoCo coordinator port, an elastic-session cohost)}
                      RL: {"template" (a DGX lineage config.json), "parent": {"checkpoint", "tokenizers"},
                           "config", "resources", "account"}},
   RL lineages also: "template_runtime" (a DGX RL runtime whose runs/ packets ship), "extras", "cloud_inputs"}
An RL lineage's task source is its config's "cases" file (shipped as an extra): any train-split natlang case JSONL
whose rows carry the semantics/oracle the runtime's collector scores. A new task mix is a new "cases" value (and its
file in "extras"), applied at the next round boundary through the controller's own hot reload.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
import cwctl  # noqa: E402
import rl_cloud_sandbox as sandbox  # noqa: E402
import job as jobs  # noqa: E402

PROJECT = ('werg', 'natlang')
PARENT_SCHEMA = 'natlang.rl-cloud-parent/1'
EXPORT_SCHEMA = 'natlang.rl-cloud-export/1'
SANDBOX_HELPER = 'scripts/coreweave/rl_cloud_sandbox.py'


# -- run files and local state -----------------------------------------------------------------------------------------

def load_run(path):
    run = json.loads(Path(path).read_text())
    if run.get('schema') != 'natlang.rl-cloud-run/1':
        raise SystemExit('not a natlang.rl-cloud-run/1 file')
    run['_path'] = str(Path(path).resolve())
    return run


def hub(run):
    path = Path(run['root']) / '.cloud'
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def store_json(path, value):
    from natlang_neuralese.common.jsonio import write_json_durable
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json_durable(Path(path), value)


def note(run, text):
    """One line in the RL decisions log (the DGX RL operations record)."""
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} cloud {run['id']}: {text}"
    with open(run['decisions_log'], 'a') as log:
        log.write(line + '\n')
    print(line, flush=True)


def file_digest(path):
    from natlang_neuralese.common.hashing import sha256_file_hex
    return sha256_file_hex(Path(path))


# -- private W&B blob store (content-addressed run files) ----------------------------------------------------------------

def _key():
    cwctl.load_env_file()
    return cwctl.cwaccounts.storage_wandb_key()


def _gql(query, variables):
    import requests
    response = requests.post('https://api.wandb.ai/graphql', auth=('api', _key()),
                             json={'query': query, 'variables': variables}, timeout=60)
    response.raise_for_status()
    data = response.json()
    if data.get('errors'):
        raise RuntimeError(str(data['errors'])[:500])
    return data['data']


def blob_run(run):
    """The W&B run whose files are this cloud run's blobs (created once, private project)."""
    registry = hub(run) / 'blob-run.json'
    if registry.exists():
        return read_json(registry)['run_path']
    os.environ['WANDB_API_KEY'] = _key()
    os.environ.setdefault('WANDB_DIR', str(hub(run) / 'wandb'))
    import wandb
    access = _gql('query { project(name:"natlang",entityName:"werg") { access } }', {})['project']['access']
    if access != 'PRIVATE':
        raise RuntimeError(f'werg/natlang is {access}, not PRIVATE')
    wb = wandb.init(entity=PROJECT[0], project=PROJECT[1], job_type='rl-cloud-blobs', name=f"{run['id']}-blobs",
                    settings=wandb.Settings(console='off', disable_code=True, disable_git=True, x_disable_stats=True,
                                            quiet=True))
    path = f'{PROJECT[0]}/{PROJECT[1]}/{wb.id}'
    wb.finish()
    store_json(registry, {'run_path': path, 'access': 'PRIVATE', 'created': time.time()})
    return path


def blobs(run):
    """Committed blobs: {sha256: {"bytes", "parts": [{"index", "offset", "bytes", "sha256", "name"}]}}. Blobs travel as
    parts of PART_BYTES (rl_cloud_sandbox.PART_BYTES), each its own private object, so a transfer failure repeats one
    part. An entry without parts is not a usable blob."""
    return {k: v for k, v in read_json(hub(run) / 'blobs.json', {}).items() if v.get('parts')}


def record_parts(run, digest, size, parts, committed):
    table = read_json(hub(run) / 'blobs.json', {})
    entry = table.get(digest) if table.get(digest, {}).get('pending') else None
    entry = entry or {'bytes': size, 'pending': [dict(p, name=sandbox.part_name(digest, p['index'])) for p in parts]}
    done = {p['index'] for p in entry.get('done', [])} | set(committed)
    entry['done'] = [p for p in entry['pending'] if p['index'] in done]
    if len(entry['done']) == len(entry['pending']):
        entry = {'bytes': size, 'parts': entry['pending'], 'committed': time.time()}
        retirement_path = hub(run) / 'rl-blob-retirement.json'
        retirement = read_json(retirement_path)
        if retirement and f'blob:{digest}' in retirement.get('retired_roles', []):  # the same content committed again
            retirement['retired_roles'].remove(f'blob:{digest}')
            retirement['operations'] = [o for o in retirement['operations'] if o.get('role') != f'blob:{digest}']
            store_json(retirement_path, retirement)
    table[digest] = entry
    store_json(hub(run) / 'blobs.json', table)


def missing_parts(run, digest, size, parts):
    """The parts of a blob not yet committed (all of them for a new blob)."""
    entry = read_json(hub(run) / 'blobs.json', {}).get(digest, {})
    if entry.get('parts'):
        return []
    done = {p['index'] for p in entry.get('done', [])}
    return [dict(p, name=sandbox.part_name(digest, p['index'])) for p in parts if p['index'] not in done]


def upload_urls(run, names):
    """Signed PUT URLs for run files ``names`` in the blob run (W&B createRunFiles), with the headers each PUT carries."""
    run_path = blob_run(run)
    mutation = ('mutation CreateRunFiles($entity:String!,$project:String!,$run:String!,$files:[String!]!) '
                '{createRunFiles(input:{entityName:$entity,projectName:$project,runName:$run,files:$files}) '
                '{runID uploadHeaders files {name uploadUrl}}}')
    out = {}
    for start in range(0, len(names), 50):
        batch = names[start:start + 50]
        data = _gql(mutation, {'entity': PROJECT[0], 'project': PROJECT[1], 'run': run_path.split('/')[-1],
                               'files': batch})['createRunFiles']
        headers = dict(h.split(':', 1) for h in data['uploadHeaders'])
        for item in data['files']:
            out[item['name']] = {'url': item['uploadUrl'], 'headers': headers}
    return out


def commit_parts(run, digest, size, parts, names):
    import wandb
    from private_export_guard import commit_uploaded_run_file
    api = wandb.Api(api_key=_key())
    for name in names:
        commit_uploaded_run_file(api, blob_run(run), name)
    by_name = {sandbox.part_name(digest, p['index']): p['index'] for p in parts}
    record_parts(run, digest, size, parts, [by_name[n] for n in names])


_REMOTE_RUNS = {}


def _remote_run(key, run_path):
    """One W&B API run handle per blob run (a launch issues hundreds of part URLs)."""
    import wandb
    if run_path not in _REMOTE_RUNS:
        _REMOTE_RUNS[run_path] = wandb.Api(api_key=key).run(run_path)
    return _REMOTE_RUNS[run_path]


def get_url(run, name, size, run_path=None):
    """A signed GET URL (a capability for one private object, not a key) at storage.googleapis.com."""
    import requests
    import wandb
    from urllib.parse import urlparse
    key = _key()
    remote = _remote_run(key, run_path or blob_run(run)).file(name)
    if remote.size != size:
        raise RuntimeError(f'{name}: remote size {remote.size} differs from {size}')
    url = remote.url
    if urlparse(url).hostname == 'api.wandb.ai':
        response = requests.get(url, auth=('api', key), stream=True, timeout=60, allow_redirects=True)
        response.close()
        response.raise_for_status()
        url = response.url
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname != 'storage.googleapis.com' or parsed.username or parsed.password:
        raise RuntimeError('no scoped read URL for the blob')
    return url


def foreign_stores(run):
    """Other runs' committed blob stores this run may read (``run["blob_stores"]``: their roots), e.g. the private
    recovery store holding full checkpoint bases, exact ZIP deltas, the current graph closure and the frozen runtime.
    Read only: this run never writes, retires or re-uploads their blobs; it leases what its launches read in the
    owning store's own lease namespace."""
    out = []
    for root in run.get('blob_stores', []):
        cloud = Path(root) / '.cloud'
        registry = read_json(cloud / 'blob-run.json')
        if not registry:
            raise RuntimeError(f'{root}: no blob run')
        table = {k: v for k, v in read_json(cloud / 'blobs.json', {}).items() if v.get('parts')}
        out.append({'root': str(root), 'hub': cloud, 'run_path': registry['run_path'], 'blobs': table})
    return out


def locate(run, digest):
    """(store, entry) of a committed blob: this run's own store first, then the foreign stores; None if absent."""
    own = blobs(run)
    if digest in own:
        return {'root': run['root'], 'hub': hub(run), 'run_path': None}, own[digest]
    for store in foreign_stores(run):
        if digest in store['blobs']:
            return store, store['blobs'][digest]
    return None


def part_urls(run, digest):
    found = locate(run, digest)
    if found is None:
        raise RuntimeError(f'blob {digest} is in no readable store')
    store, entry = found
    return [dict(p, url=get_url(run, p['name'], p['bytes'], store['run_path'])) for p in entry['parts']]


def put_local(run, path):
    """DGX file -> blob (a seed), part by part; a committed part is never sent again."""
    path = Path(path)
    def stamp():
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    before = stamp()
    def unchanged():
        try:
            actual = stamp()
        except OSError as error:
            raise RuntimeError(f'{path} disappeared during upload') from error
        if actual != before:
            raise RuntimeError(f'{path} changed during upload')
    digest, parts = sandbox.digest_parts(path)
    unchanged()
    size = before[2]
    todo = missing_parts(run, digest, size, parts)
    started = time.time()
    for part in todo:
        target = upload_urls(run, [part['name']])[part['name']]
        unchanged()
        if not sandbox.put_part(target['url'], target['headers'], path, part['offset'], part['bytes'],
                                {'name': part['name'], 'path': str(path)}):
            raise RuntimeError(f'upload of {part["name"]} ({path}) failed')
        unchanged()
        commit_parts(run, digest, size, parts, [part['name']])
    # The initial complete digest plus unchanged inode/size/mtime/ctime binds
    # every read. A second whole-file pass only repeats slow external-disk I/O.
    # This is source custody, not remote readback: exporters still GET and hash
    # independently before granting a durable recovery claim.
    unchanged()
    seconds = time.time() - started
    print(json.dumps({'event': 'blob_put', 'path': str(path), 'sha256': digest, 'bytes': size, 'parts': len(parts),
                      'sent_parts': len(todo), 'seconds': round(seconds, 1),
                      'mb_per_s': round(sum(p['bytes'] for p in todo) / 1e6 / max(seconds, 0.1), 2)}), flush=True)
    return digest, size


def get_blob(run, digest, destination):
    """Owner GET of one blob to the DGX: every part and the whole verified before the file is renamed into place."""
    destination = Path(destination)
    if destination.exists() and file_digest(destination) == digest:
        return 0.0
    started = time.time()
    temp = destination.with_name(f'.{destination.name}.get')
    sandbox.download_parts(part_urls(run, digest), temp, digest, blobs(run)[digest]['bytes'])
    temp.replace(destination)
    return time.time() - started


# -- lineage configs and seeds -------------------------------------------------------------------------------------------

def lineage_dir(run, name):
    return Path(run['root']) / name


def is_sft(spec):
    """A joint-SFT lineage of windows continues a named DGX mandatory supervised controller (``"controller"``); an RL
    lineage follows a DGX lineage config template (``"template"``)."""
    return 'controller' in spec


SGLANG_BENCH = 'natlang_neuralese.serve.sglang_bench'
DILOCO_ROLE = 'natlang_neuralese.train.diloco.cloud_role'
DILOCO_PORT = 8080


def diloco_role(spec):
    """``coordinator``/``learner`` for a DiLoCo role lineage, else None."""
    tune = spec.get('tune') or {}
    return (tune.get('diloco') or {}).get('role') if tune.get('module') == DILOCO_ROLE else None


def is_tune(spec):
    """A stage shard (``"tune"``) runs an adaptation stage module on a parent checkpoint: ``natlang_neuralese.tune.stage
    --shard I/N`` (the default ``module``) or ``natlang_neuralese.tune.learned.bank_stage`` over an ``episodes`` pool."""
    return 'tune' in spec


def is_rl(spec):
    return not is_sft(spec) and not is_tune(spec)


def kind(spec):
    return 'sft' if is_sft(spec) else 'tune' if is_tune(spec) else 'rl'


def sandbox_command(command):
    """A DGX controller command without its host-only prefixes: the memory-ledger admission wrapper
    (``[python, WRAPPER.py, '--', ...]`` whose source adopts a ledger unit and sets DGX unified-memory/NVMe controls)
    and the DGX GPU update lock. The sandbox owns its own equivalents (lfm3_rl_lineage.sh: checkpoint floor, discrete
    CUDA; one lineage per GPU needs no lock)."""
    command = list(command)
    while True:
        if (len(command) > 3 and command[2] == '--' and command[1].endswith('.py') and Path(command[1]).is_file()
                and 'memory_ledger.py' in Path(command[1]).read_text()):
            command = command[3:]
        elif command and command[0].endswith('/update_lock.sh'):
            command = command[1:]
        else:
            return command


def sandbox_controls(config, out):
    """The sandbox form of a controller config: host-only command prefixes dropped, the resident source plan beside
    its own controller, and no DGX llama.cpp build (an aarch64 tree absent from the sandbox)."""
    config = copy.deepcopy(config)
    for key in ('learner_command', 'seed_command', 'before_update', 'after_update'):
        if config.get(key):
            config[key] = sandbox_command(config[key])
    for key, value in list(config.items()):
        if isinstance(value, list) and value and any(isinstance(x, str) and x.endswith('/update_lock.sh') for x in value):
            raise ValueError(f'{key} still names the DGX update lock')
    if 'llama_cpp_converter' in config:
        config['llama_cpp_converter'] = None
    if config.get('resident_data_directory'):
        resident = str(Path(out) / 'resident-data')
        config['resident_data_directory'] = resident
        for key in ('trainer_parameters', 'seed_trainer_parameters'):
            if isinstance(config.get(key), dict) and 'resident_data_directory' in config[key]:
                config[key]['resident_data_directory'] = resident
    return config


def cloud_config(run, name):
    """An RL lineage's cloud config from its DGX template: paths moved to ROOT/<name> and ROOT/runtime (built in the
    sandbox from the bundle's code commit), host-only controls dropped (sandbox_controls), the parent checkpoint at
    ROOT/<name>/parent/checkpoint.pt, then the lineage's overrides."""
    spec = run['lineages'][name]
    template = json.loads(Path(spec['template']).read_text())
    old_out = template['out']
    runtime = str(Path(run['template_runtime']))
    cloud_runtime = str(Path(run['root']) / 'runtime')
    runtime_path = re.compile(re.escape(runtime) + r'(?=$|[/:])')
    def rebind(value):
        if isinstance(value, str):
            return runtime_path.sub(lambda _: cloud_runtime, value)
        if isinstance(value, list):
            return [rebind(item) for item in value]
        if isinstance(value, dict):
            return {key: rebind(item) for key, item in value.items()}
        return value
    text = json.dumps(rebind(template))
    text = text.replace(old_out + '/', str(lineage_dir(run, name)) + '/').replace(f'"{old_out}"', json.dumps(str(lineage_dir(run, name))))
    config = sandbox_controls(json.loads(text), lineage_dir(run, name))
    config['out'] = str(lineage_dir(run, name))
    config['checkpoint'] = str(lineage_dir(run, name) / 'parent' / 'checkpoint.pt')
    for key, value in spec.get('config', {}).items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    return config


def _tar_zst_put(run, base, members, name):
    """A deterministic tar.zst of ``members`` (relative to ``base``) as a committed blob of this run."""
    work = hub(run) / 'staging'
    work.mkdir(parents=True, exist_ok=True)
    archive = work / f'.{name}.tar.zst'
    listing = work / f'.{name}.members'
    listing.write_text(''.join(f'{m}\n' for m in sorted(members)))
    subprocess.run(['tar', '--sort=name', '--mtime=@0', '--owner=0', '--group=0', '--numeric-owner', '--no-recursion',
                    '-C', str(base), '-cf', str(archive.with_suffix('')), '-T', str(listing)], check=True)
    subprocess.run(['zstd', '-q', '-f', '-T0', '-3', '--rm', str(archive.with_suffix('')), '-o', str(archive)], check=True)
    digest, size = put_local(run, archive)
    archive.unlink()
    listing.unlink()
    return digest, size


def runtime_tool(runtime, *args):
    """scripts/neuralese_data/sft_window.py under the lineage's frozen runtime package (its own resolvers and
    continuation_config), as the runtime's frozen launches run (workdir, NATLANG_REPO and PYTHONPATH bound)."""
    runtime = str(runtime)
    env = {**os.environ, 'NATLANG_REPO': runtime, 'PYTHONPATH': f'{runtime}/training/neuralese:{runtime}',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    learner_python = REPO / '.venv-neuralese' / 'bin' / 'python'  # the DGX learner environment
    result = subprocess.run([str(learner_python), str(REPO / 'scripts' / 'neuralese_data' / 'sft_window.py'), *args],
                            cwd=runtime, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'sft_window.py {args[0]} failed: {result.stderr[-2000:]}')
    return json.loads(result.stdout.strip().splitlines()[-1])


def controller_successors(control):
    """Sibling controls (``OPS/*/config.json``) that continue ``control`` (``branch_origin.prior_control`` or a rolling
    retry's ``preparation.json`` ``prior_control``): the DGX has moved on from it."""
    control = Path(control).resolve()
    out = []
    for candidate in sorted(control.parent.parent.glob('*/config.json')):
        if candidate.resolve() == control:
            continue
        refs = []
        try:
            refs.append(((json.loads(candidate.read_text()).get('branch_origin') or {}).get('prior_control') or {}).get('path'))
            preparation = candidate.parent / 'preparation.json'
            if preparation.is_file():
                refs.append((json.loads(preparation.read_text()).get('prior_control') or {}).get('path'))
        except (ValueError, OSError):
            continue
        if any(r and Path(r).resolve() == control for r in refs):
            out.append(str(candidate))
    return out


def select_boundary(spec):
    """The named DGX controller's newest earned boundary, read now (the DGX learner keeps advancing; a run file never
    pins a stale parent). A completed window continues as the next window. A window still training (or failed) offers
    only its own parent: that window is rebuilt from its prior control's completed boundary (the same parent, recipe
    and window length), exactly as the DGX prepared it. Either way window 0 comes from ``sft_window.py next``."""
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    control = Path(spec['controller'])
    config = json.loads(control.read_text())
    if not config.get('mandatory_sft_only'):
        raise SystemExit(f'{control} is not a mandatory supervised controller')
    successors = controller_successors(control)
    if successors:
        raise SystemExit(f'{control} has successor controls {successors}: name the lineage\'s current window')
    state = read_json(Path(config['out']) / 'state.json', {})
    window = spec.get('window')
    if state.get('phase') == 'awaiting_semantic_refresh' and state.get('pending_native_checkpoint'):
        mode, previous, parent = 'next-window', control, Path(state['pending_native_checkpoint'])
    else:
        # A window still training, failed or retried in its rolling slot (a rolling control keeps the window's
        # branch origin): only its parent is a completed boundary the cloud can rebuild the window from.
        prior = ((config.get('branch_origin') or {}).get('prior_control') or {}).get('path')
        if not prior:
            raise SystemExit(f'{control} has no prior control to rebuild its window from')
        previous, parent, mode = Path(prior), Path(config['checkpoint']), 'same-window'
        prior_state = read_json(Path(json.loads(previous.read_text())['out']) / 'state.json', {})
        if (prior_state.get('phase') != 'awaiting_semantic_refresh'
                or Path(prior_state.get('pending_native_checkpoint', '')).resolve() != parent.resolve()):
            raise SystemExit(f'{previous} is not the completed boundary {control} continues')
        if window is None:
            recipe = json.loads(Path(config['production_recipe']).read_text())
            stage = next(s for s in recipe['stages'] if s['id'] == config['production_stage'])
            window = stage['parameters'].get('refresh_after_updates')
    parent = parent.resolve(strict=True)
    sha = sha256_file_hex_cached(parent)
    expected = (config.get('branch_origin') or {}).get('parent', {}).get('sha256')
    if mode == 'same-window' and expected and expected != sha:
        raise SystemExit(f'{control}: parent sha256 {sha} differs from its branch origin {expected}')
    in_flight = None
    if mode == 'same-window':
        slot = Path(config['out']) / 'learner' / 'checkpoint.pt'
        rolling = config.get('rolling_sft_resume') or {}
        in_flight = {'phase': state.get('phase'),
                     'rolling_slot': str(slot) if slot.is_file() and slot.resolve() != parent.resolve() else None,
                     'rolling_slot_global_step': (read_json(control.parent / 'preparation.json', {}) or {}).get('slot_global_step')
                     if rolling else None,
                     'note': 'a cloud rebuild of this window forks from the DGX window and does not carry its in-flight '
                             'or rolling progress; seed requires lineages.NAME.rebuild_window: true'}
    return {'mode': mode, 'controller': str(control), 'controller_sha256': file_digest(control),
            'phase': state.get('phase'), 'dgx_window_in_flight': in_flight,
            'previous_controller': str(previous), 'window': window,
            'parent': str(parent), 'parent_sha256': sha, 'parent_bytes': parent.stat().st_size,
            'runtime': (config.get('actor_environment') or {}).get('NATLANG_REPO') or str(Path(config['runtime']).parent)}


def checkpoint_bases(run):
    return {b['sha256']: b for b in run.get('checkpoint_bases', [])}


def base_specs(run, digest, out=None):
    """The exact base chain of ``digest`` (raw full checkpoint or ZIP delta over another base), each as a spec."""
    out = {} if out is None else out
    spec = checkpoint_bases(run)[digest]
    out[digest] = {k: v for k, v in spec.items() if k != 'path'}
    if spec['encoding'] == 'zip-delta':
        base_specs(run, spec['base'], out)
    return out


def put_checkpoint(run, source, target_path):
    """The checkpoint blob of a seed: an exact ZIP delta over the nearest declared full base whose bytes are on the
    DGX and whose whole chain is in a readable store (the sandbox reconstructs it at cloud bandwidth), when the delta
    is materially smaller; else the raw file. The delta is decoded once on the DGX before upload."""
    import artifact_zip_delta
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    source = Path(source).resolve(strict=True)
    sha, size = sha256_file_hex_cached(source), source.stat().st_size
    found = locate(run, sha)
    if found is not None and found[1].get('bytes') == size:
        # The exact bytes are already a committed blob (e.g. the boundary a cloud learner exported itself, uploaded
        # at cloud bandwidth): a new worker fetches them directly. Building and uploading a delta from the DGX
        # (2026-11-04: 22 min build + 5.9 GB at ~6 MB/s) only delayed the first DiLoCo session by ~1.5 h.
        return {'sha256': sha, 'bytes': size, 'encoding': 'raw', 'kind': 'checkpoint', 'path': str(target_path)}
    declared = checkpoint_bases(run).get(sha)
    if declared is not None and declared['bytes'] == size:
        chain = base_specs(run, sha)
        if all(locate(run, d) is not None for d, spec in chain.items()
               for d in ([d] if spec['encoding'] == 'raw' else [spec['receipt']['sha256'], spec['payload']['sha256']])):
            # the parent itself is a declared exact base (raw or a delta chain) already in a readable store
            return {**{k: v for k, v in declared.items() if k not in ('path', 'source')}, 'kind': 'checkpoint',
                    'path': str(target_path)}
    for base in run.get('checkpoint_bases', []):
        local = base.get('path')
        if not local or not Path(local).is_file():
            continue
        chain = base_specs(run, base['sha256'])
        absent = [d for d, s in chain.items() for d in ([d] if s['encoding'] == 'raw' else [s['receipt']['sha256'], s['payload']['sha256']])
                  if locate(run, d) is None]
        if absent:
            print(json.dumps({'event': 'base_unusable', 'base': base['sha256'], 'absent_blobs': absent}), flush=True)
            continue
        directory = hub(run) / 'deltas' / f'{base["sha256"][:16]}-{sha}'
        receipt = directory / 'receipt.json'
        if not receipt.is_file():
            started = time.time()
            built = artifact_zip_delta.build(local, source, directory)
            print(json.dumps({'event': 'delta_built', 'base': base['sha256'], 'target': sha,
                              'payload_bytes': built['payload']['bytes'], 'seconds': round(time.time() - started, 1)}),
                  flush=True)
        value = json.loads(receipt.read_text())
        if value['target'] != {'sha256': sha, 'bytes': size} or value['base']['sha256'] != base['sha256']:
            raise RuntimeError(f'{receipt} binds another base or target')
        if value['payload']['bytes'] > 0.8 * size:
            print(json.dumps({'event': 'delta_not_smaller', 'payload_bytes': value['payload']['bytes']}), flush=True)
            break
        receipt_sha = file_digest(receipt)
        artifact_zip_delta.verify(local, receipt, expected_sha256=sha, expected_bytes=size,
                                  expected_manifest_sha256=receipt_sha)
        payload = directory / value['payload']['path']
        put_local(run, receipt)
        put_local(run, payload)
        return {'kind': 'checkpoint', 'encoding': 'zip-delta', 'sha256': sha, 'bytes': size, 'path': str(target_path),
                'base': base['sha256'], 'receipt': {'sha256': receipt_sha, 'bytes': receipt.stat().st_size},
                'payload': {'sha256': value['payload']['sha256'], 'bytes': value['payload']['bytes']}}
    digest, size = put_local(run, source)
    return {'kind': 'checkpoint', 'encoding': 'raw', 'sha256': digest, 'bytes': size, 'path': str(target_path)}


def lineage_inputs(run, name, closure):
    """The run inputs of an SFT lineage: its frozen runtime archive, every closure file already committed in a readable
    store as an exact ``file`` input (no DGX upload), the rest packed once as one ``archive`` of absolute paths."""
    inputs = []
    spec = run['runtime_archive']
    runtime, archive = Path(spec['runtime']), Path(spec['archive'])
    if str(runtime) != closure['runtime']:
        raise SystemExit(f'{name}: closure runtime {closure["runtime"]} is not the run runtime {runtime}')
    if file_digest(runtime / 'runtime-input-closure.json') != spec['closure_sha256']:
        raise SystemExit(f'{runtime}: protected-input closure differs from the declared frozen runtime')
    found = locate(run, spec['sha256'])
    if found is None:
        if file_digest(archive) != spec['sha256']:
            raise SystemExit(f'{archive}: not the declared runtime archive')
        put_local(run, archive)
    inputs.append({'kind': 'runtime', 'path': str(runtime), 'sha256': spec['sha256'], 'bytes': spec['bytes'],
                   'closure_sha256': spec['closure_sha256']})
    return inputs + file_inputs(run, name, closure['files'])


# Host system trees are the sandbox's own (another architecture and distribution): never a run input.
SYSTEM_PREFIXES = tuple(f'/{name}/' for name in ('usr', 'etc', 'proc', 'sys', 'dev', 'tmp', 'var', 'run', 'lib', 'lib32',
                                                 'lib64', 'libx32', 'bin', 'sbin', 'opt', 'snap', 'boot', 'root'))


def file_inputs(run, name, files):
    """Exact files at their DGX paths: committed blobs in a readable store as ``file`` inputs, the rest packed once."""
    system = [item['path'] for item in files if item['path'].startswith(SYSTEM_PREFIXES)
              or '/.venv' in item['path'] or '/site-packages/' in item['path']]
    if system:
        raise SystemExit(f'{name}: closure names host system or Python-environment files as inputs: {system[:5]}')
    inputs, packed = [], []
    for item in files:
        found = locate(run, item['sha256'])
        if found is not None and found[1]['bytes'] == item['bytes']:
            inputs.append({'kind': 'file', 'path': item['path'], 'sha256': item['sha256'], 'bytes': item['bytes']})
        else:
            packed.append(item)
    if packed:
        digest, size = _tar_zst_put(run, '/', [p['path'].lstrip('/') for p in packed], f'{name}-closure')
        inputs.append({'kind': 'archive', 'path': f'{run["root"]}/.inputs/{name}-closure', 'sha256': digest,
                       'bytes': size, 'files': [{'path': p['path'], 'sha256': p['sha256']} for p in packed]})
    return inputs


def _main_python(code, *args):
    """``code`` in the DGX learner environment with this checkout's package (tune shards run main's code)."""
    env = {**os.environ, 'NATLANG_REPO': str(REPO), 'PYTHONPATH': f'{REPO}/training/neuralese:{REPO}',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    result = subprocess.run([str(REPO / '.venv-neuralese' / 'bin' / 'python'), '-c', code, *args], cwd=REPO, env=env,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'learner-environment step failed: {result.stderr[-2000:]}')
    return result.stdout


def custody_inputs():
    """The files ProtectedWorlds reads (common.custody) that the code snapshot does not carry (untracked packets and
    data-root source groups)."""
    listed = _main_python('import json\n'
                          'from natlang_neuralese.common.custody import _protected_source_paths\n'
                          'from natlang_neuralese.common.paths import root\n'
                          'repo = root("repo")\n'
                          'declared = json.loads((repo / "training/protected-worlds.json").read_text())\n'
                          'print(json.dumps([str(i[0]) for i in _protected_source_paths(declared, repo)]))').strip()
    out = []
    for path in json.loads(listed.splitlines()[-1]):
        relative = os.path.relpath(path, REPO)
        tracked = not relative.startswith('..') and subprocess.run(
            ['git', '-C', str(REPO), 'ls-files', '--error-unmatch', relative], capture_output=True).returncode == 0
        if not tracked and Path(path).is_file():
            out.append(path)
    return out


def _file_records(paths):
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    return [{'path': str(p), 'bytes': Path(p).stat().st_size, 'sha256': sha256_file_hex_cached(Path(p))}
            for p in sorted(set(map(str, paths)))]


def seed_tune(run, name):
    """Seed export of a tune shard: its stage command, the programs staged once on the DGX into the stage directory
    (``tune.stage.stage_programs``; the shards and the DGX merge share that directory and path), the program manifest
    and protection custody files as inputs, and the parent checkpoint as a declared exact base (no upload)."""
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    spec = run['lineages'][name]
    tune = spec['tune']
    exports = hub(run) / 'exports' / name
    if (exports / 'seed.json').exists():
        return read_json(exports / 'seed.json')
    L = lineage_dir(run, name)
    if L.exists():
        shutil.rmtree(L)  # only an interrupted seed's prepared command: nothing earned
    parent = Path(tune['parent']['path'])
    if sha256_file_hex_cached(parent.resolve(strict=True)) != tune['parent']['sha256']:
        raise SystemExit(f'{parent} is not the declared parent {tune["parent"]["sha256"]}')
    stage_dir = Path(tune['stage_dir'])
    module = tune.get('module', 'natlang_neuralese.tune.stage')
    if module == 'natlang_neuralese.tune.stage':
        programs = Path(tune['programs'])
        names = tune['program_names']
        _main_python('import json, sys\nfrom natlang_neuralese.tune.stage import stage_programs\n'
                     'stage_programs(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))',
                     str(programs), str(stage_dir), json.dumps(names))
        staged = [p for p in (stage_dir / 'programs').rglob('*') if p.is_file()]
        inputs = file_inputs(run, name, _file_records([programs, *staged, *custody_inputs()]))
        argv = ['--programs', str(programs), '--protection', str(REPO / 'training' / 'protected-worlds.json'),
                '--continue-from', str(parent), '--out', str(stage_dir), '--device', 'cuda', '--shard', tune['shard'],
                '--program-names', json.dumps(names), *tune.get('args', [])]
    elif module == 'natlang_neuralese.tune.learned.bank_stage':
        # The bank meta-learning stage: its episode pool files, the RL task source of its held task-outcome gate and
        # the protection custody as exact inputs. ``seed_stage`` also ships the stage directory as it stands (a DGX
        # run handed over to the cloud continues from its saved LoRA fits, learner state and store).
        names = []
        pool = Path(tune['episodes'])
        tasks = [Path(tune['task_source'])] if tune.get('task_source') else []
        carried = [p for p in stage_dir.rglob('*') if p.is_file()] if tune.get('seed_stage') else []
        inputs = file_inputs(run, name, _file_records([pool, pool.parent / 'rows.jsonl', *tasks, *carried,
                                                       *custody_inputs()]))
        argv = ['--episodes', str(pool), *(['--task-source', str(tasks[0])] if tasks else []),
                '--protection', str(REPO / 'training' / 'protected-worlds.json'),
                '--continue-from', str(parent), '--out', str(stage_dir), '--device', 'cuda', *tune.get('args', [])]
    elif module == SGLANG_BENCH:
        # SGLang actor throughput and actor/learner mismatch on real RL turns (serve/sglang_bench.py): the fork's venv,
        # the HF export of the parent and its server in the sandbox; reports and samples in the stage directory.
        names = []
        workload = Path(tune['workload'])
        inputs = file_inputs(run, name, _file_records([workload]))
        argv = ['all', '--out', str(stage_dir), '--workload', str(workload), '--checkpoint', str(parent),
                '--base', run['hf'].split('=', 1)[1], '--export-dir', str(stage_dir.parent / 'sglang-hf'),
                *tune.get('args', [])]
    elif module == DILOCO_ROLE:
        # One DiLoCo role (train/diloco/cloud_role.py): the coordinator (CPU sandbox, service port) or one learner. The
        # run file gives the role's exact argv; its declared input files (training/held subsets, recipe controls) ship
        # as exact inputs. Secrets (bearer token, coordinator URL and share token) arrive at launch, never in argv.
        names = []
        role = (tune.get('diloco') or {}).get('role')
        if role not in ('coordinator', 'learner') or not tune.get('args') or tune['args'][0] != role:
            raise SystemExit(f'{name}: a DiLoCo role names diloco.role and an argv starting with that role')
        inputs = file_inputs(run, name, _file_records([Path(p) for p in tune.get('files', [])]))
        argv = list(tune['args'])
    else:
        raise SystemExit(f'{name}: no cloud stage shard for module {module}')
    L.mkdir(parents=True)
    store_json(L / 'tune.json', {'schema': 'natlang.cloud-tune-shard/1', 'module': module,
                                 'argv': argv, 'stage_dir': str(stage_dir), 'parent': tune['parent'],
                                 'shard': tune.get('shard'), 'program_names': names})
    checkpoint = put_checkpoint(run, parent, parent)
    top_sha, top_bytes = _tar_zst_put(run, L, ['tune.json'], f'{name}-seed-top')
    record = {'schema': EXPORT_SCHEMA, 'seq': 'seed', 'lineage': name, 'lineage_dir': str(L), 'round': 0,
              'trained_rounds': 0, 'created': time.time(), 'state_sha256': None, 'kind': 'tune',
              'blobs': [{'kind': 'top', 'sha256': top_sha, 'bytes': top_bytes, 'encoding': 'tar.zst'}, checkpoint],
              'checkpoint': str(parent), 'checkpoints': [str(parent)], 'links': [], 'inputs': inputs,
              'parent_source': tune['parent'], 'dgx_verified': True, 'cloud_committed': True}
    store_json(exports / 'seed.json', record)
    note(run, f'{name} stage shard seed ({module}): {tune.get("shard")} of {names} on {parent} ({tune["parent"]["sha256"][:12]}, '
              f'{checkpoint["encoding"]}); {len(inputs)} inputs; stage {stage_dir}')
    return record


def seed_sft(run, name):
    """Seed export of a joint-SFT lineage from the named DGX controller's newest earned boundary: window 0's sandbox
    control (relocated, host-only controls dropped), the lineage index, the exact input closure as inputs, and the
    parent checkpoint (an exact ZIP delta over a declared full base when smaller). Nothing on the DGX lineage changes."""
    spec = run['lineages'][name]
    exports = hub(run) / 'exports' / name
    if (exports / 'seed.json').exists():
        return read_json(exports / 'seed.json')
    L = lineage_dir(run, name)
    if L.exists() and any(L.iterdir()):
        if any(L.rglob('state.json')):
            raise SystemExit(f'{L} has a controller state; a seed is only for a new lineage')
        shutil.rmtree(L)  # an interrupted seed's prepared controls: nothing earned, rebuilt from the live boundary
    if spec.get('prepared_control'):
        return seed_prepared_sft(run, name, spec, exports, L)
    boundary = select_boundary(spec)
    if boundary['mode'] == 'same-window' and spec.get('rebuild_window') is not True:
        raise SystemExit(f'{name}: {spec["controller"]} is not a completed window ({boundary["phase"]}); seeding '
                         f'rebuilds it from {boundary["parent_sha256"][:12]} and forks from the DGX window: set '
                         f'rebuild_window: true to do that deliberately, or name the DGX window once it completes')
    window, parent_path = L / 'w000000', L / 'parent' / 'checkpoint.pt'
    ident = f'lfm3-cloud-{name}-w000000'
    prepared = window / 'dgx-prepared-config.json'
    args = ['next', '--previous-config', boundary['previous_controller'], '--out', str(window), '--config-out',
            str(prepared), '--checkpoint-path', str(parent_path), '--id', ident]
    if boundary['window']:
        args += ['--window', str(boundary['window'])]
    runtime_tool(boundary['runtime'], *args)
    config = sandbox_controls(read_json(prepared), window)
    for key, value in spec.get('config', {}).items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    store_json(window / 'config.json', config)
    store_json(L / 'lineage.json', {'schema': sandbox.LINEAGE_SCHEMA, 'origin': boundary,
                                    'windows': [{'index': 0, 'out': str(window), 'config': str(window / 'config.json')}]})
    closure_path = hub(run) / 'closures' / f'{name}.json'
    closure_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_tool(boundary['runtime'], 'closure', '--config', str(window / 'config.json'), '-o', str(closure_path),
                 *observed_reads(spec))
    closure = read_json(closure_path)
    allowed = set(spec.get('allow_missing', []))
    missing = [m['path'] for m in closure['missing'] if m['path'] not in allowed]
    if missing:
        raise SystemExit(f'{name}: inputs the control names are absent on the DGX: {missing[:20]}')
    closure['files'] += origin_referenced_files(run, spec, boundary['parent'], {f['path'] for f in closure['files']})
    inputs = lineage_inputs(run, name, closure)
    checkpoint = put_checkpoint(run, boundary['parent'], parent_path)
    historical, skipped, historical_links = historical_checkpoint_blobs(run, closure, parent_path)
    if skipped:
        print(json.dumps({'event': 'historical_checkpoints_not_shipped', 'items': skipped}), flush=True)
    members = ['lineage.json'] + [str(p.relative_to(L)) for p in sorted(L.rglob('*')) if p != L / 'lineage.json'
                                  and not str(p.relative_to(L)).startswith('parent')]
    top_sha, top_bytes = _tar_zst_put(run, L, members, f'{name}-seed-top')
    record = {'schema': EXPORT_SCHEMA, 'seq': 'seed', 'lineage': name, 'lineage_dir': str(L), 'round': 0,
              'trained_rounds': 0, 'created': time.time(), 'state_sha256': None, 'kind': 'sft',
              'blobs': [{'kind': 'top', 'sha256': top_sha, 'bytes': top_bytes, 'encoding': 'tar.zst'}, checkpoint,
                        *historical],
              'checkpoint': str(parent_path), 'checkpoints': [str(parent_path)], 'links': historical_links,
              'inputs': inputs,
              'closure': {'path': str(closure_path), 'sha256': file_digest(closure_path), 'files': len(closure['files']),
                          'bytes': closure['bytes'], 'excluded': len(closure['excluded'])},
              'parent_source': boundary, 'dgx_verified': True, 'cloud_committed': True}
    store_json(exports / 'seed.json', record)
    note(run, f'{name} seed export ({boundary["mode"]}): parent {boundary["parent"]} ({boundary["parent_sha256"][:12]}, '
              f'{checkpoint["encoding"]}) from {spec["controller"]}; {len(inputs)} inputs, closure '
              f'{closure["bytes"] / 1e9:.2f} GB')
    return record


def prepared_boundary(spec):
    """An exact window control prepared on the DGX but never started there (``"prepared_control": true``), e.g. a
    corrected-mix boundary whose source correction, objective continuation and focus only its own preparer applies:
    ``sft_window.py next`` from the prior control would rebuild a different (uncorrected) window. The cloud lineage
    becomes this control's only executor, so a control the DGX has started (or moved past) is refused."""
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    control = Path(spec['controller'])
    config = json.loads(control.read_text())
    if not config.get('mandatory_sft_only'):
        raise SystemExit(f'{control} is not a mandatory supervised controller')
    if (Path(config['out']) / 'state.json').exists():
        raise SystemExit(f'{control} has started on the DGX ({config["out"]}/state.json): name its completed boundary '
                         f'instead; a prepared control has one executor')
    successors = controller_successors(control)
    if successors:
        raise SystemExit(f'{control} has successor controls {successors}')
    parent = Path(config['checkpoint']).resolve(strict=True)
    sha = sha256_file_hex_cached(parent)
    expected = ((config.get('branch_origin') or {}).get('parent') or {}).get('sha256')
    if expected and expected != sha:
        raise SystemExit(f'{control}: parent sha256 {sha} differs from its branch origin {expected}')
    return {'mode': 'prepared-control', 'controller': str(control), 'controller_sha256': file_digest(control),
            'phase': None, 'dgx_window_in_flight': None, 'previous_controller': None, 'window': None,
            'parent': str(parent), 'parent_sha256': sha, 'parent_bytes': parent.stat().st_size,
            'runtime': (config.get('actor_environment') or {}).get('NATLANG_REPO') or str(Path(config['runtime']).parent)}


def relocate_control(config, window, parent_path, ident):
    """A prepared DGX control as window 0 of a cloud lineage: its output tree moved to ``window`` (every path under the
    old ``out``), the parent at the lineage's parent path, host-only controls dropped (sandbox_controls)."""
    old_out = str(config['out']).rstrip('/')
    text = json.dumps(config)
    text = text.replace(json.dumps(old_out), json.dumps(str(window))).replace(old_out + '/', str(window) + '/')
    config = sandbox_controls(json.loads(text), window)
    config['out'] = str(window)
    config['checkpoint'] = str(parent_path)
    config['id'] = ident
    config.pop('rolling_sft_resume', None)
    return config


def observed_reads(spec):
    """``--observed-reads TRACE`` when the lineage declares the strace of a CPU controller start of this exact window
    control (``observed_reads``): every regular file that start read becomes a closure member (sft_window.traced_reads)."""
    trace = spec.get('observed_reads')
    if not trace:
        return ()
    if not Path(trace).is_file():
        raise SystemExit(f'observed_reads trace {trace} is missing')
    return ('--observed-reads', str(trace))


def historical_checkpoint_blobs(run, closure, parent_path):
    """Historical full checkpoints the window's admission records bind (execution-transition verification re-hashes
    the exact parent/child bytes at controller start), shipped at their DGX paths as declared exact bases (ZIP-delta
    chains already in a readable store: no upload; the sandbox reconstructs each). Bound checkpoints that are not
    declared bases are reported, not uploaded."""
    declared = checkpoint_bases(run)
    blobs, skipped, links, shipped = [], [], [], set()
    for item in closure.get('historical_checkpoints', []):
        digest, path = item['sha256'], item['path']
        if path == str(parent_path):
            continue
        chain = base_specs(run, digest) if digest in declared else {}
        if not chain or not all(locate(run, d) is not None for d, spec in chain.items()
                                for d in ([d] if spec['encoding'] == 'raw' else [spec['receipt']['sha256'], spec['payload']['sha256']])):
            skipped.append(item)
            continue
        # A bound path that is a symlink on the DGX stays a symlink: verifiers compare resolved paths (two records of
        # one checkpoint through a link and its target must resolve to the same file, 2026-11-03 namespace replay).
        real = os.path.realpath(path)
        if real != path:
            links.append({'path': path, 'symlink_to': real})
        if real not in shipped:
            shipped.add(real)
            blobs.append({**{k: v for k, v in declared[digest].items() if k not in ('path', 'source')},
                          'kind': 'checkpoint', 'path': real})
    return blobs, skipped, links


def origin_referenced_files(run, spec, parent, have):
    """Files of a ``historical_from`` origin lineage that the parent checkpoint's saved training frame binds (capacity
    proofs, packed geometry, memory qualifications, the window recipe): the trainer re-reads them at restore, after the
    model load (2026-11-04 corrected-mix-d1: corrected-mix/w000000/learner/packed-capacity-geometry absent). Full
    checkpoints stay historical blobs."""
    import re
    origins = [str(lineage_dir(run, origin)) + '/' for origin in spec.get('historical_from', [])]
    if not origins:
        return []
    # The orchestration venv has no torch: read the frame with the learner environment (mmap, metadata only).
    text = subprocess.run([str(REPO / '.venv-neuralese' / 'bin' / 'python'), '-c',
                           'import json,sys,torch;s=torch.load(sys.argv[1],map_location="cpu",weights_only=False,'
                           'mmap=True);print(json.dumps(s.get("training"),default=lambda v:None))', str(parent)],
                          capture_output=True, text=True, check=True).stdout
    found = []
    for prefix in origins:
        for path in sorted(set(re.findall(re.escape(prefix) + r'[^"\\]*', text))):
            if path in have or not os.path.isfile(path) or path.endswith('.pt'):
                continue
            found.append({'path': path, 'sha256': file_digest(path), 'bytes': os.path.getsize(path),
                          'from': ['origin-lineage-reference']})
    return found


def seed_prepared_sft(run, name, spec, exports, L):
    """Seed export of a joint-SFT lineage whose window 0 is an exact DGX-prepared, never-started control."""
    boundary = prepared_boundary(spec)
    window, parent_path = L / 'w000000', L / 'parent' / 'checkpoint.pt'
    window.mkdir(parents=True)
    source = json.loads(Path(spec['controller']).read_text())
    store_json(window / 'dgx-prepared-config.json', source)
    config = relocate_control(source, window, parent_path, f'lfm3-cloud-{name}-w000000')
    for key, value in spec.get('config', {}).items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    store_json(window / 'config.json', config)
    store_json(L / 'lineage.json', {'schema': sandbox.LINEAGE_SCHEMA, 'origin': boundary,
                                    'windows': [{'index': 0, 'out': str(window), 'config': str(window / 'config.json')}]})
    closure_path = hub(run) / 'closures' / f'{name}.json'
    closure_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_tool(boundary['runtime'], 'closure', '--config', str(window / 'config.json'), '-o', str(closure_path),
                 *observed_reads(spec))
    closure = read_json(closure_path)
    allowed = set(spec.get('allow_missing', []))
    missing = [m['path'] for m in closure['missing'] if m['path'] not in allowed]
    if missing:
        raise SystemExit(f'{name}: inputs the control names are absent on the DGX: {missing[:20]}')
    closure['files'] += origin_referenced_files(run, spec, boundary['parent'], {f['path'] for f in closure['files']})
    inputs = lineage_inputs(run, name, closure)
    checkpoint = put_checkpoint(run, boundary['parent'], parent_path)
    historical, skipped, historical_links = historical_checkpoint_blobs(run, closure, parent_path)
    if skipped:
        print(json.dumps({'event': 'historical_checkpoints_not_shipped', 'items': skipped}), flush=True)
    # A continuation of another cloud lineage (an elastic DiLoCo worker cloned from its boundary) re-verifies the
    # parent's folded execution admission, whose historical checkpoints only that lineage's own seed named
    # (2026-11-04: corrected-mix-d1 failed on zero-sgd-source-adoption-67 absent). Ship them as that seed did.
    for origin in spec.get('historical_from', []):
        source = read_json(hub(run) / 'exports' / origin / 'seed.json')
        origin_parent = str(lineage_dir(run, origin) / 'parent' / 'checkpoint.pt')
        have = {b['path'] for b in historical}
        historical += [b for b in source['blobs'] if b.get('kind') == 'checkpoint' and b['path'] != origin_parent
                       and b['path'] not in have]
        historical_links += [x for x in source.get('links', []) if x not in historical_links]
    members = ['lineage.json'] + [str(p.relative_to(L)) for p in sorted(L.rglob('*')) if p != L / 'lineage.json'
                                  and not str(p.relative_to(L)).startswith('parent')]
    top_sha, top_bytes = _tar_zst_put(run, L, members, f'{name}-seed-top')
    record = {'schema': EXPORT_SCHEMA, 'seq': 'seed', 'lineage': name, 'lineage_dir': str(L), 'round': 0,
              'trained_rounds': 0, 'created': time.time(), 'state_sha256': None, 'kind': 'sft',
              'blobs': [{'kind': 'top', 'sha256': top_sha, 'bytes': top_bytes, 'encoding': 'tar.zst'}, checkpoint,
                        *historical],
              'checkpoint': str(parent_path), 'checkpoints': [str(parent_path)], 'links': historical_links,
              'inputs': inputs,
              'closure': {'path': str(closure_path), 'sha256': file_digest(closure_path), 'files': len(closure['files']),
                          'bytes': closure['bytes'], 'excluded': len(closure['excluded'])},
              'parent_source': boundary, 'dgx_verified': True, 'cloud_committed': True}
    store_json(exports / 'seed.json', record)
    note(run, f'{name} seed export (prepared control): parent {boundary["parent"]} ({boundary["parent_sha256"][:12]}, '
              f'{checkpoint["encoding"]}) from {spec["controller"]}; {len(inputs)} inputs, closure '
              f'{closure["bytes"] / 1e9:.2f} GB')
    return record


def seed(run, name):
    """Seed export (seq 'seed', DGX-verified by construction: these are the DGX's own bytes)."""
    spec = run['lineages'][name]
    if is_sft(spec):
        return seed_sft(run, name)
    if is_tune(spec):
        return seed_tune(run, name)
    exports = hub(run) / 'exports' / name
    if (exports / 'seed.json').exists():
        return read_json(exports / 'seed.json')
    L = lineage_dir(run, name)
    if (L / 'state.json').exists():
        raise SystemExit(f'{L} already has a state; a seed is only for a new lineage')
    config = cloud_config(run, name)
    checkpoint = Path(spec['parent']['checkpoint']).resolve()
    tokenizers = Path(spec['parent']['tokenizers'])
    with tempfile.TemporaryDirectory(dir=hub(run)) as temp:
        stage = Path(temp) / name
        (stage / 'actor-tokenizers').mkdir(parents=True)
        shutil.copytree(tokenizers, stage / 'actor-tokenizers' / tokenizers.name)
        (stage / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        top = Path(temp) / 'top.tar.zst'
        subprocess.run(f'tar -C {stage} -cf - config.json actor-tokenizers | zstd -q -3 -c > {top}', shell=True,
                       check=True)
        top_sha, top_bytes = put_local(run, top)
        cas = Path(run['root']) / '.cas'
        cas.mkdir(parents=True, exist_ok=True)
        shutil.copy(top, cas / top_sha)
    # The template control's exact input closure (recipe resolver, stage inputs, protected custody; the runtime's own
    # resolvers), so a current production control runs without hand-listed extras: files already committed in a
    # readable store are exact ``file`` inputs, the rest one packed archive.
    closure_path = hub(run) / 'closures' / f'{name}.json'
    closure_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_tool(run['template_runtime'], 'closure', '--config', spec['template'], '-o', str(closure_path))
    closure = read_json(closure_path)
    allowed = set(spec.get('allow_missing', []))
    missing = [m['path'] for m in closure['missing'] if m['path'] not in allowed]
    if missing:
        raise SystemExit(f'{name}: inputs the template names are absent on the DGX: {missing[:20]}')
    inputs = file_inputs(run, name, closure['files'])
    checkpoint_blob = put_checkpoint(run, checkpoint, L / 'parent' / 'checkpoint.pt')
    expected = spec['parent'].get('sha256')
    if expected and expected != checkpoint_blob['sha256']:
        raise SystemExit(f'parent checkpoint sha256 {checkpoint_blob["sha256"]} != declared {expected}')
    record = {'schema': EXPORT_SCHEMA, 'seq': 'seed', 'lineage': name, 'lineage_dir': str(L), 'round': 0,
              'trained_rounds': 0, 'created': time.time(), 'state_sha256': None,
              'blobs': [{'kind': 'top', 'sha256': top_sha, 'bytes': top_bytes, 'encoding': 'tar.zst'}, checkpoint_blob],
              'inputs': inputs, 'closure': {'path': str(closure_path), 'sha256': file_digest(closure_path),
                                            'files': len(closure['files']), 'bytes': closure['bytes'],
                                            'excluded': len(closure['excluded'])},
              'links': [], 'parent_source': str(checkpoint), 'dgx_verified': True, 'cloud_committed': True}
    store_json(exports / 'seed.json', record)
    note(run, f'{name} seed export: parent {checkpoint} ({checkpoint_blob["sha256"][:12]}, {checkpoint_blob["encoding"]}), {len(inputs)} inputs, config from {spec["template"]} with '
              f'overrides {json.dumps(spec.get("config", {}))}')
    return record


def order(record):
    """Export order within a lineage (the seed first)."""
    return -1 if record['seq'] == 'seed' else record['seq']


def newest_verified(run, name):
    exports = hub(run) / 'exports' / name
    records = [read_json(p) for p in exports.glob('*.json')]
    verified = [r for r in records if r and r.get('dgx_verified')]
    if not verified:
        return None
    return max(verified, key=order)


def lineage_seed_inputs(run, name):
    """The seed's inputs plus every frozen runtime the lineage adopted in place after its seed (add-runtime): a relaunch
    resumes the active window, whose control names its own runtime, not the seed's."""
    seeded = (read_json(hub(run) / 'exports' / name / 'seed.json', {}) or {}).get('inputs', [])
    added = read_json(hub(run) / 'exports' / name / 'added-runtimes.json', {}).get('inputs', [])
    return seeded + [i for i in added if i['sha256'] not in {s['sha256'] for s in seeded}]


def add_runtime(run, name, runtime, archive):
    """Record a frozen runtime a lineage adopted in place (a window-boundary runtime transition) as a lineage input:
    its archive is committed to the store and every later launch restores it beside the seed's runtime."""
    runtime, archive = Path(runtime).resolve(strict=True), Path(archive).resolve(strict=True)
    closure = runtime / 'runtime-input-closure.json'
    if not closure.is_file():
        raise SystemExit(f'{runtime} is not a frozen runtime (no runtime-input-closure.json)')
    digest, size = file_digest(archive), archive.stat().st_size
    if locate(run, digest) is None:
        put_local(run, archive)
    item = {'kind': 'runtime', 'path': str(runtime), 'sha256': digest, 'bytes': size,
            'closure_sha256': file_digest(closure)}
    path = hub(run) / 'exports' / name / 'added-runtimes.json'
    record = read_json(path, {}) or {'schema': 'natlang.cloud-lineage-added-runtimes/1', 'inputs': []}
    record['inputs'] = [i for i in record['inputs'] if i['path'] != item['path']] + [item]
    store_json(path, record)
    return item


def launch_blobs(run, record, name=None):
    """The blobs a launch from ``record`` materializes. A tune shard's exports carry its stage directory, never its
    read-only parent, so every launch of a shard also restores the seed's parent checkpoint."""
    out = list(record['blobs'])
    seed = read_json(hub(run) / 'exports' / (name or record['lineage']) / 'seed.json', {}) or {}
    if record.get('kind') == 'tune' and not any(b['kind'] == 'checkpoint' for b in out):
        out += [b for b in seed['blobs'] if b['kind'] == 'checkpoint']
    elif seed and seed is not record:
        # Historical checkpoints the seed shipped outside the lineage directory (execution-transition and source
        # adoption proofs the earned frame verifies at every trainer start): a lineage's own exports never carry them.
        # 2026-11-04: w0 113911e0 relaunched from export 13 failed on zero-sgd-source-adoption-67-full-v2-1102.
        lineage_dir = str(record.get('lineage_dir') or seed.get('lineage_dir') or '').rstrip('/') + '/'
        have = {b['path'] for b in out if 'path' in b}
        out += [b for b in seed.get('blobs', []) if b['kind'] == 'checkpoint' and b['path'] not in have
                and not b['path'].startswith(lineage_dir)]
    return out


def launch_links(run, record, name=None):
    """The record's links plus the seed's (symlinks/hardlinks the seed's historical checkpoints are reached through)."""
    seed = read_json(hub(run) / 'exports' / (name or record['lineage']) / 'seed.json', {}) or {}
    links = list(record.get('links', []))
    paths = {link['path'] for link in links}
    return links + [link for link in seed.get('links', []) if link['path'] not in paths]


def record_digests(run, record, name=None):
    """Every blob a launch from ``record`` reads: its blobs (and a delta's receipt, payload and base chain) and the
    lineage's and run's inputs."""
    out = set()
    for blob in launch_blobs(run, record, name):
        out.add(blob['sha256'])
        if blob['encoding'] == 'zip-delta':
            out |= {blob['receipt']['sha256'], blob['payload']['sha256']}
            for digest, spec in base_specs(run, blob['base']).items():
                out |= ({digest} if spec['encoding'] == 'raw' else {spec['receipt']['sha256'], spec['payload']['sha256']})
    inputs = read_json(hub(run) / 'run-state.json', {}).get('inputs', []) + lineage_seed_inputs(run, name or record['lineage'])
    return out | {i['sha256'] for i in inputs}


def _with_parts(run, value):
    return {**value, 'parts': part_urls(run, value['sha256'])}


def parent_capability(run, record):
    inputs = read_json(hub(run) / 'run-state.json', {}).get('inputs', []) + lineage_seed_inputs(run, record['lineage'])
    out_blobs, bases = [], {}
    for blob in launch_blobs(run, record):
        if blob['encoding'] == 'zip-delta':
            out_blobs.append({**blob, 'receipt': _with_parts(run, blob['receipt']),
                              'payload': _with_parts(run, blob['payload'])})
            for digest, spec in base_specs(run, blob['base']).items():
                bases[digest] = (_with_parts(run, spec) if spec['encoding'] == 'raw' else
                                 {**spec, 'receipt': _with_parts(run, spec['receipt']),
                                  'payload': _with_parts(run, spec['payload'])})
        else:
            out_blobs.append(_with_parts(run, blob))
    return {'schema': PARENT_SCHEMA, 'lineage_dir': record['lineage_dir'], 'seq': record['seq'],
            'trained_rounds': record['trained_rounds'], 'state_sha256': record.get('state_sha256'),
            'checkpoint': record.get('checkpoint'), 'blobs': out_blobs, 'bases': bases,
            'inputs': [_with_parts(run, i) for i in inputs], 'links': launch_links(run, record)}


# -- blob retention (shared lease namespace: retention_leases.py) --------------------------------------------------------

def _lease_paths(run, store_hub=None):
    h = store_hub or hub(run)
    return {'lock_path': h / 'standalone-cloud-retention.lock',
            'state_path': h / 'standalone-retention-consumer-leases.json',
            'retirement_state_path': h / 'rl-blob-retirement.json'}


def _store_hubs(run):
    return [hub(run)] + [s['hub'] for s in foreign_stores(run)]


def lease_blobs(run, consumer, digests, release=False):
    """Register (before any capability for them is issued) or release a consumer of exact blobs, in the namespace of
    the store that holds each (a foreign store's retention then keeps what this run's launches read)."""
    from retention_leases import change_lease
    from natlang_neuralese.common.jsonio import write_json_durable
    for digest in sorted(set(digests)):
        found = locate(run, digest)
        store_hub = found[0]['hub'] if found else hub(run)
        change_lease(**_lease_paths(run, store_hub), role=f'blob:{digest}', consumer_id=consumer,
                     identity={'sha256': digest, 'run': run['id']}, release=release, write_durable=write_json_durable)


def release_consumer(run, consumer):
    from retention_leases import change_lease
    from natlang_neuralese.common.jsonio import write_json_durable
    for store_hub in _store_hubs(run):
        paths = _lease_paths(run, store_hub)
        state = read_json(paths['state_path'], {'leases': []})
        for lease in [x for x in state['leases'] if x['consumer_id'] == consumer]:
            change_lease(**paths, role=lease['role'], consumer_id=consumer, identity=lease['identity'],
                         release=True, write_durable=write_json_durable)


BLOB_GRACE_SECONDS = 24 * 3600


def retained_blobs(run):
    """Every lineage's newest DGX-verified export, anything newer (in flight), its seed (the fresh-init parent and
    the seeded config), the current and cached run inputs, and every blob committed within the grace window."""
    keep = {i['sha256'] for i in read_json(hub(run) / 'run-state.json', {}).get('inputs', [])}
    # an input archive is committed before the bundle that references it is stored: the inputs cache covers that gap
    keep |= {i['sha256'] for i in read_json(hub(run) / 'inputs.json', {}).values()}
    # any blob committed within the grace window may still be on its way into a record (bundle, seed, export)
    keep |= {k for k, v in blobs(run).items() if time.time() - v.get('committed', 0) < BLOB_GRACE_SECONDS}
    for lineage in run['lineages']:
        exports = hub(run) / 'exports' / lineage
        records = [r for r in (read_json(p) for p in exports.glob('*.json')) if r]
        newest = newest_verified(run, lineage)
        for record in records:
            if record['seq'] == 'seed' or newest is None or order(record) >= order(newest):
                keep |= record_digests(run, record, lineage)
    return keep


def prune_blobs(run, execute=True):
    """Retire superseded blobs in the private store: never a retained or leased one. Intent is durable before any
    delete; each part's absence is confirmed after its delete."""
    import fcntl
    import wandb
    from retention_leases import leased_roles
    paths = _lease_paths(run)
    candidates = sorted(set(blobs(run)) - retained_blobs(run))
    if not candidates or not execute:
        return candidates
    api = wandb.Api(api_key=_key())
    remote = api.run(blob_run(run))
    with paths['lock_path'].open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        leased = leased_roles(paths['state_path'])
        retirement = read_json(paths['retirement_state_path'], {'retired_roles': [], 'operations': []})
        table = read_json(hub(run) / 'blobs.json', {})
        keep = retained_blobs(run)  # re-read under the lock
        for digest in candidates:
            role = f'blob:{digest}'
            if role in leased or digest in keep or digest not in table:
                continue
            op = {'role': role, 'phase': 'intent', 'at': time.time(), 'parts': [p['name'] for p in table[digest]['parts']]}
            retirement['operations'].append(op)
            store_json(paths['retirement_state_path'], retirement)
            # deleting an absent file is an opaque internal error: delete only parts the listing shows
            for name in sorted({f.name for f in api.run(blob_run(run)).files()} & set(op['parts'])):
                remote.file(name).delete()
            # files(names=...) echoes requested names whether present or not: confirm against the full listing
            present = {f.name for f in api.run(blob_run(run)).files()}
            gone = not present & set(op['parts'])
            if not gone:
                raise RuntimeError(f'{role}: parts still present after delete')
            op.update(phase='deleted', deleted_at=time.time())
            retirement['retired_roles'].append(role)
            table.pop(digest, None)
            store_json(hub(run) / 'blobs.json', table)
            store_json(paths['retirement_state_path'], retirement)
            note(run, f'retired superseded blob {digest[:12]} ({len(op["parts"])} parts) from the private store')
    return candidates


# -- bundle ---------------------------------------------------------------------------------------------------------------

def fork_inputs(run, code_sha):
    """The fork source at the pinned commit (git archive) and its x86_64 lock (uv pip compile against the public
    indexes), staged under ROOT/inputs so the bundle materializes them at the same paths in the sandbox."""
    fork = json.loads(subprocess.run(['git', '-C', str(REPO), 'show', f'{code_sha}:training/neuralese/sglang-fork.json'],
                                     capture_output=True, text=True, check=True).stdout)
    commit = fork['commit']
    source = (REPO / fork['path']).resolve()
    inputs = Path(run['root']) / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    tar = inputs / f'sglang-natlang-{commit}.tar.gz'
    if not tar.exists():
        subprocess.run(f'git -C {source} archive --format=tar.gz -o {tar}.partial {commit} && mv {tar}.partial {tar}',
                       shell=True, check=True)
    lock = inputs / f'sglang-lock-{commit}-x86_64.txt'
    if not lock.exists():
        with tempfile.TemporaryDirectory() as temp:
            subprocess.run(f'git -C {source} archive {commit} python | tar -x -C {temp}', shell=True, check=True)
            subprocess.run(['uv', 'pip', 'compile', f'{temp}/python/pyproject.toml', '--python-version', '3.12',
                            '--python-platform', 'x86_64-manylinux_2_28', '--extra-index-url',
                            'https://download.pytorch.org/whl/cu130', '--index-strategy', 'unsafe-best-match',
                            '--no-header', '-o', f'{lock}.partial'], check=True)
        os.replace(f'{lock}.partial', lock)
    return [str(tar), str(lock)]


def runtime_runs(run):
    """The custody packets (runs/) of the DGX template runtime, which a runtime built in the sandbox from the code
    snapshot needs (RL lineages); a frozen runtime carries its own."""
    inputs = Path(run['root']) / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    runs = inputs / 'runtime-runs'
    target = Path(run['template_runtime']) / 'runs'
    if runs.is_symlink() and Path(os.readlink(runs)) != target:
        runs.unlink()  # the run moved to a newer template runtime (its held packets are the custody inputs)
    if not runs.exists():
        runs.symlink_to(target)
    return [str(runs)]


def _tree_key(path):
    """A cheap identity of a file or directory tree (relative path, size, mtime) to reuse its archive."""
    path = Path(path)
    items = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
    listing = [(str(p.relative_to(path.parent)), p.stat().st_size, p.stat().st_mtime_ns) for p in items]
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    return canonical_json_sha256_hex(listing)


def cloud_inputs(run):
    """Large run inputs (``run["cloud_inputs"]``: the task source, its assets) as tar.zst blobs in the private store,
    so a launch fetches them at cloud bandwidth (sha256 per part and whole) instead of pushing them over the DGX uplink.
    The DGX copies stay authoritative: an archive is rebuilt when its tree changes."""
    cache_path = hub(run) / 'inputs.json'
    cache = read_json(cache_path, {})
    out = []
    for path in run.get('cloud_inputs', []):
        path = Path(path)
        key = _tree_key(path)
        entry = cache.get(str(path))
        if not entry or entry['tree_key'] != key or entry['sha256'] not in blobs(run):
            work = hub(run) / 'inputs'
            work.mkdir(parents=True, exist_ok=True)
            archive = work / f'.{path.name}.tar.zst'
            subprocess.run(['tar', '--sort=name', '--mtime=@0', '--owner=0', '--group=0', '--numeric-owner', '-C',
                            str(path.parent), '-cf', str(archive.with_suffix('')), path.name], check=True)
            subprocess.run(['zstd', '-q', '-f', '-T0', '-3', '--rm', str(archive.with_suffix('')), '-o', str(archive)],
                           check=True)
            digest, size = put_local(run, archive)
            archive.unlink()
            entry = {'tree_key': key, 'sha256': digest, 'bytes': size}
            cache[str(path)] = entry
            store_json(cache_path, cache)
        out.append({'path': str(path), 'extract_to': str(path.parent), 'sha256': entry['sha256'],
                    'bytes': entry['bytes'], 'encoding': 'tar.zst'})
    return out


def needs_sglang(run):
    """Whether any lineage's actor runs the SGLang fork (RL lineages with ``--engine sglang``); SFT windows and native
    actors need none of its source, lock or venv."""
    for name, spec in run['lineages'].items():
        if is_tune(spec) and spec['tune'].get('module') == SGLANG_BENCH:
            return True
        if not is_rl(spec):
            continue
        if actor_uses_sglang(cloud_config(run, name)):
            return True
    return False


def actor_uses_sglang(config):
    """An RL config whose actor runs (or, through ``production.adopt_actor_engine_defaults`` at its next unbound
    collection, will run) the SGLang fork. Kept in step with the sandbox driver's copy in lfm3_rl_lineage.sh."""
    if config.get('mandatory_sft_only'):
        return False
    command = config.get('actor_command') or []
    engine = command[command.index('--engine') + 1] if '--engine' in command[:-1] else 'native'
    if engine == 'sglang':
        return True
    return bool(command) and engine == 'native' and not (
        config.get('actor_engine_keep_native') is True or config.get('coefficient_policy')
        or config.get('actor_engine', 'native-python') != 'native-python' or config.get('model') != 'lfm')


def bundle(run):
    code_sha = subprocess.run(['git', '-C', str(REPO), 'rev-parse', run.get('code', 'HEAD')], capture_output=True,
                              text=True, check=True).stdout.strip()
    inputs = cloud_inputs(run)
    remote = {i['path'] for i in inputs}
    covered = lambda p: any(p == r or p.startswith(r + '/') for r in remote)
    rl = any(is_rl(spec) for spec in run['lineages'].values())
    extras = ((fork_inputs(run, code_sha) if needs_sglang(run) else []) + (runtime_runs(run) if rl else [])
              + [e for e in run.get('extras', []) if not covered(e)])
    for spec in run['lineages'].values():
        if not is_rl(spec):
            continue
        cases = spec.get('config', {}).get('cases') or json.loads(Path(spec['template']).read_text())['cases']
        if cases not in extras and not covered(cases):
            extras.append(cases)
    out = hub(run) / 'bundles' / f'{code_sha[:12]}.json'
    command = [sys.executable, str(HERE / 'bundle.py'), 'plan', '--code-sha', code_sha, '--prebuilt', 'wheelhouse',
               '--hf', run['hf'], '-o', str(out)]
    for extra in extras:
        command += ['--extra', extra]
    subprocess.run(command, check=True)
    state = read_json(hub(run) / 'run-state.json', {})
    state['bundle'] = str(out)
    state['code_sha'] = code_sha
    state['inputs'] = inputs
    store_json(hub(run) / 'run-state.json', state)
    return out


# -- launch ---------------------------------------------------------------------------------------------------------------

def resource_amount(kind, value):
    """CPU cores or GiB of a Kubernetes-style quantity (cwctl._gib for memory)."""
    if kind == 'memory':
        return cwctl._gib(value)
    text = str(value)
    return float(text[:-1]) / 1000 if text.endswith('m') else float(text)


def render_job(run, name, rounds, check=False):
    spec = run['lineages'][name]
    text = Path(REPO / run['job']).read_text()
    for key, value in {'ROOT': run['root'], 'LINEAGE': name, 'ROUNDS': str(rounds),
                       'JOB_ID': f"{run['id']}-{name}-{'check-' if check else ''}{time.strftime('%m%d%H%M')}"}.items():
        text = text.replace(f'@@{key}@@', value)
    job = json.loads(text)
    # Exactly one GPU and explicit CPU/RAM minima per lineage (an RL lineage's actor and node workers need more than a
    # learner; AGENTS.md learner default 8 CPU / 64Gi).
    job['resources'] = {**job['resources'], **spec.get('resources', {}), 'gpu': 1}
    if diloco_role(spec) == 'coordinator':
        # The DiLoCo coordinator runs on a dedicated CPU-only sandbox (AGENTS.md: never cohosted with a learner) and
        # serves the learners through its public share-token gateway.
        job['resources'].pop('gpu')
        job['service'] = {'port': DILOCO_PORT}
        job['needs'] = {**job.get('needs', {}), 'gpu_memory_gb': 0}
    elif spec.get('service'):
        # A GPU learner that also cohosts an elastic DiLoCo coordinator (owner 2026-11-03, AGENTS 16a944ad1;
        # train/diloco/join.py starts it beside the learner): the learner job plus the coordinator's service port.
        job['service'] = {'port': DILOCO_PORT}
    if spec.get('minimum_resources'):  # the shape backoff floor (cwctl.create_with_shape_backoff)
        job['minimum_resources'] = spec['minimum_resources']
        above = [k for k in ('cpu', 'memory') if k in job['minimum_resources'] and k in job['resources']
                 and resource_amount(k, job['minimum_resources'][k]) > resource_amount(k, job['resources'][k])]
        if above:  # refused at plan time, never after a grant (the sandbox shape guard would end it at once)
            raise SystemExit(f'{name}: minimum_resources {above} exceed the lineage request {job["resources"]}')
    elif job.get('minimum_resources'):
        # The template floor belongs to the template request: a lineage that requests less keeps its own request as
        # the floor (2026-11-03: bankmeta-pca asked 40Gi under the 48Gi template floor and its granted GPU refused).
        job['minimum_resources'] = {k: (job['resources'][k] if k in job['resources'] and
                                        resource_amount(k, v) > resource_amount(k, job['resources'][k]) else v)
                                    for k, v in job['minimum_resources'].items()}
    if spec.get('allow_default_shape'):
        job['allow_default_shape'] = True
    floor = job.get('minimum_resources') or {}
    if floor:  # the sandbox refuses to start work on an assigned cgroup below the measured minimum
        job['env']['NATLANG_CW_MIN_CPU'] = str(floor.get('cpu', ''))
        job['env']['NATLANG_CW_MIN_MEMORY'] = str(floor.get('memory', ''))
    if spec.get('lifetime_hours'):  # a lineage with a declared GPU-hour ceiling (a bounded pilot shard)
        job['lifetime_hours'] = spec['lifetime_hours']
    if not is_sft(spec):  # RL lineages and tune shards run the code snapshot's ts-host
        job['env']['NATLANG_CW_BUILD_TSHOST'] = '1'
    if check:  # the CPU pipeline check: no GPU, small, short
        job['resources'] = {'cpu': '4', 'memory': '16Gi'}
        job.pop('minimum_resources', None)
        job.pop('allow_default_shape', None)
        job['env'].pop('NATLANG_CW_MIN_CPU', None)
        job['env'].pop('NATLANG_CW_MIN_MEMORY', None)
        job['lifetime_hours'] = 3
        job['kind'] = 'smoke'
        job['env']['NATLANG_RL_CHECK'] = '1'
        job['placement']['reason'] = 'CPU check of the cloud lineage pipeline without GPU work: ' + job['placement']['reason']
    return jobs.validate(job)


def started(sandbox_id):
    """Whether a sandbox whose start wait failed is nevertheless running (a transient poll error), re-read from the
    control plane; a startup timeout, failure or unreadable sandbox counts as not started."""
    try:
        status = str(cwctl.read_sandbox(sandbox_id, patience=120).status).lower()
    except Exception:
        return False
    return 'running' in status


def launch(run, name, account=None, dry_run=False, rounds=None, parent_later=False, check=False):
    current = load_run(run['_path']) if run.get('_path') else run
    if current.get('launch_hold'):
        raise ValueError(f'cloud launch held: {current["launch_hold"]}')
    state = read_json(hub(run) / 'run-state.json', {})
    if not state.get('bundle'):
        raise SystemExit('no bundle: rl_cloud.py bundle RUN.json first')
    live = state.get('lineages', {}).get(name, {})
    if live.get('sandbox_id') and live.get('status') == 'running':
        raise SystemExit(f'{name} already runs in {live["sandbox_id"]}: stop it first')
    record = newest_verified(run, name)
    if record is None and not parent_later:
        raise SystemExit(f'{name}: no verified export or seed; rl_cloud.py seed RUN.json {name}')
    record = record or {'seq': 'later', 'trained_rounds': None}
    job = render_job(run, name, run.get('rounds', 0) if rounds is None else rounds, check)
    account = account or run['lineages'][name].get('account')
    if account in (None, 'auto'):
        ledger = cwctl.account_ledger(write=False)
        busy = [v.get('account') for k, v in state.get('lineages', {}).items() if v.get('status') == 'running']
        from collections import Counter
        choice = cwctl.cwaccounts.choose_account(ledger, cwctl.cwaccounts.settings(), job['resources'],
                                                 float(job['lifetime_hours']), load=Counter(busy))
        account = choice['account']
        reason = choice['reason']
    else:
        reason = 'pinned'
    if dry_run:
        return {'dry_run': True, 'account': account, 'reason': reason, 'parent_seq': record['seq'],
                'trained_rounds': record['trained_rounds'], 'job': job['id'], 'resources': job['resources']}
    # A parent delivered later (deliver-parent) lets the sandbox build its environments while a seed still uploads.
    consumer = f"launch:{job['id']}"
    if not parent_later:
        lease_blobs(run, consumer, record_digests(run, record, name))
    secrets = {} if parent_later else {'rl-parent': json.dumps(parent_capability(run, record)).encode()}
    secrets.update(diloco_secrets(run, name, state))

    def created(sandbox_id):
        state.setdefault('lineages', {})[name] = {'sandbox_id': sandbox_id, 'account': account, 'status': 'running',
                                                  'launched': time.time(), 'job': job['id'],
                                                  'parent_seq': record['seq'], 'lease_consumer': consumer}
        store_json(hub(run) / 'run-state.json', state)
    def created_and_followed(sandbox_id):
        created(sandbox_id)
        follow_logs(run, sandbox_id)
    entry = cwctl.launch(job, False, True, None, Path(state['bundle']), secrets=secrets,
                         on_created=created_and_followed, account=account)
    if entry.get('wait_error') and not started(entry['sandbox_id']):
        # Never scheduled or failed before Running (2026-11-04: StartupTimeout "no node was assigned" was reported as
        # a grant): the lineage is not running; stop the record and fail the launch.
        subprocess.run([sys.executable, str(HERE / 'cwctl.py'), 'stop', entry['sandbox_id']], check=False)
        live = state['lineages'][name]
        live.update(status='failed', stopped=time.time(), error=entry['wait_error'])
        release_consumer(run, consumer)
        store_json(hub(run) / 'run-state.json', state)
        note(run, f'{name} sandbox {entry["sandbox_id"]} never started ({entry["wait_error"]}); launch failed')
        raise SystemExit(f'{name}: sandbox {entry["sandbox_id"]} never started: {entry["wait_error"]}')
    if entry.get('service_url'):
        state['lineages'][name]['service_url'] = entry['service_url']
        store_json(hub(run) / 'run-state.json', state)
    note(run, f'{name} launched in {entry["sandbox_id"]} ({account}: {reason}) from export {record["seq"]} '
              f'(trained_rounds {record["trained_rounds"]}), job {job["id"]}')
    return entry


def diloco_token(run, coordinator):
    """The DiLoCo run's bearer token: created once on the DGX (owner-only), delivered to every role as a secret."""
    path = hub(run) / 'diloco' / f'{coordinator}.token'
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        import secrets as token_source
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(token_source.token_urlsafe(32))
    return path.read_bytes().strip()


def diloco_secrets(run, name, state):
    """Secrets of a DiLoCo role: the bearer token; a learner also the running coordinator's URL and share token."""
    spec = run['lineages'][name]
    role = diloco_role(spec)
    if role is None:
        return {}
    if role == 'coordinator':
        return {'diloco-token': diloco_token(run, name)}
    coordinator = spec['tune']['diloco']['coordinator']
    live = state.get('lineages', {}).get(coordinator, {})
    if live.get('status') != 'running' or not live.get('service_url'):
        raise SystemExit(f'{name}: coordinator lineage {coordinator} is not running with a service URL')
    return {'diloco-token': diloco_token(run, coordinator), 'diloco-url': live['service_url'].encode(),
            'diloco-gateway': cwctl.share_token(live['sandbox_id'])}


def follow_logs(run, sandbox_id):
    """The sandbox's main-process log, followed into ROOT/.cloud/logs/<id>.log by a ledger service (a completed
    sandbox's log stream is gone, so it must be captured while it runs)."""
    logs = hub(run) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    cw = f'{sys.executable} {HERE / "cwctl.py"}'
    command = (f'while true; do {cw} logs {sandbox_id} --follow >> {logs / sandbox_id}.log 2>&1; '
               f'{cw} get {sandbox_id} 2>/dev/null | grep -qE "STATUS +(completed|failed|terminated)" && break; '
               f'sleep 20; done')
    subprocess.run(['/usr/bin/python3', str(REPO / 'scripts' / 'memory_ledger.py'), 'run', '--unit',
                    f'rl-cloud-logs-{sandbox_id[:8]}', '--budget-gb', '1', '--class', 'service', '--priority', 'medium',
                    '--wait', '300', '--detach', '--', '/bin/sh', '-c', command], capture_output=True)


def plan(run):
    """The launch plan of every lineage, without allocating or uploading anything: account (pinned, else spread over
    the eligible accounts with the plan's own load), resources and lifetime, and what the launch would start from (the
    newest verified export, else the boundary a seed would select now from the named DGX controller, with its base
    chain's availability in the readable stores)."""
    from collections import Counter
    ledger = cwctl.account_ledger(write=False)
    state = read_json(hub(run) / 'run-state.json', {})
    load = Counter(v.get('account') for v in state.get('lineages', {}).values() if v.get('status') == 'running')
    out = {'run': run['id'], 'root': run['root'], 'bundle': state.get('bundle'), 'launch_hold': run.get('launch_hold'),
           'lineages': {}}
    for name, spec in run['lineages'].items():
        job = render_job(run, name, run.get('rounds', 0))
        account = spec.get('account')
        if account in (None, 'auto'):
            choice = cwctl.cwaccounts.choose_account(ledger, cwctl.cwaccounts.settings(), job['resources'],
                                                     float(job['lifetime_hours']), load=load)
            account, reason = choice['account'], choice['reason']
        else:
            reason = 'pinned'
        load[account] += 1
        entry = {'kind': kind(spec), 'account': account, 'reason': reason,
                 'resources': job['resources'], 'lifetime_hours': job['lifetime_hours'], 'job': job['id']}
        record = newest_verified(run, name)
        if record is not None:
            entry['start'] = {'export': record['seq'], 'checkpoint': record.get('checkpoint'),
                              'phase': record.get('phase'), 'window': record.get('window')}
        elif is_tune(spec):
            entry['start'] = {'seed': 'not yet prepared', **spec['tune']}
        elif is_sft(spec):
            boundary = prepared_boundary(spec) if spec.get('prepared_control') else select_boundary(spec)
            entry['start'] = {'seed': 'not yet prepared', **boundary}
            entry['bases'] = {b['sha256']: {'local': bool(b.get('path') and Path(b['path']).is_file()),
                                            'chain_blobs_readable': all(
                                                locate(run, d) is not None for d, s in base_specs(run, b['sha256']).items()
                                                for d in ([d] if s['encoding'] == 'raw' else
                                                          [s['receipt']['sha256'], s['payload']['sha256']]))}
                              for b in run.get('checkpoint_bases', [])}
        else:
            entry['start'] = {'seed': 'not yet prepared', 'parent': spec['parent']}
        out['lineages'][name] = entry
    return out


def deliver_parent(run, name):
    """Write the rl-parent capability of the newest verified export into a running sandbox that waits for it."""
    state = read_json(hub(run) / 'run-state.json', {})
    live = state['lineages'][name]
    record = newest_verified(run, name)
    if record is None:
        raise SystemExit(f'{name}: nothing verified to deliver')
    box = cwctl.read_sandbox(live['sandbox_id'], patience=300)
    live['lease_consumer'] = live.get('lease_consumer') or f"launch:{live['job']}"
    lease_blobs(run, live['lease_consumer'], record_digests(run, record, name))
    secrets = f"{json.loads(Path(REPO / run['job']).read_text())['scratch']['path']}/.coreweave/secrets"
    capability = json.dumps(parent_capability(run, record)).encode()
    cwctl.cwaccounts.refuse_credentials({'rl-parent': capability})
    box.write_file(f'{secrets}/.rl-parent.tmp', capability).result(timeout=120)
    _exec(box, ['sh', '-c', f'chmod 600 {secrets}/.rl-parent.tmp && mv {secrets}/.rl-parent.tmp {secrets}/rl-parent'], 60)
    live['parent_seq'] = record['seq']
    store_json(hub(run) / 'run-state.json', state)
    note(run, f'{name}: parent export {record["seq"]} (trained_rounds {record["trained_rounds"]}) delivered to '
              f'{live["sandbox_id"]}')


def stop(run, name):
    state = read_json(hub(run) / 'run-state.json', {})
    live = state.get('lineages', {}).get(name)
    if not live:
        raise SystemExit(f'{name} has no sandbox')
    subprocess.run([sys.executable, str(HERE / 'cwctl.py'), 'stop', live['sandbox_id']], check=False)
    live['status'] = 'stopped'
    live['stopped'] = time.time()
    if live.get('lease_consumer'):
        release_consumer(run, live['lease_consumer'])
    store_json(hub(run) / 'run-state.json', state)
    note(run, f'{name} sandbox {live["sandbox_id"]} stopped')


# -- exporter -------------------------------------------------------------------------------------------------------------

def _exec(box, argv, timeout):
    result = box.exec(argv, timeout_seconds=timeout).result(timeout=timeout + 120)
    return result


def materialize(run, record):
    """The DGX hub copy of the lineage at its export: ROOT/<lineage> on DGX = the sandbox's lineage directory at that
    boundary (round archives, top files, parent checkpoint), so the DGX itself can resume it too."""
    L = Path(record['lineage_dir'])
    cas = Path(run['root']) / '.cas'
    L.mkdir(parents=True, exist_ok=True)
    for blob in record['blobs']:
        source = cas / blob['sha256']
        if blob['encoding'] == 'raw':
            target = Path(blob['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            if not (target.exists() and os.path.samefile(target, source)):
                temp = target.with_name(f'.{target.name}.link')
                if temp.exists():
                    temp.unlink()
                os.link(source, temp)
                temp.replace(target)
        elif blob['kind'] == 'stage':  # a tune shard's stage directory: the hub copy beside its record
            (L / 'stage').mkdir(parents=True, exist_ok=True)
            sandbox.untar_zst(source, L / 'stage')
        else:
            sandbox.untar_zst(source, L)
    for link in record.get('links', []):
        target = Path(link['path'])
        if target.exists() or target.is_symlink():
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        if 'symlink_to' in link:
            os.symlink(link['symlink_to'], target)
        else:
            os.link(link['same_as'], target)
    # The exported state is the active controller's (an SFT lineage's window directory, an RL lineage's root), as the
    # sandbox's own fetch checks it (2026-11-03: every corrected-mix export failed on <lineage>/state.json).
    state = sandbox.controller(L)[0] / 'state.json'
    if record.get('state_sha256') and file_digest(state) != record['state_sha256']:
        raise RuntimeError(f'{state} differs from export {record["seq"]}')


def export_lineage(run, name, live, log=print, get=True):
    """One exporter pass for one running lineage: newest staged export -> W&B (signed PUT from the sandbox) ->
    DGX (owner GET, verified) -> materialized hub copy; then mark it uploaded in the sandbox (prune may proceed)."""
    box = cwctl.read_sandbox(live['sandbox_id'], patience=300)
    E = f"{run['root']}/.exports/{name}"
    listing = _exec(box, ['sh', '-c', f'ls {E} 2>/dev/null'], 120).stdout.split()
    if 'parent-seq' in listing and live.get('lease_consumer'):  # the parent and inputs are fetched and verified
        release_consumer(run, live['lease_consumer'])
    staged = sorted(n for n in listing if re.fullmatch(r'\d{6}\.json', n))
    uploaded = {n.split('.')[0] for n in listing if n.endswith('.uploaded')}
    if not staged:
        return {'lineage': name, 'idle': 'nothing staged'}
    newest = staged[-1]
    seq = newest[:6]
    exports = hub(run) / 'exports' / name
    record = read_json(exports / f'{int(seq):06d}.json')
    if record is None:
        record = json.loads(box.read_file(f'{E}/{newest}').result(timeout=300))
        if record.get('schema') != EXPORT_SCHEMA or record.get('lineage') != name:
            raise RuntimeError(f'{name}: staged export {newest} is not this lineage\'s export')
        record.update(sandbox_id=live['sandbox_id'], account=live['account'], cloud_committed=False,
                      dgx_verified=False)
        store_json(exports / f'{int(seq):06d}.json', record)  # immutable intent before any byte moves
    if not record.get('cloud_committed'):
        todo = [(b, p) for b in record['blobs'] for p in missing_parts(run, b['sha256'], b['bytes'], b['parts'])]
        if todo:
            urls = upload_urls(run, [p['name'] for _, p in todo])
            items = [{'name': p['name'], 'sha256': p['sha256'], 'offset': p['offset'], 'bytes': p['bytes'],
                      'source': f"{E}/cas/{b['sha256']}", **urls[p['name']]} for b, p in todo]
            cap = f'{E}/.capability-{seq}.json'
            box.write_file(cap, json.dumps({'items': items}).encode()).result(timeout=120)
            _exec(box, ['chmod', '600', cap], 60)
            started = time.time()
            try:
                result = _exec(box, ['python3', str(REPO / SANDBOX_HELPER), 'upload', cap], 6 * 3600)
            finally:
                _exec(box, ['rm', '-f', cap], 60)
            events = [json.loads(l) for l in result.stdout.splitlines() if l.startswith('{')]
            ok = {e['name'] for e in events if e.get('event') == 'uploaded'}
            for blob in record['blobs']:
                names = [p['name'] for b, p in todo if b is blob and p['name'] in ok]
                if names:
                    commit_parts(run, blob['sha256'], blob['bytes'], blob['parts'], names)
            if result.returncode != 0 or ok != {p['name'] for _, p in todo}:
                raise RuntimeError(f'{name}: upload of export {seq} incomplete: {events[-3:]} {result.stderr[-500:]}')
            seconds = round(time.time() - started, 1)
            sent = sum(p['bytes'] for _, p in todo)
            record['upload'] = {'bytes': sent, 'parts': len(todo), 'seconds': seconds,
                                'mb_per_s': round(sent / 1e6 / max(seconds, 0.1), 1)}
        record['cloud_committed'] = True
        store_json(exports / f'{int(seq):06d}.json', record)
        box.write_file(f'{E}/{seq}.uploaded', b'').result(timeout=120)
        log(json.dumps({'event': 'cloud_committed', 'lineage': name, 'seq': seq,
                        'trained_rounds': record['trained_rounds'], **record.get('upload', {})}))
    elif seq not in uploaded:
        box.write_file(f'{E}/{seq}.uploaded', b'').result(timeout=120)
    if get:
        verify_newest(run, name)
        record = read_json(exports / f'{int(seq):06d}.json')
    return {'lineage': name, 'seq': seq, 'trained_rounds': record['trained_rounds'],
            'dgx_verified': bool(record.get('dgx_verified'))}



def verify_newest(run, name):
    """Owner GET of the newest cloud-committed export of a lineage that the DGX has not verified (W&B only; the
    sandbox may already be gone). Older unverified exports are superseded and skipped."""
    exports = hub(run) / 'exports' / name
    records = [r for r in (read_json(p) for p in exports.glob('[0-9]*.json')) if r and r.get('cloud_committed')]
    if not records:
        return None
    record = max(records, key=lambda r: r['seq'])
    if record.get('dgx_verified'):
        return record
    seq = record['seq']
    cas = Path(run['root']) / '.cas'
    cas.mkdir(parents=True, exist_ok=True)
    started, total = time.time(), 0
    for blob in record['blobs']:
        if not (cas / blob['sha256']).exists():
            get_blob(run, blob['sha256'], cas / blob['sha256'])
            total += blob['bytes']
    materialize(run, record)
    record['dgx_verified'] = True
    record['dgx_get'] = {'bytes': total, 'seconds': round(time.time() - started, 1), 'at': time.time()}
    store_json(exports / f'{int(seq):06d}.json', record)
    note(run, f'{name} export {seq} (trained_rounds {record["trained_rounds"]}) verified on DGX: '
              f'{total / 1e9:.2f} GB in {record["dgx_get"]["seconds"]} s; hub copy {record["lineage_dir"]}')
    prune_hub(run, name)
    return record

def prune_hub(run, name):
    """Keep the blobs of every lineage's newest verified export (and anything newer); drop older local CAS copies.
    W&B objects stay until a later retention pass (they are the cloud tier of the same exports)."""
    keep = set()
    for lineage in run['lineages']:
        exports = hub(run) / 'exports' / lineage
        records = [read_json(p) for p in exports.glob('*.json')]
        newest = newest_verified(run, lineage)
        for record in records:
            if record and (record is newest or (newest and order(record) >= order(newest))):
                keep |= {b['sha256'] for b in record['blobs']}
    for blob in (Path(run['root']) / '.cas').glob('[0-9a-f]' * 64):
        if blob.name not in keep and blob.stat().st_nlink == 1:
            blob.unlink()


def export_loop(run, once=False, interval=120):
    while True:
        state = read_json(hub(run) / 'run-state.json', {})
        # uploads first (cloud bandwidth, frees sandbox disk), then the slower DGX GETs
        for get in (False, True):
            for name, live in state.get('lineages', {}).items():
                try:
                    if get:
                        record = verify_newest(run, name)
                        print(json.dumps({'lineage': name, 'verified_seq': record and record['seq']}), flush=True)
                    elif live.get('status') == 'running':
                        print(json.dumps(export_lineage(run, name, live, get=False)), flush=True)
                except Exception as error:  # one lineage's failure must not stop the others' exports
                    print(json.dumps({'event': 'export_error', 'lineage': name,
                                      'error': f'{type(error).__name__}: {error}'[:800]}), flush=True)
        for name, live in read_json(hub(run) / 'run-state.json', {}).get('lineages', {}).items():
            record = newest_verified(run, name)
            if live.get('status') == 'running' and record and record.get('final'):
                try:
                    note(run, f'{name}: final export {record["seq"]} ({record.get("phase")}) verified on DGX; '
                              f'shutting down {live["sandbox_id"]}')
                    stop(run, name)
                except Exception as error:  # retried next pass
                    print(json.dumps({'event': 'stop_error', 'lineage': name,
                                      'error': f'{type(error).__name__}: {error}'[:500]}), flush=True)
        try:
            retired = prune_blobs(run)
            if retired:
                print(json.dumps({'event': 'blobs_retired', 'count': len(retired)}), flush=True)
        except Exception as error:  # retention must never stop exports
            print(json.dumps({'event': 'prune_error', 'error': f'{type(error).__name__}: {error}'[:800]}), flush=True)
        if once:
            return
        time.sleep(interval)


def status(run):
    state = read_json(hub(run) / 'run-state.json', {})
    out = {'bundle': state.get('bundle'), 'code_sha': state.get('code_sha'), 'lineages': {}}
    for name in run['lineages']:
        newest = newest_verified(run, name)
        out['lineages'][name] = {**state.get('lineages', {}).get(name, {}),
                                 'newest_verified': newest and {'seq': newest['seq'],
                                                                'trained_rounds': newest['trained_rounds']}}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('seed', 'launch', 'stop', 'deliver-parent'):
        p = sub.add_parser(command)
        p.add_argument('run')
        p.add_argument('lineage')
        if command == 'launch':
            p.add_argument('--account')
            p.add_argument('--rounds', type=int)
            p.add_argument('--dry-run', action='store_true')
            p.add_argument('--parent-later', action='store_true', help='deliver the parent with deliver-parent')
            p.add_argument('--check', action='store_true', help='CPU sandbox: pipeline check without GPU work')
    p = sub.add_parser('add-runtime', help='record a frozen runtime the lineage adopted in place (shipped on relaunch)')
    p.add_argument('run')
    p.add_argument('lineage')
    p.add_argument('--runtime', required=True)
    p.add_argument('--archive', required=True, help='its archive_local_recovery_runtime.py tar.zst')
    for command in ('bundle', 'status', 'export', 'prune', 'plan'):
        p = sub.add_parser(command)
        p.add_argument('run')
        if command == 'prune':
            p.add_argument('--execute', action='store_true', help='without it, list the blobs that would be retired')
        if command == 'export':
            p.add_argument('--once', action='store_true')
            p.add_argument('--interval', type=float, default=120)
    args = parser.parse_args(argv)
    run = load_run(args.run)
    if args.command == 'seed':
        print(json.dumps(seed(run, args.lineage), indent=1))
    elif args.command == 'bundle':
        print(bundle(run))
    elif args.command == 'launch':
        print(json.dumps(launch(run, args.lineage, args.account, args.dry_run, args.rounds, args.parent_later,
                                args.check),
                         indent=1, default=str))
    elif args.command == 'add-runtime':
        print(json.dumps(add_runtime(run, args.lineage, args.runtime, args.archive)))
    elif args.command == 'deliver-parent':
        deliver_parent(run, args.lineage)
    elif args.command == 'stop':
        stop(run, args.lineage)
    elif args.command == 'export':
        export_loop(run, args.once, args.interval)
    elif args.command == 'prune':
        print(json.dumps(prune_blobs(run, execute=args.execute), indent=1))
    elif args.command == 'plan':
        print(json.dumps(plan(run), indent=1, default=str))
    elif args.command == 'status':
        print(json.dumps(status(run), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
