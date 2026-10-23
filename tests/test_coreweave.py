import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'coreweave'))


def load(name):
    spec = importlib.util.spec_from_file_location(f'cw_{name}', ROOT / 'scripts' / 'coreweave' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


job = load('job')
bundle = load('bundle')
cwctl = load('cwctl')


def base_job(**changes):
    value = {'schema': job.SCHEMA, 'id': 'eval-one', 'kind': 'eval', 'placement': {'reason': 'DGX is full'},
             'image': 'nvcr.io/nvidia/pytorch:26.09-py3', 'resources': {'cpu': '8', 'memory': '32Gi', 'gpu': 1},
             'lifetime_hours': 2, 'command': ['python', '-c', 'print(1)']}
    value.update(changes)
    return value


@pytest.mark.parametrize('changes, message', [
    ({'resources': {'cpu': '8', 'memory': '32Gi', 'gpu': 2}}, 'exactly 1 GPU'),
    ({'env': {'WANDB_API_KEY': 'x'}}, 'must not carry credentials'),
    ({'lifetime_hours': 721}, 'at most 720'),
    ({'placement': {}}, 'placement.reason'),
    ({'resources': {'cpu': '8'}}, 'needs cpu and memory'),
    ({'resources': {'cpu': '8', 'memory': '32', 'gpu': 1}}, 'binary unit'),
    ({'id': 'Has Spaces'}, 'lower-case'),
    ({'bundle': 'bundles/x.json'}, 'set bucket'),
])
def test_job_validation_names_the_fix(changes, message):
    with pytest.raises(job.JobError, match=message):
        job.validate(base_job(**changes))


def test_multi_gpu_allowed_only_on_cks():
    assert job.validate(base_job(placement_mode='cks', resources={'cpu': '8', 'memory': '32Gi', 'gpu': 4}))


def test_sandbox_request_shape():
    request = job.sandbox_request(base_job(bucket='natlang-cw', bundle='bundles/abc.json', region='US-EAST-04A',
                                           outputs=['runs/x']), 'echo boot')
    assert request['args'] == [job.BOOTSTRAP_PATH, 'python', '-c', 'print(1)']
    assert request['resources'] == {'cpu': '8', 'memory': '32Gi', 'gpu': {'count': 1}}
    assert request['max_lifetime_seconds'] == 7200
    assert request['mounted_files'] == {job.BOOTSTRAP_PATH: 'echo boot'}
    assert request['object_storage_access'] == {'buckets': ['natlang-cw'], 'permission': 'read_write'}
    env = request['environment_variables']
    assert env['NATLANG_CW_BUNDLE'] == 'bundles/abc.json' and env['AWS_REGION'] == 'US-EAST-04A'
    assert env['NATLANG_CW_OUTPUTS'] == 'runs/x' and env['NATLANG_CW_OUTPUT_PREFIX'] == 'outputs/eval-one/'
    assert 'natlang' in request['tags'] and 'job-eval-one' in request['tags']
    assert 'max_timeout_seconds' not in request  # removed in cwsandbox 1.x (TypeError)


def test_cost_estimate_is_request_based_and_gpu_rate_unknown_by_default():
    cost = job.cost_estimate(base_job())
    assert cost['cpu_memory_per_hour'] == round(8 * 0.128 + 32 * 0.0212, 4)
    assert cost['gpu_rate_known'] is False and cost['max_total'] is None
    priced = job.cost_estimate(base_job(), gpu_hour_price=2.5)
    assert priced['max_total'] == round((8 * 0.128 + 32 * 0.0212 + 2.5) * 2, 2)


@pytest.mark.parametrize('resources', [{'gpu': 1}, {'gpu': 1, 'memory': '24Gi'}])
def test_one_gpu_request_keeps_only_declared_resource_filters(resources):
    declaration=base_job(resources=resources)
    request=job.sandbox_request(declaration,'echo boot')
    assert request['resources']=={**{k:v for k,v in resources.items() if k!='gpu'},'gpu':{'count':1}}
    cost=job.cost_estimate(declaration,gpu_hour_price=2.5)
    assert cost['gpu_rate_known'] and cost['gpu_per_hour']==2.5
    assert cost['cpu_memory_per_hour'] is None and not cost['cpu_memory_rate_known']
    assert cost['max_total'] is None  # Assigned CPU/RAM cannot be priced before allocation.


def test_snapshot_restore_is_explicit_and_requires_a_mounted_volume():
    scratch={'path':'/work','size':'200Gi','restore_from_snapshot_id':'retained-snapshot-id'}
    request=job.sandbox_request(base_job(resources={'gpu':1},scratch=scratch),'echo boot')
    assert request['file_system_snapshot']=={'name':'work','mount_path':'/work','size':'200Gi',
        'file_system_snapshot_id':'retained-snapshot-id'}
    assert 'volumes' not in request and request['resources']=={'gpu':{'count':1}}
    assert job.validate(job.validate(base_job(scratch=scratch)))['scratch']=={**scratch,'storage':'volume'}
    with pytest.raises(job.JobError,match='requires volume storage'):
        job.validate(base_job(scratch={**scratch,'storage':'root'}))
    for bad in ('', ' ', ' id ', 17):
        with pytest.raises(job.JobError,match='nonempty snapshot ID'):
            job.validate(base_job(scratch={**scratch,'restore_from_snapshot_id':bad}))


def test_root_work_path_omits_vendor_volume_without_changing_bootstrap_or_outputs():
    declaration = base_job(scratch={'path': '/work', 'size': '50Gi', 'storage': 'root'}, outputs=['runs/x'])
    request = job.sandbox_request(declaration, 'echo boot')
    assert 'volumes' not in request
    assert request['mounted_files'] == {job.BOOTSTRAP_PATH: 'echo boot'}
    assert request['environment_variables']['NATLANG_CW_WORK'] == '/work'
    assert request['environment_variables']['NATLANG_CW_OUTPUTS'] == 'runs/x'
    legacy = job.sandbox_request(base_job(), 'echo boot')
    assert legacy['volumes'] == [{'name': 'work', 'mount_path': '/work', 'size': '200Gi'}]
    with pytest.raises(job.JobError, match='scratch.storage'):
        job.validate(base_job(scratch={'path': '/work', 'size': '50Gi', 'storage': 'persistent'}))


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.modified = {}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            error = Exception('404')
            error.response = {'Error': {'Code': '404'}}
            raise error
        return {'ContentLength': len(self.objects[Key])}

    def upload_file(self, source, bucket, key):
        self.objects[key] = Path(source).read_bytes()

    def upload_fileobj(self, stream, bucket, key):
        self.objects[key] = stream.read()

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            error = Exception('NoSuchKey')
            error.response = {'Error': {'Code': 'NoSuchKey'}}
            raise error
        import io
        return {'Body': io.BytesIO(self.objects[Key])}

    def list_objects_v2(self, Bucket, Prefix='', ContinuationToken=None, MaxKeys=1000):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {'Contents': [{'Key': k, 'Size': len(self.objects[k]), 'LastModified': self.modified.get(k)}
                             for k in keys], 'IsTruncated': False}

    def delete_objects(self, Bucket, Delete):
        for item in Delete['Objects']:
            self.objects.pop(item['Key'], None)
        return {}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body

    def download_file(self, bucket, key, target):
        Path(target).write_bytes(self.objects[key])


def make_files(tmp_path):
    (tmp_path / 'runs' / 'a').mkdir(parents=True)
    (tmp_path / 'runs' / 'a' / 'x.jsonl').write_text('{"a": 1}\n')
    (tmp_path / 'runs' / 'a' / 'y.bin').write_bytes(b'\0' * 1000)
    (tmp_path / 'runs' / 'a' / 'same.bin').write_bytes(b'\0' * 1000)
    return bundle.extra_entries('runs/a', tmp_path)


def test_bundle_round_trip_dedups_and_verifies(tmp_path):
    pairs = make_files(tmp_path)
    manifest = bundle.build_manifest(pairs)
    bundle.check_manifest(manifest)
    assert [f['path'] for f in manifest['files']] == ['runs/a/same.bin', 'runs/a/x.jsonl', 'runs/a/y.bin']
    s3 = FakeS3()
    sources = {item['key']: source for item, source in pairs}
    stats = bundle.upload(manifest, sources, 'b', s3, log=lambda _: None)
    assert stats['uploaded'] == 2  # same.bin and y.bin share one content-addressed object
    assert bundle.upload(manifest, sources, 'b', s3, log=lambda _: None)['skipped'] == 2
    assert bundle.manifest_key(manifest) in s3.objects
    dest = tmp_path / 'dest'
    receipt = bundle.fetch(manifest, 'b', s3, dest, log=lambda _: None)
    assert receipt['fetched'] == 3 and receipt['verified']
    assert bundle.verify(manifest, dest) == []
    assert bundle.fetch(manifest, 'b', s3, dest, log=lambda _: None)['present'] == 3


def test_fetch_refuses_corrupt_object(tmp_path):
    pairs = make_files(tmp_path)
    manifest = bundle.build_manifest(pairs)
    s3 = FakeS3()
    bundle.upload(manifest, {item['key']: source for item, source in pairs}, 'b', s3, log=lambda _: None)
    key = next(f['key'] for f in manifest['files'] if f['path'].endswith('x.jsonl'))
    s3.objects[key] = b'{"a": 2}\n'
    with pytest.raises(bundle.BundleError, match='downloaded sha256'):
        bundle.fetch(manifest, 'b', s3, tmp_path / 'dest', log=lambda _: None)
    assert not (tmp_path / 'dest' / 'runs' / 'a' / 'x.jsonl').exists()


def test_tampered_manifest_is_refused(tmp_path):
    manifest = bundle.build_manifest(make_files(tmp_path))
    manifest['files'][0]['path'] = 'runs/a/other.bin'
    with pytest.raises(bundle.BundleError, match='does not match its content'):
        bundle.check_manifest(manifest)


def test_unsafe_paths_and_remap():
    with pytest.raises(bundle.BundleError):
        bundle._safe_relative('../etc/passwd')
    assert bundle.remap('/work/out/run-v4/x.pt', [('/work/out', '/data/cw')]) == '/data/cw/run-v4/x.pt'
    assert bundle.remap('/work/outer/x', [('/work/out', '/data')]) == '/work/outer/x'


def test_recipe_entries_use_pinned_hashes_and_stage_selection(tmp_path):
    data = tmp_path / 'runs' / 'd.jsonl'
    data.parent.mkdir(parents=True)
    data.write_text('x\n')
    other = tmp_path / 'runs' / 'e.jsonl'
    other.write_text('y\n')
    pinned = hashlib.sha256(b'x\n').hexdigest()
    recipe = tmp_path / 'recipe.json'
    recipe.write_text(json.dumps({'schema': 's', 'id': 'r', 'input_bindings': {
        'text': {'path': 'runs/d.jsonl', 'sha256': pinned}, 'other': 'runs/e.jsonl'},
        'stages': [{'id': 'one', 'inputs': {'text_data': 'text'}}, {'id': 'two', 'inputs': {'text_data': 'other'}}]}))
    entries = bundle.recipe_entries(recipe, ['one'], tmp_path)
    assert [(e['path'], e['sha256']) for e, _ in entries] == [('runs/d.jsonl', pinned)]
    assert len(bundle.recipe_entries(recipe, [], tmp_path)) == 2
    with pytest.raises(bundle.BundleError, match='no stage three'):
        bundle.recipe_entries(recipe, ['three'], tmp_path)


def test_directory_manifest_skips_space_reservations(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'checkpoint.pt').write_bytes(b'state')
    (out / '.checkpoint-space.reserve').write_bytes(b'\0' * 64)
    (out / 'checkpoint.pending').write_bytes(b'half')
    manifest, sources = bundle.directory_manifest(out)
    bundle.close_sources(sources)
    assert [Path(f['path']).name for f in manifest['files']] == ['checkpoint.pt']
    assert manifest['files'][0]['path'] == str(out / 'checkpoint.pt')


def test_env_file_must_be_private_and_never_overrides(tmp_path, monkeypatch):
    env = tmp_path / 'coreweave.env'
    env.write_text('export CWSANDBOX_API_KEY="secret-value"\nNATLANG_CW_BUCKET=natlang-cw\n')
    os.chmod(env, 0o644)
    with pytest.raises(cwctl.ControllerError, match='chmod 600'):
        cwctl.load_env_file(env)
    os.chmod(env, 0o600)
    monkeypatch.delenv('CWSANDBOX_API_KEY', raising=False)
    monkeypatch.setenv('NATLANG_CW_BUCKET', 'mine')
    assert cwctl.load_env_file(env) == ['CWSANDBOX_API_KEY']
    assert os.environ['NATLANG_CW_BUCKET'] == 'mine'
    assert cwctl.api_key_name() == 'CWSANDBOX_API_KEY'
    assert 'secret-value' not in json.dumps(cwctl.doctor())


def test_coreweave_api_token_alias(monkeypatch):
    monkeypatch.delenv('CWSANDBOX_API_KEY', raising=False)
    monkeypatch.setenv('COREWEAVE_API_TOKEN', 'tok')
    assert cwctl.api_key_name() == 'COREWEAVE_API_TOKEN'
    assert os.environ['CWSANDBOX_API_KEY'] == 'tok'


def test_dry_run_hides_bootstrap_content_and_needs_no_key(monkeypatch):
    monkeypatch.delenv('CWSANDBOX_API_KEY', raising=False)
    monkeypatch.delenv('COREWEAVE_API_TOKEN', raising=False)
    result = cwctl.launch(job.validate(cwctl.smoke_job(True)), True, False, None)
    files = result['request']['mounted_files']
    assert set(files[job.BOOTSTRAP_PATH]) == {'sha256', 'bytes'}
    with pytest.raises(cwctl.ControllerError, match='no credential'):
        cwctl.launch(job.validate(cwctl.smoke_job(True)), False, False, None)


def test_installed_client_accepts_the_request():
    pytest.importorskip('cwsandbox')
    wire = cwctl.wire_request(job.sandbox_request(base_job(bucket='natlang-cw'), 'echo boot'))
    assert isinstance(wire, dict), wire
    container = wire['sandbox']['spec']['containers'][0]
    assert container['files'][0]['path'] == job.BOOTSTRAP_PATH
    assert wire['sandbox']['spec']['objectStorageAccess']['buckets'] == ['natlang-cw']


def test_large_file_upload_is_the_hashed_inode_even_if_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, 'SMALL_FILE_BYTES', 4)
    out = tmp_path / 'run'
    out.mkdir()
    (out / 'checkpoint.pt').write_bytes(b'old-state')
    manifest, sources = bundle.directory_manifest(out)
    (out / 'checkpoint.pending').write_bytes(b'new-state!')
    (out / 'checkpoint.pending').replace(out / 'checkpoint.pt')  # the trainer's atomic rename, mid-publish
    s3 = FakeS3()
    bundle.upload(manifest, sources, 'b', s3, log=lambda _: None)
    bundle.close_sources(sources)
    key = manifest['files'][0]['key']
    assert s3.objects[key] == b'old-state' and key.endswith(hashlib.sha256(b'old-state').hexdigest())


def test_watch_publishes_each_checkpoint_and_resume_fetches_latest(tmp_path):
    out = tmp_path / 'runs' / 'stage'
    out.mkdir(parents=True)
    s3 = FakeS3()
    steps = iter(range(1, 10))

    def trainer_step(_seconds):  # between watch rounds the job writes a new checkpoint, or nothing
        step = next(steps)
        if step in (1, 3):
            (out / 'checkpoint.pt').write_bytes(f'state {step}'.encode())
        (out / 'train.jsonl').open('a').write(f'{{"step": {step}}}\n')

    published = bundle.watch([out], 'b', 'outputs/job/', s3, rounds=5, sleep=trainer_step, clock=lambda: 0.0,
                             log=lambda _: None)
    assert published == 2  # one per new checkpoint; log growth alone does not publish
    pointer = json.loads(s3.objects['outputs/job/stage/latest.json'])
    assert pointer['previous'] and pointer['manifest_key'] != pointer['previous']
    manifest = json.loads(s3.objects[pointer['manifest_key']])
    restored = tmp_path / 'restored'
    bundle.fetch(manifest, 'b', s3, tmp_path, [(str(out), str(restored))], log=lambda _: None)
    assert (restored / 'checkpoint.pt').read_bytes() == b'state 3'


def test_watch_republishes_after_max_interval_without_trigger(tmp_path):
    out = tmp_path / 'stage'
    out.mkdir()
    (out / 'checkpoint.pt').write_bytes(b'x')
    now = [0.0]

    def advance(_seconds):
        now[0] += 30

    count = bundle.watch([out], 'b', 'o/', FakeS3(), rounds=4, max_interval=50, sleep=advance,
                         clock=lambda: now[0], log=lambda _: None)
    assert count == 2  # at t=0 (new checkpoint) and t=60 (due); t=30 and t=90 are within 50 s of a publish


def test_resume_sets_env_and_needs_outputs():
    with pytest.raises(job.JobError, match='resume'):
        job.validate(base_job(resume=True, bucket='natlang-cw'))
    env = job.bootstrap_env(job.validate(base_job(resume=True, bucket='natlang-cw', outputs=['runs/x'])))
    assert env['NATLANG_CW_RESUME'] == '1'


def test_gc_keeps_live_outputs_and_recent_inputs_and_drops_the_rest(tmp_path):
    import datetime
    s3 = FakeS3()
    old = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    now = datetime.datetime(2026, 10, 12, tzinfo=datetime.timezone.utc).timestamp()
    for name, payload in (('live', b'live-state'), ('dead', b'dead-state')):
        out = tmp_path / name / 'stage'
        out.mkdir(parents=True)
        (out / 'checkpoint.pt').write_bytes(payload)
        bundle.publish_directory(out, 'b', f'outputs/{name}/', s3, log=lambda _: None)
    inputs = bundle.build_manifest(make_files(tmp_path / 'in'))
    s3.objects[bundle.manifest_key(inputs)] = json.dumps(inputs).encode()
    for key in s3.objects:
        s3.modified[key] = old
    s3.objects['cas/sha256/ff/' + 'f' * 64] = b'in flight'
    s3.modified['cas/sha256/ff/' + 'f' * 64] = datetime.datetime(2026, 10, 11, 23, tzinfo=datetime.timezone.utc)
    plan = bundle.gc_plan(s3, 'b', drop_jobs=['dead'], now=now)
    deleted = {item['key'] for item in plan['delete']}
    dead_key = bundle.object_key(hashlib.sha256(b'dead-state').hexdigest())
    live_key = bundle.object_key(hashlib.sha256(b'live-state').hexdigest())
    assert dead_key in deleted and live_key not in deleted
    assert 'outputs/dead/stage/latest.json' in deleted and 'outputs/live/stage/latest.json' not in deleted
    assert bundle.manifest_key(inputs) in deleted  # old input bundle, not kept by id
    assert 'cas/sha256/ff/' + 'f' * 64 not in deleted  # younger than min_age_hours
    kept = bundle.gc_plan(s3, 'b', drop_jobs=['dead'], keep_bundles=[inputs['id']], now=now)
    assert bundle.manifest_key(inputs) not in {item['key'] for item in kept['delete']}
    assert bundle.gc_execute(s3, 'b', plan) == len(plan['delete'])
    assert live_key in s3.objects and dead_key not in s3.objects


placement = load('placement')
NEEDS = {'id': 'j', 'kind': 'eval', 'gpu_memory_gb': 34.0, 'hours': 2.0, 'needs_teacher': False, 'lineage_owner': 'dgx',
         'moves_active_lineage': False, 'coreweave_cost_usd': 12.0}
FLEET = {'dgx_headroom_gb': 4.0, 'dgx_waiting_ahead': 10, 'coreweave_ready': True, 'coreweave_gpu_memory_gb': 96.0,
         'coreweave_fit_margin_gb': 6.0, 'coreweave_active': 0, 'coreweave_max_parallel': 10,
         'coreweave_budget_left_usd': None}


@pytest.mark.parametrize('needs, fleet, where', [
    ({}, {}, 'coreweave'),
    ({'lineage_owner': 'pop'}, {}, 'pop'),
    ({'moves_active_lineage': True}, {}, 'ask'),
    ({'needs_teacher': True}, {}, 'queue-dgx'),
    ({'needs_teacher': True}, {'dgx_headroom_gb': 40.0}, 'dgx'),
    ({}, {'dgx_headroom_gb': 40.0, 'dgx_waiting_ahead': 0}, 'dgx'),
    ({}, {'dgx_headroom_gb': 40.0}, 'coreweave'),  # fits DGX memory but would wait behind 10 units
    ({}, {'coreweave_ready': False}, 'queue-dgx'),
    ({'gpu_memory_gb': 91.0}, {}, 'queue-dgx'),
    ({}, {'coreweave_active': 10}, 'queue-dgx'),
    ({'coreweave_cost_usd': None}, {'coreweave_budget_left_usd': 100.0}, 'queue-dgx'),
    ({'coreweave_cost_usd': 50.0}, {'coreweave_budget_left_usd': 100.0}, 'coreweave'),
])
def test_crisp_placement_follows_the_numbered_steps(needs, fleet, where):
    assert placement.crisp_decision({**NEEDS, **needs}, {**FLEET, **fleet})['where'] == where


def test_placement_facts_from_ledger_and_records():
    status = {'headroom_gb': 3.94, 'waiting': [{'priority': 'high'}, {'priority': 'medium'}, {'priority': 'low'}]}
    assert placement.dgx_facts('eval', status) == {'dgx_headroom_gb': 3.9, 'dgx_waiting_ahead': 2}
    assert placement.dgx_facts('train', status)['dgx_waiting_ahead'] == 1
    import calendar, time as _time
    now = calendar.timegm(_time.strptime('2026-10-12T12:00:00Z', '%Y-%m-%dT%H:%M:%SZ'))
    records = [
        {'event': 'gpu_smoke_ok', 'at': '2026-10-12T09:00:00Z'},
        {'event': 'launched', 'sandbox_id': 'a', 'at': '2026-10-12T11:00:00Z', 'lifetime_hours': 4, 'cost': {'max_total': 30}},
        {'event': 'launched', 'sandbox_id': 'b', 'at': '2026-10-12T11:00:00Z', 'lifetime_hours': 4, 'cost': {'max_total': 30}},
        {'event': 'stop', 'at': '2026-10-12T11:30:00Z', 'args': ['b']},
        {'event': 'launched', 'sandbox_id': 'c', 'at': '2026-10-12T01:00:00Z', 'lifetime_hours': 1, 'cost': {'max_total': 5}},
    ]
    config = {**placement.settings(), 'weekly_budget_usd': 100}
    facts = placement.coreweave_facts(config, records, True, True, now)
    assert facts['coreweave_ready'] and facts['coreweave_active'] == 1 and facts['coreweave_budget_left_usd'] == 35
    assert not placement.coreweave_facts(config, records[1:], True, True, now)['coreweave_ready']


def test_launch_refuses_a_job_placed_elsewhere(monkeypatch):
    monkeypatch.setenv('CWSANDBOX_API_KEY', 'x')
    placed = job.validate(base_job(placement={'where': 'queue-dgx', 'reason': 'CoreWeave is not ready'}))
    with pytest.raises(cwctl.ControllerError, match="queue-dgx"):
        cwctl.launch(placed, False, False, None)
    assert cwctl.launch(placed, True, False, None)['dry_run']  # the dry run still shows the request


def test_run_call_matches_the_installed_client_signature():
    cwsandbox = pytest.importorskip('cwsandbox')
    import inspect
    request = job.sandbox_request(base_job(bucket='natlang-cw'), 'echo boot')
    signature = inspect.signature(cwsandbox.Sandbox.run)
    signature.bind(*cwctl.run_arguments(request), auth=None, **cwctl.run_keywords(request))
    assert cwctl.run_arguments(request)[:2] == ['/bin/bash', job.BOOTSTRAP_PATH]


def test_throttle_spaces_creates_and_backs_off_only_on_retryable(tmp_path, monkeypatch):
    monkeypatch.setattr(cwctl, 'STATE', tmp_path)
    monkeypatch.setattr(cwctl, 'CREATE_INTERVAL_SECONDS', 45.0)
    now, sleeps = [1000.0], []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    class Throttled(Exception):
        reason, metadata, retry_delay = 'CWSANDBOX_SERVERLESS_CREATE_THROTTLED', {}, None

    calls = []

    def create():
        calls.append(now[0])
        if len(calls) == 1:
            raise Throttled('429')
        return 'sandbox'

    assert cwctl.throttled_create(create, sleep=sleep, clock=lambda: now[0]) == 'sandbox'
    assert len(calls) == 2 and calls[1] - calls[0] >= 45
    assert cwctl.throttled_create(lambda: 'again', sleep=sleep, clock=lambda: now[0]) == 'again'
    assert sleeps[-1] == pytest.approx(45.0)  # the next create still waits the minimum interval

    class Invalid(Exception):
        reason, metadata, retry_delay = 'CWSANDBOX_INVALID_REQUEST', {}, None

    def bad():
        raise Invalid('bad request')

    before = len(sleeps)
    with pytest.raises(Invalid):
        cwctl.throttled_create(bad, sleep=sleep, clock=lambda: now[0])
    assert len(sleeps) - before <= 1  # the interval wait only; no retry loop

    class Quota(Exception):
        reason, retry_delay = 'CWSANDBOX_OCCUPANCY_QUOTA_EXCEEDED', None
        metadata = {'limit': '20', 'current': '20'}

    def full():
        raise Quota('quota')

    with pytest.raises(Quota):
        cwctl.throttled_create(full, sleep=sleep, clock=lambda: now[0], attempts=2)
    observed = [r for r in cwctl.records() if r['event'] == 'quota_observed']
    assert observed and observed[0]['limit'] == '20'


def test_training_jobs_default_to_the_maximum_lifetime():
    train = {k: v for k, v in base_job(kind='train').items() if k != 'lifetime_hours'}
    assert job.validate(train)['lifetime_hours'] == 720
    assert job.sandbox_request(train, 'x')['max_lifetime_seconds'] == 2592000


def test_fetch_from_inbox_and_url_without_a_bucket(tmp_path):
    import functools
    import http.server
    import threading
    served = tmp_path / 'served'
    served.mkdir()
    (served / 'weights.bin').write_bytes(b'public weights')
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        pairs = make_files(tmp_path / 'src')
        url_item = {'path': str(tmp_path / 'models' / 'weights.bin'),
                    'sha256': hashlib.sha256(b'public weights').hexdigest(), 'bytes': 14,
                    'url': f'http://127.0.0.1:{server.server_port}/weights.bin'}
        url_item['key'] = bundle.object_key(url_item['sha256'])
        manifest = bundle.build_manifest(pairs + [(url_item, None)])
        inbox = tmp_path / 'inbox'
        inbox.mkdir()
        for item, source in pairs:
            (inbox / item['sha256']).write_bytes(source.read_bytes())
        receipt = bundle.fetch(manifest, None, None, tmp_path / 'dest', inbox=inbox, log=lambda _: None)
        assert receipt['fetched'] == 4
        assert (tmp_path / 'models' / 'weights.bin').read_bytes() == b'public weights'
        assert bundle.verify(manifest, tmp_path / 'dest') == []
        bad = dict(url_item, sha256='0' * 64, key=bundle.object_key('0' * 64), path=str(tmp_path / 'models' / 'bad.bin'))
        with pytest.raises(bundle.BundleError, match='downloaded sha256'):
            bundle.fetch(bundle.build_manifest([(bad, None)]), None, None, tmp_path, log=lambda _: None)
    finally:
        server.shutdown()


def test_snapshot_manifest_keeps_its_id_with_the_directory_field(tmp_path):
    out = tmp_path / 'run'
    out.mkdir()
    (out / 'report.json').write_text('{}')
    target = tmp_path / 'm.json'
    assert bundle.main(['snapshot', str(out), '-o', str(target)]) == 0
    bundle.check_manifest(json.loads(target.read_text()))


def test_api_transport_is_the_default_without_a_bucket():
    assert job.validate(base_job())['transport'] == 'api'
    assert job.bootstrap_env(job.validate(base_job()))['NATLANG_CW_TRANSPORT'] == 'api'
    with pytest.raises(job.JobError, match='transport'):
        job.validate(base_job(transport='bucket'))


def test_read_sandbox_bounds_sdk_future_by_existing_patience(monkeypatch):
    from types import SimpleNamespace
    now = [0.0]
    observed = []

    class StuckFuture:
        def result(self, timeout=None):
            observed.append(timeout)
            now[0] += timeout + 0.01
            raise TimeoutError('SDK read did not complete')

    fake = SimpleNamespace(from_id=lambda *args, **kwargs: StuckFuture())
    monkeypatch.setitem(sys.modules, 'cwsandbox', SimpleNamespace(Sandbox=fake))
    monkeypatch.setattr(cwctl, 'sdk_auth', lambda: None)
    monkeypatch.setattr(cwctl.time, 'time', lambda: now[0])
    monkeypatch.setattr(cwctl.time, 'sleep', lambda seconds: pytest.fail('expired read must not sleep'))
    with pytest.raises(TimeoutError, match='did not complete'):
        cwctl.read_sandbox('existing-id', patience=10)
    assert observed == [10]
