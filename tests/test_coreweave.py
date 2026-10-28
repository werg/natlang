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


def test_service_port_becomes_a_share_token_https_endpoint():
    request = job.sandbox_request(base_job(service={'port': 8080}), '#!/bin/bash')
    assert request['services'] == [{'port': 8080, 'visibility': 'public', 'endpoint': {
        'kind': 'https', 'auth': 'share_token', 'request_timeout_seconds': job.SERVICE_TIMEOUT_SECONDS}}]
    assert request['environment_variables']['NATLANG_CW_SECRETS'] == '/work/.coreweave/secrets'
    assert 'services' not in job.sandbox_request(base_job(), '#!/bin/bash')
    with pytest.raises(job.JobError, match='unprivileged'):
        job.validate(base_job(service={'port': 80}))


def test_diloco_run_composes_ordinary_jobs(tmp_path):
    diloco = load('diloco')
    trainer = ['python', '-m', 'natlang_neuralese.train.native_state_sft', '--continue-from', '/work/in/parent.pt']
    (tmp_path / 'learner.json').write_text(json.dumps(base_job(id='lfm', kind='train', command=trainer)))
    (tmp_path / 'coordinator.json').write_text(json.dumps(base_job(
        id='coord', resources={'cpu': '8', 'memory': '32Gi'}, transport='api')))
    run = diloco.validate({'schema': diloco.SCHEMA, 'id': 'lfm-dl', 'resident': 'off', 'settings': {'run_id': 'lfm-dl', 'workers': 2},
        'coordinator': {'job': str(tmp_path / 'coordinator.json'), 'bundle_file': 'b.json', 'parent': '/work/in/parent.pt'},
        'learners': {'job': str(tmp_path / 'learner.json'), 'bundle_file': 'b.json'}})
    learner = diloco.learner_job(run, 1, 'https://coordinator.example')
    assert learner['id'] == 'lfm-dl-learner-1' and learner['env']['NATLANG_DILOCO_WORKER_ID'] == '1'
    assert learner['command'][:-2] == trainer and learner['command'][-2] == '--diloco'
    assert json.loads(learner['command'][-1]) == {
        'run_id': 'lfm-dl', 'workers': 2, 'coordinator_url': 'https://coordinator.example'}
    coordinator = diloco.coordinator_job(run, trainer[3:])
    assert coordinator['service'] == {'port': diloco.PORT}
    assert 'native_diloco init' in coordinator['command'][-1] and 'diloco serve' in coordinator['command'][-1]
    with pytest.raises(diloco.cwctl.ControllerError, match='filled in'):
        diloco.validate({**run, 'settings': {**run['settings'], 'coordinator_url': 'x'}})


def test_diloco_cohosted_job_uses_one_gpu_and_separate_owned_roots(tmp_path):
    import subprocess
    diloco = load('diloco')
    trainer = ['python', '-m', 'natlang_neuralese.train.native_state_sft', '--continue-from', '/work/in/parent.pt']
    (tmp_path / 'learner.json').write_text(json.dumps(base_job(id='lfm', kind='train', command=trainer, resources={'gpu':1,'memory':'64Gi'})))
    run = {'id': 'lfm-dl', 'settings': {'run_id': 'lfm-dl', 'workers': 4},
           'coordinator': {'parent': '/work/in/parent.pt'}, 'learners': {'job': str(tmp_path / 'learner.json')}}
    job = diloco.cohosted_job(run, trainer[3:])
    assert job['resources'] == {'gpu':1,'memory':'64Gi'}
    request=diloco.jobs.sandbox_request(job,'bootstrap')
    assert request['resources']=={'gpu':{'count':1},'memory':'64Gi'}
    overridden=diloco.cohosted_job({**run,'coordinator':{**run['coordinator'],'cohost_resources':{'gpu':1,'cpu':'8','memory':'96Gi'}}},trainer[3:])
    assert overridden['resources']=={'gpu':1,'cpu':'8','memory':'96Gi'}
    with pytest.raises(diloco.cwctl.ControllerError,match='cohost resources'):
        diloco.cohosted_job({**run,'coordinator':{**run['coordinator'],'cohost_resources':{'gpu':1,'gpu_type':'x'}}},trainer[3:])
    assert job['env']['NATLANG_DILOCO_WORKER_ID'] == '0'
    assert '/work/diloco-coordinator' in job['outputs'] and '/work/diloco-cohost-logs' in job['outputs']
    script = job['command'][2]
    assert "CUDA_VISIBLE_DEVICES=''" in script and 'http://127.0.0.1:8080/v1/status' in script
    assert script.index('/v1/status') < script.index('native_state_sft')
    assert 'coordinator remains available' in script and 'wait -n -p finished' in script
    subprocess.run(['bash', '-n'], input=script, text=True, check=True)


@pytest.mark.parametrize('placement,expected_count', [('cohosted', 4), ('separate', 5)])
def test_diloco_launch_records_shared_ownership_once(tmp_path, monkeypatch, placement, expected_count):
    diloco = load('diloco')
    monkeypatch.setattr(diloco, 'RUNS', tmp_path / 'records')
    trainer = ['python', '-m', 'natlang_neuralese.train.native_state_sft', '--continue-from', '/work/in/parent.pt']
    learner = tmp_path / 'learner.json'
    coord = tmp_path / 'coord.json'
    learner.write_text(json.dumps(base_job(id='lfm', kind='train', command=trainer)))
    coord.write_text(json.dumps(base_job(id='coord', resources={'cpu': '8', 'memory': '32Gi'}, transport='api')))
    run = {'schema': diloco.SCHEMA, 'id': 'run', 'resident': 'off', 'placement': placement,
           'settings': {'run_id': 'run', 'workers': 4},
           'coordinator': {'job': str(coord), 'bundle_file': 'coord.bundle', 'parent': '/work/in/parent.pt'},
           'learners': {'job': str(learner), 'bundle_file': 'learner.bundle'}}
    calls = []
    def fake_launch(job, *args, **kwargs):
        calls.append((job['id'], str(args[3])))
        result = {'sandbox_id': job['id'], 'push': {}}
        if job.get('service'):
            result['service_url'] = 'https://coord.example'
        return result
    monkeypatch.setattr(diloco.cwctl, 'launch', fake_launch)
    monkeypatch.setattr(diloco.cwctl, 'share_token', lambda sandbox_id: b'gateway')
    monkeypatch.setattr(diloco, 'wait_coordinator', lambda *args, **kwargs: None)
    monkeypatch.setattr(diloco.cwctl, 'record', lambda event: None)
    record = diloco.launch(run)
    assert len(calls) == expected_count and len(record['learners']) == 4
    assert len(record['ownership']) == expected_count and record['state'] == 'running'
    assert (record['coordinator'] == record['learners'][0]) == (placement == 'cohosted')
    assert json.loads((tmp_path / 'records/run.json').read_text()) == record
    if placement == 'cohosted':
        assert record['ownership'][record['coordinator']] == ['coordinator', 'learner-0']
        assert calls[0][1] == 'learner.bundle'


def test_diloco_cohost_failure_journals_roles_and_stops_each_sandbox_once(tmp_path, monkeypatch):
    diloco = load('diloco')
    monkeypatch.setattr(diloco, 'RUNS', tmp_path / 'records')
    trainer = ['python', '-m', 'natlang_neuralese.train.native_state_sft', '--continue-from', '/work/in/parent.pt']
    learner, coord = tmp_path / 'learner.json', tmp_path / 'coord.json'
    learner.write_text(json.dumps(base_job(id='lfm', kind='train', command=trainer)))
    coord.write_text(json.dumps(base_job(id='coord', resources={'cpu': '8', 'memory': '32Gi'})))
    run = {'schema': diloco.SCHEMA, 'id': 'run', 'resident': 'off', 'settings': {'run_id': 'run', 'workers': 2},
           'coordinator': {'job': str(coord), 'bundle_file': 'c', 'parent': '/work/in/parent.pt'},
           'learners': {'job': str(learner), 'bundle_file': 'l'}}
    def fake_launch(job, *args, **kwargs):
        if job.get('service'):
            return {'sandbox_id': 'shared', 'push': {}, 'service_url': 'https://coord.example'}
        return {'sandbox_id': 'worker1', 'wait_error': 'failed'}
    stops = []
    monkeypatch.setattr(diloco.cwctl, 'launch', fake_launch)
    monkeypatch.setattr(diloco.cwctl, 'share_token', lambda sandbox_id: b'gateway')
    monkeypatch.setattr(diloco, 'wait_coordinator', lambda *args, **kwargs: None)
    monkeypatch.setattr(diloco.cwctl, 'passthrough', lambda args: stops.append(args[1]))
    with pytest.raises(diloco.cwctl.ControllerError, match='did not start'):
        diloco.launch(run)
    record = json.loads((tmp_path / 'records/run.json').read_text())
    assert record['state'] == 'failed-stop-requested'
    assert record['ownership'] == {'shared': ['coordinator', 'learner-0'], 'worker1': ['learner-1']}
    assert sorted(stops) == ['shared', 'worker1']


@pytest.mark.parametrize('scenario', ['worker_fail', 'coordinator_fail', 'worker_success', 'signal', 'prepare_success', 'prepare_fail'])
def test_diloco_cohost_process_lifecycle(tmp_path, monkeypatch, scenario):
    """Execute the generated shell with authenticated HTTP and independently supervised processes."""
    import socket
    import subprocess
    import time

    diloco = load('diloco')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    monkeypatch.setattr(diloco, 'PORT', port)
    work = tmp_path / 'work'
    work.mkdir()
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    dispatcher = bin_dir / 'python'
    dispatcher.write_text(f'''#!{sys.executable}
import http.server, os, pathlib, signal, sys, time
root = pathlib.Path(os.environ['TEST_ROOT'])
args = sys.argv[1:]
if args[0] == '-c':
    os.execv({sys.executable!r}, [{sys.executable!r}, *args])
role = args[1]
def write(name, value=''):
    (root / name).write_text(value)
if role == 'seed_preparer':
    write('prepare-env', os.environ.get('CUDA_VISIBLE_DEVICES', '<absent>'))
    if os.environ['TEST_SCENARIO'] == 'prepare_fail': sys.exit(23)
    write('prepare-done'); sys.exit(0)
if role.endswith('native_diloco'):
    if os.environ['TEST_SCENARIO'] == 'prepare_success' and not (root / 'prepare-done').exists(): sys.exit(33)
    write('init-env', os.environ.get('CUDA_VISIBLE_DEVICES', '<absent>'))
    sys.exit(0)
if role.endswith('train.diloco'):
    write('server-env', os.environ.get('CUDA_VISIBLE_DEVICES', '<absent>'))
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            ok = self.path == '/v1/status' and self.headers.get('Authorization') == 'Bearer ' + os.environ['NATLANG_DILOCO_TOKEN']
            self.send_response(200 if ok else 401); self.end_headers(); self.wfile.write(b'{{}}')
        def log_message(self, *a): pass
    server = http.server.HTTPServer(('127.0.0.1', int(args[args.index('--port') + 1])), Handler)
    server.timeout = .05
    def stop(*a):
        write('server-stop', 'checkpoint=' + str((root / 'checkpoint').exists()))
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    while True:
        server.handle_request()
        if (root / 'crash-server').exists(): sys.exit(9)
if role.endswith('native_state_sft'):
    write('worker-env', os.environ.get('CUDA_VISIBLE_DEVICES', '<absent>'))
    def stop(*a):
        time.sleep(.2)
        write('checkpoint')
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    write('worker-started')
    scenario = os.environ['TEST_SCENARIO']
    if scenario == 'worker_fail': sys.exit(17)
    if scenario in ('worker_success', 'prepare_success'): write('checkpoint'); sys.exit(0)
    while True: time.sleep(.05)
sys.exit(2)
''')
    dispatcher.chmod(0o755)
    trainer = ['python', '-m', 'natlang_neuralese.train.native_state_sft', '--continue-from', '/parent']
    learner = tmp_path / 'learner.json'
    learner.write_text(json.dumps(base_job(id='lfm', kind='train', command=trainer,
                                           scratch={'path': str(work), 'size': '200Gi'})))
    run = {'id': 'lfm-dl', 'settings': {'run_id': 'lfm-dl', 'workers': 4},
           'coordinator': {'parent': '/parent'}, 'learners': {'job': str(learner)}}
    if scenario.startswith('prepare_'):
        run['coordinator']['prepare_parent'] = ['python', '-m', 'seed_preparer']
    job = diloco.cohosted_job(run, trainer[3:])
    env = {**os.environ, 'PATH': str(bin_dir) + os.pathsep + os.environ['PATH'],
           'TEST_ROOT': str(tmp_path), 'TEST_SCENARIO': scenario, 'CUDA_VISIBLE_DEVICES': '0',
           'NATLANG_DILOCO_TOKEN': 'x' * 32}
    process = subprocess.Popen(job['command'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    def wait_for(path):
        deadline = time.monotonic() + 12
        while not path.exists():
            if process.poll() is not None:
                raise AssertionError(f'cohost exited early {process.returncode}: {process.communicate()}')
            if time.monotonic() > deadline:
                raise AssertionError(f'timed out waiting for {path}')
            time.sleep(.05)
    try:
        if scenario == 'prepare_fail':
            assert process.wait(timeout=8) == 23
            assert not (tmp_path / 'init-env').exists()
            assert (tmp_path / 'prepare-env').read_text() == '0'
            return
        wait_for(tmp_path / 'worker-started')
        assert (tmp_path / 'init-env').read_text() == ''
        assert (tmp_path / 'server-env').read_text() == ''
        assert (tmp_path / 'worker-env').read_text() == '0'
        if scenario == 'worker_fail':
            assert process.wait(timeout=8) == 17
        elif scenario == 'coordinator_fail':
            (tmp_path / 'crash-server').write_text('')
            assert process.wait(timeout=8) != 0
            assert (tmp_path / 'checkpoint').exists()
        elif scenario in ('worker_success', 'prepare_success'):
            if scenario == 'prepare_success':
                assert (tmp_path / 'prepare-env').read_text() == '0'
                assert (tmp_path / 'prepare-done').exists()
            wait_for(tmp_path / 'checkpoint')
            time.sleep(.2)
            assert process.poll() is None and not (tmp_path / 'server-stop').exists()
            process.terminate()
            assert process.wait(timeout=8) == 143
        else:
            process.terminate()
            assert process.wait(timeout=8) == 143
            assert (tmp_path / 'checkpoint').exists()
            assert (tmp_path / 'server-stop').read_text() == 'checkpoint=True'
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=8)


def test_diloco_resident_args_bound_into_learner_and_coordinator_identity(tmp_path):
    diloco = load('diloco')
    trainer = ['python', '-m', 'natlang_neuralese.train.native_state_sft', '--continue-from', '/parent']
    learner = tmp_path / 'learner.json'
    learner.write_text(json.dumps(base_job(id='lfm', kind='train', command=trainer)))
    run = {'id': 'lfm-dl', 'settings': {'run_id': 'lfm-dl', 'workers': 4},
           'coordinator': {'parent': '/parent'}, 'learners': {'job': str(learner)},
           'resident': {'directory': '/work/run/resident-data', 'geometry_qualification': '/work/geometry.json'}}
    job = diloco.learner_job(run, 2, 'https://coordinator')
    assert job['command'][-6:-2] == ['--resident-data-directory', '/work/run/resident-data',
                                      '--resident-geometry-qualification', '/work/geometry.json']
    trainer_args = diloco.learner_job(run, 0, 'http://127.0.0.1:8080')['command'][len(diloco.TRAINER):-2]
    cohost = diloco.cohosted_job(run, trainer_args)
    assert cohost['command'][-1].count('--resident-data-directory') == 2


def test_diloco_resident_publication_stages_all_before_cas_and_resumes(tmp_path, monkeypatch):
    resident = load('diloco_resident')
    source = tmp_path / 'recipe.json'; source.write_bytes(b'recipe')
    manifest = {'schema': resident.SCHEMA, 'recipe': '/work/v2/recipe.json', 'stage': 'sft',
                'tokenizer': 't', 'roles': [1, 2, 3, 4], 'files': [{
                    'source': str(source), 'target': '/work/v2/recipe.json', 'bytes': 6,
                    'sha256': hashlib.sha256(b'recipe').hexdigest()}]}
    record = {'run': {'id': 'run', 'settings': {'workers': 4},
                      'resident': {'directory': '/work/resident'}}, 'learners': ['w0', 'w1', 'w2', 'w3']}
    events, fail = [], [True]
    class Result:
        returncode = 0
        def result(self): return self
    class FakeSandbox:
        def __init__(self, name): self.name = name
        def exec(self, *args, **kwargs): return Result()
        def write_file_streaming(self, path, chunks, **kwargs):
            assert b''.join(chunks) == b'recipe'
            return Result()
    monkeypatch.setattr(resident.cwctl, 'read_sandbox', lambda name: FakeSandbox(name))
    monkeypatch.setattr(resident, 'ensure_helper', lambda box: '/verified-helper')
    def call(box, args, helper=None):
        events.append((box.name, args[0]))
        if args[0] == 'prepare':
            return {'request': {'path': '/request', 'sha256': box.name},
                    'source_identity': {'train_sha256': 'same', 'ce_sha256': 'ce'}}
        if args[0] == 'publish':
            assert sum(event[1] == 'prepare' for event in events) == 4
            if box.name == 'w2' and fail[0]: raise RuntimeError('interrupted publication')
            return {'generation': box.name}
        return {}
    monkeypatch.setattr(resident, 'call', call)
    journal = tmp_path / 'publication.json'
    with pytest.raises(RuntimeError, match='interrupted'):
        resident.publish(record, manifest, journal)
    state = json.loads(journal.read_text())
    assert state['state'] == 'incomplete' and state['workers']['0']['state'] == 'published'
    fail[0] = False
    final = resident.publish(record, manifest, journal, resume=True)
    assert final['state'] == 'published'
    assert sum(event[1] == 'prepare' for event in events) == 4
    assert all(item['state'] == 'published' for item in final['workers'].values())


def test_diloco_resident_is_enabled_by_default_and_explicitly_opted_out():
    diloco = load('diloco')
    run = {'schema': diloco.SCHEMA, 'id': 'run', 'settings': {'run_id': 'run', 'workers': 4},
           'coordinator': {'job': 'c', 'bundle_file': 'cb', 'parent': '/parent'},
           'learners': {'job': 'l', 'bundle_file': 'lb'}}
    assert diloco.validate(run)['resident'] == {'directory': '/work/diloco-resident/resident-data',
                                                'geometry_qualification': '/work/diloco-resident/geometry.json'}
    assert diloco.validate({**run, 'resident': 'off'})['resident'] == 'off'


def test_diloco_resident_helper_is_staged_verified_and_immutable(tmp_path, monkeypatch):
    import subprocess
    resident = load('diloco_resident')
    monkeypatch.setattr(resident, 'CONTROL_DIRECTORY', str(tmp_path / 'control'))
    class Result:
        def __init__(self, result=None):
            self.returncode = result.returncode if result else 0
            self.stderr = result.stderr if result else ''
        def result(self): return self
    class LocalSandbox:
        def exec(self, args, **kwargs):
            return Result(subprocess.run(args, capture_output=True, text=True))
        def write_file(self, path, data):
            Path(path).write_bytes(data)
            return Result()
    sandbox = LocalSandbox()
    target = Path(resident.ensure_helper(sandbox))
    assert target.read_bytes() == resident.HELPER_SOURCE.read_bytes()
    assert resident.ensure_helper(sandbox) == str(target)
    target.write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='verification failed'):
        resident.ensure_helper(sandbox)


def test_sft_seed_publishes_actual_full_checkpoint_and_rejects_stale(tmp_path):
    import torch
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
    from natlang_neuralese.train.native_state_sft import TRAINING_SCHEMA
    seed = load('sft_seed_parent_verify')
    out = tmp_path / 'fresh'; (out / 'train').mkdir(parents=True)
    checkpoint = out / 'train/checkpoint.pt'
    training = {'schema': TRAINING_SCHEMA, 'global_step': 51, 'optimizer': {'state': {'x': 1}},
                'rng': {'cpu': torch.tensor([1])}, 'source_receipts': [{'sha256': 'source'}],
                'exposure_counts': {'a': 50}}
    torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': {'source_architecture': 'lfm3'},
                'step': 51, 'training': training}, checkpoint)
    (out / 'train/progress.json').write_text(json.dumps({'checkpoint_step': 50}))
    target = tmp_path / 'parent.pt'
    receipt = seed.publish(out, target, 50)
    assert receipt['global_step'] == 51 and receipt['checkpoint_step_observed'] == 50
    assert receipt['sha256'] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert receipt['exposure_visits'] == 50 and (tmp_path / 'parent.pt.seed.json').is_file()
    with pytest.raises((ValueError, FileExistsError)):
        seed.publish(out, target, 50)
    training.pop('rng'); torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': {'source_architecture': 'lfm3'},
                                    'step': 51, 'training': training}, checkpoint)
    with pytest.raises(ValueError, match='full optimizer/RNG/exposure'):
        seed.publish(out, tmp_path / 'bad.pt', 50)


def test_diloco_rejects_parent_preparation_on_separate_coordinator():
    diloco = load('diloco')
    run = {'schema': diloco.SCHEMA, 'id': 'run', 'placement': 'separate', 'settings': {'workers': 4},
           'coordinator': {'job': 'c', 'bundle_file': 'b', 'parent': '/p', 'prepare_parent': ['seed']},
           'learners': {'job': 'l', 'bundle_file': 'b'}}
    with pytest.raises(diloco.cwctl.ControllerError, match='requires cohosted'):
        diloco.validate(run)


def test_sft_seed_shell_requires_fresh_output_and_runs_phase_a(tmp_path):
    import subprocess
    import torch
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
    from natlang_neuralese.train.native_state_sft import TRAINING_SCHEMA
    training = {'schema': TRAINING_SCHEMA, 'global_step': 50, 'optimizer': {'state': {'x': 1}},
                'rng': {'cpu': torch.tensor([1])}, 'source_receipts': [{'sha256': 'source'}],
                'exposure_counts': {'a': 50}}
    fixture = tmp_path / 'checkpoint.pt'
    torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': {'source_architecture': 'lfm3'},
                'step': 50, 'training': training}, fixture)
    fake = tmp_path / 'fake-check.sh'
    fake.write_text('#!/bin/bash\nset -e\n[ "$CHECK_PHASE_B" = 0 ]\nmkdir -p "$1/train"\n'
                    'cp "$TEST_SEED_FIXTURE" "$1/train/checkpoint.pt"\n'
                    'printf \'{"checkpoint_step":50}\\n\' > "$1/train/progress.json"\n')
    fake.chmod(0o755)
    args = tmp_path / 'args.json'
    args.write_text(json.dumps(['--base', '@@BASE@@', '--epochs', '1', '--checkpoint-every', '50']))
    out, parent = tmp_path / 'fresh', tmp_path / 'target/parent.pt'
    env = {**os.environ, 'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'],
           'NATLANG_SFT_CHECK_SCRIPT': str(fake), 'TEST_SEED_FIXTURE': str(fixture)}
    command = ['bash', str(ROOT / 'scripts/coreweave/sft_seed_parent.sh'), str(out), str(args), str(parent), '50']
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert parent.is_file() and json.loads((parent.parent / 'parent.pt.seed.json').read_text())['global_step'] == 50
    again = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    assert again.returncode != 0 and 'fresh seed output' in again.stderr


def test_diloco_seed_pause_then_resume_launches_only_remaining_workers(tmp_path, monkeypatch):
    diloco = load('diloco'); monkeypatch.setattr(diloco, 'RUNS', tmp_path / 'records')
    trainer = ['python', '-m', 'natlang_neuralese.train.native_state_sft', '--continue-from', '/parent']
    learner = tmp_path / 'learner.json'; learner.write_text(json.dumps(base_job(id='lfm', kind='train', command=trainer)))
    run = {'schema': diloco.SCHEMA, 'id': 'run', 'resident': 'off', 'settings': {'run_id': 'run', 'workers': 4},
           'coordinator': {'job': 'unused', 'bundle_file': 'unused', 'parent': '/parent',
                           'prepare_parent': ['bash', 'sft_seed_parent.sh']},
           'learners': {'job': str(learner), 'bundle_file': 'learner.bundle'}}
    calls = []
    def fake_launch(job, *args, **kwargs):
        calls.append(job['id'])
        return {'sandbox_id': job['id'], 'push': {}, **({'service_url': 'https://coordinator'} if job.get('service') else {})}
    monkeypatch.setattr(diloco.cwctl, 'launch', fake_launch)
    monkeypatch.setattr(diloco.cwctl, 'share_token', lambda sandbox_id: b'gateway')
    monkeypatch.setattr(diloco.cwctl, 'record', lambda event: None)
    monkeypatch.setattr(diloco.cwctl, 'read_sandbox', lambda ident: type('Box', (), {'status': 'running', 'get_status': lambda self: None})())
    monkeypatch.setattr(diloco, 'wait_coordinator', lambda *args, **kwargs: None)
    first = diloco.launch(run, coordinator_only=True)
    assert first['state'] == 'coordinator-ready' and calls == ['run-learner-0']
    final = diloco.launch_remaining('run')
    assert final['state'] == 'running' and sorted(calls) == [f'run-learner-{i}' for i in range(4)]
    with pytest.raises(diloco.cwctl.ControllerError, match='not a paused'):
        diloco.launch_remaining('run')


def test_diloco_detects_terminal_sandbox_during_seed_startup(monkeypatch):
    from types import SimpleNamespace
    diloco = load('diloco')
    monkeypatch.setattr(diloco.urllib.request, 'urlopen', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('not ready')))
    monkeypatch.setattr(diloco.cwctl, 'read_sandbox', lambda *args, **kwargs: SimpleNamespace(status='failed'))
    with pytest.raises(diloco.cwctl.ControllerError, match='ended during startup'):
        diloco.wait_coordinator('http://localhost', 'x' * 32, b'g', timeout=30, sandbox_id='worker0')


def test_diloco_seed_startup_failure_is_journaled_before_stop(tmp_path, monkeypatch):
    diloco = load('diloco'); monkeypatch.setattr(diloco, 'RUNS', tmp_path / 'records')
    trainer = ['python', '-m', 'natlang_neuralese.train.native_state_sft', '--continue-from', '/parent']
    learner = tmp_path / 'learner.json'; learner.write_text(json.dumps(base_job(id='lfm', kind='train', command=trainer)))
    run = {'schema': diloco.SCHEMA, 'id': 'run', 'resident': 'off', 'settings': {'run_id': 'run', 'workers': 4},
           'coordinator': {'job': 'unused', 'bundle_file': 'unused', 'parent': '/parent',
                           'prepare_parent': ['bash', 'sft_seed_parent.sh']},
           'learners': {'job': str(learner), 'bundle_file': 'learner.bundle'}}
    monkeypatch.setattr(diloco.cwctl, 'launch', lambda *args, **kwargs: {
        'sandbox_id': 'shared', 'push': {}, 'service_url': 'https://coordinator'})
    monkeypatch.setattr(diloco.cwctl, 'share_token', lambda sandbox_id: b'gateway')
    monkeypatch.setattr(diloco, 'wait_coordinator', lambda *args, **kwargs: (_ for _ in ()).throw(
        diloco.cwctl.ControllerError('seed failed')))
    stopped = []
    monkeypatch.setattr(diloco.cwctl, 'passthrough', lambda args: stopped.append(args[1]))
    with pytest.raises(diloco.cwctl.ControllerError, match='seed failed'):
        diloco.launch(run, coordinator_only=True)
    record = json.loads((tmp_path / 'records/run.json').read_text())
    assert record['state'] == 'failed-stop-requested' and record['learners'] == ['shared', None, None, None]
    assert record['ownership'] == {'shared': ['coordinator', 'learner-0']} and stopped == ['shared']


def test_diloco_restored_workers_isolate_stale_control_and_verify_seed(tmp_path):
    import subprocess
    diloco = load('diloco')
    seed = tmp_path / 'parent.pt'; seed.write_bytes(b'full-seed')
    expected = hashlib.sha256(seed.read_bytes()).hexdigest()
    (tmp_path / 'parent.pt.seed.json').write_text(json.dumps({'sha256': expected, 'global_step': 50,
        'requested_stop': 50, 'full_optimizer_rng_exposure': True}))
    fakebin = tmp_path / 'bin'; fakebin.mkdir()
    fakepython = fakebin / 'python'; fakepython.write_text('#!/bin/sh\nprintf started > "$WORKER_MARKER"\n'); fakepython.chmod(0o755)
    trainer = ['python', '-m', 'natlang_neuralese.train.native_state_sft', '--continue-from', str(seed)]
    template = tmp_path / 'job.json'; template.write_text(json.dumps(base_job(id='orig', kind='train', command=trainer,
        env={'NATLANG_DILOCO_SEED_SHA256': expected}, scratch={'path':'/work','size':'200Gi','restore_from_snapshot_id':'s'})))
    run = {'id':'run', 'settings':{'run_id':'run','workers':4}, 'coordinator':{'parent':str(seed)},
           'learners':{'job':str(template)}}
    target = diloco.learner_job(run, 1, 'https://coordinator')
    request = diloco.jobs.sandbox_request(target, 'bootstrap')
    assert target['env']['NATLANG_CW_INSTANCE']=='run-learner-1'
    assert request['environment_variables']['NATLANG_CW_SECRETS']=='/work/.coreweave/instances/run-learner-1/secrets'
    env = {**os.environ, 'PATH':str(fakebin)+os.pathsep+os.environ['PATH'], 'WORKER_MARKER':str(tmp_path/'started'), 'NATLANG_CW_WORK':str(tmp_path/'sandbox-work')}
    result = subprocess.run(target['command'], env=env, capture_output=True, text=True)
    assert result.returncode == 0 and (tmp_path/'started').exists()
    (tmp_path/'started').unlink(); seed.write_bytes(b'wrong')
    result = subprocess.run(target['command'], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and not (tmp_path/'started').exists()
    bootstrap = (ROOT/'scripts/coreweave/bootstrap.sh').read_text()
    assert 'CW="$CW/instances/$NATLANG_CW_INSTANCE"' in bootstrap
    assert 'until [ -f "$CW/inputs-ready" ]' in bootstrap


def test_diloco_post_seed_snapshot_download_and_resume_are_idempotent(tmp_path, monkeypatch):
    post = load('diloco_post_seed')
    seedsha = hashlib.sha256(b'seed').hexdigest()
    original = tmp_path/'learner.json'
    definition=base_job(id='job',kind='train',scratch={'path':'/work','size':'200Gi'}, env={})
    original.write_text(json.dumps(definition))
    record_path=tmp_path/'run.json'
    record={'coordinator':'box0','placement':'cohosted','state':'coordinator-ready',
            'learners':['box0',None,None,None], 'ownership':{'box0':['coordinator','learner-0']},
            'run':{'id':'run','settings':{'workers':4},'learners':{'job':str(original)},
                   'coordinator':{'parent':'/work/parent.pt'}}}
    record_path.write_text(json.dumps(record))
    import threading
    launch_event=threading.Event()
    counts={'cache':0,'snapshot':0,'download':0,'private':0,'launch':0,'restore':0}
    class FakeCloud:
        sandbox_id='box0'; parent='/work/parent.pt'
        def seed_receipt(self):return {'checkpoint':self.parent,'sha256':seedsha,'bytes':4,'global_step':50,
                                       'requested_stop':50,'full_optimizer_rng_exposure':True}
        def cache(self):counts['cache']+=1;return {'objects':1,'bytes':4}
        def capture(self,request_id):counts['snapshot']+=1;return 'snap-1'
        def inspect(self,snapshot):return {'snapshot_id':snapshot,'status':'READY','source_sandbox_id':'box0','size_bytes':4}
        def download(self,receipt):counts['download']+=1;return {'verified':True,'sha256':seedsha,'bytes':4}
        def runtime_binding(self):return {'code_sha256':'runtime','plan_sha256':'plan'}
        def private_export(self,receipt,intent):
            counts['private']+=1;assert counts['snapshot']==1
            assert intent['global_step']==50 and intent['runtime']['code_sha256']=='runtime'
            assert json.loads(journal.read_text())['private_export_intent']==intent
            assert launch_event.wait(5), 'private verification blocked worker launch'
            return {'private':True,'identity':{'sha256':seedsha,'bytes':4},'independent_verification':{'sha256':seedsha,'bytes':4}}
        def verify_restore(self,snapshot,worker,receipt):counts['restore']+=1;return {'passed':True,'worker':worker}
    def launch_remaining(run_id):
        counts['launch']+=1
        launch_event.set()
        value=json.loads(record_path.read_text()); assert value['state']=='coordinator-ready'
        new=json.loads(Path(value['run']['learners']['job']).read_text())
        assert new['scratch']['restore_from_snapshot_id']=='snap-1' and new['lifetime_hours']==720
        assert new['env']['NATLANG_DILOCO_SEED_SHA256']==seedsha
        assert json.loads(original.read_text())==definition
        value.update(state='running',learners=['box0','box1','box2','box3']);record_path.write_text(json.dumps(value));return value
    monkeypatch.setattr(post.diloco,'launch_remaining',launch_remaining)
    journal=tmp_path/'journal.json'
    result=post.run(record_path,journal,FakeCloud(),wait_seconds=3,interval=.01)
    assert result['phase']=='complete' and len(result['restored'])==3
    assert counts=={'cache':1,'snapshot':1,'download':1,'private':1,'launch':1,'restore':3}
    post.run(record_path,journal,FakeCloud(),wait_seconds=3,interval=.01)
    assert counts=={'cache':1,'snapshot':1,'download':1,'private':1,'launch':1,'restore':3}


def test_restored_worker_quarantines_inherited_state_once(tmp_path):
    import subprocess
    diloco = load('diloco')
    work=tmp_path/'work';work.mkdir()
    seed=work/'parent.pt';seed.write_bytes(b'full-seed');sha=hashlib.sha256(seed.read_bytes()).hexdigest()
    (work/'parent.pt.seed.json').write_text(json.dumps({'sha256':sha,'global_step':50,
        'requested_stop':50,'full_optimizer_rng_exposure':True}))
    out=work/'learner';out.mkdir();(out/'checkpoint.pt').write_text('worker-zero')
    resident=work/'resident';resident.mkdir();(resident/'initial.json').write_text('old-owner')
    (work/'recipe-plan.json').write_text('preserve-plan')
    fakebin=tmp_path/'bin';fakebin.mkdir();fake=fakebin/'python';fake.write_text('#!/bin/sh\nexit 0\n');fake.chmod(0o755)
    trainer=['python','-m','natlang_neuralese.train.native_state_sft','--out',str(out),'--continue-from',str(seed)]
    template=tmp_path/'job.json';template.write_text(json.dumps(base_job(id='orig',kind='train',command=trainer,
        env={'NATLANG_DILOCO_SEED_SHA256':sha},scratch={'path':str(work),'size':'200Gi','restore_from_snapshot_id':'s'})))
    run={'id':'run','settings':{'run_id':'run','workers':4},'coordinator':{'parent':str(seed)},
         'learners':{'job':str(template)},'resident':{'directory':str(resident),'geometry_qualification':str(work/'geo')}}
    job=diloco.learner_job(run,1,'https://coordinator')
    env={**os.environ,'PATH':str(fakebin)+os.pathsep+os.environ['PATH'],'NATLANG_CW_WORK':str(work)}
    result=subprocess.run(job['command'],env=env,capture_output=True,text=True)
    assert result.returncode==0, result.stderr
    stash=work/'.coreweave/instances/run-learner-1/inherited'
    assert (stash/'0/checkpoint.pt').read_text()=='worker-zero' and (stash/'1/initial.json').read_text()=='old-owner'
    assert (work/'recipe-plan.json').read_text()=='preserve-plan' and not out.exists() and not resident.exists()
    out.mkdir();(out/'checkpoint.pt').write_text('worker-one')
    result=subprocess.run(job['command'],env=env,capture_output=True,text=True)
    assert result.returncode==0 and (out/'checkpoint.pt').read_text()=='worker-one'


def test_postseed_cache_keeps_live_markers_and_secrets(tmp_path):
    import subprocess
    post=load('diloco_post_seed');work=tmp_path/'work'; repo=tmp_path/'repo';repo.mkdir()
    cw=work/'.coreweave';(cw/'secrets').mkdir(parents=True)
    (cw/'inputs-ready').write_text('live');(cw/'secrets/token').write_text('secret')
    source=repo/'data.json';source.write_bytes(b'important')
    code=cw/'code.tar.gz';code.write_bytes(b'code')
    def item(path,content):return {'path':path,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
    manifest={'files':[item('data.json',b'important')], 'code':item('ignored',b'code')}
    (cw/'bundle.json').write_text(json.dumps(manifest))
    def run():return subprocess.run([sys.executable,'-c',post.CACHE_PROGRAM,str(work),str(repo),str(cw)],capture_output=True,text=True)
    result=run();assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['objects']==2
    assert (cw/'inputs-ready').read_text()=='live' and (cw/'secrets/token').read_text()=='secret'
    assert source.read_bytes()==b'important' and code.read_bytes()==b'code'
    assert (cw/'cas'/manifest['files'][0]['sha256']).read_bytes()==b'important'
    assert run().returncode==0
    (cw/'cas'/manifest['files'][0]['sha256']).write_bytes(b'corrupt')
    assert run().returncode!=0


def test_restore_origin_uses_provider_get_sandbox_rpc(monkeypatch):
    import asyncio
    import types
    from types import SimpleNamespace
    recovery = load('recovery_snapshots')
    class Request:
        def __init__(self, sandbox_id): self.sandbox_id=sandbox_id
    proto=types.ModuleType('cwsandbox._proto');proto.sandbox_pb2=SimpleNamespace(GetSandboxRequest=Request)
    package=types.ModuleType('cwsandbox');package._proto=proto
    monkeypatch.setitem(sys.modules,'cwsandbox',package)
    monkeypatch.setitem(sys.modules,'cwsandbox._proto',proto)
    calls=[]
    class Stub:
        async def GetSandbox(self, req, timeout, metadata):
            calls.append((req.sandbox_id,timeout,metadata))
            vol=SimpleNamespace(scratch=SimpleNamespace(restore_from_snapshot_id='actual-snapshot'),
                                HasField=lambda name:name=='scratch')
            return SimpleNamespace(spec=SimpleNamespace(volumes=[vol]))
    class Destination:
        _stub=Stub();_auth_metadata=[('auth','x')]
        _loop_manager=SimpleNamespace(run_sync=lambda coro: asyncio.run(coro))
        async def _ensure_client(self):pass
    destination=Destination()
    assert recovery.inspect_destination_snapshots(destination,'worker')==['actual-snapshot']
    assert calls==[('worker',30,[('auth','x')])]


def test_initial_seed_restored_snapshot_has_fresh_isolated_control(tmp_path):
    diloco=load('diloco')
    trainer=['python','-m','natlang_neuralese.train.native_state_sft','--continue-from','/work/new-parent.pt']
    template=tmp_path/'job.json';template.write_text(json.dumps(base_job(id='job',kind='train',command=trainer,
        scratch={'path':'/work','size':'300Gi','restore_from_snapshot_id':'failed-seed-snapshot'})))
    run={'id':'fresh-run','settings':{'run_id':'fresh-run','workers':4},
         'coordinator':{'parent':'/work/new-parent.pt','prepare_parent':['bash','prepare-new-seed.sh']},
         'learners':{'job':str(template)}}
    args=diloco.learner_job(run,0,'http://127.0.0.1:8080')['command'][len(diloco.TRAINER):-2]
    cohost=diloco.cohosted_job(run,args)
    request=diloco.jobs.sandbox_request(cohost,'bootstrap')
    assert request['environment_variables']['NATLANG_CW_INSTANCE']=='fresh-run-learner-0'
    assert request['environment_variables']['NATLANG_CW_SECRETS']=='/work/.coreweave/instances/fresh-run-learner-0/secrets'
    assert 'prepare-new-seed.sh' in cohost['command'][-1]


def test_postseed_cache_reads_actual_isolated_cohost_manifest(tmp_path):
    import subprocess
    post=load('diloco_post_seed');work=tmp_path/'work';repo=tmp_path/'repo';repo.mkdir()
    global_cw=work/'.coreweave';instance=global_cw/'instances/new-run-learner-0';instance.mkdir(parents=True)
    (global_cw/'bundle.json').write_text('{"files":[]}')
    source=repo/'new';source.write_bytes(b'new-data')
    sha=hashlib.sha256(b'new-data').hexdigest()
    (instance/'bundle.json').write_text(json.dumps({'files':[{'path':'new','sha256':sha,'bytes':8}]}))
    result=subprocess.run([sys.executable,'-c',post.CACHE_PROGRAM,str(work),str(repo),str(instance)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert (global_cw/'cas'/sha).read_bytes()==b'new-data'
    assert json.loads(result.stdout)['objects']==1


def test_postseed_private_export_binds_actual_sandbox_and_independent_get(tmp_path, monkeypatch):
    from types import SimpleNamespace
    post=load('diloco_post_seed');sha='a'*64;receipt={'sha256':sha,'bytes':123,'global_step':50}
    intent={'role':'lfm3-v3-seed-'+sha[:12]};cloud=post.Cloud('actual-box','/work/parent.pt',tmp_path)
    seen=[]
    class Result:
        returncode=0
        def result(self):return self
    monkeypatch.setattr(cloud,'sandbox',lambda: SimpleNamespace(exec=lambda args: Result()))
    def fake_run(argv,env,check):
        seen.append((argv,env))
        value={'sandbox_id':'actual-box','source':'/work/parent.pt','private':True,
               'identity':{'sha256':sha,'bytes':123},'independent_verification':{'sha256':sha,'bytes':123},
               'verified_at':1,'run':'private-run','name':'object'}
        Path(env['CW_EXPORT_RECEIPT_FILE']).write_text(json.dumps(value))
    monkeypatch.setattr(post.subprocess,'run',fake_run)
    verified=cloud.private_export(receipt,intent)
    argv,env=seen[0]
    assert argv[-2:]==[intent['role'],'/work/parent.pt']
    assert env['CW_EXPORT_SANDBOX_ID']=='actual-box' and env['CW_EXPORT_EXPECTED_SHA256']==sha
    assert env['CW_EXPORT_EXPECTED_BYTES']=='123' and env['CW_EXPORT_VERIFY_ON_OWNER']=='1'
    assert verified['independent_verification']['sha256']==sha
    def bad_run(argv,env,check):
        value={'sandbox_id':'actual-box','source':'/work/parent.pt','private':True,
               'identity':{'sha256':sha,'bytes':123},'independent_verification':{'sha256':'b'*64,'bytes':123},
               'verified_at':1,'run':'private-run','name':'object'}
        Path(env['CW_EXPORT_RECEIPT_FILE']).write_text(json.dumps(value))
    monkeypatch.setattr(post.subprocess,'run',bad_run)
    with pytest.raises(ValueError,match='independent full-object'):
        cloud.private_export(receipt,intent)


def test_code_archive_removes_only_verified_predecessor_modules(tmp_path):
    import io
    import tarfile
    archive=load('code_archive');repo=tmp_path/'repo';work=tmp_path/'work';repo.mkdir();work.mkdir()
    package=repo/archive.PACKAGE;package.mkdir(parents=True)
    def make_tar(name, files):
        target=tmp_path/name
        with tarfile.open(target,'w') as tar:
            for path,data in files.items():
                info=tarfile.TarInfo(path);info.size=len(data);tar.addfile(info,io.BytesIO(data))
        return target
    old={'__init__.py':b'old-init','removed.py':b'old-file'}
    first=make_tar('first.tar',{archive.PACKAGE+'/'+k:v for k,v in old.items()})
    with tarfile.open(first) as tar:archive.extract_code_archive(tar,repo,work,'a'*64)
    protected=repo/'data/source.json';protected.parent.mkdir();protected.write_text('preserve')
    new={'__init__.py':b'new-init','added.py':b'new-file'}
    second=make_tar('second.tar',{archive.PACKAGE+'/'+k:v for k,v in new.items()})
    with tarfile.open(second) as tar:result=archive.extract_code_archive(tar,repo,work,'b'*64)
    assert result['removed_predecessor_files']==1 and not (package/'removed.py').exists()
    assert protected.read_text()=='preserve' and (package/'added.py').read_bytes()==b'new-file'
    extra=package/'untracked.py';extra.write_bytes(b'user-work')
    with tarfile.open(second) as tar, pytest.raises(ValueError,match='untracked native Python'):
        archive.extract_code_archive(tar,repo,work,'b'*64)
    assert extra.read_bytes()==b'user-work'


def test_code_archive_refuses_to_delete_modified_predecessor(tmp_path):
    import io
    import tarfile
    archive=load('code_archive');repo=tmp_path/'repo';work=tmp_path/'work';repo.mkdir();work.mkdir()
    package=repo/archive.PACKAGE;package.mkdir(parents=True)
    def add(name, files):
        path=tmp_path/name
        with tarfile.open(path,'w') as tar:
            for k,v in files.items():
                info=tarfile.TarInfo(archive.PACKAGE+'/'+k);info.size=len(v);tar.addfile(info,io.BytesIO(v))
        return path
    before=add('before.tar',{'__init__.py':b'x','retired.py':b'original'})
    after=add('after.tar',{'__init__.py':b'x'})
    with tarfile.open(before) as tar:archive.extract_code_archive(tar,repo,work,'a'*64)
    (package/'retired.py').write_bytes(b'user-changes')
    with tarfile.open(after) as tar, pytest.raises(ValueError,match='existing native code changed'):
        archive.extract_code_archive(tar,repo,work,'b'*64)
    assert (package/'retired.py').read_bytes()==b'user-changes'
    assert '@@NATLANG_CODE_ARCHIVE_HELPER@@' not in cwctl.bootstrap_text()


def test_postseed_download_defers_full_decode_to_separate_ledger_unit(tmp_path,monkeypatch):
    post=load('diloco_post_seed');sha='a'*64;receipt={'sha256':sha,'bytes':123,'global_step':50}
    cloud=post.Cloud('sandbox','/work/parent.pt',tmp_path)
    seen=[]
    def run(argv,**kwargs):
        seen.append(argv)
        if argv[1].endswith('recovery_pull.py'):
            latest=tmp_path/'latest';latest.mkdir();(latest/'receipt.json').write_text(json.dumps(
                {'verified':True,'sha256':sha,'bytes':123,'local':str(tmp_path/'checkpoint.pt')}))
        else:
            assert argv[2]=='run' and argv[argv.index('--budget-gb')+1]=='18'
            proof=Path(argv[-1]);proof.write_text(json.dumps({'passed':True,'sha256':sha,'bytes':123,'global_step':50}))
        return SimpleNamespace(returncode=0)
    from types import SimpleNamespace
    monkeypatch.setattr(post.subprocess,'run',run)
    result=cloud.download(receipt)
    assert len(seen)==2 and result['full_state_verification']['passed']
    assert result['verification_unit'].startswith('lfm3-seed-decode-')


def test_seed_recovery_decoder_verifies_actual_full_state(tmp_path):
    import torch
    import hashlib
    import subprocess
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
    from natlang_neuralese.train.native_state_sft import TRAINING_SCHEMA
    path=tmp_path/'seed.pt';state={'schema':CHECKPOINT_SCHEMA,'step':50,'model_config':{'base':'test'},
        'backbone_state':{'weight':torch.ones(1)},'interface_state':{'weight':torch.ones(1)},
        'training':{'schema':TRAINING_SCHEMA,'global_step':50,
        'optimizer':{'state':{'x':1}},'rng':{'x':1},'exposure_counts':{'row':1},'source_receipts':[{'sha256':'x'}]}}
    torch.save(state,path);sha=hashlib.sha256(path.read_bytes()).hexdigest();out=tmp_path/'receipt.json'
    command=[sys.executable,str(ROOT/'scripts/coreweave/qualify_seed_recovery.py'),str(path),sha,str(path.stat().st_size),'50',str(out)]
    subprocess.run(command,check=True,capture_output=True,text=True)
    assert json.loads(out.read_text())['passed'] is True
    result=subprocess.run([*command[:-2],'51',str(tmp_path/'wrong.json')],capture_output=True,text=True)
    assert result.returncode!=0 and not (tmp_path/'wrong.json').exists()


def test_legacy_snapshot_bootstrap_authenticates_predecessor_archive(tmp_path):
    import io, tarfile, hashlib
    archive=load('code_archive'); repo=tmp_path/'repo'; work=tmp_path/'work'
    package=repo/archive.PACKAGE; package.mkdir(parents=True)
    root=work/'.coreweave'; cas=root/'cas'; cas.mkdir(parents=True)
    def build(path, files):
        with tarfile.open(path,'w') as tar:
            for name,data in files.items():
                info=tarfile.TarInfo(archive.PACKAGE+'/'+name);info.size=len(data)
                tar.addfile(info,io.BytesIO(data))
    old=tmp_path/'old.tar';new=tmp_path/'new.tar'
    build(old,{'native_memory.py':b'old','retired.py':b'original'})
    build(new,{'native_memory.py':b'alias fix','added.py':b'new'})
    oldsha=hashlib.sha256(old.read_bytes()).hexdigest(); (cas/oldsha).write_bytes(old.read_bytes())
    manifest=root/'bundle.json';manifest.write_text(json.dumps({'code':{'sha256':oldsha,'bytes':old.stat().st_size}}))
    manifestsha=hashlib.sha256(manifest.read_bytes()).hexdigest()
    (package/'native_memory.py').write_bytes(b'old');(package/'retired.py').write_bytes(b'original')
    (package/'retired.py').write_bytes(b'user change')
    with tarfile.open(new) as tar, pytest.raises(ValueError,match='existing native code changed'):
        archive.extract_code_archive(tar,repo,work,'b'*64,predecessor_manifest_sha256=manifestsha)
    (package/'retired.py').write_bytes(b'original')
    with tarfile.open(new) as tar, pytest.raises(ValueError,match='manifest does not match'):
        archive.extract_code_archive(tar,repo,work,'b'*64,predecessor_manifest_sha256='a'*64)
    with tarfile.open(new) as tar:
        result=archive.extract_code_archive(tar,repo,work,'b'*64,predecessor_manifest_sha256=manifestsha)
    assert result['removed_predecessor_files']==1
    assert (package/'native_memory.py').read_bytes()==b'alias fix' and not (package/'retired.py').exists()


def test_startup_log_capture_preserves_failure_and_redacts_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv('WANDB_API_KEY','secret-value')
    class Sandbox:
        def stream_logs(self,follow):
            assert follow is True
            yield '{"bootstrap":"waiting-for-inputs"}\n'
            yield 'Traceback: secret-value\n'
            yield '{"bootstrap":"failed","step":"code"}\n'
    path=tmp_path/'startup.log';ready=tmp_path/'ready'
    cwctl.capture_startup_logs(Sandbox(),path,ready)
    assert ready.read_text().strip()=='ready'
    content=path.read_text()
    assert 'Traceback: [redacted]' in content and 'secret-value' not in content and '"step":"code"' in content
    assert path.stat().st_mode & 0o077 == 0


def test_startup_log_capture_redacts_run_command(tmp_path):
    class Sandbox:
        def stream_logs(self,follow):
            yield '{"bootstrap":"run","command":"secret argument"}\n'
            yield 'trainer output must not be captured'
    path=tmp_path/'startup.log';ready=tmp_path/'ready'
    cwctl.capture_startup_logs(Sandbox(),path,ready)
    assert 'secret argument' not in path.read_text() and 'trainer output' not in path.read_text()
    assert ready.read_text().strip()=='ready'


def test_coordinator_identity_uses_unwrapped_restored_trainer_argv(tmp_path,monkeypatch):
    diloco=load('diloco');monkeypatch.setattr(diloco,'RUNS',tmp_path/'records')
    trainer=['python','-m','natlang_neuralese.train.native_state_sft','--continue-from','/work/parent.pt','--weight-ema','off']
    learner=base_job(id='lfm',kind='train',command=trainer)
    learner['scratch']={'path':'/work','size':'200Gi','restore_from_snapshot_id':'snapshot'}
    learner['env']={'NATLANG_DILOCO_SEED_SHA256':'a'*64}
    path=tmp_path/'learner.json';path.write_text(json.dumps(learner))
    run={'schema':diloco.SCHEMA,'id':'run','placement':'cohosted','settings':{'run_id':'run','workers':4},
         'resident':{'directory':'/work/diloco-resident/resident-data','geometry_qualification':'/work/diloco-resident/geometry.json'},
         'coordinator':{'job':str(path),'bundle_file':'coord.bundle','parent':'/work/parent.pt'},
         'learners':{'job':str(path),'bundle_file':'learner.bundle'}}
    wrapped=diloco.learner_job(run,0,'http://127.0.0.1:8080')['command']
    assert wrapped[:2]==['bash','-c']
    actual=diloco.cohosted_job;captured=[]
    def cohost(run,argv):captured.extend(argv);return actual(run,argv)
    monkeypatch.setattr(diloco,'cohosted_job',cohost)
    monkeypatch.setattr(diloco.cwctl,'launch',lambda *a,**k:{'sandbox_id':'s0','push':{},'service_url':'https://coord'})
    monkeypatch.setattr(diloco.cwctl,'share_token',lambda _:b'gateway')
    monkeypatch.setattr(diloco,'wait_coordinator',lambda *a,**k:None)
    monkeypatch.setattr(diloco.cwctl,'record',lambda _:None)
    result=diloco.launch(run,coordinator_only=True)
    assert result['state']=='coordinator-ready'
    assert captured==trainer[3:]+['--resident-data-directory','/work/diloco-resident/resident-data',
                                  '--resident-geometry-qualification','/work/diloco-resident/geometry.json']


def test_prepare_earned_parent_quarantines_roles_idempotently(tmp_path):
    import subprocess, sys, torch
    root=tmp_path/'work';root.mkdir();parent=root/'parent.pt'
    torch.save({'training':{'global_step':59,'options':{'recipe_binding':json.dumps({'sha256':'recipe'})},
        'optimizer':{'state':1},'rng':{'state':1},'exposure_counts':{'x':1},'ema':{'updates':59}}},parent)
    sha=hashlib.sha256(parent.read_bytes()).hexdigest()
    receipt={'sha256':sha,'bytes':parent.stat().st_size,'global_step':59,'requested_stop':50,'full_optimizer_rng_exposure':True}
    Path(str(parent)+'.seed.json').write_text(json.dumps(receipt))
    pin=root/'pin.json';pin.write_text(json.dumps({'path':str(parent),**receipt}))
    plan=root/'plan.json';plan.write_text(json.dumps({'recipe_sha256':'recipe','handoffs':[{}]}))
    for name in ('diloco-coordinator','diloco-learner','diloco-resident'):
        (root/name).mkdir();(root/name/'old').write_text(name)
    env={**os.environ,'NATLANG_CW_WORK':str(root),'NATLANG_CW_INSTANCE':'new-learner-0'}
    command=[sys.executable,str(ROOT/'scripts/coreweave/diloco_prepare_continuation.py'),str(pin),
             hashlib.sha256(pin.read_bytes()).hexdigest(),str(plan)]
    for _ in range(2):subprocess.run(command,env=env,check=True,capture_output=True,text=True)
    dest=root/'diloco-resident/recipe-plan.json'
    assert json.loads(dest.read_text())['handoffs'][-1]['seed_checkpoint']['global_step']==59
    quarantine=root/'.coreweave/instances/new-learner-0/inherited-previous-run'
    for name in ('diloco-coordinator','diloco-learner','diloco-resident'):
        assert (quarantine/name/'old').read_text()==name
    assert hashlib.sha256(parent.read_bytes()).hexdigest()==sha


def test_private_parent_fetch_verifies_complete_bytes_and_preserves_existing(tmp_path):
    import io
    fetch=load('diloco_fetch_parent');body=b'full optimizer rng exposure';sha=hashlib.sha256(body).hexdigest()
    cap=tmp_path/'cap';cap.write_text(json.dumps({'url':'https://storage.googleapis.com/bucket/scoped?signature=private',
        'sha256':sha,'bytes':len(body),'global_step':59,'requested_stop':50,'full_optimizer_rng_exposure':True}))
    target=tmp_path/'parent.pt';calls=[]
    def opener(url,timeout):calls.append(url);return io.BytesIO(body)
    assert fetch.fetch(cap,target,opener=opener)['global_step']==59
    assert target.read_bytes()==body and json.loads(Path(str(target)+'.seed.json').read_text())['sha256']==sha
    fetch.fetch(cap,target,opener=opener);assert len(calls)==1
    receipt_path=Path(str(target)+'.seed.json');existing=json.loads(receipt_path.read_text())
    existing['final_save_evidence']={'cadence_progress_stale':True};receipt_path.write_text(json.dumps(existing))
    fetch.fetch(cap,target,opener=opener)
    assert json.loads(receipt_path.read_text())['final_save_evidence']['cadence_progress_stale'] is True
    target.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='existing parent differs'):fetch.fetch(cap,target,opener=opener)
    bad=tmp_path/'bad.pt'
    with pytest.raises(ValueError,match='differs from exact pin'):fetch.fetch(cap,bad,opener=lambda *a,**k:io.BytesIO(b'bad'))
    assert not bad.exists() and not Path(str(bad)+'.seed.json').exists()


def test_parent_transfer_cohost_fetches_before_init_and_workers_fetch_before_verify(tmp_path):
    diloco=load('diloco')
    trainer=['python','-m','natlang_neuralese.train.native_state_sft','--continue-from','/work/parent.pt','--out','/work/diloco-learner']
    job=base_job(id='lfm',kind='train',command=trainer,env={'NATLANG_DILOCO_SEED_SHA256':'a'*64},
                 scratch={'path':'/work','size':'200Gi'})
    path=tmp_path/'job.json';path.write_text(json.dumps(job))
    run={'id':'private-run','settings':{'run_id':'private-run','workers':4},'parent_transfer':{'pin':'pin','receipt':'receipt'},
         'resident':{'directory':'/work/diloco-resident/resident-data','geometry_qualification':'/work/diloco-resident/geometry.json'},
         'coordinator':{'parent':'/work/parent.pt','prepare_parent':['true']},'learners':{'job':str(path)}}
    worker=diloco.learner_job(run,1,'https://coord')
    assert worker['env']['NATLANG_CW_INSTANCE']=='private-run-learner-1'
    assert worker['command'][2].index('diloco_fetch_parent')<worker['command'][2].index('sha256_file_hex')
    cohost=diloco.cohosted_job(run,diloco.trainer_command(run,diloco._template(str(path)))[len(diloco.TRAINER):])
    script=cohost['command'][2]
    assert script.index('diloco_fetch_parent')<script.index('native_diloco init')


def test_fresh_parent_base_fetch_pins_revision_without_credentials(tmp_path):
    fetch=load('diloco_fetch_parent');revision='a'*40;cache=tmp_path/'hf'
    snapshot=cache/'models--org--model/snapshots'/revision
    observed=[]
    def download(**kwargs):observed.append(kwargs);snapshot.mkdir(parents=True,exist_ok=True);return str(snapshot)
    base={'repo_id':'org/model','revision':revision,'cache':str(cache),'path':str(snapshot)}
    assert fetch.ensure_base(base,download=download)['revision']==revision
    assert observed==[{'repo_id':'org/model','revision':revision,'cache_dir':str(cache),'token':False}]
    snapshot.rmdir()
    with pytest.raises(ValueError,match='different snapshot'):
        fetch.ensure_base(base,download=lambda **kwargs:str(tmp_path))


def test_parent_read_secret_requires_verified_receipt_and_binds_capability(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    diloco=load('diloco');pin={'path':'/work/parent.pt','sha256':'a'*64,'bytes':123,'global_step':59,
                              'requested_stop':50,'full_optimizer_rng_exposure':True}
    identity={'sha256':pin['sha256'],'bytes':pin['bytes']}
    receipt={'private':True,'identity':identity,'independent_verification':identity,
             'run':'entity/project/run','name':'object','source':'/work/parent.pt'}
    pin_path=tmp_path/'pin';pin_path.write_text(json.dumps(pin));receipt_path=tmp_path/'receipt';receipt_path.write_text(json.dumps(receipt))
    job=base_job(id='lfm',kind='train',env={'NATLANG_DILOCO_SEED_SHA256':pin['sha256']})
    job_path=tmp_path/'job';job_path.write_text(json.dumps(job))
    remote=SimpleNamespace(size=123,url='https://storage.googleapis.com/private/object?signature=SECRET')
    fake=SimpleNamespace(Api=lambda:SimpleNamespace(run=lambda x:SimpleNamespace(file=lambda y:remote)))
    monkeypatch.setitem(sys.modules,'wandb',fake)
    run={'parent_transfer':{'pin':str(pin_path),'pin_sha256':hashlib.sha256(pin_path.read_bytes()).hexdigest(),'receipt':str(receipt_path),'receipt_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest()},'coordinator':{'parent':pin['path']},'learners':{'job':str(job_path)}}
    cap=json.loads(diloco.parent_read_secret(run)['parent-read'])
    assert cap['sha256']==pin['sha256'] and cap['global_step']==59 and cap['url']==remote.url
    receipt['independent_verification']={'sha256':'b'*64,'bytes':123};receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(diloco.cwctl.ControllerError,match='binding changed'):diloco.parent_read_secret(run)
    run['parent_transfer']['receipt_sha256']=hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='independent verification'):diloco.parent_read_secret(run)


def test_seed_publisher_accepts_only_hash_bound_final_report_for_stale_cadence(tmp_path):
    import torch
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
    from natlang_neuralese.train.native_state_sft import TRAINING_SCHEMA
    seed=load('sft_seed_parent_verify');out=tmp_path/'fresh';train=out/'train';train.mkdir(parents=True)
    checkpoint=train/'checkpoint.pt'
    training={'schema':TRAINING_SCHEMA,'global_step':750,'step':750,'optimizer':{'state':{'x':1}},
              'rng':{'cpu':torch.tensor([1])},'source_receipts':[{'sha256':'source'}],'exposure_counts':{'a':1}}
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':{'source_architecture':'lfm3'},'step':750,'training':training},checkpoint)
    (train/'progress.json').write_text(json.dumps({'checkpoint_step':700,'global_step':750}))
    report={'schema':'natlang.neuralese-native-sft-report/1','step':750,'checkpoint':str(checkpoint),'checkpoint_sha256':'0'*64}
    (train/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='does not bind'):
        seed.publish(out,tmp_path/'bad.pt',750)
    report['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (train/'report.json').write_text(json.dumps(report))
    result=seed.publish(out,tmp_path/'good.pt',750)
    assert result['checkpoint_step_observed']==700 and result['global_step']==750
    assert result['final_save_evidence']['checkpoint_sha256']==report['checkpoint_sha256']
    assert result['final_save_evidence']['cadence_progress_stale'] is True


def test_diloco_incomplete_cohort_retries_only_missing_slots(tmp_path, monkeypatch):
    diloco = load('diloco'); monkeypatch.setattr(diloco, 'RUNS', tmp_path)
    run = {'id': 'run', 'settings': {'workers': 4}}
    record = {'run': run, 'state': 'launch-incomplete', 'placement': 'cohosted',
              'coordinator': 'zero', 'url': 'https://coordinator',
              'learners': ['zero', 'one', None, None],
              'ownership': {'zero': ['coordinator', 'learner-0'], 'one': ['learner-1']}}
    (tmp_path / 'run.json').write_text(json.dumps(record))
    (tmp_path / 'run').mkdir(); (tmp_path / 'run' / 'token').write_text('token')
    monkeypatch.setattr(diloco.cwctl, 'read_sandbox', lambda ident: type('Box', (), {'status': 'running', 'get_status': lambda self: None})())
    monkeypatch.setattr(diloco.cwctl, 'share_token', lambda ident: b'gateway')
    monkeypatch.setattr(diloco.cwctl, 'record', lambda event: None)
    monkeypatch.setattr(diloco, 'wait_coordinator', lambda *a, **kw: None)
    def launch_missing(run, url, token, gateway, started, **kw):
        assert kw['worker_ids'] == [2, 3]
        for i in kw['worker_ids']: kw['on_started'](i, f'new-{i}')
    monkeypatch.setattr(diloco, 'launch_learners', launch_missing)
    result = diloco.launch_remaining('run')
    assert result['learners'] == ['zero', 'one', 'new-2', 'new-3']
    assert result['state'] == 'running'


def test_create_retry_backoff_releases_cross_process_lock(tmp_path, monkeypatch):
    import fcntl
    cwctl = load('cwctl'); monkeypatch.setattr(cwctl, 'STATE', tmp_path)
    monkeypatch.setattr(cwctl, 'CREATE_INTERVAL_SECONDS', 0)
    monkeypatch.setattr(cwctl, 'record', lambda event: None)
    monkeypatch.setattr(cwctl, 'error_facts', lambda error: {'reason': 'capacity', 'metadata': {}, 'retryable': True, 'retry_delay_seconds': 0, 'error': 'capacity'})
    calls = []
    def create():
        calls.append(1)
        if len(calls) == 1: raise RuntimeError('capacity')
        return 'ok'
    def sleep(_):
        with (tmp_path / 'create.lock').open('a+') as other:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(other, fcntl.LOCK_UN)
    assert cwctl.throttled_create(create, sleep=sleep, attempts=2) == 'ok'
