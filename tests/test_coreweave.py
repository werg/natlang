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
    ({'resources': {'cpu': '8', 'gpu': 1}}, 'needs cpu and memory'),
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


class FakeS3:
    def __init__(self):
        self.objects = {}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            error = Exception('404')
            error.response = {'Error': {'Code': '404'}}
            raise error
        return {'ContentLength': len(self.objects[Key])}

    def upload_file(self, source, bucket, key):
        self.objects[key] = Path(source).read_bytes()

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
    manifest, sources = bundle.directory_manifest(out)
    assert [Path(f['path']).name for f in manifest['files']] == ['checkpoint.pt']
    assert manifest['files'][0]['path'] == str(out.resolve() / 'checkpoint.pt')


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
