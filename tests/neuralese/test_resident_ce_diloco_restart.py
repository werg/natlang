"""Real resident objective handoff through an HTTP coordinator across a crash."""
import copy
import json
import socket
import threading
import time

import pytest
import torch

from test_native_model import tiny_source
import test_native_resident_ce as ce_fixture
from test_native_resident_ce import fixture
from test_native_source_fetch_recovery import equal
from test_own_output_supervision import objective
from natlang_neuralese.common.jsonio import write_json_durable as write_json
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.model.native_model import load_native_checkpoint, load_native_model
from natlang_neuralese.train import native_resident as resident, native_state_sft as trainer
from natlang_neuralese.train.own_output_supervision import identity as objective_identity, policy_objective
from natlang_neuralese.train.recipe import load_recipe, native_recipe_binding, resolve_stage_inputs
from natlang_neuralese.train.diloco.coordinator import initialize_coordinator, serve_coordinator, TOKEN_ENV
from natlang_neuralese.train.diloco.protocol import TensorSurface
from natlang_neuralese.train.diloco.worker import DilocoSettings, Transport
from natlang_neuralese.train.native_diloco import identity


def test_real_same_source_ce_http_commit_crash_resume(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    monkeypatch.setenv(TOKEN_ENV, 'x' * 32)
    base = tmp_path / 'base'; tiny_source(base)
    _, _, roles, config = load_native_model(base, dtype=torch.float32, rank=4)
    original_ordinary = ce_fixture.ordinary
    def templated_ordinary():
        row = original_ordinary()
        row['nodes'][0]['token_ids'] = [10, 12, 40, 17, 10, 13, 41, 17]
        row['nodes'][0]['target_weights'] = [0., 0., 0., 0., 0., 0., 1., 1.]
        row['conversion']['source_suffix_start'] = 6
        return row
    monkeypatch.setattr(ce_fixture, 'ordinary', templated_ordinary)
    paths, _, _ = fixture(tmp_path, config['tokenizer_fingerprint'])
    source = NativeGraphSource(paths['old_train'], config['tokenizer_fingerprint'], required_split='train', role_ids=roles.ids)
    mapping = tmp_path / 'family.json'
    write_json(mapping, {'schema': 'natlang.logical-family-map/1', 'source_sha256': source.file_sha256,
                         'rows': [{'id': e['id'], 'binding': e['binding'], 'family': 'ordinary'} for e in source.entries]})
    before = json.loads(paths['old_ce'].read_text())
    before['objective'] = objective(mass='token', family_balance={'key': 'logical-family', 'max_scale': 8.,
        'normalization': 'bounded-mean-one/1', 'map': resident.binding(mapping)})
    before['objective']['role_tokens']['tokenizer_fingerprint'] = config['tokenizer_fingerprint']
    write_json(paths['old_ce'], before)
    after = copy.deepcopy(before)
    after['objective']['family_balance'].update(normalization='declared/1', scales={'ordinary': 1.2})
    after['objective_transition'] = {'from': objective_identity(policy_objective(before))['sha256']}
    write_json(paths['new_ce'], after)
    old_recipe = json.loads(paths['old_recipe'].read_text())
    old_recipe['input_bindings']['ce'] = resident.binding(paths['old_ce'])
    write_json(paths['old_recipe'], old_recipe)
    new_recipe = copy.deepcopy(old_recipe); new_recipe['input_bindings']['ce'] = resident.binding(paths['new_ce'])
    write_json(paths['new_recipe'], new_recipe)
    loaded = load_recipe(paths['old_recipe'])
    plan = {'schema': 'natlang.neuralese-recipe-plan/1', 'recipe': loaded,
            'recipe_sha256': resident.sha(paths['old_recipe']), 'device': 'cpu',
            'code': resident.code_inventory(__import__('pathlib').Path(resident.__file__).parents[1]),
            'inputs': {}, 'stage_inputs': {'sft': resolve_stage_inputs(loaded, loaded['stages'][0], {})}}
    out = tmp_path / 'run' / 'sft'; out.mkdir(parents=True)
    directory = out.parent / 'resident-data'; write_json(out.parent / 'recipe-plan.json', plan)
    seedout = tmp_path / 'seed'
    common = ['--base', str(base), '--dtype', 'float32', '--rank', '4', '--device', 'cpu',
              '--native-graphs', str(paths['old_train']), '--held-native-graphs', str(paths['held']),
              '--ce-supervision-policy', str(paths['old_ce']), '--epochs', '2', '--batch-graphs', '1',
              '--accumulate-batches', '1', '--eval-every', '1', '--checkpoint-every', '1', '--checkpoint-seconds', '0']
    trainer.main(common + ['--out', str(seedout), '--initialize-only'])
    seed = seedout / 'checkpoint.pt'; model, *_ = load_native_checkpoint(seed, base=base, device='cpu')
    with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    settings = DilocoSettings(run_id='ce-crash', workers=1, worker_id=0, coordinator_url=url,
                              allow_private_http=True, local_steps=1, wire_dtype='fp32', heartbeat_seconds=.1)
    binding = native_recipe_binding(paths['old_recipe'], plan['recipe_sha256'])
    argv = common + ['--out', str(out), '--continue-from', str(seed), '--resident-data-directory', str(directory),
                     '--recipe-binding', json.dumps(binding), '--stage-id', 'sft', '--diloco', json.dumps(settings.__dict__)]
    refs = [resident.binding(p) for p in (paths['old_train'], paths['old'], paths['held'], paths['old_ce'], mapping, paths['old_recipe'])]
    role = {'kind': 'sft', 'identity': identity(seed, argv, settings), 'dependencies': refs}
    authority = {'schema': 'natlang.diloco-heterogeneous-identity/1', 'baseline_sha256': resident.sha(seed),
                 'recipe_lineage_sha256': binding['lineage_sha256'], 'diloco': settings.identity(), 'roles': {'0': [role]}}
    root = tmp_path / 'outer'; initialize_coordinator(root, run_id='ce-crash',
        masters=TensorSurface(dict(trainer.native_trainable_parameters(model))).masters, identity=authority,
        workers=1, min_quorum=1, grace_seconds=0, fragment_bytes=1 << 28, seed_source={'sha256': resident.sha(seed)})
    stop = threading.Event(); thread = threading.Thread(target=serve_coordinator,
        args=(root, '127.0.0.1', port, 'x' * 32), kwargs={'stop': stop}, daemon=True); thread.start()
    transport = Transport(url, 'x' * 32, 10, allow_http=True)
    for _ in range(100):
        try: transport.json('GET', '/v1/manifest'); break
        except OSError: time.sleep(.02)
    original_init = resident.ResidentAdoption.__init__; original_save = trainer._save
    published = []
    def initialize(self, *args, **kwargs):
        proof = tmp_path / 'evidence.json'; write_json(proof, {'passed': True})
        capacity = tmp_path / 'capacity.json'; write_json(capacity, {'schema': 'natlang.native-resident-geometry-qualification/1',
            'passed': True, 'identity': kwargs['geometry_identity'], 'runtime_code_sha256': resident.digest(plan['code']),
            'limits': {'forward_tokens': 1000000, 'kv_slots': 1000000}, 'evidence': resident.binding(proof)})
        kwargs['geometry_qualification'] = capacity; original_init(self, *args, **kwargs)
    def save(policy, *args, **kwargs):
        result = original_save(policy, *args, **kwargs)
        if kwargs['step'] == 1 and not published:
            published.append(resident.prepare(directory, paths['new_recipe'], 'sft',
                tokenizer_fingerprint=config['tokenizer_fingerprint'], role_ids=roles.ids))
        return result
    original_poll = resident.ResidentAdoption.poll
    class Crash(BaseException): pass
    def crash(self, step, **kwargs):
        result = original_poll(self, step, **kwargs)
        if result is not None: raise Crash()
        return result
    try:
        with monkeypatch.context() as patch:
            patch.setattr(resident.ResidentAdoption, '__init__', initialize)
            patch.setattr(trainer, '_save', save); patch.setattr(resident.ResidentAdoption, 'poll', crash)
            with pytest.raises(Crash): trainer.main(argv)
        assert resident.current(directory)
        checkpoint = torch.load(out / 'checkpoint.pt', weights_only=False, map_location='cpu')
        assert checkpoint['training']['global_step'] == 1
        original_update = trainer.native_training_update
        resumed = []
        def observe_update(model, optimizer, batch, **kwargs):
            if not resumed:
                resumed.append(True)
                equal(optimizer.state_dict(), checkpoint['training']['optimizer'])
                equal(trainer.capture_training_rng_state('cpu'), checkpoint['training']['rng'])
            return original_update(model, optimizer, batch, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(resident.ResidentAdoption, '__init__', initialize)
            patch.setattr(trainer, 'native_training_update', observe_update)
            assert trainer.main(argv) == 0
        assert resumed
        final = torch.load(out / 'checkpoint.pt', weights_only=False, map_location='cpu')
        assert final['training']['global_step'] == 2
        assert final['training']['exposure_counts'] == {'old': 2}
        assert final['training']['options']['ce_supervision_policy_binding'] == resident.binding(paths['new_ce'])
        assert final['training']['diloco']['contract_history']
        assert transport.json('GET', '/v1/status')['versions'][0] >= 1
        assert final['training']['optimizer'] and final['training']['rng']
    finally:
        stop.set(); thread.join(timeout=5)
