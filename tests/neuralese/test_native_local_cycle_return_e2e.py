"""Local-cycle return wiring through the real bundle and owner CAS.

The tiny native participants, saved checkpoints, shared VAE/joint/distill
updates, role releases, portable bundle, pointer and adoption are real. Only
the expensive external task and complete numerical qualification boundaries
are replaced with structurally valid fixture evidence; this is not a production
readiness result.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import contextlib
import io
import sys

import torch

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.model.native_model import (CHECKPOINT_SCHEMA, load_native_model,
    model_state, native_identity)
from natlang_neuralese.rl import native_role_release as roles
from natlang_neuralese.rl import local_cycle_semantic_return as returned
from natlang_neuralese.train import native_semantic_stage as stage
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.train.native_state_sft import make_native_optimizer
from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA, collect_sft_progress
from natlang_neuralese.train.loop import capture_training_rng_state
from test_native_full_updates import source
from test_native_role_release import fixture as role_fixture
from test_native_semantic import setup_pair
from test_native_semantic_stage import _fixture as semantic_fixture
from test_native_task_execution_receipt import score_fixture
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'coreweave'))
import native_role_exchange as role_relay
from native_role_exchange_io import NativeRoleExchangeIO


def test_local_role_exchange_remote_matches_sdk_json_value_contract(tmp_path):
    runtime=tmp_path/'runtime';runtime.mkdir()
    provider={'sandboxes':{'cloud-fixture':{'runtime':str(runtime)}},
        'receipts':str(tmp_path/'receipts'),
        'local_source_host':{'source_host_id':'local-fixture','runtime':str(runtime),
            'python':sys.executable,'read_roots':[str(tmp_path)],
            'write_roots':[str(tmp_path)]}}
    io=NativeRoleExchangeIO(provider)
    assert io.remote('local-fixture','import json;print(json.dumps({"value":7}))')=={'value':7}
    assert io.remote('local-fixture','pass') is None


def _ids_and_sources(tmp_path):
    result = {}
    for index, name in enumerate(('lfm', 'mellum')):
        base = tmp_path / ('base-' + name)
        dtype = source(base, name)
        model, tokenizer, _, config = load_native_model(base, dtype=dtype, rank=4)
        config = {**config, 'base_inventory_sha256': canonical_json_sha256_hex({'base': name})}
        optimizer, _ = make_native_optimizer(model, lr=3e-4 if name == 'lfm' else 7e-5,
            latent_ns='cubic5' if name == 'lfm' else 'quintic', in_backward=False)
        # Give each participant real inherited optimizer history at its source
        # boundary; semantic training then continues those exact moments.
        sum(parameter.float().square().mean() for parameter in model.parameters()
            if parameter.requires_grad).backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        directory = tmp_path / ('owner-' + name); directory.mkdir()
        checkpoint = directory / 'parent.pt'
        rng = capture_training_rng_state('cpu')
        source_identity = native_identity(model, config, checkpoint_sha256='0' * 64)
        training = {'schema': TRAINING_SCHEMA, 'global_step': 1, 'step': 1,
            'optimizer': optimizer.state_dict(), 'rng': rng, 'source_rng': rng,
            'identity': {'optimizer_options': {'lr': optimizer.param_groups[0]['lr'],
                'weight_decay': 0.0, 'latent_ns': 'cubic5' if name == 'lfm' else 'quintic',
                'learned_row_lr': None, 'interface_row_lr': None, 'native_optimizer_migration': None},
                'fp8': {'kind': 'off', 'recipe': 'delayed', 'amax_history_len': 16}},
            'options': {'stage_id': 'native-graph', 'epochs': 1,
                'native_snapshot_ancestors': [], 'optimizer_state_storage': 'device'},
            'source_receipts': [{'kind': 'fixture-source', 'sha256': ('a' if name == 'lfm' else 'b') * 64}],
            'exposure_counts': {'parent-row': 1}, 'sft_progress': collect_sft_progress({
                'schema': TRAINING_SCHEMA, 'step': 1, 'global_step': 1,
                'identity': {'objective': None}, 'source_row_bindings': {'parent-row': 'a' * 64},
                'source_row_order': ['parent-row'], 'exposure_counts': {'parent-row': 1},
                'options': {'stage_id': 'native-graph', 'epochs': 1}})}
        torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': config,
            **model_state(model), 'training': training}, checkpoint)
        identity = native_identity(model, config, checkpoint_sha256=sha256_file_hex(checkpoint))
        source_identity = {**source_identity, 'checkpoint_sha256': identity['checkpoint_sha256']}
        result[name] = {'base': base, 'model': model, 'tokenizer': tokenizer,
            'config': config, 'optimizer': optimizer, 'checkpoint': checkpoint,
            'identity': identity, 'training': training, 'source_identity': source_identity}
    return result


def _use_actual_tokenizers(row, participants, aliases, *, suffix):
    from natlang_neuralese.common.hashing import sha256_hex
    text = 't4'
    for role, view in row['models'].items():
        owner_name = next(name for name in participants if aliases[name] == role)
        owner = participants[owner_name]
        encoded = owner['tokenizer'](text, add_special_tokens=False, return_offsets_mapping=True)
        payload = list(encoded['input_ids'])
        offsets = [tuple(pair) for pair in encoded['offset_mapping']]
        assert payload and any(end > start for start, end in offsets)
        produce_open, produce_close, consume_open, consume_close = owner['model'].marker_ids
        graph = view['graph']
        graph['id'] = role + ':' + suffix
        graph['tokenizer_fingerprint'] = owner['identity']['tokenizer_fingerprint']
        graph['nodes'] = [
            {'name': 'writer', 'token_ids': [1, produce_open, *payload, produce_close],
             'target_weights': [0.] + [1.] * (len(payload) + 2),
             'producing': [[2, 2 + len(payload)]]},
            {'name': 'consumer', 'token_ids': [3, consume_open, consume_close, 12, 13],
             'target_weights': [0., 1., 0., 1., 1.],
             'imports': [{'at': 2, 'source': 'writer', 'start': 2, 'stop': 2 + len(payload)}]}]
        view['export'] = {'node': 'writer', 'start': 2, 'stop': 2 + len(payload),
            'text': text, 'offsets': offsets}
        view['consumer_nodes'] = ['consumer']
        view['witness']['payload_sha256'] = sha256_hex(text.encode())
        graph['source_pairing'] = {'consumer_source_row_id': 'reader:' + suffix,
            'consumer_original_text_sha256': sha256_hex(text.encode()), 'consumed_occurrences': [{
                'source_row_id': 'reader:' + suffix, 'payload_sha256': sha256_hex(text.encode()),
                'wrapped_token_start': 2, 'wrapped_token_stop': 2 + len(payload), 'consumer_import_at': 2}]}
        graph['source_pairing']['communication_sha256'] = row['communication_sha256']
    row['payload_sha256'] = sha256_hex(text.encode())
    return row


def test_role_tick_exports_real_descendants_and_local_owner_adopts_them(tmp_path, monkeypatch):
    from natlang_neuralese.train.native_resident import binding
    from natlang_neuralese.train.native_readiness import validate_ready
    from natlang_neuralese.train.native_semantic import SemanticParticipant
    from natlang_neuralese.model.native_bundle import verify_bundle
    from natlang_neuralese.eval import native_backend
    from natlang_neuralese.eval import native_semantic_runtime
    from natlang_neuralese.train.native_continuation import progress_obligations
    from test_native_readiness import ready

    # Keep the genuine CheckpointPolicy reader for all checkpoint and bundle
    # verification; the older protocol helper replaces it only while building
    # its unrelated fixture.
    real_checkpoint_load = CheckpointPolicy.load
    participants = _ids_and_sources(tmp_path)
    aliases = {name: f"{participants[name]['config']['architecture']}:"
        f"{participants[name]['config']['base_inventory_sha256']}" for name in participants}
    rows = role_fixture(tmp_path, monkeypatch, names=('lfm', 'mellum'))
    monkeypatch.setattr(CheckpointPolicy, 'load', real_checkpoint_load)

    # Turn each ready fixture into exact source custody for a real full-state
    # tiny native checkpoint. Qualification facts remain fixture-scoped.
    for role, row in rows.items():
        owner = participants[role]
        old = row['receipt']
        readiness_path = Path(old['path']) if 'path' in old else Path(row['state']['models'][role]['readiness']['path'])
        receipt = json.loads(readiness_path.read_text())
        identity = owner['identity']; model_id = aliases[role]
        checkpoint_ref = {**binding(owner['checkpoint']), 'schema': CHECKPOINT_SCHEMA,
            'bytes': owner['checkpoint'].stat().st_size}
        receipt.update(model_id=model_id, checkpoint=checkpoint_ref, native_identity=identity)
        report_path = Path(receipt['training_receipt']['path'])
        report = json.loads(report_path.read_text())
        report['checkpoint_sha256'] = checkpoint_ref['sha256']
        report['model_config'] = {'architecture': owner['config']['architecture'],
            'base_inventory_sha256': owner['config']['base_inventory_sha256'],
            'marker_ids': [58, 59, 60, 61]}
        profile = {'model_config_sha256': canonical_json_sha256_hex(report['model_config']),
            'fp8': {'kind': 'off'}}
        report['qualification_profile'] = profile
        write_json_durable(report_path, report)
        receipt['training_receipt'] = binding(report_path)
        for field in ('runtime_certificate', 'task_execution_receipt'):
            path = Path(receipt[field]['path']); value = json.loads(path.read_text())
            value['native_identity'] = identity
            if field == 'runtime_certificate':
                value['qualification_profile'] = profile
                complete = value['complete_gradient_qualification']
                complete['native_identity'] = identity
                complete['model_config'] = report['model_config']
                complete['model_config_sha256'] = profile['model_config_sha256']
                value['complete_gradient_qualification'] = complete
            write_json_durable(path, value); receipt[field] = binding(path)
        write_json_durable(readiness_path, receipt)
        row['receipt'] = receipt
        row['requirement'] = row['plan']['models'][0]['handoff_requirement']
        row['requirement']['model_id'] = model_id
        row['requirement']['readiness'] = str(readiness_path)
        row['actual_model_id'] = model_id
        row['actual_readiness_path'] = readiness_path
        row['actual_recipe'] = receipt['recipe']
    for row in rows.values():
        for declaration in row['plan']['models']:
            source_role = declaration['model']
            declaration['handoff_requirement']['model_id'] = aliases[source_role]
            declaration['handoff_requirement']['readiness'] = str(rows[source_role]['actual_readiness_path'])
            declaration['native_participant'] = aliases[source_role]
            declaration['production_recipe'] = rows[source_role]['actual_recipe']
            for field in ('cases', 'seed_native_graphs', 'held_native_graphs'):
                path = tmp_path / (source_role + '-' + field + '.json')
                if not path.is_file():
                    write_json_durable(path, {'fixture': field})
                declaration[field] = binding(path)
        row['state']['models'][row['plan']['owned_models'][0]].update(
            checkpoint={**binding(participants[row['plan']['owned_models'][0]]['checkpoint']),
                'schema': CHECKPOINT_SCHEMA, 'bytes': participants[row['plan']['owned_models'][0]]['checkpoint'].stat().st_size},
            readiness=binding(rows[row['plan']['owned_models'][0]]['actual_readiness_path']))

    exchange_dir = tmp_path / 'shared-exchange'; exchange_dir.mkdir()
    for role, row in rows.items():
        row['plan']['role_exchange']['directory'] = str(exchange_dir)
        row['config']['out'] = str(tmp_path / role / 'production')
        row['config']['native_actor_mode'] = 'raw-state'
        row['config']['native_participant'] = aliases[role]
        owner = participants[role]
        runtime = tmp_path / ('runtime-' + role); (runtime / 'scripts').mkdir(parents=True)
        collector = runtime / 'scripts' / 'run_bonsai_queue.py'; collector.write_text('# fixture asset\n')
        row['runtime'] = runtime
        plan_path = row['plan_path']; write_json_durable(plan_path, row['plan'])
        out = Path(row['config']['out']); out.mkdir(parents=True)
        config_path = tmp_path / role / 'rl-config.json'
        write_json_durable(config_path, row['config'])
        cycles_out = tmp_path / role / 'cycles'; cycles_out.mkdir()
        lock = tmp_path / role / 'optimizer-owner.lock'; lock.touch()
        cycle_plan_path = tmp_path / role / 'cycles-plan.json'
        write_json_durable(cycle_plan_path, {'out': str(cycles_out), 'rl_config': str(config_path),
            'optimizer_owner_lock': str(lock), 'local_cycle_role_plan': binding(plan_path)})
        owner_identity_path = tmp_path / role / 'owner.json'
        owner_identity = {'pid': 99999999, 'start_ticks': '1', 'hostname': 'fixture'}
        write_json_durable(owner_identity_path, owner_identity)
        state = {'schema': 'natlang.rl-production-state/1', 'checkpoint': str(owner['checkpoint']),
            'native_participant': aliases[role], 'phase': 'collect', 'released': True,
            'owner': owner_identity,
            'readiness': binding(Path(row['state']['models'][role]['readiness']['path']))}
        state_path = out / 'state.json'; write_json_durable(state_path, state)
        cursor_path = cycles_out / 'cursor.json'; write_json_durable(cursor_path,
            {'cycle': 0, 'phase': 'supervised', 'stage_index': 1})
        row['plan']['local_cycle_source'] = {'schema': 'natlang.native-local-cycle-source/1',
            'cycles_plan': str(cycle_plan_path), 'rl_config': binding(config_path),
            'readiness': str(row['state']['models'][role]['readiness']['path']),
            'state_path': str(state_path), 'cursor_path': str(cursor_path),
            'optimizer_owner_lock': str(lock), 'owner_identity_path': str(owner_identity_path),
            'return_directory': str(exchange_dir)}
        write_json_durable(plan_path, row['plan'])
        cycle_plan = json.loads(cycle_plan_path.read_text())
        cycle_plan['local_cycle_role_plan'] = binding(plan_path)
        write_json_durable(cycle_plan_path, cycle_plan)
        row.update(config_path=config_path, state_path=state_path, cursor_path=cursor_path,
            lock=lock, cycle_plan_path=cycle_plan_path, owner_identity_path=owner_identity_path)

    # Publish real full-state source releases through the normal current-owner
    # validator. Both roles share a fixed cohort and actual ready identities.
    releases = {}
    for role, row in rows.items():
        releases[role] = roles.publish_local_cycle(row['plan_path'], kind='released')
        assert releases[role]
        assert roles.validate(releases[role], row['plan'], kind='released')[0] == role

    # Prepare the actual paired source and run the native VAE/joint/distill
    # trainer. It receives the real parent models/optimizers; only the paired
    # row's miniature tokenization wrapper comes from the CPU fixture.
    stage_fixture = tmp_path / 'stage-fixture'; stage_fixture.mkdir()
    config_path, manifest_path, config, pair = semantic_fixture(stage_fixture, monkeypatch)
    fresh_pairs, _, _ = setup_pair()
    ids = aliases
    pair['models'] = {ids[name]: pair['models'][name] for name in pair['models']}
    pair['id'] = 'actual-native-paired-parent'
    _use_actual_tokenizers(pair, participants, ids, suffix=pair['id'])
    train_path = Path(config['pairs']['path']); train_path.write_text(json.dumps(pair) + '\n')
    config['pairs']['sha256'] = sha256_file_hex(train_path)
    held_path = Path(config['held_pairs']['path'])
    held = json.loads(held_path.read_text())
    held['models'] = {ids[name]: held['models'][name] for name in held['models']}
    _use_actual_tokenizers(held, participants, ids, suffix='held:' + held['id'])
    held_path.write_text(json.dumps(held) + '\n')
    config['held_pairs']['sha256'] = sha256_file_hex(held_path)
    manifest = json.loads(manifest_path.read_text())
    manifest['models'] = []
    source_participants = {}
    for role, owner in participants.items():
        model_id = ids[role]
        source_participants[model_id] = owner
        item = next(binding_row for binding_row in json.loads(manifest_path.read_text())['models']
            if binding_row['model_id'] == role)
        item['model_id'] = model_id
        item['checkpoint'] = {**binding(owner['checkpoint']), 'schema': CHECKPOINT_SCHEMA,
            'bytes': owner['checkpoint'].stat().st_size}
        item['native_identity'] = owner['identity']
        item.pop('readiness', None)
        manifest['models'].append(item)
        item['native_identity'] = owner['identity']
    manifest['signature'] = canonical_json_sha256_hex(manifest['models'])
    write_json_durable(manifest_path, manifest)
    config['models'] = {ids[name]: {'fp8_training': 'off', 'base': str(owner['base'])}
        for name, owner in participants.items()}
    # The returned checkpoint must be based on the exact released pair.
    consumer_plan = rows['mellum']['plan_path']
    config['snapshot_adoption'] = {'schema': 'natlang.native-snapshot-adoption/1',
        'plan': binding(consumer_plan), 'releases': [releases['lfm'], releases['mellum']]}
    write_json_durable(config_path, config)
    original_loader = stage.load_participant
    # Keep participant weights/checkpoints/optimizer state actual; the only
    # override is avoiding a GPU-only Mellum FP8 backend in this CPU wiring test.
    def load_source(name, bind, config_value, device):
        owner = source_participants[name]
        fake = fresh_pairs['lfm' if name == ids['lfm'] else 'mellum']
        participant = SemanticParticipant(owner['model'], owner['tokenizer'], owner['config'], owner['identity'],
            owner['optimizer'], sft_progress=owner['training']['sft_progress'], global_step=1,
            recipe_binding=rows['lfm' if name == ids['lfm'] else 'mellum']['receipt']['recipe'])
        participant.optimizer_options = owner['training']['identity']['optimizer_options']
        participant.optimizer_state_storage = owner['training']['options']['optimizer_state_storage']
        participant.fp8_profile = owner['training']['identity']['fp8']
        participant.source_rng = owner['training']['source_rng']
        participant.snapshot_ancestors = []
        return participant
    monkeypatch.setattr(stage, 'load_participant', load_source)

    # Fixture-only qualification boundaries emit strictly bound receipts. The
    # task score, completion schema, checkpoint, adapter, source releases and
    # downstream verifiers remain real and are checked byte-for-byte.
    def fake_tasks(live, export, config_value, out, step, device, **kwargs):
        from natlang_neuralese.eval.native_state import task_execution_receipt
        result = {}
        for name in live:
            directory = Path(out) / ('fixture-task-' + canonical_json_sha256_hex(name)); directory.mkdir(parents=True)
            summary, _, _ = score_fixture(directory)
            identity = export['models'][name]['native_identity']
            receipt = task_execution_receipt(summary, identity,
                checkpoint_sha256=export['checkpoint']['sha256'])
            receipt['semantic_adapter_sha256'] = sha256_file_hex(Path(out) / 'semantic-adapter.json')
            path = directory / 'native-task-execution-receipt.json'; write_json_durable(path, receipt)
            result[name] = {'path': str(path.resolve()), 'sha256': sha256_file_hex(path), 'passed': True}
        return result
    monkeypatch.setattr(stage, 'measure_tasks', fake_tasks)
    template_root = tmp_path / 'gradient-fixtures'; template_root.mkdir()
    template_proofs = {}
    for role, owner in participants.items():
        ready_path, receipt, _ = ready(template_root, ids[role])
        template_proofs[ids[role]] = receipt['runtime_certificate']
    def fake_complete(model, tokenizer, roles_arg, model_config, *, checkpoint_sha256, **kwargs):
        from natlang_neuralese.model.native_model import native_identity as identify
        identity = identify(model, model_config, checkpoint_sha256=checkpoint_sha256)
        profile = {'model_config_sha256': canonical_json_sha256_hex({'architecture': model_config['architecture']}),
            'fp8': {'kind': 'off'}}
        model_name = next(name for name, p in source_participants.items()
            if p['config']['architecture'] == model_config['architecture'])
        proof = json.loads(Path(template_proofs[model_name]['path']).read_text())['complete_gradient_qualification']
        proof.update(native_identity=identity, model_config={'architecture': model_config['architecture']},
            model_config_sha256=profile['model_config_sha256'])
        return {'schema': 'natlang.native-runtime-certificate/1', 'passed': True,
            'native_identity': identity, 'backbone_optimizer_updates': 1,
            'qualification_profile': profile, 'complete_gradient_qualification': proof}
    monkeypatch.setattr(native_backend, 'qualify_complete_runtime', fake_complete)
    def fake_transport(live, channel, **kwargs):
        return {'schema': 'natlang.native-semantic-transport-certificate/1', 'passed': True,
            'adapter_sha256': channel.adapter_sha256,
            'directions': [{'source': a, 'target': b, 'passed': True,
                'mode': 'learned-semantic-state-deltas', 'wire_layer_tensors': 0,
                'source_native_identity': channel.bindings[a], 'target_native_identity': channel.bindings[b]}
                for a in live for b in live if a != b]}
    monkeypatch.setattr(native_semantic_runtime, 'qualify_semantic_transport', fake_transport)
    # Avoid accidentally retaining the original stage fixture callback in a
    # local variable; the real stage runner uses the actual parents above.
    del original_loader
    stage_out = tmp_path / 'stage-out'
    completion = stage.run(config_path, manifest_path, stage_out, device='cpu')
    assert completion['completed'] and completion['production_ready']
    completed_curriculum = json.loads(Path(completion['curriculum']['path']).read_text())
    assert completed_curriculum['snapshot_adoption']['plan'] == binding(rows['mellum']['plan_path']), (
        completed_curriculum['snapshot_adoption']['plan'], binding(rows['mellum']['plan_path']))

    # Run the actual relay tick. Its fake IO only transports already bound
    # local files and executes the source-host Python role helpers in-process.
    source_ids = {role: row['plan']['role_exchange']['roles'][role]['source_host_id']
        for role, row in rows.items()}
    relay_config = {'schema': 'natlang.native-role-exchange-service/1',
        'hosts': {role: {'source_host_id': source_ids[role], 'plan': binding(row['plan_path']),
            'raw_state': str(row['state_path'])} for role, row in rows.items()},
        'sandboxes': {source_ids[role]: {'runtime': str(row['runtime'])} for role, row in rows.items()},
        'task_eval_collector_assets': {role: {'kind': 'task-source-file',
            'path': str(row['runtime'] / 'scripts/run_bonsai_queue.py'),
            'bytes': (row['runtime'] / 'scripts/run_bonsai_queue.py').stat().st_size,
            'sha256': sha256_file_hex(row['runtime'] / 'scripts/run_bonsai_queue.py')}
            for role, row in rows.items()},
        'local_cycle_returns': {'schema': 'natlang.native-local-cycle-return-transport/1',
            'consumer': 'mellum', 'completion_path': str(stage_out / 'shared-stage-completion.json'),
            'directory': str(exchange_dir), 'role_plans': {role: binding(row['plan_path'])
                for role, row in rows.items()}}}
    def test_scoped_binding(reference):
        path = Path(reference['path']).resolve(strict=True)
        if not path.is_relative_to(tmp_path.resolve()):
            raise ValueError('fixture transport escaped its isolated source namespace')
        actual = {'path': str(path), 'sha256': sha256_file_hex(path), 'bytes': path.stat().st_size}
        if any(reference.get(key) != actual[key] for key in ('path', 'sha256', 'bytes') if key in reference):
            raise ValueError('fixture transport changed a bound source byte')
        return actual
    monkeypatch.setattr(role_relay, 'checked_binding', test_scoped_binding)
    class LocalIO:
        local = {'source_host_id': source_ids['lfm']}
        config = relay_config
        def has_host(self, sid): return sid in source_ids.values()
        def dependency_code(self, sid): return ''
        def remote(self, sid, code, timeout=120):
            capture = io.StringIO()
            # Match the provider adapters: execute the exact role code in the
            # source-host namespace and decode its single JSON result.
            with contextlib.redirect_stdout(capture): exec(code, {'json': json})
            output = capture.getvalue().strip()
            return json.loads(output) if output else None
        def read(self, sid, path): return Path(path).read_bytes()
        def read_optional(self, sid, path): return Path(path).read_bytes() if Path(path).is_file() else None
        def mirror_many(self, source_sid, target_sid, files):
            for item in files:
                path = Path(item['path'])
                if path.stat().st_size != item.get('bytes', path.stat().st_size) or sha256_file_hex(path) != item['sha256']:
                    raise ValueError('fixture transport saw changed bound bytes')
            return True
        def verify_assets(self, sid, assets):
            for item in assets:
                path = Path(item['path'])
                assert path.stat().st_size == item['bytes'] and sha256_file_hex(path) == item['sha256']
    relay_result = role_relay.tick(relay_config, LocalIO())
    assert relay_result['status'] == 'local-cycle-descendants-returned'
    assert len(relay_result['participants']) == 2
    for row in relay_result['participants']:
        bundle = verify_bundle(Path(row['receipt']['path']).parent / 'semantic-return-bundles' / row['participant'] / completion['checkpoint']['sha256'])
        assert bundle['participant'] == row['participant'] and bundle['semantic_adapter']

    # The actual local owner consumes the pointer through the ordinary durable
    # state/cursor adoption transaction, with zero synthetic training credit.
    lfm = rows['lfm']; owner_id = aliases['lfm']
    owner_plan = json.loads(lfm['cycle_plan_path'].read_text())
    owner_plan['semantic_return'] = {'return_directory': str(exchange_dir),
        'role_plan': str(lfm['plan_path']), 'role': 'lfm'}
    owner_config = {'native_participant': owner_id}
    adopted = returned.adopt_at_boundary(owner_config, owner_plan,
        json.loads(lfm['cursor_path'].read_text()), state_path=lfm['state_path'], cursor_path=lfm['cursor_path'])
    assert adopted and adopted['training_credit'] == 0
    state = json.loads(lfm['state_path'].read_text()); cursor = json.loads(lfm['cursor_path'].read_text())
    assert state['checkpoint'] == adopted['checkpoint']['path']
    assert state['semantic_adapter'] == adopted['semantic_adapter']['path']
    assert cursor['checkpoint'] == adopted['checkpoint']
    assert 'training_credit' not in cursor and 'training_credit' not in state

