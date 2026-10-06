from natlang_neuralese.train.memory_estimator import AdaptiveGraphMemory, geometry_bytes, selective_writer_fits


def test_batch_admission_and_observation_share_current_namespace():
    from natlang_neuralese.train.memory_estimator import writer_batch_kind
    model = AdaptiveGraphMemory()
    for selective in (False, True):
        kind = writer_batch_kind(2, selective=selective)
        assert 'tape-v2' in kind
        for _ in range(3):
            model.observe(kind, 4400, 48, 1000, 1600)
        assert model.predict(kind, 4400, 48, 1000) == 1680
        # Row-normalized geometry learns only from the same live-tape policy.
        assert model.predict(writer_batch_kind(3, selective=selective), 4400, 48, 1500) == 2520
        assert model.predict(writer_batch_kind(3, selective=selective), 4400, 145, 1500) == 1575


def test_batch_calibration_excludes_legacy_auxiliary_tapes_and_other_policies():
    from natlang_neuralese.train.memory_estimator import writer_batch_kind
    model = AdaptiveGraphMemory()
    for _ in range(3):
        model.observe('writer-batch-selective:2', 4400, 48, 1000, 9000)
        model.observe(writer_batch_kind(2), 4400, 48, 1000, 5000)
    kind = writer_batch_kind(3, selective=True)
    assert not model.calibration(kind, 4400, 48)
    assert model.predict(kind, 4400, 48, 1500) == 1575


def test_adaptation_uses_executed_measurements_and_roundtrips():
    model = AdaptiveGraphMemory()
    assert model.predict('writer', 16000, 8, 1000) == 1050
    for ratio in [.4, .5, .45, .6, .55]:
        model.observe('writer', 16000, 8, 1000, ratio * 1000)
    assert model.predict('writer', 16000, 8, 1000) == 630
    restored = AdaptiveGraphMemory(model.state_dict())
    assert restored.predict('writer', 16000, 8, 1000) == 630
    # Unseen shapes start from geometry rather than another bin's cheap case.
    assert restored.predict('writer', 40000, 64, 1000) == 1050


def test_more_context_vectors_and_targets_raise_geometry_cost():
    args = dict(width=1024, layers=16, intermediate=4096, kv_width=1024,
                dtype_bytes=2, checkpointed=True)
    base = geometry_bytes(16000, 8, **args)
    assert geometry_bytes(32000, 8, **args) > base
    assert geometry_bytes(16000, 32, **args) > base
    assert geometry_bytes(16000, 8, target_tokens=400, vocab_size=65536, **args) > base
    assert geometry_bytes(16000, 8, **{**args, 'checkpointed': False}) > base
    mixed = geometry_bytes(16000, 8, uncheckpointed_layers=12, **args)
    assert base < mixed < geometry_bytes(16000, 8, **{**args, 'checkpointed': False})


def test_joint_miss_adapts_related_shapes_without_forcing_large_machine_to_stage():
    model = AdaptiveGraphMemory()
    plan = {'writers': [(18000, 8)] * 3, 'reader_context': 18000}
    model.observe_joint(plan, 6000, 6800, failed=True)
    predicted = model.adjust_joint(plan, 6000)
    assert predicted > 6800
    assert predicted > 6500  # small envelope stages
    assert predicted < 20000  # large envelope remains joint
    restored = AdaptiveGraphMemory(model.state_dict())
    assert restored.adjust_joint(plan, 6000) == predicted


def test_changed_prefix_geometry_discards_only_stale_memory_calibration():
    old = AdaptiveGraphMemory()
    old.observe('writer', 4096, 128, 100, 500)
    new = AdaptiveGraphMemory(old.state_dict(), geometry_version='shared-prefix-v1')
    assert new.samples == {} and new.joint_ratios == {}
    assert new.reset_reason == 'geometry changed from full-prefix-v1 to shared-prefix-v1'
    new.observe('writer', 4096, 128, 100, 200)
    restored = AdaptiveGraphMemory(new.state_dict(), geometry_version='shared-prefix-v1')
    assert restored.samples == new.samples
    assert not restored.calibration_reset
    args = dict(width=1024, layers=16, intermediate=4096, kv_width=1024,
                dtype_bytes=2, checkpointed=True)
    shared = geometry_bytes(4096, 128, shared_kv_prefix=True, **args)
    assert shared < geometry_bytes(4096, 128, **args)
    assert geometry_bytes(4096, 256, shared_kv_prefix=True, **args) > shared


def test_selective_writer_uses_live_budget_and_preserves_large_context_checkpoints():
    layout = dict(width=1024, layers=16, intermediate=4096, kv_width=1024,
                  dtype_bytes=2, checkpointed=True, shared_kv_prefix=True,
                  uncheckpointed_layers=0)
    kw = dict(baseline=2**30, budget=5.8 * 2**30, plain_layers=12)
    assert selective_writer_fits(4096, 128, layout, **kw)
    assert not selective_writer_fits(32768, 128, layout, **kw)
    assert selective_writer_fits(32768, 128, layout, **{**kw, 'budget': 32 * 2**30})
    assert not selective_writer_fits(4096, 128, layout, **{**kw, 'baseline': 6 * 2**30})
    assert layout['uncheckpointed_layers'] == 0


def test_replay_resource_choice_pins_true_and_false_without_remeasuring():
    from natlang_neuralese.train.memory_estimator import ReplayResourceChoice
    for decision in (False, True):
        choice = ReplayResourceChoice()
        assert choice.resolve(lambda: decision) == decision
        def changed_live_budget():
            raise AssertionError('replay must not reconsider its execution path')
        assert choice.resolve(changed_live_budget) == decision


def test_joint_producer_geometry_includes_native_gold_tape():
    from natlang_neuralese.train.memory_estimator import producer_geometry_bytes
    layout = dict(width=1024, layers=16, intermediate=4096, kv_width=1024,
                  dtype_bytes=2, checkpointed=True, shared_kv_prefix=True)
    writer = geometry_bytes(4400, 150, **layout)
    gold = geometry_bytes(4550, 0, target_tokens=150, vocab_size=65536, **layout)
    assert producer_geometry_bytes(4400, 150, layout) == writer
    assert producer_geometry_bytes(4400, 150, layout, native_gold=True, vocab_size=65536) == writer + gold
    old = AdaptiveGraphMemory(geometry_version='shared-prefix-v1')
    old.observe('writer', 4400, 150, writer, writer / 2)
    fresh = AdaptiveGraphMemory(old.state_dict(), geometry_version='shared-prefix-v1:native-gold-tape-v1')
    assert fresh.calibration_reset and not fresh.samples


def test_estimator_only_change_preserves_known_joint_failure_routes():
    old = {'geometry_version': 'shared-prefix-v1', 'samples': {'writer:12:7': [2.0]}}
    extended = AdaptiveGraphMemory(old, geometry_version='shared-prefix-v1:native-gold-tape-v1')
    assert extended.calibration_reset and not extended.samples
    assert extended.joint_routes_compatible
    changed_execution = AdaptiveGraphMemory(old, geometry_version='full-prefix-v1:native-gold-tape-v1')
    assert changed_execution.calibration_reset and not changed_execution.joint_routes_compatible


def test_local_stage_branch_tape_and_workspace_use_actual_group_shapes():
    from natlang_neuralese.train.memory_estimator import geometry_bytes, local_stage_kv_workspace
    layout = dict(width=10, layers=4, intermediate=30, kv_width=20, dtype_bytes=4, checkpointed=True)
    ordinary = geometry_bytes(100, 5, **layout)
    local = geometry_bytes(100, 5, **layout, stage_group_size=2, sketch_cutoff=1)
    # Branch tokens2²+2²+1²=9; shallow history5; upper history4.
    assert local - ordinary == (9*4 + 5*1 + 4*3) * 10 * 4
    assert geometry_bytes(100, 0, **layout, stage_group_size=2, sketch_cutoff=1) == geometry_bytes(100, 0, **layout)
    assert local_stage_kv_workspace(100, 5, 2, max_layer_kv_width=20, dtype_bytes=4) == 2*104*20*4


def test_formula_only_local_stage_upgrade_preserves_known_joint_failure_routes():
    from natlang_neuralese.train.memory_estimator import AdaptiveGraphMemory
    old = 'shared-prefix-v1:writer-latent-sketch-v2:local_stage:k4:local-group16'
    estimator = AdaptiveGraphMemory({'geometry_version': old, 'samples': {'x': [1.2]}},
                                    geometry_version=old+':local-stage-geometry-v1')
    assert estimator.joint_routes_compatible
    assert estimator.calibration_reset
    assert estimator.samples == {}
