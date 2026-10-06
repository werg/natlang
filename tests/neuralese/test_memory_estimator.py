from natlang_neuralese.train.memory_estimator import AdaptiveGraphMemory, geometry_bytes, selective_writer_fits


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
