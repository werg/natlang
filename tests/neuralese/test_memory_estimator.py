from natlang_neuralese.train.memory_estimator import AdaptiveGraphMemory, geometry_bytes


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
