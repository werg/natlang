import pytest

from natlang_neuralese.train.memory_estimator import AdaptiveGraphMemory, geometry_bytes, selective_writer_fits


def test_direct_sft_sparse_exact_bin_uses_upper_tail_not_q80():
    model = AdaptiveGraphMemory(geometry_version='direct-sft-window-v2:fp32-rmsnorm-casts')
    kind = 'direct-sft-sequential-checkpointed-0'
    for ratio in [1.20] * 9 + [1.40]:
        model.observe(kind, 5107, 244, 1000, int(ratio * 1000))
    assert model.calibration_info(kind, 5107, 244)['quantile'] == .95
    assert model.predict(kind, 5107, 244, 1000) == 1470


def test_direct_sft_prefers_mature_exact_and_same_context_calibration():
    model = AdaptiveGraphMemory(geometry_version='direct-sft-window-v2:fp32-rmsnorm-casts')
    kind = 'direct-sft-sequential-checkpointed-0'
    exact_context, adjacent_context = 3000, 5000  # bins 11 and 12
    for ratio in [2.0] * 20:
        model.observe(kind, adjacent_context, 80, 1000, int(1000 * ratio))
    for ratio in [1.20, 1.21, 1.22]:
        model.observe(kind, exact_context, 80, 1000, int(1000 * ratio))
    # Exact-bin evidence takes precedence over the much larger adjacent ratios.
    details = model.calibration_info(kind, exact_context, 80)
    assert details['source'] == 'exact-target-bin'
    assert details['sample_count'] == 3
    assert details['context_bin_distance'] == 0
    assert model.predict(kind, exact_context, 80, 1000) == 1281

    # Same-context target pooling also wins over adjacent-context evidence.
    for ratio in [1.3, 1.4, 1.5]:
        model.observe(kind, exact_context, 256, 1000, int(1000 * ratio))
    pooled = model.calibration_info(kind, exact_context, 128)
    assert pooled['source'] == 'same-context-target-pool'
    assert pooled['sample_count'] == 6
    assert pooled['context_bin_distance'] == 0


def test_direct_sft_adjacent_context_fallback_is_qualified_and_isolated():
    version = 'direct-sft-window-v2:fp32-rmsnorm-casts'
    model = AdaptiveGraphMemory(geometry_version=version)
    kind = 'direct-sft-sequential-checkpointed-0'
    adjacent_context = 3000  # target context is bin 12; this is bin 11
    target_context = 5000
    for i in range(20):
        model.observe(kind, adjacent_context, 64 if i % 2 else 128,
                      1000, 1000 + 10 * i)
    details = model.calibration_info(kind, target_context, 80)
    assert details['source'] == 'adjacent-context'
    assert details['source_context_bin'] == 11
    assert details['source_bins'] == sorted(details['source_bins'])
    assert len(details['source_bins']) == 2  # eligible target bins pool together
    assert details['sample_count'] == 20
    assert details['quantile'] == .95
    # Nearest-rank q95 selects observation 19 of 20; the existing 5% margin remains.
    assert model.predict(kind, target_context, 80, 1000) == int(1.18 * 1000 * 1.05)

    # Do not bridge a distance-two context gap, another execution/checkpoint
    # family, or a different estimator geometry namespace.
    assert model.calibration_info(kind, 9000, 80)['source'] == 'uncalibrated'
    other_kind = 'direct-sft-packed-checkpointed-0'
    assert model.calibration_info(other_kind, target_context, 80)['source'] == 'uncalibrated'
    other_checkpoint = 'direct-sft-sequential-checkpointed-1'
    assert model.calibration_info(other_checkpoint, target_context, 80)['source'] == 'uncalibrated'
    other_geometry = AdaptiveGraphMemory(model.state_dict(), geometry_version=version + ':different')
    assert other_geometry.calibration_info(kind, target_context, 80)['source'] == 'uncalibrated'


def test_direct_sft_adjacent_requires_minimum_evidence_and_stays_within_memory_budget():
    from natlang_neuralese.train.memory_estimator import DIRECT_SFT_ADJACENT_MIN_SAMPLES
    from natlang_neuralese.train.memory_policy import choose_layer_checkpointing

    model = AdaptiveGraphMemory(geometry_version='direct-sft-window-v2:fp32-rmsnorm-casts')
    kind = 'direct-sft-sequential-checkpointed-0'
    for _ in range(DIRECT_SFT_ADJACENT_MIN_SAMPLES - 1):
        model.observe(kind, 3000, 80, 1000, 1200)
    target_context = 5000
    details = model.calibration_info(kind, target_context, 80)
    assert details['source'] == 'uncalibrated'
    assert details['sample_count'] == 0

    # One more successful observation qualifies the adjacent bin. The normal
    # device-floor admission still decides whether the uncheckpointed route fits.
    model.observe(kind, 3000, 80, 1000, 1200)
    details = model.calibration_info(kind, target_context, 80)
    assert details['source'] == 'adjacent-context'
    assert details['sample_count'] == 20
    calibration = {**details, 'sample_count': details['sample_count']}
    raw = 1000
    estimate = lambda checkpointed: (raw, model.predict(kind, target_context, 80, raw), calibration)
    need = estimate(False)[1]
    floor = 250
    device_available = need + floor + 100
    claim_available = need + floor
    checkpointed, _, predicted, used = choose_layer_checkpointing(
        estimate, floor, min(device_available, claim_available))
    assert not checkpointed
    assert predicted == need and used['source'] == 'adjacent-context'
    # Even though the device has enough space, a tighter ledger claim remains
    # the controlling bound for the same calibrated forecast and floor.
    claim_available = need + floor - 1
    checkpointed, _, _, _ = choose_layer_checkpointing(
        estimate, floor, min(device_available, claim_available))
    assert checkpointed


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


def test_joint_calibration_scales_only_the_tape_and_carries_to_an_unseen_bin():
    """A joint attempt's fixed part (resident weights, optimizer state, gradient buffers) is counted exactly; only the
    graph tape is calibrated, and a first two-writer record borrows the one-writer record's miss (Mellum E2E smoke,
    2026-10-11: 39.8 GB predicted, 72 GB observed at step 0, out of memory at step 1)."""
    gib = 2**30
    model = AdaptiveGraphMemory()
    one = {'writers': [(4582, 46)], 'reader_context': 4600}
    two = {'writers': [(4443, 46), (4582, 46)], 'reader_context': 4600}
    assert model.observe_joint(one, 40 * gib, 72 * gib, fixed=30 * gib) == pytest.approx(4.2)
    # The next update's fixed part has grown by the optimizer state; the tape ratio, not the whole-peak ratio, applies.
    predicted = model.adjust_joint(two, 74 * gib, fixed=54 * gib)
    assert predicted == int(54 * gib + 20 * gib * 4.2)
    assert predicted > 92 * gib  # routed to staging rather than attempted jointly
    assert model.adjust_joint(two, 74 * gib, fixed=74 * gib) == 74 * gib  # no tape: nothing to scale
    restored = AdaptiveGraphMemory(model.state_dict())
    assert restored.adjust_joint(two, 74 * gib, fixed=54 * gib) == predicted
    # Older whole-peak ratios are a different quantity and are not reused.
    legacy = AdaptiveGraphMemory({'joint_ratios': {'joint:0:12': [9.0]}})
    assert legacy.adjust_joint(one, 40 * gib, fixed=30 * gib) == 40 * gib
