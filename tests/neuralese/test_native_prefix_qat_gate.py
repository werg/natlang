from natlang_neuralese.train.native_state_sft import _communication_prefix_allowed


def test_communication_prefixes_allow_inline_qat_but_exclude_external_backend():
    assert _communication_prefix_allowed(None, None)
    assert _communication_prefix_allowed(None, (object(), 'bf16', 0.0, 1.0))
    assert _communication_prefix_allowed(None, (object(), 'fp8', 0.06, 1.5))
    assert _communication_prefix_allowed(None, (object(), 'nvfp4', 0.5, 1.5))
    assert not _communication_prefix_allowed(object(), None)
    assert not _communication_prefix_allowed(object(), (object(), 'bf16', 0.0, 1.0))
