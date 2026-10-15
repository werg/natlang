from natlang_neuralese.common.attention_backend import flex_kernel_options


def test_decode_tiles_divide_shared_sparse_blocks_and_preserve_precision():
    for fp32 in (False, True):
        options = flex_kernel_options(float32=fp32)
        assert 128 % options['fwd_BLOCK_M'] == 0
        assert 128 % options['fwd_BLOCK_N'] == 0
        assert ('FLOAT32_PRECISION' in options) == fp32
        assert 'BLOCK_M' not in options and 'BLOCK_N' not in options
    assert flex_kernel_options(float32=True)['FLOAT32_PRECISION'] == "'tf32x3'"
    changed = flex_kernel_options()
    changed['fwd_BLOCK_M'] = 1024
    assert flex_kernel_options()['fwd_BLOCK_M'] == 32


def test_short_query_initial_decode_tiles_are_valid_before_prefixed_override():
    for queries in [2, 28, 33, 127]:
        options = flex_kernel_options(query_tokens=queries)
        assert options['BLOCK_M'] == options['fwd_BLOCK_M'] == 32
        assert options['BLOCK_N'] == options['fwd_BLOCK_N'] == 64
    for queries in [128, 32768]:
        options = flex_kernel_options(query_tokens=queries)
        assert 'BLOCK_M' not in options and 'BLOCK_N' not in options
