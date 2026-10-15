"""One process-wide SDPA selection for training and its evaluator threads/jobs."""
import os


def configure_sdpa_backend(backend='default'):
    if backend not in ('default', 'math', 'flash'):
        raise ValueError('SDPA backend must be default, math or flash')
    if backend != 'default':
        import torch
        cuda = torch.backends.cuda
        cuda.enable_flash_sdp(backend == 'flash')
        cuda.enable_math_sdp(backend == 'math')
        cuda.enable_mem_efficient_sdp(False)
        cuda.enable_cudnn_sdp(False)
    os.environ['NATLANG_NEURALESE_SDPA_BACKEND'] = backend
    return backend


def flex_kernel_options(*, float32=False, query_tokens=None):
    """Decode tiles must divide the shared 128-token sparse mask blocks."""
    options = {'fwd_BLOCK_M': 32, 'fwd_BLOCK_N': 64}
    # Some Torch decode tuners prune candidates before applying fwd_ overrides.
    # Pin their initial tiles too for short queries; keep long training autotuned.
    if query_tokens is not None and query_tokens < 128:
        options.update(BLOCK_M=32, BLOCK_N=64)
    if float32:
        options['FLOAT32_PRECISION'] = "'tf32x3'"
    return options
