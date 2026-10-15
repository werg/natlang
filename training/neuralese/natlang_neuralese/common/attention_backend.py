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
