"""Restore the exact execution settings declared by a capacity certificate.

This is diagnostic plumbing, not a choice of training policy or arithmetic.
Keep the observer's actual model settings and its identity reconstruction equal.
"""
from types import SimpleNamespace
import os
import torch


def restore_qualification_execution(model, identity, *, device='cuda'):
    arithmetic = identity.get('execution_arithmetic')
    if arithmetic is not None:
        if (not isinstance(arithmetic, dict)
                or set(arithmetic) != {'native_compile', 'bf16_reduced_precision_reduction'}
                or arithmetic['native_compile'] not in ('0', '1')
                or type(arithmetic['bf16_reduced_precision_reduction']) is not bool):
            raise ValueError('invalid qualification execution arithmetic')
        os.environ['NATLANG_NATIVE_COMPILE'] = arithmetic['native_compile']
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = arithmetic['bf16_reduced_precision_reduction']
    args = SimpleNamespace(**{key: identity[key] for key in (
        'batch_graphs', 'batch_tokens', 'checkpoint_layers', 'chunk_tokens',
        'checkpoint_input_offload')},
        checkpoint_input_offload_directory=identity.get('checkpoint_input_offload_directory'),
        checkpoint_direct_token_limit=identity.get('checkpoint_direct_token_limit', 0),
        checkpoint_layer_group_size=identity.get('checkpoint_layer_group_size', 1),
        ffn_chunk_tokens=identity.get('ffn_chunk_tokens', 0),
        cotangent_accumulator_backend=identity.get('cotangent_accumulator_backend', 'cpu_file'),
        share_causal_prefixes=identity.get('share_causal_prefixes', False),
        device=device, fp8_training=identity['fp8']['kind'],
        fp8_recipe=identity['fp8']['recipe'],
        fp8_amax_history_len=identity['fp8']['amax_history_len'])
    if type(args.checkpoint_layer_group_size) is not int or args.checkpoint_layer_group_size < 1:
        raise ValueError('invalid qualification checkpoint layer grouping')
    model.port.checkpoint_layer_group_size = args.checkpoint_layer_group_size
    model.share_causal_prefixes = args.share_causal_prefixes
    model.checkpoint_direct_token_limit = args.checkpoint_direct_token_limit
    model.native_cotangent_accumulator_backend = args.cotangent_accumulator_backend
    model.port.ffn_chunk_tokens = args.ffn_chunk_tokens
    model.port.checkpoint_layers = args.checkpoint_layers
    return args
