"""Optional exact CPU storage for large saved attention tensors, without recomputation."""
from contextlib import contextmanager
import weakref

import torch


@contextmanager
def offload_attention_tensors(budget_bytes: int = 0, min_tokens: int = 1024, *, activations=False, persistent_tensors=()):
    stats = {'offloaded_bytes': 0, 'offloaded_tensors': 0}
    if budget_bytes <= 0:
        yield stats
        return

    persistent = {t.untyped_storage().data_ptr() for t in persistent_tensors if t.is_cuda}
    copies = {}

    def pack(tensor):
        # Only attention's long K/V-shaped tensors. Avoid copying frozen weights
        # repeatedly, and leave short queries/outputs and all other tensors alone.
        size = tensor.numel() * tensor.element_size()
        eligible = (tensor.ndim >= 2 and size >= 2**20) if activations else (tensor.ndim == 4 and tensor.shape[-2] >= min_tokens)
        if tensor.is_cuda and eligible and tensor.untyped_storage().data_ptr() not in persistent:
            owner = tensor
            while owner._base is not None:
                owner = owner._base
            key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape),
                   tuple(tensor.stride()), tensor.dtype, tensor._version)
            # Allocator addresses can be recycled during the SAME forward. A
            # live original owner must still match before a CPU copy is reused.
            if key in copies and copies[key][0]() is owner:
                return tensor.device, copies[key][1]
            if stats['offloaded_bytes'] + size > budget_bytes:
                return None, tensor
            stats['offloaded_bytes'] += size
            stats['offloaded_tensors'] += 1
            copies[key] = weakref.ref(owner), tensor.detach().to('cpu')
            return tensor.device, copies[key][1]
        return None, tensor

    def unpack(saved):
        device, tensor = saved
        return tensor if device is None else tensor.to(device)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield stats
