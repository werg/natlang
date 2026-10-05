"""Optional exact CPU storage for large saved attention tensors, without recomputation."""
from contextlib import contextmanager
import weakref

import torch


@contextmanager
def offload_attention_tensors(budget_bytes: int = 0, min_tokens: int = 1024, *, activations=False, persistent_tensors=()):
    stats = {'offloaded_bytes': 0, 'offloaded_tensors': 0,
             'live_offloaded_bytes': 0, 'peak_offloaded_bytes': 0}
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
            previous = copies.get(key)
            cached = previous[1]() if previous and previous[0]() is owner else None
            if cached is not None:
                return tensor.device, cached
            if stats['live_offloaded_bytes'] + size > budget_bytes:
                return None, tensor.detach()
            copy = tensor.detach().to('cpu')
            stats['offloaded_bytes'] += size
            stats['offloaded_tensors'] += 1
            stats['live_offloaded_bytes'] += size
            stats['peak_offloaded_bytes'] = max(stats['peak_offloaded_bytes'], stats['live_offloaded_bytes'])
            def release():
                stats['live_offloaded_bytes'] -= size
                reference = copies.get(key, (None, None))[1]
                if reference is not None and reference() is None:
                    copies.pop(key)
            # Completed staged graphs must release CPU storage and return their
            # budget. The cache itself must not retain those discarded tensors.
            copies[key] = weakref.ref(owner), weakref.ref(copy)
            weakref.finalize(copy, release)
            return tensor.device, copy
        # Saving the original tensor through a Python hook retains its
        # grad_fn, which can create an invisible C++ autograd reference cycle.
        # Preserve storage/version without retaining the graph itself.
        return None, tensor.detach()

    def unpack(saved):
        device, tensor = saved
        return tensor if device is None else tensor.to(device)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield stats
