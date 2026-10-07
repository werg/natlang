"""The device a model runs on when the caller does not name one."""
from __future__ import annotations


def auto_device(min_free_gb: float = 4.0) -> str:
    """``cuda`` when a CUDA device is visible and has ``min_free_gb`` free, else ``cpu``.

    On unified memory (the DGX) CUDA's free memory is the system's, so there is room on the GPU whenever there is room
    for the CPU; on a discrete GPU that training is using, the model falls back to the CPU. ``CUDA_VISIBLE_DEVICES=``
    hides the GPU and so selects the CPU.
    """
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    free, _ = torch.cuda.mem_get_info()
    return "cuda" if free >= min_free_gb * 2**30 else "cpu"


def cap_cuda_memory(device: str, memory_gb: float | None) -> None:
    """Hold this process's CUDA allocations to ``memory_gb`` (the ledger's ``NATLANG_CUDA_MEMORY_GB``) on a CUDA device."""
    import torch

    if memory_gb and str(device).startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, memory_gb * 2**30 / total))
