"""Pure preflight policy for budgeted saved-activation offload.

The caller supplies a geometry/calibration estimate for the next update's
incremental CUDA peak, current device free/total bytes, and an estimate of the
eligible saved-activation bytes.  This module intentionally does not inspect
CUDA or run a model forward, so the same policy can be tested on CPU.
"""
from dataclasses import dataclass
import math


def text_warmup_update_geometry_bytes(prefix_tokens: int, target_tokens: int,
                                      sequence_passes: int, batch_size: int,
                                      full_layout: dict, shallow_layout: dict,
                                      *, cutoff: int, vocab_size: int,
                                      readout_chunk_tokens: int = 128):
    """Estimate the peak retained geometry of one text-warmup update.

    Pass zero is a single gold-history forward. Later sequence passes retain a
    shallow producer and a full consumer/replay at the same time, so that
    overlap is added and compared with pass zero. Passes themselves are
    backpropagated and released serially: their estimates are never multiplied
    by ``sequence_passes``. The consumer readout uses checkpointed chunks, so
    at most one configured logit chunk is counted. Flex attention uses its
    fused/tiled path; this estimate intentionally has no quadratic T-by-T
    attention-mask term. The current ``sequence_completions`` path forwards its
    ``group_size`` argument, but ``replay_sequence_inputs`` does not chunk on
    it and calls ``isolated_sequence`` for the full target; this estimate uses
    that actual full-target call shape.

    The caller supplies the actual model layouts. ``shallow_layout`` must use
    the actual number of layers and summed K/V projection widths through the
    cutoff; this avoids estimating a nonexistent full-depth producer.
    """
    from .memory_estimator import geometry_bytes

    counts = (prefix_tokens, target_tokens, sequence_passes, batch_size,
              cutoff, vocab_size, readout_chunk_tokens)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in counts):
        raise TypeError('geometry dimensions must be integers')
    if min(prefix_tokens, target_tokens, sequence_passes, batch_size,
           cutoff, vocab_size, readout_chunk_tokens) < 1:
        raise ValueError('geometry dimensions must be positive')
    if full_layout.get('layers', 0) < cutoff or shallow_layout.get('layers', 0) != cutoff:
        raise ValueError('full and shallow layouts must match the actual cutoff')
    if full_layout.get('width') != shallow_layout.get('width'):
        raise ValueError('full and shallow layouts must have the same width')

    readout_tokens = min(target_tokens, readout_chunk_tokens)
    pass_zero = geometry_bytes(prefix_tokens + target_tokens - 1, 0,
                               target_tokens=readout_tokens, vocab_size=vocab_size,
                               **full_layout)
    peak_per_row = pass_zero
    if sequence_passes > 1:
        producer_prefix = geometry_bytes(prefix_tokens, 0, **full_layout)
        producer_shallow = geometry_bytes(target_tokens - 1, 0, **shallow_layout)
        consumer_prefix = geometry_bytes(prefix_tokens, 0, **full_layout)
        consumer_history = geometry_bytes(target_tokens - 1, 0, **full_layout)
        consumer_branch = geometry_bytes(target_tokens - 1, 0,
                                         target_tokens=readout_tokens,
                                         vocab_size=vocab_size, **full_layout)
        # isolated_sequence keeps both ordinary history and the changed-input
        # branch live while the shallow producer output remains differentiable.
        iterative_overlap = (producer_prefix + producer_shallow + consumer_prefix +
                            consumer_history + consumer_branch)
        peak_per_row = max(pass_zero, iterative_overlap)
    return int(peak_per_row * batch_size)


def effective_cuda_free_bytes(device_free_bytes: int, total_bytes: int,
                              allocator_reserved_bytes: int,
                              allocator_allocated_bytes: int):
    """Count reusable PyTorch cache without mistaking it for live memory.

    ``cuda.mem_get_info`` free bytes exclude PyTorch's cached-but-unused
    reserved blocks. Those blocks are immediately reusable by this process,
    so add ``reserved - allocated`` back. Device-wide free bytes already
    account for live non-PyTorch users. The total-device cap protects against
    stale/inconsistent counters.
    """
    values = (device_free_bytes, total_bytes, allocator_reserved_bytes,
              allocator_allocated_bytes)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise TypeError('CUDA memory byte counts must be integers')
    if min(device_free_bytes, allocator_reserved_bytes, allocator_allocated_bytes) < 0 or total_bytes <= 0:
        raise ValueError('memory byte counts must be nonnegative and total_bytes positive')
    if device_free_bytes > total_bytes or allocator_allocated_bytes > total_bytes:
        raise ValueError('free or allocated bytes cannot exceed total_bytes')
    reclaimable_cache = max(0, allocator_reserved_bytes - allocator_allocated_bytes)
    return min(total_bytes, device_free_bytes + reclaimable_cache)


@dataclass(frozen=True)
class SavedActivationPlan:
    """A targeted offload budget and whether it should fit within headroom."""

    offload_budget_bytes: int
    required_gpu_reduction_bytes: int
    predicted_residual_overage_bytes: int
    usable_free_bytes: int
    predicted_total_incremental_peak_bytes: int

    @property
    def should_offload(self):
        return self.offload_budget_bytes > 0

    @property
    def predicted_fit(self):
        return self.predicted_residual_overage_bytes == 0


def plan_saved_activation_offload(predicted_update_increment_bytes: int,
                                  free_bytes: int,
                                  total_bytes: int,
                                  eligible_saved_activation_bytes: int,
                                  *, headroom_fraction: float = 0.05,
                                  assumed_gpu_bytes_freed_per_cpu_byte: float = 0.5):
    """Choose only the offload needed to fit one predicted update.

    ``predicted_update_increment_bytes`` includes the complete predicted peak
    increment over the measured pre-update baseline, including activation,
    gradient, and any optimizer-step allocation exactly once. It is produced
    from current geometry plus successful-update calibration. ``free_bytes``
    is the live device-wide free-memory measurement at that boundary, so
    unrelated CUDA allocations are accounted for. A reserve is held back from
    total device capacity. The caller should refuse a preflight with
    ``predicted_fit == False`` before running a model forward; this function
    never suggests truncating context or dropping work.

    ``assumed_gpu_bytes_freed_per_cpu_byte`` is a policy assumption in (0, 1], not a
    measurement. It maps bytes saved-tensor hooks place on CPU to the estimated
    CUDA bytes freed for preflight planning. Actual offloaded update peaks are
    censored and must not be used to calibrate the no-offload predictor. Do not
    confuse the 5% device reserve here with ``AdaptiveGraphMemory``'s prediction
    margin; callers must avoid applying both as separate headroom allowances.
    """
    values = (predicted_update_increment_bytes, free_bytes, total_bytes,
              eligible_saved_activation_bytes)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise TypeError('memory byte counts must be integers')
    if predicted_update_increment_bytes < 0 or free_bytes < 0 or total_bytes <= 0 or eligible_saved_activation_bytes < 0:
        raise ValueError('memory byte counts must be nonnegative and total_bytes positive')
    if free_bytes > total_bytes:
        raise ValueError('free_bytes cannot exceed total_bytes')
    if not math.isfinite(headroom_fraction) or not 0 <= headroom_fraction < 1:
        raise ValueError('headroom_fraction must be finite and in [0, 1)')
    if (not math.isfinite(assumed_gpu_bytes_freed_per_cpu_byte) or
            not 0 < assumed_gpu_bytes_freed_per_cpu_byte <= 1):
        raise ValueError('assumed offload savings must be finite and in (0, 1]')

    reserve = math.ceil(total_bytes * headroom_fraction)
    usable_free = max(0, free_bytes - reserve)
    incremental_peak = predicted_update_increment_bytes
    required_reduction = max(0, incremental_peak - usable_free)
    if required_reduction == 0:
        return SavedActivationPlan(0, 0, 0, usable_free, incremental_peak)

    requested_budget = math.ceil(required_reduction / assumed_gpu_bytes_freed_per_cpu_byte)
    budget = min(requested_budget, eligible_saved_activation_bytes)
    expected_reduction = math.floor(budget * assumed_gpu_bytes_freed_per_cpu_byte)
    residual = max(0, required_reduction - expected_reduction)
    return SavedActivationPlan(budget, required_reduction, residual, usable_free, incremental_peak)
