"""Pure preflight policy for budgeted saved-activation offload.

The caller supplies a geometry/calibration estimate for the next update's
incremental CUDA peak, current device free/total bytes, and an estimate of the
eligible saved-activation bytes.  This module intentionally does not inspect
CUDA or run a model forward, so the same policy can be tested on CPU.
"""
from dataclasses import dataclass
import math

from .memory_estimator import CALIBRATED_SAMPLES



def effective_cuda_free_bytes(device_free_bytes: int, total_bytes: int,
                              allocator_reserved_bytes: int,
                              allocator_allocated_bytes: int,
                              inactive_split_bytes: int | None = None):
    """Count reusable PyTorch cache without mistaking it for live memory.

    ``cuda.mem_get_info`` free bytes exclude PyTorch's cached-but-unused
    reserved blocks. Add cached blocks back, except native-allocator inactive
    splits, which PyTorch reports as non-releasable and which cannot satisfy a
    new allocation. ``inactive_split_bytes`` is optional because its stat is
    backend-specific; when unavailable, retain the previous cache estimate.
    Device-wide free bytes already account for live non-PyTorch users. The
    total-device cap protects against stale/inconsistent counters.
    """
    values = (device_free_bytes, total_bytes, allocator_reserved_bytes,
              allocator_allocated_bytes)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise TypeError('CUDA memory byte counts must be integers')
    if min(device_free_bytes, allocator_reserved_bytes, allocator_allocated_bytes) < 0 or total_bytes <= 0:
        raise ValueError('memory byte counts must be nonnegative and total_bytes positive')
    if device_free_bytes > total_bytes or allocator_allocated_bytes > total_bytes:
        raise ValueError('free or allocated bytes cannot exceed total_bytes')
    if (inactive_split_bytes is not None and
            (not isinstance(inactive_split_bytes, int) or isinstance(inactive_split_bytes, bool) or
             inactive_split_bytes < 0)):
        raise ValueError('inactive_split_bytes must be a nonnegative integer when provided')
    reclaimable_cache = max(0, allocator_reserved_bytes - allocator_allocated_bytes)
    if inactive_split_bytes is not None:
        reclaimable_cache -= min(reclaimable_cache, inactive_split_bytes)
    return min(total_bytes, device_free_bytes + reclaimable_cache)


GIB = 2 ** 30


def ledger_claim_margin_bytes(budget_bytes: int) -> int:
    """Slack kept below a ledger claim: CUDA context, allocator rounding and host spikes the plan does not model."""
    return max(4 * GIB, budget_bytes // 20)


def ledger_reusable_bytes(budget_bytes: int, host_bytes: int, allocator_reserved_bytes: int,
                          allocator_allocated_bytes: int, inactive_split_bytes: int | None = None,
                          margin_bytes: int | None = None) -> int:
    """Bytes a new allocation may use inside this process's ledger claim.

    On unified memory (DGX GB10) device-free memory is the whole machine's: sizing an update against it lets a job
    outgrow its claim, and the ledger guard stops it under pressure. The claim minus a margin, minus what the job holds
    now (host memory as the guard counts it, plus PyTorch's reserved pool), plus the reusable part of that pool
    (exactly as in ``effective_cuda_free_bytes``) is what the job may still allocate.
    """
    values = (budget_bytes, host_bytes, allocator_reserved_bytes, allocator_allocated_bytes)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError('ledger and allocator byte counts must be nonnegative integers')
    margin = ledger_claim_margin_bytes(budget_bytes) if margin_bytes is None else margin_bytes
    reclaimable_cache = max(0, allocator_reserved_bytes - allocator_allocated_bytes)
    if inactive_split_bytes is not None:
        reclaimable_cache -= min(reclaimable_cache, inactive_split_bytes)
    return max(0, budget_bytes - margin - host_bytes - allocator_reserved_bytes + reclaimable_cache)


_LEDGER_MODULE = []


def _ledger_module():
    """scripts/memory_ledger.py, loaded once: its own claim and cgroup readers (one implementation per concept)."""
    if not _LEDGER_MODULE:
        import importlib.util
        import os
        from pathlib import Path
        from ..common.paths import root
        script = Path(os.environ.get('NATLANG_REPO') or root('repo')) / 'scripts' / 'memory_ledger.py'
        module = None
        if script.is_file():
            spec = importlib.util.spec_from_file_location('natlang_memory_ledger', script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        _LEDGER_MODULE.append(module)
    return _LEDGER_MODULE[0]


def ledger_claim(unit: str | None = None):
    """``{'unit', 'budget_bytes', 'host_bytes', 'source'}`` for this process's ledger claim, or None without a ledger.

    The unit is ``NATLANG_LEDGER_UNIT`` (set by ``memory_ledger.py run``). The budget is the claim's current budget in
    the ledger state (the guard may raise it); if the state cannot be read, the ``NATLANG_CUDA_MEMORY_GB`` the job was
    launched with. Host usage is the guard's own cgroup reading for this process (0 when unreadable)."""
    import os
    unit = unit or os.environ.get('NATLANG_LEDGER_UNIT')
    if not unit:
        return None
    ledger = _ledger_module()
    budget = ledger.claimed_budget(unit) if ledger is not None else None
    source = 'ledger-state'
    if budget is None and os.environ.get('NATLANG_CUDA_MEMORY_GB'):
        budget, source = int(float(os.environ['NATLANG_CUDA_MEMORY_GB']) * GIB), 'NATLANG_CUDA_MEMORY_GB'
    if budget is None:
        return None
    discrete_cuda = os.environ.get('NATLANG_DISCRETE_CUDA_MEMORY') == '1'
    # Pop's CUDA allocation lives outside host RAM. The unified GB10 claim
    # subtracts host usage from its shared pool; doing so on a discrete card
    # double-charges the same job and can leave no apparent training headroom.
    host = 0 if discrete_cuda else ((ledger.own_host_usage() if ledger is not None else None) or 0)
    return {'unit': unit, 'budget_bytes': int(budget), 'host_bytes': int(host),
            'discrete_cuda': discrete_cuda, 'source': source}


# An uncalibrated no-checkpoint forecast is raw geometry; it is trusted only with this much room to spare.
UNCALIBRATED_NO_CHECKPOINT_FACTOR = 2.0


def choose_layer_checkpointing(estimate, floor_bytes: int, available_bytes: int | None):
    """Least checkpointing that fits ``available_bytes``: ``(checkpointed, raw, predicted, calibration)``.

    ``estimate(checkpointed)`` returns ``(raw, predicted, calibration)`` with ``calibration['sample_count']``. Without
    a memory figure (CPU) layers are checkpointed. Running without checkpoints needs a calibrated forecast that fits;
    an uncalibrated one (its bin never measured, e.g. sized from another bin's ratio or raw geometry) must fit
    ``UNCALIBRATED_NO_CHECKPOINT_FACTOR`` times over, else the update checkpoints until that bin is measured.
    """
    if available_bytes is not None:
        raw, predicted, calibration = estimate(False)
        need = predicted if calibration['sample_count'] >= CALIBRATED_SAMPLES else int(
            predicted * UNCALIBRATED_NO_CHECKPOINT_FACTOR)
        if need + floor_bytes <= available_bytes:
            return False, raw, predicted, calibration
    return (True, *estimate(True))


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


def reclaimable_file_bytes() -> int:
    """Clean file-backed pages the kernel could drop (MemAvailable − MemFree, from /proc/meminfo); 0 when unknown."""
    try:
        fields = {line.split(':')[0]: int(line.split()[1]) * 1024
                  for line in open('/proc/meminfo') if line.startswith(('MemAvailable:', 'MemFree:'))}
        return max(0, fields['MemAvailable'] - fields['MemFree'])
    except (OSError, ValueError, IndexError, KeyError):
        return 0


def release_page_cache(timeout: float = 900.0) -> dict:
    """Turn reclaimable page cache into free memory with the ledger's own mechanism (scripts/memory_ledger.py
    release-cache: posix_fadvise DONTNEED on large files of the data roots, then the balloon). On the GB10 a CUDA
    allocation can use only MemFree, so a preflight counts page cache as usable only after this has freed it."""
    import json
    import subprocess
    import sys

    from ..common.paths import root
    script = root('repo') / 'scripts' / 'memory_ledger.py'
    if not script.exists():
        return {'skipped': 'no ledger script at ' + str(script)}
    try:
        out = subprocess.run([sys.executable, str(script), 'release-cache'], capture_output=True, text=True,
                             timeout=timeout).stdout.strip().splitlines()
        return json.loads(out[-1]) if out else {'skipped': 'no output'}
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        return {'error': f'{type(error).__name__}: {error}'[:300]}

