"""Execution profiles, unreachable-host DGX fallback, and cost telemetry (plans/neuralese/lfm3opti.md,
plans/neuralese/async-train.md).

Three profiles are supported:
- ``dgx``: local DGX GB10 unified-memory machine (128 GiB combined, managed via memory_ledger.py). It runs the held
  series (async-train §1: SGLang, vendor projection, FP8 KV, BF16 weights, 4 samples per task, attested by the held
  manifest's engine.json), so the series never depends on a Colab session or on Pop.
- ``colab`` (old name ``colab_g4`` accepted): a Colab session with a discrete GPU whose hardware varies by session (a
  G4's RTX PRO 6000 Blackwell 96 GB, an A100-SXM4-80GB, ...): learner and collection actors, sized from the session's
  attested hardware (``colab_layout``). It has no held series (async-train §1 supersedes lfm3opti §4's G4 series).
- ``pop``: Pop laptop with discrete RTX 4060 (8 GB VRAM, 15 GiB host RAM): the legacy held series (2 samples).

Colab L4 (~300 GB/s bandwidth, same as GB10) is explicitly excluded per lfm3opti.md §1 as a profile; an L4 session
still serves actors (fills, collection), but round SFT goes there only when ``colab_layout(...)['sft_beats_dgx']``.

Cost: Colab bills a session from its start to its stop, busy or idle, so ``CostTelemetry`` charges sessions
(``open_session``/``close_session``, an open one up to now); stage entries attribute wall time to stages, and a stage
entry covered by a session adds no compute units of its own.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .jsonio import write_json_durable

EXCLUDED_PROFILES = frozenset({'l4', 'colab_l4', 'colab-l4'})
DEFAULT_FALLBACK_WAIT_MINUTES = 30.0
DEFAULT_COLAB_STATE_SYNC_MINUTES = 15.0
DEFAULT_G4_CU_PER_HOUR = 13.0
PROFILE_ALIASES = {'colab_g4': 'colab'}     # the profile's old, hardware-specific name

# The LoRA learner's measured memory model (DECISIONS 2026-11-22 "LoRA core fixes", GB10 and compiled LoRA r=128):
# CUDA peak ~= LEARNER_FIXED_GIB (BF16 base, FP32 factors and AdamW, workspace) + LEARNER_ACTIVATION_FACTOR x
# --activation-gb; saved activations cost ACTIVATION_BYTES_PER_TOKEN_LAYER per token and layer (24 layers).
LEARNER_FIXED_GIB = 5.2
LEARNER_ACTIVATION_FACTOR = 1.2
ACTIVATION_BYTES_PER_TOKEN_LAYER = 127 * 1024
LFM3_LAYERS = 24
ALLOCATOR_SLACK_GIB = 2.0       # fragmentation and the CUDA context beside the modelled peak
ADMISSION_RESERVE_GIB = 1.0     # what scripts/colab/admission.py keeps free
VRAM_UNREPORTED_GIB = 2.0       # nvidia-smi total less what a process can allocate (96 -> 94 on the G4)
# An SGLang actor: BF16 weights 3.4 GB + CUDA graphs and allocator slack ~6.6 GB besides its KV pool. Beside the
# learner (MPS) its KV takes a quarter of the GPU and its running cap scales with the KV (64 at the G4's 24 GB).
ACTOR_FIXED_GIB = 10.0
ACTOR_BESIDE_KV_SHARE = 0.25
ACTOR_MEM_FRACTION = 0.88       # alone: SGLang's static memory fraction
# Alone on a large GPU the pool is capped: 128 running used 20% of the L4's 11.2 GB FP8 pool (1.8M LFM3 tokens, mostly
# conv state; DECISIONS 2026-11-23 13:50), so 24 GB holds ~10x what the actor's sequences use, and the rest of an A100
# or G4 (~44 / 60 GiB) is free for tenants between rounds (program tuning, PROGRAM_TUNING_A100_20261124.md). An L4's
# fraction-sized pool is below the cap and unchanged. NATLANG_COLAB_ACTOR_FULL=1 opts out (the fraction alone).
ACTOR_ALONE_KV_CAP_GB = 24
# Tenants of an actor-alone session (scripts/colab/admission.py --preemptible: a round's learner or an actor served
# again evicts them): a program fit (tune.build, BF16 base + FP32 rank-16 LoRA; 6.3 GB CUDA measured on the GB10) and a
# measurement actor (the base, optionally with an adapter release on its SGLang slots, for held_eval arms).
TENANT_FIT = {'vram_gib': 12.0, 'host_ram_gib': 16.0}
TENANT_MEASURE_KV_GB = 8
# Round SFT goes to a Colab session only when it trains faster than the DGX. The learner is compute bound (A100 measured
# 9 s/step against the GB10's 36.8 s on the same mix), so the session's GPU needs at least SFT_MIN_SPEEDUP x the
# GB10's dense BF16 throughput (~125 TFLOPS) and room for SFT_MIN_ACTIVATION_GIB of saved activations (below that most
# layers recompute and long chains split). An L4 (121 TFLOPS, 300 GB/s, ~10 GiB of activations) is the GB10's class
# with an eighth of its memory: slower, so its rounds stay on the DGX. An unknown GPU stays on the DGX.
GB10_BF16_TFLOPS = 125.0
BF16_DENSE_TFLOPS_BY_NAME = (('A100', 312.0), ('H100', 756.0), ('H200', 989.0), ('RTX PRO 6000', 500.0),
                             ('L40', 362.0), ('A10', 125.0), ('L4', 121.0), ('T4', 65.0))
SFT_MIN_SPEEDUP = 2.0
SFT_MIN_ACTIVATION_GIB = 16
FP8_MIN_COMPUTE_CAPABILITY = (8, 9)    # SGLang FP8 weight GEMMs: Ada / Hopper / Blackwell, not Ampere (sm_80)
# compute capability by GPU name when a session's ready file predates its probe (bootstrap probes compute_cap now)
COMPUTE_CAPABILITY_BY_NAME = (('A100', '8.0'), ('A10', '8.6'), ('L4', '8.9'), ('L40', '8.9'), ('H100', '9.0'),
                              ('H200', '9.0'), ('B200', '10.0'), ('RTX PRO 6000', '12.0'))


def compute_capability(hardware: dict) -> tuple[int, int] | None:
    """(major, minor) from a hardware record's ``compute_cap`` (nvidia-smi), else from its GPU name."""
    raw = hardware.get('compute_cap')
    if not raw:
        name = str(hardware.get('gpu_name') or '')
        raw = next((cap for key, cap in COMPUTE_CAPABILITY_BY_NAME if key in name), None)
    if not raw:
        return None
    major, _, minor = str(raw).partition('.')
    return int(major), int(minor or 0)


# measured round-SFT speed against the GB10 on one mix (the A100's 9 s/step against 36.8 s above); a GPU without a
# measurement plans with its BF16 TFLOPS ratio, which undershoots the A100's (2.5x against 4.1x measured)
MEASURED_SFT_SPEEDUP_BY_NAME = (('A100', round(36.8 / 9.0, 2)),)


def sft_speedup(hardware: dict, measured: dict | None = None) -> tuple[float | None, str]:
    """(round SFT speed of the session's GPU against the DGX's, its source): ``measured`` ({gpu name: speedup}, what
    this loop measured on a round that moved), else the built-in measurement, else the BF16 TFLOPS ratio (None when
    the GPU is unknown)."""
    name = str(hardware.get('gpu_name') or '')
    if measured and measured.get(name):
        return float(measured[name]), 'measured by this loop'
    known = next((x for key, x in MEASURED_SFT_SPEEDUP_BY_NAME if key in name), None)
    if known:
        return known, 'measured (execution_profile)'
    tflops = bf16_dense_tflops(hardware)
    return (round(tflops / GB10_BF16_TFLOPS, 2), 'BF16 TFLOPS ratio') if tflops else (None, 'unknown GPU')


def bf16_dense_tflops(hardware: dict) -> float | None:
    """Peak dense BF16 TFLOPS of a session's GPU by name (None when unknown)."""
    name = str(hardware.get('gpu_name') or '')
    return next((tflops for key, tflops in BF16_DENSE_TFLOPS_BY_NAME if key in name), None)


def colab_layout(hardware: dict) -> dict:
    """Admission claims and engine sizes of a Colab session from its attested hardware (``colab-ready.json``
    ``hardware``: ``vram_total_gib``, ``host_ram_total_gib``, ``gpu_name``, ``compute_cap``), by the measured memory
    model above, for the four layouts: the learner alone, the learner beside a capped actor (MPS), that actor, and an
    actor alone. Every claim fits the GPU: learner + actor + reserve <= usable VRAM. ``fp8``: whether FP8 weights
    serve here (an sm_80 A100 has no FP8 GEMMs: an FP8 request picks BF16)."""
    total = float(hardware['vram_total_gib'])
    host = float(hardware.get('host_ram_total_gib') or 0.) or 64.0
    usable = total - VRAM_UNREPORTED_GIB
    cap = compute_capability(hardware)

    def learner(claim):
        activation = int((claim - LEARNER_FIXED_GIB - ALLOCATOR_SLACK_GIB) / LEARNER_ACTIVATION_FACTOR)
        return {'vram_gib': round(claim, 1), 'host_ram_gib': round(min(40.0, 0.4 * host), 1),
                'activation_gib': activation,
                'peak_gib': round(LEARNER_FIXED_GIB + LEARNER_ACTIVATION_FACTOR * activation, 1),
                'resident_tokens': int(activation * 2**30 // (ACTIVATION_BYTES_PER_TOKEN_LAYER * LFM3_LAYERS))}
    kv = round(ACTOR_BESIDE_KV_SHARE * usable)
    beside = {'kv_gb': kv, 'max_running': max(8, min(64, int(64 * kv / 24))), 'vram_gib': float(kv + ACTOR_FIXED_GIB),
              'host_ram_gib': round(min(24.0, 0.2 * host), 1)}
    # alone, 128 running on every session: LFM3's KV is small (mostly conv layers), so even an L4's pool (11.2 GB FP8
    # after mem fraction 0.88) holds 1.8M tokens, 14k per running sequence; decode is bandwidth bound and gains from them
    alone = {'kv_gb': None, 'max_running': 128, 'vram_gib': float(round(ACTOR_MEM_FRACTION * usable + 1.0)),
             'host_ram_gib': round(min(24.0, 0.2 * host), 1), 'mem_fraction': ACTOR_MEM_FRACTION}
    if (ACTOR_MEM_FRACTION * usable - ACTOR_FIXED_GIB > ACTOR_ALONE_KV_CAP_GB
            and os.environ.get('NATLANG_COLAB_ACTOR_FULL') != '1'):
        alone = {'kv_gb': ACTOR_ALONE_KV_CAP_GB, 'max_running': 128,
                 'vram_gib': float(ACTOR_ALONE_KV_CAP_GB + ACTOR_FIXED_GIB), 'host_ram_gib': alone['host_ram_gib']}
    measure = {'kv_gb': TENANT_MEASURE_KV_GB, 'max_running': 64, 'vram_gib': float(TENANT_MEASURE_KV_GB + ACTOR_FIXED_GIB),
               'host_ram_gib': 16.0}
    tenants = {'vram_gib': round(usable - alone['vram_gib'] - ADMISSION_RESERVE_GIB, 1), 'fit': dict(TENANT_FIT),
               'measure_actor': measure}
    learner_alone = learner(usable - ADMISSION_RESERVE_GIB)
    tflops = bf16_dense_tflops(hardware)
    speedup = round(tflops / GB10_BF16_TFLOPS, 2) if tflops else None
    sft = bool(speedup and speedup >= SFT_MIN_SPEEDUP and learner_alone['activation_gib'] >= SFT_MIN_ACTIVATION_GIB)
    return {'gpu_name': hardware.get('gpu_name'), 'vram_total_gib': total, 'usable_gib': usable,
            'compute_capability': '.'.join(map(str, cap)) if cap else None,
            'fp8': bool(cap and cap >= FP8_MIN_COMPUTE_CAPABILITY),
            'sft_beats_dgx': sft,
            'sft_reason': (f'{hardware.get("gpu_name")}: {speedup}x the GB10\'s BF16 throughput (need '
                           f'{SFT_MIN_SPEEDUP}), learner alone {learner_alone["activation_gib"]} GiB of activations '
                           f'(need {SFT_MIN_ACTIVATION_GIB})'),
            'learner_alone': learner_alone,
            'learner_beside_actor': learner(usable - beside['vram_gib'] - ADMISSION_RESERVE_GIB),
            'actor_beside_learner': beside, 'actor_alone': alone, 'tenants': tenants}


G4_HARDWARE = {'gpu_name': 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'vram_total_gib': 96.0,
               'host_ram_total_gib': 176.0, 'compute_cap': '12.0'}


@dataclass(frozen=True)
class ExecutionProfile:
    """Hardware-specific knobs for a training, collection, or evaluation stage."""

    name: str
    is_discrete_gpu: bool
    vram_budget_gib: float | None
    host_ram_budget_gib: float | None
    combined_budget_gib: float | None
    activation_gib: float
    sglang_kv_cache_dtype: str
    sglang_max_running: int
    sglang_mem_fraction: float | None
    sglang_export_format: str
    collect_slots: int
    held_samples: int
    cu_per_hour: float = 0.0
    fallback_profile: str | None = None
    fallback_wait_minutes: float = DEFAULT_FALLBACK_WAIT_MINUTES
    state_sync_minutes: float = 0.0
    hosts: tuple[str, ...] = ()
    scratch_dir: str | None = None
    held_series_host: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data['hosts'] = list(self.hosts)
        return data


BUILTIN_PROFILES: dict[str, ExecutionProfile] = {
    'dgx': ExecutionProfile(
        name='dgx',
        is_discrete_gpu=False,
        vram_budget_gib=None,
        host_ram_budget_gib=None,
        combined_budget_gib=100.0,
        activation_gib=56.0,
        sglang_kv_cache_dtype='fp8_e4m3',
        sglang_max_running=128,
        sglang_mem_fraction=None,
        sglang_export_format='sglang-dgx-gb10-bf16-vendor-projection-kv-fp8_e4m3',
        collect_slots=128,
        held_samples=4,
        cu_per_hour=0.0,
        fallback_profile=None,
        fallback_wait_minutes=0.0,
        state_sync_minutes=0.0,
        held_series_host='dgx',
    ),
    'colab': ExecutionProfile(
        name='colab',
        is_discrete_gpu=True,
        vram_budget_gib=92.0,
        host_ram_budget_gib=150.0,
        combined_budget_gib=None,
        # a session's own sizes come from its hardware (colab_layout); this default is a G4's learner beside an actor
        activation_gib=float(colab_layout(G4_HARDWARE)['learner_beside_actor']['activation_gib']),
        sglang_kv_cache_dtype='fp8_e4m3',
        sglang_max_running=128,
        sglang_mem_fraction=0.88,
        sglang_export_format='sglang-colab-bf16-vendor-projection-kv-fp8_e4m3',
        collect_slots=128,
        held_samples=0,               # no G4 held series (async-train §1): held runs on the DGX
        cu_per_hour=DEFAULT_G4_CU_PER_HOUR,
        fallback_profile='dgx',
        fallback_wait_minutes=DEFAULT_FALLBACK_WAIT_MINUTES,
        state_sync_minutes=DEFAULT_COLAB_STATE_SYNC_MINUTES,
        scratch_dir='/content/natlang-work',
        held_series_host=None,
    ),
    'pop': ExecutionProfile(
        name='pop',
        is_discrete_gpu=True,
        vram_budget_gib=7.5,
        host_ram_budget_gib=10.5,
        combined_budget_gib=None,
        activation_gib=6.0,
        sglang_kv_cache_dtype='fp8_e4m3',
        sglang_max_running=20,
        sglang_mem_fraction=0.77,
        sglang_export_format='sglang-pop-rtx4060-bf16-vendor-projection-kv-fp8_e4m3',
        collect_slots=20,
        held_samples=2,
        cu_per_hour=0.0,
        fallback_profile='dgx',
        fallback_wait_minutes=DEFAULT_FALLBACK_WAIT_MINUTES,
        state_sync_minutes=0.0,
        held_series_host=None,
    ),
}


def parse_colab_hosts(raw: str | Sequence[str] | None = None, config_path: str | Path | None = None) -> tuple[str, ...]:
    """Resolve ordered Colab G4 SSH host aliases from explicit argument, NATLANG_COLAB_HOSTS, or config JSON."""
    items: list[str] = []
    if raw:
        if isinstance(raw, str):
            items.extend(part.strip() for part in raw.split(',') if part.strip())
        else:
            for entry in raw:
                items.extend(part.strip() for part in str(entry).split(',') if part.strip())
    elif os.environ.get('NATLANG_COLAB_HOSTS'):
        items.extend(part.strip() for part in os.environ['NATLANG_COLAB_HOSTS'].split(',') if part.strip())
    elif config_path and Path(config_path).is_file():
        cfg = json.loads(Path(config_path).read_text())
        raw_cfg = cfg.get('colab_hosts') or cfg.get('colab_g4_hosts') or []
        if isinstance(raw_cfg, str):
            items.extend(part.strip() for part in raw_cfg.split(',') if part.strip())
        else:
            items.extend(str(h).strip() for h in raw_cfg if str(h).strip())
    # Deduplicate preserving order
    return tuple(dict.fromkeys(items))


def resolve_profile(
    name: str,
    *,
    hosts: str | Sequence[str] | None = None,
    config_path: str | Path | None = None,
    cu_per_hour: float | None = None,
    fallback_wait_minutes: float | None = None,
) -> ExecutionProfile:
    """Return an ExecutionProfile by name; explicitly rejects excluded L4 profiles."""
    key = str(name).strip().lower().replace('-', '_')
    if key in EXCLUDED_PROFILES:
        raise ValueError(
            f'ExecutionProfile {name!r} is excluded (lfm3opti.md §1: L4 bandwidth ~300 GB/s matches GB10 '
            'and buys no speed). Supported profiles: dgx, colab, pop.'
        )
    key = PROFILE_ALIASES.get(key, key)
    if key not in BUILTIN_PROFILES:
        raise ValueError(f'Unknown ExecutionProfile {name!r}; expected one of {sorted(BUILTIN_PROFILES)}')
    base = BUILTIN_PROFILES[key]
    resolved_hosts = parse_colab_hosts(hosts, config_path) if key == 'colab' else (
        tuple(hosts) if isinstance(hosts, (list, tuple)) else ((hosts,) if isinstance(hosts, str) and hosts else base.hosts)
    )
    return ExecutionProfile(
        **{
            **asdict(base),
            'hosts': resolved_hosts,
            'cu_per_hour': float(cu_per_hour) if cu_per_hour is not None else base.cu_per_hour,
            'fallback_wait_minutes': (
                float(fallback_wait_minutes) if fallback_wait_minutes is not None else base.fallback_wait_minutes
            ),
        }
    )


def select_stage_profile(
    preferred: str | ExecutionProfile,
    *,
    reachable_fn: Callable[[str], bool],
    hosts: Sequence[str] | None = None,
    wait_minutes: float | None = None,
    poll_seconds: float = 30.0,
    fallback: str = 'dgx',
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
) -> tuple[ExecutionProfile, list[str], str | None]:
    """Select the active profile for a stage, falling back to DGX if remote host(s) are unreachable past the wait.

    Returns ``(active_profile, reachable_hosts, fallback_reason)``.
    """
    profile = resolve_profile(preferred, hosts=hosts) if isinstance(preferred, str) else preferred
    if profile.name == 'dgx':
        return profile, ['local'], None
    candidate_hosts = list(hosts) if hosts else list(profile.hosts)
    if not candidate_hosts:
        fb = resolve_profile(fallback or profile.fallback_profile or 'dgx')
        return fb, ['local'], f'{profile.name}: no hosts configured; falling back to {fb.name}'
    wait_s = max(0.0, (profile.fallback_wait_minutes if wait_minutes is None else float(wait_minutes)) * 60.0)
    deadline = clock_fn() + wait_s
    while True:
        up = [h for h in candidate_hosts if reachable_fn(h)]
        if up:
            return profile, up, None
        remaining = deadline - clock_fn()
        if remaining <= 0:
            fb = resolve_profile(fallback or profile.fallback_profile or 'dgx')
            return fb, ['local'], (
                f'{profile.name} hosts {candidate_hosts} unreachable after {wait_s / 60.0:.1f} min; '
                f'falling back to {fb.name}'
            )
        sleep_fn(min(poll_seconds, remaining))


@dataclass
class StageCostEntry:
    stage: str
    profile: str
    wall_seconds: float
    sessions: int
    cu_per_hour: float
    cu_spent: float
    canonical_chain_tokens: int = 0
    note: str = ''


@dataclass
class SessionCostEntry:
    """One remote session (a Colab G4 actor or learner) charged from its start to its stop, idle time included."""
    host: str
    profile: str
    started: float
    stopped: float | None = None
    cu_per_hour: float = 0.0
    note: str = ''

    def seconds(self, now: float | None = None) -> float:
        end = self.stopped if self.stopped is not None else (time.time() if now is None else now)
        return max(0.0, end - self.started)

    def cu(self, now: float | None = None) -> float:
        return self.cu_per_hour * self.seconds(now) / 3600.0


@dataclass
class CostTelemetry:
    """Per-cycle cost & efficiency telemetry (lfm3opti.md §6 & v2 §6.4): remote sessions are charged from start to
    stop (``open_session``/``close_session``); a stage entry with ``session_host`` is attribution only."""

    cycle_name: str
    entries: list[StageCostEntry] = field(default_factory=list)
    sessions: list[SessionCostEntry] = field(default_factory=list)

    def open_session(self, host: str, profile: str | ExecutionProfile, *, at: float | None = None,
                     cu_per_hour: float | None = None, note: str = '') -> SessionCostEntry:
        """Start charging ``host`` (idempotent: an open session of the host is kept)."""
        for entry in self.sessions:
            if entry.host == host and entry.stopped is None:
                return entry
        name = profile.name if isinstance(profile, ExecutionProfile) else PROFILE_ALIASES.get(str(profile), str(profile))
        rate = float(cu_per_hour) if cu_per_hour is not None else (
            profile.cu_per_hour if isinstance(profile, ExecutionProfile)
            else BUILTIN_PROFILES.get(name, BUILTIN_PROFILES['dgx']).cu_per_hour)
        entry = SessionCostEntry(host=host, profile=name, started=time.time() if at is None else float(at),
                                 cu_per_hour=round(rate, 4), note=note)
        self.sessions.append(entry)
        return entry

    def close_session(self, host: str, *, at: float | None = None) -> SessionCostEntry | None:
        for entry in self.sessions:
            if entry.host == host and entry.stopped is None:
                entry.stopped = time.time() if at is None else float(at)
                return entry
        return None

    def record_stage(
        self,
        stage: str,
        profile: str | ExecutionProfile,
        wall_seconds: float,
        *,
        sessions: int = 1,
        canonical_chain_tokens: int = 0,
        cu_per_hour: float | None = None,
        note: str = '',
        session_host: str | None = None,
    ) -> StageCostEntry:
        prof_name = profile.name if isinstance(profile, ExecutionProfile) else PROFILE_ALIASES.get(str(profile),
                                                                                                   str(profile))
        rate = (
            float(cu_per_hour)
            if cu_per_hour is not None
            else (profile.cu_per_hour if isinstance(profile, ExecutionProfile)
                  else BUILTIN_PROFILES.get(prof_name, BUILTIN_PROFILES['dgx']).cu_per_hour)
        )
        cu = rate * (max(0.0, float(wall_seconds)) / 3600.0) * max(1, int(sessions))
        if session_host:     # the session charges this time (start to stop): the stage is attribution only
            cu = 0.0
            note = (note + ' ' if note else '') + f'charged by session {session_host}'
        entry = StageCostEntry(
            stage=stage,
            profile=prof_name,
            wall_seconds=round(float(wall_seconds), 2),
            sessions=max(1, int(sessions)),
            cu_per_hour=round(rate, 4),
            cu_spent=round(cu, 4),
            canonical_chain_tokens=int(canonical_chain_tokens),
            note=note,
        )
        self.entries.append(entry)
        return entry

    def summary(self, now: float | None = None) -> dict:
        session_cu = sum(e.cu(now) for e in self.sessions)
        total_cu = sum(e.cu_spent for e in self.entries) + session_cu
        total_wall = sum(e.wall_seconds for e in self.entries)
        total_tokens = sum(e.canonical_chain_tokens for e in self.entries)
        by_stage: dict[str, dict] = {}
        for e in self.entries:
            slot = by_stage.setdefault(e.stage, {'wall_seconds': 0.0, 'cu_spent': 0.0, 'profiles': [], 'tokens': 0})
            slot['wall_seconds'] = round(slot['wall_seconds'] + e.wall_seconds, 2)
            slot['cu_spent'] = round(slot['cu_spent'] + e.cu_spent, 4)
            slot['tokens'] += e.canonical_chain_tokens
            if e.profile not in slot['profiles']:
                slot['profiles'].append(e.profile)
        tokens_per_cu = (total_tokens / total_cu) if (total_cu > 0 and total_tokens > 0) else None
        cu_per_m_units = (total_cu * 1e6 / total_tokens) if (total_tokens > 0 and total_cu > 0) else None
        return {
            'schema': 'natlang.cost-telemetry/1',
            'cycle': self.cycle_name,
            'total_wall_seconds': round(total_wall, 2),
            'total_cu_spent': round(total_cu, 4),
            'canonical_chain_tokens': int(total_tokens),
            'canonical_chain_tokens_per_cu': None if tokens_per_cu is None else round(tokens_per_cu, 1),
            'cu_per_million_units': None if cu_per_m_units is None else round(cu_per_m_units, 4),
            'by_stage': by_stage,
            'session_cu_spent': round(session_cu, 4),
            'open_sessions': [e.host for e in self.sessions if e.stopped is None],
            'entries': [asdict(e) for e in self.entries],
            'sessions': [{**asdict(e), 'seconds': round(e.seconds(now), 1), 'cu_spent': round(e.cu(now), 4)}
                         for e in self.sessions],
        }

    def save(self, path: str | Path) -> dict:
        data = self.summary()
        write_json_durable(Path(path), data)
        return data

    @classmethod
    def load(cls, path: str | Path, cycle_name: str) -> 'CostTelemetry':
        p = Path(path)
        if not p.is_file():
            return cls(cycle_name=cycle_name)
        try:
            raw = json.loads(p.read_text())
            entries = [StageCostEntry(**item) for item in raw.get('entries') or ()]
            fields = set(SessionCostEntry.__dataclass_fields__)
            sessions = [SessionCostEntry(**{k: v for k, v in item.items() if k in fields})
                        for item in raw.get('sessions') or ()]
            return cls(cycle_name=raw.get('cycle') or cycle_name, entries=entries, sessions=sessions)
        except Exception:
            return cls(cycle_name=cycle_name)


__all__ = [
    'BUILTIN_PROFILES',
    'CostTelemetry',
    'DEFAULT_COLAB_STATE_SYNC_MINUTES',
    'DEFAULT_FALLBACK_WAIT_MINUTES',
    'DEFAULT_G4_CU_PER_HOUR',
    'EXCLUDED_PROFILES',
    'bf16_dense_tflops',
    'ExecutionProfile',
    'G4_HARDWARE',
    'PROFILE_ALIASES',
    'colab_layout',
    'compute_capability',
    'SessionCostEntry',
    'StageCostEntry',
    'parse_colab_hosts',
    'resolve_profile',
    'select_stage_profile',
]
