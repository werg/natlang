"""Execution profiles, unreachable-host DGX fallback, and cost telemetry (plans/neuralese/lfm3opti.md).

Three profiles are supported:
- ``dgx``: local DGX GB10 unified-memory machine (128 GiB combined, managed via memory_ledger.py).
- ``colab_g4``: Colab G4 VM with discrete RTX PRO 6000 Blackwell (96 GB VRAM, ~1.8 TB/s, ~13 CU/h).
- ``pop``: Pop laptop with discrete RTX 4060 (8 GB VRAM, 15 GiB host RAM).

Colab L4 (~300 GB/s bandwidth, same as GB10) is explicitly excluded per lfm3opti.md §1.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .jsonio import write_json_durable

EXCLUDED_PROFILES = frozenset({'l4', 'colab_l4', 'colab-l4'})
DEFAULT_FALLBACK_WAIT_MINUTES = 30.0
DEFAULT_COLAB_STATE_SYNC_MINUTES = 15.0
DEFAULT_G4_CU_PER_HOUR = 13.0


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
        held_samples=2,
        cu_per_hour=0.0,
        fallback_profile=None,
        fallback_wait_minutes=0.0,
        state_sync_minutes=0.0,
        held_series_host=None,
    ),
    'colab_g4': ExecutionProfile(
        name='colab_g4',
        is_discrete_gpu=True,
        vram_budget_gib=92.0,
        host_ram_budget_gib=150.0,
        combined_budget_gib=None,
        activation_gib=80.0,
        sglang_kv_cache_dtype='fp8_e4m3',
        sglang_max_running=128,
        sglang_mem_fraction=0.88,
        sglang_export_format='sglang-colab-g4-bf16-vendor-projection-kv-fp8_e4m3',
        collect_slots=128,
        held_samples=4,
        cu_per_hour=DEFAULT_G4_CU_PER_HOUR,
        fallback_profile='dgx',
        fallback_wait_minutes=DEFAULT_FALLBACK_WAIT_MINUTES,
        state_sync_minutes=DEFAULT_COLAB_STATE_SYNC_MINUTES,
        scratch_dir='/content/natlang-work',
        held_series_host='colab_g4',
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
            'and buys no speed). Supported profiles: dgx, colab_g4, pop.'
        )
    if key not in BUILTIN_PROFILES:
        raise ValueError(f'Unknown ExecutionProfile {name!r}; expected one of {sorted(BUILTIN_PROFILES)}')
    base = BUILTIN_PROFILES[key]
    resolved_hosts = parse_colab_hosts(hosts, config_path) if key == 'colab_g4' else (
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
class CostTelemetry:
    """Per-cycle cost & efficiency telemetry (lfm3opti.md §6 & v2 §6.4)."""

    cycle_name: str
    entries: list[StageCostEntry] = field(default_factory=list)

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
    ) -> StageCostEntry:
        prof_name = profile.name if isinstance(profile, ExecutionProfile) else str(profile)
        rate = (
            float(cu_per_hour)
            if cu_per_hour is not None
            else (profile.cu_per_hour if isinstance(profile, ExecutionProfile)
                  else BUILTIN_PROFILES.get(prof_name, BUILTIN_PROFILES['dgx']).cu_per_hour)
        )
        cu = rate * (max(0.0, float(wall_seconds)) / 3600.0) * max(1, int(sessions))
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

    def summary(self) -> dict:
        total_cu = sum(e.cu_spent for e in self.entries)
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
            'entries': [asdict(e) for e in self.entries],
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
            return cls(cycle_name=raw.get('cycle') or cycle_name, entries=entries)
        except Exception:
            return cls(cycle_name=cycle_name)


__all__ = [
    'BUILTIN_PROFILES',
    'CostTelemetry',
    'DEFAULT_COLAB_STATE_SYNC_MINUTES',
    'DEFAULT_FALLBACK_WAIT_MINUTES',
    'DEFAULT_G4_CU_PER_HOUR',
    'EXCLUDED_PROFILES',
    'ExecutionProfile',
    'StageCostEntry',
    'parse_colab_hosts',
    'resolve_profile',
    'select_stage_profile',
]
