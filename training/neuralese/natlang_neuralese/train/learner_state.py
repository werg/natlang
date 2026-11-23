"""Versioned resumable learner state, atomic staging, immutable merged BF16 export, and provenance manifests
(`lfm3opti.md` §3, §5, §6, v2 §7.1 & §8).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..common.code_provenance import code_provenance
from ..common.hashing import sha256_file_hex
from ..common.jsonio import write_json_durable
from .lfm3_learner import (
    AdapterSpec,
    master_rows,
    merge_policy_state,
    LayerAdapterBundle,
    Lfm3Learner,
    LinearFactorLeaf,
    TrainableRows,
    capture_rng_state,
    restore_rng_state,
)

STATE_SCHEMA = 'natlang.lfm3-learner-state/1'
FACTOR_EXPORT_SCHEMA = 'natlang.lfm3-adapter-factors/1'


def hardware_profile(device=None) -> dict:
    """Capture GPU SKU, usable GiB, host RAM GiB, driver/CUDA/PyTorch versions, and TF32 flags."""
    profile: dict = {
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'allow_tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
        'allow_tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
        'float32_matmul_precision': (torch.get_float32_matmul_precision()
                                     if hasattr(torch, 'get_float32_matmul_precision') else None),
    }
    try:
        with open('/proc/meminfo') as handle:
            mem = {}
            for line in handle:
                key, _, rest = line.partition(':')
                if key in ('MemTotal', 'MemAvailable'):
                    mem[key] = round(int(rest.split()[0]) / 2**20, 2)
        profile['host_ram_total_gib'] = mem.get('MemTotal')
        profile['host_ram_available_gib'] = mem.get('MemAvailable')
    except OSError:
        pass
    if torch.cuda.is_available():
        dev = torch.device(device) if device is not None else torch.device('cuda', torch.cuda.current_device())
        if dev.type == 'cuda':
            props = torch.cuda.get_device_properties(dev)
            free_b, total_b = torch.cuda.mem_get_info(dev)
            profile.update({
                'gpu_sku': props.name,
                'sm_capability': f'{props.major}.{props.minor}',
                'sm_count': int(props.multi_processor_count),
                'vram_total_gib': round(total_b / 2**30, 2),
                'vram_usable_gib': round(free_b / 2**30, 2),
            })
            try:
                drv = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                    text=True, timeout=10, stderr=subprocess.DEVNULL,
                ).strip().splitlines()[0].strip()
                profile['driver_version'] = drv
            except Exception:
                profile['driver_version'] = None
    return profile


def _serialize_bundles(bundles: tuple[LayerAdapterBundle, ...] | None, rows: TrainableRows | None,
                       target_map: dict[str, tuple[int, str, str]]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    if bundles:
        for param_name in sorted(target_map):
            layer_idx, _kind, short_name = target_map[param_name]
            leaf: LinearFactorLeaf | None = getattr(bundles[layer_idx], short_name)
            if leaf is not None:
                tensors[f'lora_A.{param_name}'] = leaf.a.detach().to(device='cpu', dtype=torch.float32).contiguous()
                tensors[f'lora_B.{param_name}'] = leaf.b.detach().to(device='cpu', dtype=torch.float32).contiguous()
    if rows is not None:
        tensors['trainable_rows.rows'] = rows.rows.detach().to(device='cpu', dtype=torch.float32).contiguous()
        tensors['trainable_rows.token_ids'] = rows.token_ids.detach().to(device='cpu', dtype=torch.int64).contiguous()
    return tensors


def save_learner_state(state_dir, learner: Lfm3Learner, optimizer: torch.optim.Optimizer | None = None, *,
                       learner_step: int, source_checkpoint_id: str, source_step: int = 0,
                       base_weights_sha256: str = '', objective_version: str = 'sft_v1',
                       optimizer_config: dict | None = None, ambient_state: dict | None = None,
                       data_manifest_sha256: str | None = None, batch_cursor: dict | None = None,
                       rl_rollout_ledger: dict | None = None,
                       trained_rounds: list[str] | None = None,
                       ref_bundles: tuple[LayerAdapterBundle, ...] | None = None,
                       ref_rows: TrainableRows | None = None,
                       keep_previous: bool = True, extra_meta: dict | None = None,
                       trainer_state: dict | None = None) -> Path:
    """Atomically save resumable LoRA+rows learner state into ``state_dir`` (staging dir -> atomic manifest commit).

    ``trainer_state``: the trainer's own resume state (any ``torch.save``-able dict: e.g. exp0_sft's OOM plan, replanned
    schedule and kept-best leaves), returned by ``load_learner_state`` as ``trainer_state``.
    ``trained_rounds``: the round ids whose export committed (async-train §2/§3 dedup ledger: a resumed learner never
    retrains a round). The episodes a round trained are its ``round.json`` ``episode_ids`` (sha-tied to its chains)."""
    root = Path(state_dir)
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = root.parent / f'.{root.name}.staging-{uuid.uuid4().hex[:10]}'
    stage.mkdir(parents=True, exist_ok=True)
    try:
        factors = _serialize_bundles(learner.bundles, learner.trainable_rows, learner.target_map)
        save_file(factors, str(stage / 'adapter_factors_fp32.safetensors'), metadata={'format': 'pt'})
        factors_sha = sha256_file_hex(stage / 'adapter_factors_fp32.safetensors')

        has_ref = bool(ref_bundles) or ref_rows is not None
        ref_sha = base_weights_sha256
        if has_ref:
            ref_tensors = _serialize_bundles(ref_bundles, ref_rows, learner.target_map)
            if ref_tensors:
                save_file(ref_tensors, str(stage / 'ref_factors_fp32.safetensors'), metadata={'format': 'pt'})
                ref_sha = sha256_file_hex(stage / 'ref_factors_fp32.safetensors')
            else:
                has_ref = False

        if optimizer is not None:
            torch.save(optimizer.state_dict(), stage / 'optimizer_state.pt')

        if trainer_state is not None:
            torch.save(trainer_state, stage / 'trainer_state.pt')

        rounds_ledger = list(dict.fromkeys(str(r) for r in (trained_rounds or ())))
        torch.save({
            'rng': capture_rng_state(),
            'batch_cursor': batch_cursor,
            'rl_rollout_ledger': rl_rollout_ledger,
            'ambient_state': ambient_state,
            'trained_rounds': rounds_ledger,
        }, stage / 'rng_and_cursors.pt')

        effective_sha = learner.effective_policy_sha256(base_weights_sha256)
        manifest = {
            'schema_version': STATE_SCHEMA,
            'git_sha': code_provenance().get('git_commit'),
            'code': code_provenance(),
            'saved_at': time.time(),
            'learner_step': int(learner_step),
            'source_checkpoint_id': str(source_checkpoint_id),
            'source_step': int(source_step),
            'base_weights_sha256': base_weights_sha256,
            'effective_policy_sha256': effective_sha,
            'reference_policy_sha256': ref_sha,
            'has_ref_factors': bool(has_ref),
            'adapter_factors_sha256': factors_sha,
            'adapter_spec': learner.spec.to_dict(),
            'optimizer_config': dict(optimizer_config or {}),
            'objective_version': objective_version,
            'ambient_state': ambient_state,
            'data_manifest_sha256': data_manifest_sha256,
            'batch_cursor': batch_cursor,
            'rl_rollout_ledger': rl_rollout_ledger,
            'trained_rounds': rounds_ledger,
            'hardware_profile': hardware_profile(learner.device),
            **(extra_meta or {}),
        }
        # Commit manifest last inside staging, then atomically rotate root -> root.prev, stage -> root.
        write_json_durable(stage / 'learner_manifest.json', manifest)
        prev = root.parent / f'{root.name}.prev'
        if root.exists():
            if prev.exists():
                shutil.rmtree(prev)
            if keep_previous:
                os.replace(root, prev)
            else:
                shutil.rmtree(root)
        os.replace(stage, root)
        return root
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def regroup_optimizer_state(saved: dict, optimizer: torch.optim.Optimizer) -> dict:
    """``saved`` (an optimizer ``state_dict``) laid out in ``optimizer``'s parameter groups when only the grouping
    differs: a state saved before the trainable rows had their own group (one group: factors then rows) resumes into
    the factor and row groups, and the reverse. The per-parameter moments keep their flat order; the live groups'
    hyperparameters (``lr_scale`` included) apply. A different parameter count is left to ``load_state_dict``."""
    groups = optimizer.param_groups
    if len(saved['param_groups']) == len(groups):
        return saved
    flat = [index for group in saved['param_groups'] for index in group['params']]
    if len(flat) != sum(len(group['params']) for group in groups):
        return saved
    regrouped, start = [], 0
    for group in groups:
        count = len(group['params'])
        regrouped.append({**{k: v for k, v in group.items() if k != 'params'}, 'params': flat[start:start + count]})
        start += count
    return {'state': saved['state'], 'param_groups': regrouped}


def load_learner_state(state_dir, learner: Lfm3Learner, optimizer: torch.optim.Optimizer | None = None, *,
                       restore_rng: bool = True, verify_base_sha: str | None = None) -> dict:
    """Restore FP32 LoRA factors, trainable rows, optimizer state, reference factors, and RNG/cursors."""
    root = Path(state_dir)
    manifest = json.loads((root / 'learner_manifest.json').read_text())
    if manifest.get('schema_version') != STATE_SCHEMA:
        raise ValueError(f"unsupported learner state schema {manifest.get('schema_version')!r}")
    if verify_base_sha and manifest.get('base_weights_sha256') and manifest['base_weights_sha256'] != verify_base_sha:
        raise ValueError(f"learner state base_weights_sha256 {manifest['base_weights_sha256']} != {verify_base_sha}")
    saved_spec = AdapterSpec.from_dict(manifest['adapter_spec'])
    if saved_spec != learner.spec:
        raise ValueError(f'learner state AdapterSpec {saved_spec} != live {learner.spec} (rank/target change is a new lineage)')

    device = learner.device
    with safe_open(str(root / 'adapter_factors_fp32.safetensors'), framework='pt', device='cpu') as handle:
        keys = set(handle.keys())
        with torch.no_grad():
            for name, leaf in learner.named_trainable_tensors().items():
                if name not in keys:
                    raise ValueError(f'learner state missing tensor {name}')
                leaf.copy_(handle.get_tensor(name).to(device=device, dtype=torch.float32))

    if optimizer is not None and (root / 'optimizer_state.pt').is_file():
        opt_state = torch.load(root / 'optimizer_state.pt', map_location=device, weights_only=False)
        optimizer.load_state_dict(regroup_optimizer_state(opt_state, optimizer))

    ref_bundles: tuple[LayerAdapterBundle, ...] = ()
    ref_rows: TrainableRows | None = None
    if manifest.get('has_ref_factors') and (root / 'ref_factors_fp32.safetensors').is_file():
        with safe_open(str(root / 'ref_factors_fp32.safetensors'), framework='pt', device='cpu') as handle:
            rkeys = set(handle.keys())
            per_layer: dict[int, dict[str, LinearFactorLeaf]] = {i: {} for i in range(len(learner.bundles))}
            any_factor = False
            for param_name, (layer_idx, _kind, short_name) in learner.target_map.items():
                ak, bk = f'lora_A.{param_name}', f'lora_B.{param_name}'
                if ak in rkeys and bk in rkeys:
                    any_factor = True
                    live_leaf = getattr(learner.bundles[layer_idx], short_name)
                    per_layer[layer_idx][short_name] = LinearFactorLeaf(
                        a=handle.get_tensor(ak).to(device=device, dtype=torch.float32).requires_grad_(False),
                        b=handle.get_tensor(bk).to(device=device, dtype=torch.float32).requires_grad_(False),
                        scale=float(live_leaf.scale),
                    )
            if any_factor:
                ref_bundles = tuple(
                    LayerAdapterBundle(layer_idx=b.layer_idx, layer_kind=b.layer_kind, **per_layer[b.layer_idx])
                    for b in learner.bundles
                )
            if 'trainable_rows.rows' in rkeys and learner.trainable_rows is not None:
                ref_rows = TrainableRows(
                    token_ids=handle.get_tensor('trainable_rows.token_ids').to(device=device, dtype=torch.long),
                    rows=handle.get_tensor('trainable_rows.rows').to(device=device, dtype=torch.float32).requires_grad_(False),
                    marker_ids=learner.trainable_rows.marker_ids,
                    override_ids=learner.trainable_rows.override_ids,
                )

    cursors = {}
    if (root / 'rng_and_cursors.pt').is_file():
        cursors = torch.load(root / 'rng_and_cursors.pt', map_location='cpu', weights_only=False)
        if restore_rng and 'rng' in cursors:
            restore_rng_state(cursors['rng'])

    trainer_state = None
    if (root / 'trainer_state.pt').is_file():
        trainer_state = torch.load(root / 'trainer_state.pt', map_location='cpu', weights_only=False)

    return {
        'manifest': manifest,
        'trainer_state': trainer_state,
        'ref_bundles': ref_bundles,
        'ref_rows': ref_rows,
        'batch_cursor': cursors.get('batch_cursor', manifest.get('batch_cursor')),
        'rl_rollout_ledger': cursors.get('rl_rollout_ledger', manifest.get('rl_rollout_ledger')),
        'ambient_state': cursors.get('ambient_state', manifest.get('ambient_state')),
        'trained_rounds': list(cursors.get('trained_rounds', manifest.get('trained_rounds') or ())),
    }


def _write_merged(out_path, state, backbone, interface, *, learner_step, adapter_spec, effective_sha, source,
                  extra=None, tensors=None) -> dict:
    """Write a merged native checkpoint (``merge_policy_state``'s result) staged and atomically; returns its record.
    ``tensors`` (the merged factor tensors): their trainable rows also go in as the FP32 master copy
    ``trainable_rows_fp32`` (``lfm3_learner.master_rows``)."""
    master = master_rows(tensors or {})
    if master is not None:
        extra = {**(extra or {}), 'trainable_rows_fp32': master}
    source_step = int(state['step'])
    step_val = int(learner_step)
    out = {
        'schema': state['schema'],
        'model_config': state['model_config'],
        'interface_state': interface,
        'backbone_state': backbone,
        'step': step_val,
        'learner_step': step_val,
        'source_step': source_step,
        'parameterization': 'lora+rows',
        'adapter_spec': adapter_spec,
        'effective_policy_sha256': effective_sha,
        'inference_only': True,
        'optimizer_pruned': True,
        'optimizer_pruned_keys': ['training'],
        'inference_source': {'schema': 'natlang.native-inference-source/1', 'step': step_val, 'learner_step': step_val,
                             'source_step': source_step, 'effective_policy_sha256': effective_sha, **source},
        **(extra or {}),
    }
    from .checkpoint_offload import staged_write
    out_path = Path(out_path)
    expected_bytes = sum(v.numel() * v.element_size() for k, v in backbone.items() if k != 'lm_head.weight')
    with staged_write(out_path, expected_bytes=expected_bytes) as target:
        torch.save(out, target.path)
    return {'path': str(out_path), 'checkpoint_sha256': sha256_file_hex(out_path), 'effective_policy_sha256': effective_sha,
            'learner_step': step_val, 'source_step': source_step}


def export_merged_checkpoint(learner: Lfm3Learner, state: dict, out_path, *, source_sha256: str,
                             learner_step: int) -> dict:
    """Export an immutable merged BF16 native checkpoint of the learner over its source ``state`` without mutating live
    weights (``merge_policy_state``, the merge a shipped factor directory gets on the DGX). Records ``learner_step``
    and ``source_step`` separately (`lfm3opti.md` §3)."""
    tensors, scales = learner.factor_tensors()
    backbone, interface, changed = merge_policy_state(state, tensors, scales)
    effective_sha = learner.effective_policy_sha256(source_sha256)
    record = _write_merged(out_path, state, backbone, interface, learner_step=learner_step,
                           adapter_spec=learner.spec.to_dict(), effective_sha=effective_sha,
                           source={'training_checkpoint_sha256': source_sha256, 'training_checkpoint_bytes': 0},
                           tensors=tensors)
    return {**record, 'changed_elements': changed}


def export_adapter_factors(
    learner: Lfm3Learner,
    out_dir,
    *,
    base_checkpoint_sha256: str,
    learner_step: int,
    round_id: str | None = None,
    source_checkpoint_id: str = '',
    objective_version: str = 'sft_v1',
    extra_meta: dict | None = None,
) -> dict:
    """Export FP32 LoRA factors and trainable Neuralese rows for WAN shipping (async-train §2.4 & §7.1).

    Commit order is strictly blob-first: ``adapter_factors_fp32.safetensors`` is written and checksummed inside a
    staging directory, ``policy_manifest.json`` is committed last, and the directory is atomically renamed into place.
    Size is ~0.5 GB at r=128 (SFT) and ~130 MB at r=32 (GRPO).
    """
    dest = Path(out_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = dest.parent / f'.{dest.name}.staging-{uuid.uuid4().hex[:10]}'
    stage.mkdir(parents=True, exist_ok=True)
    try:
        tensors, scales = learner.factor_tensors()
        factors_file = stage / 'adapter_factors_fp32.safetensors'
        save_file(tensors, str(factors_file), metadata={'format': 'pt'})
        factors_sha = sha256_file_hex(factors_file)
        effective_sha = learner.effective_policy_sha256(base_checkpoint_sha256)
        manifest = {
            'schema': FACTOR_EXPORT_SCHEMA,
            'git_sha': code_provenance().get('git_commit'),
            'created_at': time.time(),
            'round_id': round_id,
            'learner_step': int(learner_step),
            'base_checkpoint_sha256': str(base_checkpoint_sha256),
            'source_checkpoint_id': str(source_checkpoint_id),
            'effective_policy_sha256': effective_sha,
            'adapter_factors_file': 'adapter_factors_fp32.safetensors',
            'adapter_factors_sha256': factors_sha,
            'adapter_factors_bytes': factors_file.stat().st_size,
            'adapter_spec': learner.spec.to_dict(),
            'scales': scales,
            'objective_version': objective_version,
            'has_trainable_rows': learner.trainable_rows is not None,
            **(extra_meta or {}),
        }
        write_json_durable(stage / 'policy_manifest.json', manifest)
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(stage, dest)
        return {'dir': str(dest), **manifest}
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def merge_factors_to_checkpoint(base_checkpoint_path, factors_dir, out_checkpoint_path, *,
                                learner_step: int | None = None) -> dict:
    """Merge a shipped factor directory (``export_adapter_factors``) into its base checkpoint: ``merge_policy_state``,
    the learner's own export merge, so the result equals ``export_merged_checkpoint`` of the same learner bit for bit
    (async-train §2.5 & §7.1). Fail-closed: the manifest schema, the factor file's sha256 and the base checkpoint's
    sha256 (``base_checkpoint_sha256``) are checked before merging; every target needs its recorded scale."""
    base_path = Path(base_checkpoint_path).resolve(strict=True)
    fdir = Path(factors_dir).resolve(strict=True)
    manifest = json.loads((fdir / 'policy_manifest.json').read_text())
    if manifest.get('schema') != FACTOR_EXPORT_SCHEMA:
        raise ValueError(f'unsupported factor export schema {manifest.get("schema")!r}')
    factors_path = fdir / manifest.get('adapter_factors_file', 'adapter_factors_fp32.safetensors')
    actual_factors_sha = sha256_file_hex(factors_path)
    if actual_factors_sha != manifest.get('adapter_factors_sha256'):
        raise ValueError(f'factor checksum mismatch: {actual_factors_sha} != {manifest.get("adapter_factors_sha256")}')
    actual_base_sha = sha256_file_hex(base_path)
    if actual_base_sha != manifest.get('base_checkpoint_sha256'):
        raise ValueError(f'base checkpoint mismatch: {actual_base_sha} != manifest base_checkpoint_sha256 '
                         f'{manifest.get("base_checkpoint_sha256")}')
    from .checkpoint_policy import CheckpointPolicy
    state = CheckpointPolicy.load(base_path)
    with safe_open(str(factors_path), framework='pt', device='cpu') as handle:
        tensors = {k: handle.get_tensor(k) for k in handle.keys()}
    backbone, interface, changed = merge_policy_state(state, tensors, manifest.get('scales') or {})
    step_val = int(manifest['learner_step']) if learner_step is None else int(learner_step)
    effective_sha = manifest['effective_policy_sha256']
    record = _write_merged(out_checkpoint_path, state, backbone, interface, learner_step=step_val,
                           adapter_spec=manifest['adapter_spec'], effective_sha=effective_sha,
                           source={'training_checkpoint_sha256': actual_base_sha, 'training_checkpoint_bytes':
                                   base_path.stat().st_size, 'adapter_factors_sha256': actual_factors_sha},
                           extra={'round_id': manifest.get('round_id')}, tensors=tensors)
    return {**record, 'base_checkpoint_sha256': actual_base_sha, 'adapter_factors_sha256': actual_factors_sha,
            'merged_matrices': len(manifest.get('scales') or {}), 'changed_elements': changed,
            'round_id': manifest.get('round_id')}
