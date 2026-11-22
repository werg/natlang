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
                       trained_episode_ids: list[str] | None = None,
                       ref_bundles: tuple[LayerAdapterBundle, ...] | None = None,
                       ref_rows: TrainableRows | None = None,
                       keep_previous: bool = True, extra_meta: dict | None = None,
                       trainer_state: dict | None = None) -> Path:
    """Atomically save resumable LoRA+rows learner state into ``state_dir`` (staging dir -> atomic manifest commit).

    ``trainer_state``: the trainer's own resume state (any ``torch.save``-able dict: e.g. exp0_sft's OOM plan, replanned
    schedule and kept-best leaves), returned by ``load_learner_state`` as ``trainer_state``.
    ``trained_rounds`` / ``trained_episode_ids``: async-train §2/§3 dedup ledger preventing round double-training."""
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
        episodes_ledger = list(dict.fromkeys(str(e) for e in (trained_episode_ids or ())))
        torch.save({
            'rng': capture_rng_state(),
            'batch_cursor': batch_cursor,
            'rl_rollout_ledger': rl_rollout_ledger,
            'ambient_state': ambient_state,
            'trained_rounds': rounds_ledger,
            'trained_episode_ids': episodes_ledger,
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
            'trained_episode_count': len(episodes_ledger),
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
        optimizer.load_state_dict(opt_state)

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
        'trained_episode_ids': list(cursors.get('trained_episode_ids', ())),
    }


def export_merged_checkpoint(learner: Lfm3Learner, state: dict, out_path, *, source_sha256: str,
                             learner_step: int, special_ids=None, base_embedding_rows=None,
                             tied_head: bool = True) -> dict:
    """Export an immutable merged BF16 native checkpoint (``bf16(fp32(W0) + s * B A)``) without mutating live weights.

    Records ``learner_step`` and ``source_step`` separately (fixing `lfm3opti.md` §3). Updated trainable Neuralese rows
    are written into ``interface_state`` (and ``model.embed_tokens.weight`` keeps original base rows at special indices
    if ``base_embedding_rows`` is given, matching ``load_native_checkpoint``'s merge contract).
    """
    out_path = Path(out_path)
    backbone, interface = learner.merged_backbone_and_interface(device='cpu', dtype=torch.bfloat16)
    if not interface and state.get('interface_state'):
        interface = {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                     for k, v in state['interface_state'].items()}
        if learner.trainable_rows is not None:
            mc = state.get('model_config') or {}
            marker_ids = list(mc.get('marker_ids') or ())
            override_ids = list(mc.get('token_override_ids') or ())
            tr_map = {int(tid): row.detach().to(device='cpu', dtype=torch.bfloat16)
                      for tid, row in zip(learner.trainable_rows.token_ids.tolist(), learner.trainable_rows.rows)}
            if marker_ids and 'input_rows' in interface and all(t in tr_map for t in marker_ids):
                interface['input_rows'] = torch.stack([tr_map[t] for t in marker_ids]).to(interface['input_rows'].dtype)
            if override_ids and 'extra_input_rows' in interface and all(t in tr_map for t in override_ids):
                interface['extra_input_rows'] = torch.stack([tr_map[t] for t in override_ids]).to(interface['extra_input_rows'].dtype)
    original_backbone = state.get('backbone_state') or {}
    changed = 0
    if special_ids and base_embedding_rows is not None:
        embed = backbone['model.embed_tokens.weight'].clone()
        embed[torch.tensor(list(special_ids), dtype=torch.long)] = base_embedding_rows.to(embed.dtype)
        backbone['model.embed_tokens.weight'] = embed.contiguous()
    elif special_ids and 'model.embed_tokens.weight' in original_backbone:
        embed = backbone['model.embed_tokens.weight'].clone()
        idx = torch.tensor(list(special_ids), dtype=torch.long)
        embed[idx] = original_backbone['model.embed_tokens.weight'][idx].to(embed.dtype)
        backbone['model.embed_tokens.weight'] = embed.contiguous()
    for name, tensor in backbone.items():
        if name in original_backbone:
            changed += int((tensor != original_backbone[name].to(tensor.dtype)).sum())
    if tied_head or 'lm_head.weight' in original_backbone:
        backbone['lm_head.weight'] = backbone['model.embed_tokens.weight']

    source_step = int(state['step'])
    step_val = int(learner_step)
    effective_sha = learner.effective_policy_sha256(source_sha256)
    out = {
        'schema': state['schema'],
        'model_config': state['model_config'],
        'interface_state': interface,
        'backbone_state': backbone,
        'step': step_val,
        'learner_step': step_val,
        'source_step': source_step,
        'parameterization': 'lora+rows',
        'adapter_spec': learner.spec.to_dict(),
        'effective_policy_sha256': effective_sha,
        'inference_only': True,
        'optimizer_pruned': True,
        'optimizer_pruned_keys': ['training'],
        'inference_source': {
            'schema': 'natlang.native-inference-source/1',
            'training_checkpoint_sha256': source_sha256,
            'training_checkpoint_bytes': 0,
            'step': step_val,
            'learner_step': step_val,
            'source_step': source_step,
            'effective_policy_sha256': effective_sha,
        },
    }
    from .checkpoint_offload import staged_write
    expected_bytes = sum(v.numel() * v.element_size() for k, v in backbone.items() if k != 'lm_head.weight')
    with staged_write(out_path, expected_bytes=expected_bytes) as target:
        torch.save(out, target.path)
    ckpt_sha = sha256_file_hex(out_path)
    return {
        'path': str(out_path),
        'checkpoint_sha256': ckpt_sha,
        'effective_policy_sha256': effective_sha,
        'learner_step': step_val,
        'source_step': source_step,
        'changed_elements': changed,
    }


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
        tensors = _serialize_bundles(learner.bundles, learner.trainable_rows, learner.target_map)
        factors_file = stage / 'adapter_factors_fp32.safetensors'
        save_file(tensors, str(factors_file), metadata={'format': 'pt'})
        factors_sha = sha256_file_hex(factors_file)
        effective_sha = learner.effective_policy_sha256(base_checkpoint_sha256)
        scales = {}
        for param_name, (layer_idx, _kind, short_name) in sorted(learner.target_map.items()):
            leaf = getattr(learner.bundles[layer_idx], short_name)
            if leaf is not None:
                scales[param_name] = float(leaf.scale)
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


def merge_factors_to_checkpoint(
    base_checkpoint_path,
    factors_dir,
    out_checkpoint_path,
    *,
    learner_step: int | None = None,
    verify_base_sha: bool = True,
) -> dict:
    """Merge shipped FP32 LoRA factors + trainable Neuralese rows into ``base_checkpoint_path`` on CPU in FP32,
    round to BF16 once, and write an immutable merged native checkpoint (async-train §2.5 & §7.1).

    Fail-closed checks: verifies ``policy_manifest.json`` schema, ``base_checkpoint_sha256`` against the base
    checkpoint file, and ``adapter_factors_sha256`` against ``adapter_factors_fp32.safetensors`` before merging.
    """
    base_path = Path(base_checkpoint_path).resolve(strict=True)
    fdir = Path(factors_dir).resolve(strict=True)
    out_path = Path(out_checkpoint_path)
    manifest = json.loads((fdir / 'policy_manifest.json').read_text())
    if manifest.get('schema') != FACTOR_EXPORT_SCHEMA:
        raise ValueError(f'unsupported factor export schema {manifest.get("schema")!r}')
    factors_path = fdir / manifest.get('adapter_factors_file', 'adapter_factors_fp32.safetensors')
    actual_factors_sha = sha256_file_hex(factors_path)
    if manifest.get('adapter_factors_sha256') and actual_factors_sha != manifest['adapter_factors_sha256']:
        raise ValueError(
            f'factor checksum mismatch: {actual_factors_sha} != {manifest["adapter_factors_sha256"]}'
        )
    actual_base_sha = sha256_file_hex(base_path)
    expected_base_sha = manifest.get('base_checkpoint_sha256')
    if verify_base_sha and expected_base_sha and actual_base_sha != expected_base_sha:
        raise ValueError(
            f'base checkpoint mismatch: {actual_base_sha} != manifest base_checkpoint_sha256 {expected_base_sha}'
        )

    from .checkpoint_policy import CheckpointPolicy
    state = CheckpointPolicy.load(base_path)
    backbone = {k: v.detach().cpu().clone() for k, v in state['backbone_state'].items()}
    tied_head = 'lm_head.weight' in backbone
    backbone.pop('lm_head.weight', None)
    interface = {
        k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
        for k, v in (state.get('interface_state') or {}).items()
    }
    spec = AdapterSpec.from_dict(manifest['adapter_spec'])
    scales = manifest.get('scales') or {}

    merged_matrices = 0
    changed = 0
    with safe_open(str(factors_path), framework='pt', device='cpu') as handle:
        keys = set(handle.keys())
        a_keys = sorted(k for k in keys if k.startswith('lora_A.'))
        for ak in a_keys:
            param_name = ak[len('lora_A.'):]
            bk = f'lora_B.{param_name}'
            if bk not in keys:
                raise ValueError(f'missing matching factor tensor {bk}')
            if param_name not in backbone:
                raise ValueError(f'factor target {param_name} absent from base checkpoint backbone')
            a = handle.get_tensor(ak).to(torch.float32)
            b = handle.get_tensor(bk).to(torch.float32)
            scale = float(scales.get(param_name, spec.alpha / float(a.shape[0])))
            base_w = backbone[param_name]
            merged_w = (base_w.to(torch.float32) + scale * (b @ a)).to(torch.bfloat16).contiguous()
            changed += int((merged_w != base_w.to(torch.bfloat16)).sum())
            backbone[param_name] = merged_w
            merged_matrices += 1
        if 'trainable_rows.rows' in keys and 'trainable_rows.token_ids' in keys:
            tr_ids = handle.get_tensor('trainable_rows.token_ids').to(torch.long).tolist()
            tr_rows = handle.get_tensor('trainable_rows.rows').to(torch.bfloat16)
            tr_map = {int(tid): row for tid, row in zip(tr_ids, tr_rows)}
            mc = state.get('model_config') or {}
            marker_ids = [int(i) for i in (mc.get('marker_ids') or ())]
            override_ids = [int(i) for i in (mc.get('token_override_ids') or ())]
            if marker_ids and 'input_rows' in interface and all(t in tr_map for t in marker_ids):
                new_in = torch.stack([tr_map[t] for t in marker_ids]).to(interface['input_rows'].dtype)
                changed += int((new_in != interface['input_rows']).sum())
                interface['input_rows'] = new_in
            if override_ids and 'extra_input_rows' in interface and all(t in tr_map for t in override_ids):
                new_ex = torch.stack([tr_map[t] for t in override_ids]).to(interface['extra_input_rows'].dtype)
                changed += int((new_ex != interface['extra_input_rows']).sum())
                interface['extra_input_rows'] = new_ex

    if tied_head:
        backbone['lm_head.weight'] = backbone['model.embed_tokens.weight']

    source_step = int(state['step'])
    step_val = int(manifest['learner_step']) if learner_step is None else int(learner_step)
    effective_sha = manifest.get('effective_policy_sha256') or actual_factors_sha
    out = {
        'schema': state['schema'],
        'model_config': state['model_config'],
        'interface_state': interface,
        'backbone_state': backbone,
        'step': step_val,
        'learner_step': step_val,
        'source_step': source_step,
        'parameterization': 'lora+rows',
        'adapter_spec': manifest['adapter_spec'],
        'effective_policy_sha256': effective_sha,
        'round_id': manifest.get('round_id'),
        'inference_only': True,
        'optimizer_pruned': True,
        'optimizer_pruned_keys': ['training'],
        'inference_source': {
            'schema': 'natlang.native-inference-source/1',
            'training_checkpoint_sha256': actual_base_sha,
            'training_checkpoint_bytes': base_path.stat().st_size,
            'step': step_val,
            'learner_step': step_val,
            'source_step': source_step,
            'effective_policy_sha256': effective_sha,
            'adapter_factors_sha256': actual_factors_sha,
        },
    }
    from .checkpoint_offload import staged_write
    expected_bytes = sum(v.numel() * v.element_size() for k, v in backbone.items() if k != 'lm_head.weight')
    with staged_write(out_path, expected_bytes=expected_bytes) as target:
        torch.save(out, target.path)
    ckpt_sha = sha256_file_hex(out_path)
    return {
        'path': str(out_path),
        'checkpoint_sha256': ckpt_sha,
        'effective_policy_sha256': effective_sha,
        'base_checkpoint_sha256': actual_base_sha,
        'adapter_factors_sha256': actual_factors_sha,
        'learner_step': step_val,
        'source_step': source_step,
        'merged_matrices': merged_matrices,
        'changed_elements': changed,
        'round_id': manifest.get('round_id'),
    }
