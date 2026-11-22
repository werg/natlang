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
                       ref_bundles: tuple[LayerAdapterBundle, ...] | None = None,
                       ref_rows: TrainableRows | None = None,
                       keep_previous: bool = True, extra_meta: dict | None = None) -> Path:
    """Atomically save resumable LoRA+rows learner state into ``state_dir`` (staging dir -> atomic manifest commit)."""
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

        torch.save({
            'rng': capture_rng_state(),
            'batch_cursor': batch_cursor,
            'rl_rollout_ledger': rl_rollout_ledger,
            'ambient_state': ambient_state,
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

    return {
        'manifest': manifest,
        'ref_bundles': ref_bundles,
        'ref_rows': ref_rows,
        'batch_cursor': cursors.get('batch_cursor', manifest.get('batch_cursor')),
        'rl_rollout_ledger': cursors.get('rl_rollout_ledger', manifest.get('rl_rollout_ledger')),
        'ambient_state': cursors.get('ambient_state', manifest.get('ambient_state')),
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


def export_merged_hf_safetensors(learner: Lfm3Learner, template_export_dir, out_dir, *,
                                 producer_from_embedding: bool = True) -> dict:
    """Materialize a complete immutable merged BF16 HF safetensors export directory for SGLang/native serving
    (v2 §7.1 atomic staging -> checksum verify -> atomic rename)."""
    template = Path(template_export_dir)
    dest = Path(out_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = dest.parent / f'.{dest.name}.staging-{uuid.uuid4().hex[:10]}'
    stage.mkdir(parents=True, exist_ok=True)
    try:
        for item in template.iterdir():
            if item.is_file() and item.name != 'model.safetensors':
                shutil.copyfile(item, stage / item.name)
        merged, _interface = learner.merged_backbone_and_interface(device='cpu', dtype=torch.bfloat16)
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(template / 'model.safetensors'), framework='pt', device='cpu') as handle:
            for name in handle.keys():
                if name in merged:
                    out_tensors[name] = merged[name]
                elif name == 'model.natlang_producer_embed' and producer_from_embedding:
                    out_tensors[name] = merged['model.embed_tokens.weight'].clone()
                else:
                    out_tensors[name] = handle.get_tensor(name).to(torch.bfloat16).contiguous()
        weights_file = stage / 'model.safetensors'
        save_file(out_tensors, str(weights_file), metadata={'format': 'pt'})
        weights_sha = sha256_file_hex(weights_file)
        effective_sha = learner.effective_policy_sha256()
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(stage, dest)
        return {
            'dir': str(dest),
            'weights_path': str(dest / 'model.safetensors'),
            'weights_sha256': weights_sha,
            'effective_policy_sha256': effective_sha,
            'tensor_count': len(out_tensors),
        }
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
