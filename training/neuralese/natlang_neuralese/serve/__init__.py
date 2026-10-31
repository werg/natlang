"""Native shared-state model serving loaders."""

from __future__ import annotations


SMALL_GPU_PREFILL_LIMIT_BYTES = 12 * 1024**3
SMALL_GPU_PREFILL_TOKENS = 2048
LARGE_GPU_PREFILL_TOKENS = 8192


def native_prefill_configuration(device: str, requested: int | None = None, *,
                                 total_memory_bytes: int | None = None) -> dict:
    """Choose and bind the native prompt-prefill chunk size for this actor.

    Chunking changes only how the same complete prompt is forwarded. It does
    not truncate or rewrite prompt tokens. Small GPUs use smaller activation
    batches; larger GPUs retain the established chunk size.
    """
    import torch
    if requested is not None and (type(requested) is not int or requested <= 0):
        raise ValueError("prefill_tokens must be a positive integer")
    device_obj = torch.device(device)
    if device_obj.type == "cuda":
        if total_memory_bytes is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA prefill selection requested but CUDA is unavailable")
            total_memory_bytes = int(torch.cuda.get_device_properties(device_obj).total_memory)
        elif type(total_memory_bytes) is not int or total_memory_bytes <= 0:
            raise ValueError("device_total_memory_bytes must be a positive integer")
    if requested is None:
        chosen = (SMALL_GPU_PREFILL_TOKENS if total_memory_bytes is not None
                  and total_memory_bytes <= SMALL_GPU_PREFILL_LIMIT_BYTES
                  else LARGE_GPU_PREFILL_TOKENS)
        selection = "auto-small-gpu-2048-large-gpu-8192-v1"
    else:
        chosen = requested
        selection = "explicit"
    return {"schema": "natlang.native-prefill-config/1", "device": str(device_obj),
        "device_total_memory_bytes": total_memory_bytes, "prefill_tokens": chosen,
        "selection": selection, "full_prompt_preserved": True,
        "identity_sha256": None}



def configure_native_prefix_cache(engine, budget_gb: float = 2.0, block: int = 256):
    """Bind prefill grid and cache capacity to the policy identity before serving."""
    import math
    from ..common.hashing import canonical_json_sha256_hex
    from .prefix_cache import PrefixCache
    if (isinstance(budget_gb, bool) or not isinstance(budget_gb, (int, float))
            or not math.isfinite(budget_gb) or budget_gb < 0
            or type(block) is not int or block <= 0):
        raise ValueError('prefix cache needs a nonnegative finite budget and positive integer block')
    if budget_gb and block > engine.prefill_tokens:
        raise ValueError('prefix cache block exceeds the bound prefill token limit')
    budget = int(budget_gb * 2**30)
    if budget_gb and not budget:
        raise ValueError('positive prefix cache budget rounds to zero bytes')
    if getattr(engine, '_thread', None) is not None:
        raise ValueError('prefix cache identity cannot change after engine start')
    engine.prefix_cache = PrefixCache(budget, block) if budget else None
    config = dict(engine.serving_config)
    config['prefix_cache'] = {'schema': 'natlang.native-prefix-cache/1',
        'enabled': bool(budget), 'budget_bytes': budget, 'block_tokens': block,
        'scope': 'fresh-native-prefill-only', 'key': 'model-identity-adapter-binding-and-exact-token-prefix',
        'execution': 'fixed-block-grid' if budget else 'native-prefill-chunks'}
    config['identity_sha256'] = canonical_json_sha256_hex(
        {key: value for key, value in config.items() if key != 'identity_sha256'})
    engine.serving_config = config
    if hasattr(engine, 'rl_policy_binding'):
        engine.rl_policy_binding['serving_config_sha256'] = config['identity_sha256']
    return config

def load_native_engine(checkpoint: str, *, base: str | None = None, device: str = "cpu", dtype=None,
                       prompt_rendering: str = "full", participant: str | None = None,
                       actor_view: dict | None = None, transported_actor: dict | None = None,
                       store=None, engine_class=None, prefill_tokens: int | None = None):
    """Restore a native-state checkpoint through the shared model loader and serving engine."""
    from ..common.hashing import sha256_file_hex_cached
    from ..model.native_model import load_native_checkpoint, native_identity
    from .engine import Engine
    from .store import TensorStore
    from ..common.hashing import canonical_json_sha256_hex

    serving_config = native_prefill_configuration(device, prefill_tokens)
    view_options = ({'actor_view':actor_view} if actor_view is not None else
                    {'transported_actor':transported_actor} if transported_actor is not None else {})
    model, tokenizer, roles, config, _state = load_native_checkpoint(
        checkpoint, base=base, device=device, dtype=dtype, participant=participant, allow_weights_only=True,
        **view_options)
    # Native policy inference runs the stored master weights without an
    # active inline QAT point. The SFT objective may sample other points;
    # those training passes do not silently change the deployed policy.
    serving_config['precision'] = {'schema':'natlang.native-policy-precision/1',
        'weights':'checkpoint-master', 'inline_qat_point':'inactive',
        'ordinary_precision':config.get('native_ordinary_precision'), 'model_dtype':config.get('dtype')}
    serving_config['identity_sha256'] = canonical_json_sha256_hex(
        {key:value for key,value in serving_config.items() if key != 'identity_sha256'})
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint_sha256 = (_state['inference_source']['training_checkpoint_sha256'] if transported_actor is not None
                         else sha256_file_hex_cached(checkpoint))
    identity = native_identity(model, config, checkpoint_sha256=checkpoint_sha256)
    engine = (engine_class or Engine)(model.port, tokenizer, store if store is not None else TensorStore(), f"native-state/{config['architecture']}",
                    model_name=f"native-{config['architecture']}", device=device,
                    prompt_rendering=prompt_rendering, shared_model=model, role_tokens=roles,
                    model_config=config, checkpoint_sha256=identity['checkpoint_sha256'],
                    embedding_weight_sha256=identity['embedding_weight_sha256'],
                    prefill_tokens=serving_config["prefill_tokens"])
    engine.native_identity = identity
    engine.serving_config = serving_config
    engine.native_checkpoint = str(checkpoint)
    engine.native_training_state = (_state.get('training') or _state.get('shared_training') or {})
    engine.native_actor_view = _state.get('actor_view')
    engine.native_inference_only = bool(engine.native_actor_view is not None
        or _state.get('inference_only') or _state.get('optimizer_pruned'))
    engine.rl_policy_binding = {'schema': 'natlang.rl-policy-binding/2',
        'checkpoint_sha256': checkpoint_sha256, 'communication': 'native-state',
        'native_identity': dict(identity),
        'history_reasoning': dict(tokenizer.natlang_history_reasoning),
        'serving_precision':dict(serving_config['precision']),
        'serving_config_sha256':serving_config['identity_sha256'],
        **({'native_participant': participant} if participant is not None else {})}
    base_restore = getattr(model, 'native_base_restore_receipt', None)
    if base_restore is not None:
        engine.rl_policy_binding['base_restore'] = dict(base_restore)
    if _state.get('inference_source') is not None:
        engine.rl_policy_binding['inference_source'] = dict(_state['inference_source'])
    if engine.native_actor_view is not None:
        engine.rl_policy_binding['actor_view'] = dict(engine.native_actor_view)
    from ..common.runtime_inventory import python_runtime_binding
    engine.rl_policy_binding['python_runtime'] = python_runtime_binding()
    return engine
