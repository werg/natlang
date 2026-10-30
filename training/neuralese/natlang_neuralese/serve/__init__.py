"""Native shared-state model serving loaders."""

from __future__ import annotations


def load_native_engine(checkpoint: str, *, base: str | None = None, device: str = "cpu", dtype=None,
                       prompt_rendering: str = "full", participant: str | None = None,
                       actor_view: dict | None = None, transported_actor: dict | None = None,
                       store=None, engine_class=None):
    """Restore a native-state checkpoint through the shared model loader and serving engine."""
    from ..common.hashing import sha256_file_hex_cached
    from ..model.native_model import load_native_checkpoint, native_identity
    from .engine import Engine
    from .store import TensorStore

    view_options = ({'actor_view':actor_view} if actor_view is not None else
                    {'transported_actor':transported_actor} if transported_actor is not None else {})
    model, tokenizer, roles, config, _state = load_native_checkpoint(
        checkpoint, base=base, device=device, dtype=dtype, participant=participant, allow_weights_only=True,
        **view_options)
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
                    embedding_weight_sha256=identity['embedding_weight_sha256'])
    engine.native_identity = identity
    engine.native_checkpoint = str(checkpoint)
    engine.native_training_state = (_state.get('training') or _state.get('shared_training') or {})
    engine.native_actor_view = _state.get('actor_view')
    engine.native_inference_only = bool(engine.native_actor_view is not None
        or _state.get('inference_only') or _state.get('optimizer_pruned'))
    engine.rl_policy_binding = {'schema': 'natlang.rl-policy-binding/2',
        'checkpoint_sha256': checkpoint_sha256, 'communication': 'native-state',
        'native_identity': dict(identity),
        'history_reasoning': dict(tokenizer.natlang_history_reasoning),
        **({'native_participant': participant} if participant is not None else {})}
    if _state.get('inference_source') is not None:
        engine.rl_policy_binding['inference_source'] = dict(_state['inference_source'])
    if engine.native_actor_view is not None:
        engine.rl_policy_binding['actor_view'] = dict(engine.native_actor_view)
    from ..common.runtime_inventory import python_runtime_binding
    engine.rl_policy_binding['python_runtime'] = python_runtime_binding()
    return engine
