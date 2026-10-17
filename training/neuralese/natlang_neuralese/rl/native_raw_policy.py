"""Exact bare native policy/tokenizer binding without constructing a second model."""
from pathlib import Path
from types import SimpleNamespace

from ..common.hashing import sha256_file_hex_cached


def require_raw_training(state):
    """One metadata contract for the actual resumable single-model optimizer."""
    from ..model.native_model import CHECKPOINT_SCHEMA
    from ..train.native_continuation import TRAINING_SCHEMA
    training=state.get('training') or {}
    if (state.get('schema')!=CHECKPOINT_SCHEMA or state.get('optimizer_pruned')
            or not training.get('optimizer') or training.get('schema')!=TRAINING_SCHEMA or not training.get('rng')):
        raise ValueError('raw native RL requires its actual full model optimizer, native training schedule and RNG state')
    return training


def prepare_raw_policy(checkpoint, *, base, tokenizer_out, require_optimizer=True, verified_checkpoint=None):
    import torch
    from transformers import AutoTokenizer
    from ..train.checkpoint_policy import CheckpointPolicy
    from ..model.native_model import CHECKPOINT_SCHEMA, native_identity
    from ..model.shared_state import provision_role_tokenizer
    from ..data.text_corpus import tokenizer_fingerprint
    checkpoint = Path(checkpoint).resolve(strict=True)
    if verified_checkpoint is None:
        state = CheckpointPolicy.load(checkpoint)
    else:
        reference,state=verified_checkpoint
        if (Path(reference['path']).resolve(strict=True)!=checkpoint or
                sha256_file_hex_cached(checkpoint)!=reference['sha256']):
            raise ValueError('reused raw checkpoint state belongs to another exact parent')
    if state.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError('raw native RL requires an actual single-model native checkpoint')
    if require_optimizer:require_raw_training(state)
    config = state['model_config']
    interface = state['interface_state']
    embedding_keys = [key for key in state['backbone_state'] if key == 'embed_tokens.weight' or key.endswith('.embed_tokens.weight')]
    if len(embedding_keys) != 1:
        raise ValueError('native checkpoint must identify its one actual input embedding tensor')
    dtype = {'torch.bfloat16': torch.bfloat16, 'torch.float32': torch.float32,
             'torch.float16': torch.float16}[config['dtype']]
    # These are views of saved tensors, not modules or randomly initialized
    # weights. Reuse serving's exact identity law, including activation casts.
    attention_mode=interface.get('native_attention_precision','ieee')
    proxy = SimpleNamespace(port=SimpleNamespace(embedding_weight=state['backbone_state'][embedding_keys[0]].to(dtype),
        native_attention_precision=attention_mode),
        native_attention_precision=attention_mode,
        native_ordinary_precision=config.get('native_ordinary_precision'),
        _native_norm_parameter_dtypes=tuple(value.dtype for name,value in state['backbone_state'].items()
            if name.startswith('model.layers.') and name.endswith('norm.weight') and value.ndim == 1),
        input_rows=interface['input_rows'], adapter=SimpleNamespace(state_dict=lambda: interface['adapter']),
        native_linear_precision=interface.get('native_linear_precision', 'ieee'),
        token_override_ids=tuple(interface.get('token_override_ids', ())),
        tied=interface.get('output_rows') is None,
        extra_input_rows=interface.get('extra_input_rows'), extra_output_rows=interface.get('extra_output_rows'))
    identity = native_identity(proxy, config, checkpoint_sha256=sha256_file_hex_cached(checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    roles = provision_role_tokenizer(tokenizer)
    if (tuple(roles) != tuple(config['marker_ids'])
            or tokenizer_fingerprint(tokenizer) != identity['tokenizer_fingerprint']):
        raise ValueError('raw native RL tokenizer/role entries differ from the actual trained checkpoint')
    tokenizer_out = Path(tokenizer_out)
    tokenizer_out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_out)
    return {'native_identity': identity, 'native_participant': None, 'semantic_adapter_sha256': None,
            'tokenizer': str(tokenizer_out.resolve()), 'checkpoint': str(checkpoint)}
