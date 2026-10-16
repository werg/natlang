"""One loader/checkpoint contract for native-state training and inference."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
from types import SimpleNamespace

import torch

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from .shared_state import SharedStateModel, provision_role_tokens

CHECKPOINT_SCHEMA = 'natlang.neuralese-native-checkpoint/1'
SHARED_CHECKPOINT_SCHEMA = 'natlang.neuralese-native-shared-checkpoint/1'
COMPOSITE_CHECKPOINT_SCHEMA = 'natlang.neuralese-native-composite-checkpoint/1'


def reasoning_token_ids(tokenizer):
    """Use only existing exact vocabulary entries; never add or resize tokens."""
    vocab = tokenizer.get_vocab()
    found = tuple(vocab[token] for token in ('<think>', '</think>') if token in vocab)
    if found and (len(found) != 2 or len(set(found)) != 2):
        raise ValueError('reasoning boundaries require two distinct existing vocabulary IDs')
    return found


def upgrade_reasoning_tokens(model, tokenizer, config, *, dtype=torch.float32, sparse_row_dtype=torch.float32):
    """Explicit output-preserving sparse-row addition to an existing native parent."""
    ids = reasoning_token_ids(tokenizer)
    promoted = {}
    promotion_names = ('input_rows',) + (() if model.tied else ('output_rows',))
    if model.token_override_ids:
        promotion_names += ('extra_input_rows',) + (() if model.tied else ('extra_output_rows',))
    for name in promotion_names:
        parameter = getattr(model, name)
        desired = dtype if name.startswith('extra_') else sparse_row_dtype
        if desired is not None and parameter.dtype != desired:
            if desired != torch.float32 or parameter.dtype not in (torch.bfloat16, torch.float16):
                raise ValueError('explicit sparse role precision repair permits only exact FP32 promotion')
            old_dtype = parameter.dtype
            with torch.no_grad():parameter.data = parameter.detach().to(dtype=torch.float32)
            promoted[name] = {'from':str(old_dtype), 'to':'torch.float32'}
    if promoted:
        config['sparse_row_dtype'] = str(model.input_rows.dtype)
        if model.token_override_ids:config['token_override_dtype'] = str(model.extra_input_rows.dtype)
    changed = model.add_token_overrides(ids, dtype=dtype)
    if changed:
        config['token_override_ids'] = list(ids)
        config['token_override_dtype'] = str(model.extra_input_rows.dtype)
    if changed or promoted:config['sparse_row_dtype'] = str(model.input_rows.dtype)
    event = {'schema':'natlang.native-token-row-upgrade/1','added':changed,'token_ids':list(ids),
             'initialization':'exact-pinned-base-rows','existing_parameters':'unchanged'}
    if promoted:
        event['promoted_parameters'] = promoted
        event['existing_parameters'] = 'exact-numeric-values; declared sparse-role dtype promotion only'
    if changed or promoted:model._native_token_row_upgrade = event
    return event


def configure_native_precision():
    """Shared process policy: stock FP32 torch operations remain IEEE.

    NGC enables TF32 by default, which changed real LFM graph/reference CE and
    gradients on H100. Faster native GEMM modes are selected per model via
    native_linear_precision, never by letting the last model loader change a
    different participant's math. BF16 source tensor cores retain their dtype.
    """
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {'float32_matmul': 'highest', 'cuda_matmul_tf32': False, 'cudnn_tf32': False}


def native_identity(model, config, *, checkpoint_sha256):
    """Fingerprint the actual restored embedding path once per served checkpoint."""
    from ..common.jsonio import canonical_json_bytes
    digest = hashlib.sha256(b'natlang.native-embedding-path/1\0')
    mode = config.get('native_linear_precision', 'ieee')
    if mode != getattr(model, 'native_linear_precision', 'ieee'):
        raise ValueError('native identity arithmetic differs from the actual model')
    for key, actual in (('sparse_row_dtype', str(model.input_rows.dtype)),
                        ('token_override_ids', list(model.token_override_ids))):
        if key in config and config[key] != actual:
            raise ValueError('native identity sparse row configuration differs: ' + key)
    if model.token_override_ids and 'token_override_dtype' in config and config['token_override_dtype'] != str(model.extra_input_rows.dtype):
        raise ValueError('native identity sparse token dtype differs from the actual model')
    if mode != 'ieee':
        digest.update(canonical_json_bytes({'native_linear_precision': mode}))
    tensors = [('base_embedding', model.port.embedding_weight), ('role_input_rows', model.input_rows),
               *[('adapter.' + name, value) for name, value in model.adapter.state_dict().items()]]
    if model.token_override_ids:
        digest.update(canonical_json_bytes({'token_override_ids': model.token_override_ids}))
        tensors.append(('extra_input_rows', model.extra_input_rows))
        if not model.tied:tensors.append(('extra_output_rows', model.extra_output_rows))
    for name, value in tensors:
        tensor = value.detach().contiguous().cpu()
        digest.update(canonical_json_bytes({'name': name, 'dtype': str(tensor.dtype), 'shape': list(tensor.shape)}))
        digest.update(b'\0')
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return {'checkpoint_sha256': checkpoint_sha256,
            'tokenizer_fingerprint': config['tokenizer_fingerprint'],
            'embedding_weight_sha256': digest.hexdigest(), 'architecture': config['architecture']}


def _input_inventory(base, *, include_weights):
    """Shared exact local HF input inventory, optionally including weight shards."""
    root = Path(base).resolve(strict=True)
    names = {'config.json', 'generation_config.json', 'model.safetensors.index.json',
             'tokenizer.json', 'tokenizer.model', 'spiece.model', 'tokenizer_config.json',
             'special_tokens_map.json', 'chat_template.jinja', 'vocab.json', 'merges.txt',
             'added_tokens.json'}
    files = {str(path.relative_to(root)): sha256_file_hex_cached(path)
             for path in sorted(root.rglob('*'))
             if path.is_file() and (include_weights and path.suffix == '.safetensors' or path.name in names)}
    if 'config.json' not in files or include_weights and not any(name.endswith('.safetensors') for name in files):
        raise ValueError('native model needs a local pinned configuration and safetensors weights')
    return files


def tokenizer_inventory(base):
    """Pin tokenizer/configuration inputs without reading model weight shards."""
    return _input_inventory(base, include_weights=False)


def base_inventory(base):
    """Pin actual model/tokenizer inputs, reusing the shared verified file cache."""
    return _input_inventory(base, include_weights=True)

def load_native_model(base, *, rank=8, device='cpu', dtype=torch.bfloat16, inventory=None, native_linear_precision="ieee", token_override_ids=None, token_override_dtype=torch.float32, sparse_row_dtype=torch.float32):
    """Return (SharedStateModel, tokenizer, RoleTokens, model_config).

    No PortHeads or projected communication module is created. Provision the
    four actual role tokens before constructing the shared backbone adapter;
    no legacy marker names or reserved rows are needed.
    """
    precision = configure_native_precision()
    base = Path(base).resolve(strict=True)
    actual = base_inventory(base)
    if inventory is not None and actual != inventory:
        raise ValueError('native checkpoint base inventory differs from the pinned model inputs')
    architecture = json.loads((base / 'config.json').read_text())['model_type']
    if architecture in {'mellum', 'maple'}:
        from ..maple.student import load_student
        from ..maple.maple_port import MaplePortBackbone
        if dtype != torch.bfloat16:
            raise ValueError('native Mellum/Maple loader requires its declared BF16 source precision')
        hf, tokenizer = load_student(str(base), device=device, precision='bf16')
        port_type = MaplePortBackbone
    elif architecture in {'lfm2', 'lfm2_moe'}:
        from .lfm2_port import load_backbone, PortBackbone
        hf, tokenizer = load_backbone(str(base), dtype=dtype, device=device)
        port_type = PortBackbone
    elif architecture in {'qwen3', 'qwen2'}:
        from .hf_port import load_qwen_backbone, QwenPortBackbone
        hf, tokenizer = load_qwen_backbone(str(base), dtype=dtype, device=device)
        port_type = QwenPortBackbone
    else:
        raise ValueError(f'unsupported native-state backbone architecture {architecture!r}')
    roles = provision_role_tokens(SimpleNamespace(hf=hf, config=hf.config,
                                  embedding_weight=hf.get_input_embeddings().weight), tokenizer)
    from .lfm2_port import ControlTokens
    port = port_type(hf, ControlTokens(roles.produce_open, roles.produce_close), markers=False)
    if token_override_ids is None:
        token_override_ids = reasoning_token_ids(tokenizer)
    model = SharedStateModel(port, marker_ids=roles.ids, rank=rank, token_override_ids=token_override_ids,
                             token_override_dtype=token_override_dtype, sparse_row_dtype=sparse_row_dtype)
    from .native_precision import set_native_linear_precision
    set_native_linear_precision(model, native_linear_precision)
    from ..data.text_corpus import tokenizer_fingerprint
    config = {'base': str(base), 'base_inventory': actual,
              'base_inventory_sha256': canonical_json_sha256_hex(actual),
              'architecture': str(port.config.model_type), 'source_architecture': architecture,
              'rank': rank, 'marker_ids': list(roles.ids), 'precision': precision,
              'native_linear_precision': native_linear_precision,
              'tokenizer_fingerprint': tokenizer_fingerprint(tokenizer), 'dtype': str(dtype)}
    config['sparse_row_dtype'] = str(model.input_rows.dtype)
    if model.token_override_ids:
        config['token_override_ids'] = list(model.token_override_ids)
        config['token_override_dtype'] = str(model.extra_input_rows.dtype)
    return model, tokenizer, roles, config


def model_state(model):
    """State includes native backbone and role/embedding adapter, never old heads."""
    from .shared_state import channel_checkpoint
    return {'backbone_state': model.port.hf.state_dict(), 'interface_state': channel_checkpoint(model)}


def restore_model_state(model, state):
    model.port.hf.load_state_dict(state['backbone_state'], strict=True)
    from .shared_state import restore_channel_checkpoint
    restore_channel_checkpoint(model, state['interface_state'])


def _bind_backbone_revision(model, checkpoint_sha256):
    """Remember the exact immutable parent whose backbone is currently loaded."""
    model._native_parameter_binding = {'checkpoint_sha256': checkpoint_sha256,
        'parameters': tuple((name,id(parameter),parameter._version) for name,parameter in model.named_parameters())}
    model._native_backbone_binding = {
        'checkpoint_sha256': checkpoint_sha256, 'token_override_ids': model.token_override_ids,
        'sparse_row_dtype': str(model.input_rows.dtype),
        'token_override_dtype': str(model.extra_input_rows.dtype) if model.token_override_ids else None,
        'parameters': tuple((name, id(parameter), parameter._version)
                            for name, parameter in model.port.hf.named_parameters()),
    }


def load_native_checkpoint(path, *, base=None, device='cpu', dtype=None, participant=None, allow_weights_only=False):
    """Return (model, tokenizer, roles, model_config, full_checkpoint_state)."""
    from pathlib import Path
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        prefix = stream.read(1)
    if prefix == b'{':
        manifest = json.loads(path.read_text())
        if manifest.get('schema') == COMPOSITE_CHECKPOINT_SCHEMA:
            from .native_composite import load_native_composite_checkpoint
            return load_native_composite_checkpoint(path, base=base, device=device, dtype=dtype)
    from ..train.checkpoint_policy import CheckpointPolicy
    state = CheckpointPolicy.load(path, shared_participant=participant)
    if state.get('optimizer_pruned') and not allow_weights_only:
        from ..train.checkpoint_offload import require_optimizer
        require_optimizer(state, path)
    if state.get('schema') == SHARED_CHECKPOINT_SCHEMA:
        if participant not in state['participants']:
            raise ValueError('shared native checkpoint requires an explicit existing participant')
        entry = state['participants'][participant]
        shared_training = {'optimizer': entry.get('optimizer'), 'fp8': entry.get('fp8'),
            'sft_progress': entry.get('sft_progress', {}), 'global_step': entry.get('global_step', state['training']['step']),
            'rng': state['training'].get('rng'), 'step': state['training']['step'],
            'options': {**(state['training'].get('config_history') or [{}])[-1].get('config', {}),
                        **entry.get('training_options', {})}}
        state = {'schema': CHECKPOINT_SCHEMA, 'model_config': entry['model_config'],
                 **entry['model_state'], 'shared_participant': participant,
                 'shared_step': state['training']['step'], 'shared_training': shared_training}
    elif participant is not None:
        raise ValueError('participant selection requires a shared native checkpoint')
    if state.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError('checkpoint is not a native-state model; legacy projection resumes are unsupported')
    saved = state['model_config']
    requested_dtype = dtype or {'torch.bfloat16': torch.bfloat16, 'torch.float32': torch.float32,
                                'torch.float16': torch.float16}[saved['dtype']]
    model, tokenizer, roles, config = load_native_model(base or saved['base'], rank=saved['rank'],
        device=device, dtype=requested_dtype, inventory=saved['base_inventory'],
        native_linear_precision=saved.get('native_linear_precision', 'ieee'),
        token_override_ids=saved.get('token_override_ids', ()),
        token_override_dtype={'torch.float32':torch.float32,'torch.bfloat16':torch.bfloat16,
            'torch.float16':torch.float16}[saved.get('token_override_dtype', saved['dtype'])],
        sparse_row_dtype={'torch.float32':torch.float32,'torch.bfloat16':torch.bfloat16,
            'torch.float16':torch.float16}[saved.get('sparse_row_dtype', saved['dtype'])])
    if 'token_override_dtype' not in saved and str(requested_dtype) == saved['dtype']:
        config.pop('token_override_dtype', None)  # exact pre-master-row configuration contract
    if 'sparse_row_dtype' not in saved and str(requested_dtype) == saved['dtype']:
        config.pop('sparse_row_dtype', None)
    for key in ('architecture', 'rank', 'marker_ids', 'tokenizer_fingerprint', 'base_inventory_sha256', 'precision'):
        if config[key] != saved.get(key):
            raise ValueError(f'native checkpoint {key} changed during restoration')
    if config.get('token_override_ids', []) != saved.get('token_override_ids', []):
        raise ValueError('native checkpoint sparse token overrides changed during restoration')
    restore_model_state(model, state)
    _bind_backbone_revision(model, sha256_file_hex_cached(path))
    return model, tokenizer, roles, config, state
