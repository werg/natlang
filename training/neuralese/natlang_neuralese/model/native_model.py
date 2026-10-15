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


def configure_native_precision():
    """Keep FP32 operations IEEE on training and serving destination GPUs.

    NGC enables TF32 by default, which changed real LFM graph/reference CE and
    gradients on H100. This contract affects FP32 operations; native BF16
    tensor-core computation retains its declared dtype.
    """
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {'float32_matmul': 'highest', 'cuda_matmul_tf32': False, 'cudnn_tf32': False}


def native_identity(model, config, *, checkpoint_sha256):
    """Fingerprint the actual restored embedding path once per served checkpoint."""
    from ..common.jsonio import canonical_json_bytes
    digest = hashlib.sha256(b'natlang.native-embedding-path/1\0')
    tensors = [('base_embedding', model.port.embedding_weight), ('role_input_rows', model.input_rows),
               *[('adapter.' + name, value) for name, value in model.adapter.state_dict().items()]]
    for name, value in tensors:
        tensor = value.detach().contiguous().cpu()
        digest.update(canonical_json_bytes({'name': name, 'dtype': str(tensor.dtype), 'shape': list(tensor.shape)}))
        digest.update(b'\0')
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return {'checkpoint_sha256': checkpoint_sha256,
            'tokenizer_fingerprint': config['tokenizer_fingerprint'],
            'embedding_weight_sha256': digest.hexdigest(), 'architecture': config['architecture']}


def base_inventory(base):
    """Pin actual model/tokenizer inputs, reusing the shared verified file cache."""
    root = Path(base).resolve(strict=True)
    names = {'config.json', 'generation_config.json', 'model.safetensors.index.json',
             'tokenizer.json', 'tokenizer.model', 'spiece.model', 'tokenizer_config.json',
             'special_tokens_map.json', 'chat_template.jinja', 'vocab.json', 'merges.txt',
             'added_tokens.json'}
    files = {str(path.relative_to(root)): sha256_file_hex_cached(path)
             for path in sorted(root.rglob('*'))
             if path.is_file() and (path.suffix == '.safetensors' or path.name in names)}
    if 'config.json' not in files or not any(name.endswith('.safetensors') for name in files):
        raise ValueError('native model needs a local pinned configuration and safetensors weights')
    return files


def load_native_model(base, *, rank=8, device='cpu', dtype=torch.bfloat16, inventory=None):
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
    model = SharedStateModel(port, marker_ids=roles.ids, rank=rank)
    from ..data.text_corpus import tokenizer_fingerprint
    config = {'base': str(base), 'base_inventory': actual,
              'base_inventory_sha256': canonical_json_sha256_hex(actual),
              'architecture': str(port.config.model_type), 'source_architecture': architecture,
              'rank': rank, 'marker_ids': list(roles.ids), 'precision': precision,
              'tokenizer_fingerprint': tokenizer_fingerprint(tokenizer), 'dtype': str(dtype)}
    return model, tokenizer, roles, config


def model_state(model):
    """State includes native backbone and role/embedding adapter, never old heads."""
    from .shared_state import channel_checkpoint
    return {'backbone_state': model.port.hf.state_dict(), 'interface_state': channel_checkpoint(model)}


def restore_model_state(model, state):
    model.port.hf.load_state_dict(state['backbone_state'], strict=True)
    from .shared_state import restore_channel_checkpoint
    restore_channel_checkpoint(model, state['interface_state'])


def load_native_checkpoint(path, *, base=None, device='cpu', dtype=None, participant=None):
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
    state = CheckpointPolicy.load(path)
    if state.get('schema') == SHARED_CHECKPOINT_SCHEMA:
        if participant not in state['participants']:
            raise ValueError('shared native checkpoint requires an explicit existing participant')
        entry = state['participants'][participant]
        state = {'schema': CHECKPOINT_SCHEMA, 'model_config': entry['model_config'],
                 **entry['model_state'], 'shared_participant': participant,
                 'shared_step': state['training']['step']}
    elif participant is not None:
        raise ValueError('participant selection requires a shared native checkpoint')
    if state.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError('checkpoint is not a native-state model; legacy projection resumes are unsupported')
    saved = state['model_config']
    requested_dtype = dtype or {'torch.bfloat16': torch.bfloat16, 'torch.float32': torch.float32,
                                'torch.float16': torch.float16}[saved['dtype']]
    model, tokenizer, roles, config = load_native_model(base or saved['base'], rank=saved['rank'],
        device=device, dtype=requested_dtype, inventory=saved['base_inventory'])
    for key in ('architecture', 'rank', 'marker_ids', 'tokenizer_fingerprint', 'base_inventory_sha256', 'precision'):
        if config[key] != saved.get(key):
            raise ValueError(f'native checkpoint {key} changed during restoration')
    restore_model_state(model, state)
    return model, tokenizer, roles, config, state
