"""Checkpoint-bound shared semantic state encoding and target-native decoding."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from .semantic_vae import SemanticStateVAE, alignment_weights
from .shared_state import Invocation, compile_graph, capture_block
from .native_model import SHARED_CHECKPOINT_SCHEMA


class SemanticChannel:
    """Use learned state deltas only for exactly bound source/target models."""
    def __init__(self, vae, bindings, tokenizers, *, adapter_sha256):
        if set(bindings) != set(vae.layouts) or set(tokenizers) != set(bindings):
            raise ValueError('semantic channel layouts, native identities and tokenizers differ')
        self.vae, self.bindings, self.tokenizers = vae, dict(bindings), dict(tokenizers)
        self.adapter_sha256 = adapter_sha256

    @classmethod
    def load(cls, receipt_path, *, device='cpu'):
        from ..train.checkpoint_policy import CheckpointPolicy
        from ..data.text_corpus import tokenizer_fingerprint
        from transformers import AutoTokenizer
        path = Path(receipt_path).resolve(strict=True)
        receipt = json.loads(path.read_text())
        if receipt.get('schema') != 'natlang.native-semantic-adapter/1':
            raise ValueError('unsupported shared semantic adapter export')
        checkpoint = Path(receipt['checkpoint']['path']).resolve(strict=True)
        if sha256_file_hex_cached(checkpoint) != receipt['checkpoint']['sha256']:
            raise ValueError('semantic adapter checkpoint bytes changed')
        state = CheckpointPolicy.load(checkpoint)
        if state.get('schema') != SHARED_CHECKPOINT_SCHEMA:
            raise ValueError('semantic adapter requires its complete shared-training checkpoint')
        spec = state['vae_specification']
        if canonical_json_sha256_hex(spec) != receipt['specification_sha256']:
            raise ValueError('semantic adapter layout differs from its exact trained checkpoint')
        vae = SemanticStateVAE.from_specification(spec).to(device)
        vae.load_state_dict(state['vae_state'], strict=True)
        vae.eval().requires_grad_(False)
        tokenizers, bindings = {}, {}
        for name, model in receipt['models'].items():
            identity = model['native_identity']
            if identity['checkpoint_sha256'] != receipt['checkpoint']['sha256']:
                raise ValueError('semantic adapter model identity differs from its trained checkpoint')
            binding = receipt['tokenizers'][name]
            root = (path.parent/binding['path']).resolve(strict=True)
            if path.parent not in root.parents:
                raise ValueError('semantic tokenizer must remain within its adapter export')
            for filename, digest in binding['files'].items():
                artifact = (root/filename).resolve(strict=True)
                if root not in artifact.parents or sha256_file_hex_cached(artifact) != digest:
                    raise ValueError('semantic adapter tokenizer bytes changed')
            tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
            if tokenizer_fingerprint(tokenizer) != identity['tokenizer_fingerprint']:
                raise ValueError('semantic adapter source tokenizer differs from its exact model binding')
            tokenizers[name], bindings[name] = tokenizer, identity
        return cls(vae, bindings, tokenizers, adapter_sha256=sha256_file_hex_cached(path))

    def model_name(self, block):
        identity = {key: getattr(block, key) for key in ('checkpoint_sha256', 'tokenizer_fingerprint',
                                                       'embedding_weight_sha256', 'architecture')}
        names = [name for name, binding in self.bindings.items() if binding == identity]
        if len(names) != 1:
            raise ValueError('semantic source is not an exactly bound unique trained participant')
        return names[0]

    def require_target(self, name, identity):
        if name not in self.bindings or self.bindings[name] != identity:
            raise ValueError('semantic adapter is not bound to these actual target weights/tokenizer/embeddings')

    def encode(self, block):
        name = self.model_name(block)
        block.require_native(**self.bindings[name])
        # Serving uses the posterior mean as the deterministic dialect token.
        # Training uses actual reparameterized posterior samples and KL.
        mean, _ = self.vae.encode(name, block)
        return mean, {'source_model': name, 'adapter_sha256': self.adapter_sha256,
                      'posterior': 'mean', 'source_native_identity': self.bindings[name]}

    def message(self, block):
        from .semantic_message import SemanticStateMessage
        source_name = self.model_name(block)
        source_tokenizer = self.tokenizers[source_name]
        source_tokens = source_tokenizer(block.text, add_special_tokens=False, return_offsets_mapping=True)
        if list(source_tokens['input_ids']) != block.token_ids.detach().cpu().tolist():
            raise ValueError('semantic source payload tokens need an explicit retained-offset alignment repair')
        device = next(self.vae.parameters()).device
        from .state_block import NativeStateBlock
        local = NativeStateBlock.from_tensors(block.metadata(), {name: value.to(device)
                                                  for name, value in block.tensors().items()})
        code, _ = self.encode(local)
        return SemanticStateMessage(text=block.text, token_ids=local.token_ids,
            positions=local.positions, embeddings=local.embeddings, layers=(),
            **{key: getattr(block, key) for key in ('architecture', 'tokenizer_fingerprint',
                'checkpoint_sha256', 'embedding_weight_sha256')},
            code=code, offsets=[list(pair) for pair in source_tokens['offset_mapping']],
            source_model=source_name, adapter_sha256=self.adapter_sha256).validate()

    def adapt(self, block, *, target_name, target_model, target_tokenizer, target_identity,
              allow_nearest=False):
        from .semantic_message import SemanticStateMessage
        self.require_target(target_name, target_identity)
        message = block if isinstance(block, SemanticStateMessage) else self.message(block)
        message.validate()
        source_name = self.model_name(message)
        if message.adapter_sha256 != self.adapter_sha256 or message.source_model != source_name:
            raise ValueError('semantic message requires the exact trained adapter and source model')
        source_tokens = self.tokenizers[source_name](message.text, add_special_tokens=False, return_offsets_mapping=True)
        if (source_tokens['input_ids'] != message.token_ids.detach().cpu().tolist()
                or [list(pair) for pair in source_tokens['offset_mapping']] != message.offsets):
            raise ValueError('semantic message source token IDs/offsets differ from its bound tokenizer')
        if message.code.shape[1] != self.vae.latent_width:
            raise ValueError('semantic message code width differs from its trained adapter')
        target_tokens = target_tokenizer(block.text, add_special_tokens=False, return_offsets_mapping=True)
        ids = tuple(target_tokens['input_ids'])
        op, close = target_model.marker_ids[:2]
        plan = compile_graph([Invocation('__semantic_payload', (op, *ids, close), producing=((1, 1+len(ids)),))])
        baseline = capture_block(target_model(plan), '__semantic_payload', 1, 1+len(ids),
                                 text=block.text, **target_identity)
        device = next(self.vae.parameters()).device
        code = message.code.to(device)
        receipt = {'source_model': source_name, 'adapter_sha256': self.adapter_sha256,
                   'posterior': 'mean', 'source_native_identity': self.bindings[source_name]}
        alignment, matching = alignment_weights(message.offsets, target_tokens['offset_mapping'],
            device=device, allow_nearest=allow_nearest)
        if baseline.embeddings.device != device:
            raise ValueError('semantic adapter and target model must use the declared serving device')
        decoded = self.vae.decode(target_name, code, baseline, alignment=alignment)
        return decoded, {**receipt, 'target_model': target_name, 'alignment': matching,
                         'mode': 'learned-semantic-state-deltas', 'target_native_identity': target_identity}
