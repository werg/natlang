"""Native serving front whose ordinary-token generation runs on natlang's SGLang fork (training/neuralese/sglang-fork.json).

The front is the native `Engine`: it renders and tokenizes requests, inserts privileged guidance, parses replies and
builds the exact `x_natlang_rl_capture` the learner replays. Only generation is delegated: the rendered prompt ids go
to SGLang's `/generate`, which samples from the checkpoint's HF export (`export/native_hf.py`, sparse rows merged) and
returns per token the exact post-temperature law it drew from, compactly (`custom_params.natlang_top_k`: one base64
record per token, `<f4 sampled logp | <i4 ids[k] | <f2 logps[k]`; the fork's sampler, `natlang_capture_top_k`).

SGLang serves turns over rendered text and privileged text guidance, sampling or greedy, RL capture with behaviour
top-k, and produce spans: the fork applies the producer adapter inside a span (the export's producing embedding table),
the front follows the native marker protocol token by token (`_sglang_token`, the native engine's decode step) and,
when a span closes, has SGLang recompute the span with a capture request (`custom_params.natlang_capture`: pre-RoPE
keys, values and conv inputs of the span positions) that becomes the same `NativeStateBlock` the native engine stores.
Native state crosses between the front and the fork only as files of the store's own format: the fork writes a
capture as a native-state file (`serve.store.write_native_state_file`, the front's metadata), the store adopts that
file by renaming it in after verifying it from the file, and an import hands the fork the stored file's path. The front
never decodes a state's tensors.
No marker is banned when the export's `serving.banned_token_ids` is empty (the native law); protocol violations are
the native engine's errors and recoverable failures. Streamed requests stream text and native-state deltas. Every
other request (imported native state, templated writes/decodes, guided generation, forced teacher replay, native
adapters) runs on the native engine of the same checkpoint, exactly as `serve_native_checkpoint.py` serves it without
`--engine sglang`; that model is loaded only when such a turn first arrives. `/metrics` counts both paths.
"""
from __future__ import annotations

import base64
import hashlib
import itertools
import json
import os
import threading
import time
import uuid
import urllib.error
import urllib.request
from concurrent.futures import Future

import numpy as np

from .chat import RequestError
from .engine import TEXT_IN_CONTEXT_MODE, Engine, Sequence, _insert_privileged, derive_seed, role_bans
from ..common.sampling import TOP_K_MASS_FIELD
from .native_protocol import FAILURE_CODES, capture_enabled, failure_descriptor

TOP_K_KEY = 'natlang_top_k'
TOP_K_MASS_KEY = 'natlang_top_k_mass'
CAPTURE_KEY = 'natlang_capture'
IMPORT_KEY = 'natlang_import'
ADAPTER_KEY = 'natlang_adapter'


def adapter_cache_key(body):
    """``body`` with SGLang's radix-cache ``extra_key`` naming its adapter stack. SGLang matches a cached prefix (KV and
    the conv/recurrent states) on the token ids and ``extra_key`` only, and the fork's slot adapters
    (``custom_params.natlang_adapter``) are not in it: a prompt state computed under one stack (or none) was reused by a
    request with another, so adapted and base requests on one prompt read each other's prefix (T3, 2026-11-24:
    base turns of a program prompt 0.70 nats from the native engine, adapted 2.6). Each stack gets its own prefix
    namespace; base requests keep the empty one. ``NATLANG_SGLANG_ADAPTER_CACHE_KEY=0`` opts out."""
    adapter = ((body.get('sampling_params') or {}).get('custom_params') or {}).get(ADAPTER_KEY)
    if not adapter or os.environ.get('NATLANG_SGLANG_ADAPTER_CACHE_KEY', '1') == '0':
        return body
    key = f"natlang-adapter:{adapter['id']}"
    extra = body.get('extra_key')
    return {**body, 'extra_key': f'{extra}|{key}' if extra else key}
TRACK_INPUT_END_KEY = 'natlang_track_input_end'
# The reply tokens a request's input already holds, which its grammar takes first (fork NATLANG_GRAMMAR_PREFIX).
GRAMMAR_PREFIX_KEY = 'natlang_grammar_prefix'
BAN_BIAS = -1e30


def decode_record(text, k):
    """(sampled logp, ids[k], logps[k]) of one fork record carrying K >= k entries (and optionally a trailing tail mass)."""
    raw = base64.b64decode(text)
    width = (len(raw) - 4 - (len(raw) - 4) % 6) // 6
    if (len(raw) - 4) % 6 not in (0, 4) or width < k:
        raise RequestError('sglang-actor-capture', f'top-k record has {width} entries, need {k}')
    logp = float(np.frombuffer(raw[:4], dtype='<f4')[0])
    ids = np.frombuffer(raw[4:4 + 4 * width], dtype='<i4')[:k]
    logps = np.frombuffer(raw[4 + 4 * width:4 + 6 * width], dtype='<f2')[:k]
    return logp, ids, logps


def record_tail_mass(text, k):
    """Exact FP32 probability outside the first k entries of one fork record, or None without the fork's tail field
    (`natlang_top_k_mass`). The fork's tail follows its K recorded entries; entries k..K are added back from their
    float16 values (only batches mixing different k carry any)."""
    raw = base64.b64decode(text)
    if (len(raw) - 4) % 6 != 4:
        return None
    width = (len(raw) - 8) // 6
    tail = float(np.frombuffer(raw[-4:], dtype='<f4')[0])
    if width > k:
        tail += float(np.exp(np.frombuffer(raw[4 + 4 * width:4 + 6 * width], dtype='<f2')[k:].astype(np.float64)).sum())
    return tail


EMBEDDING = 'model.embed_tokens.weight'
PRODUCER_TABLE = 'model.natlang_producer_embed'   # export.native_hf: the producing input embedding of every token
TENSOR_IDENTITY_SCHEMA = b'natlang.hf-tensor-identity/1\0'
IDENTITY_STAGING_BYTES = 64 << 20


def tensor_identity(tensors):
    """SHA-256 over ``{name: tensor}`` in name order: each name, shape and the tensor's BF16 bytes (floating tensors are
    compared as the BF16 the exports hold; a BF16 value held in FP32 casts back exactly). One scheme for every party
    that attests served weights: the trainer computes it from the export it wrote, the actor front from the native
    model it reloaded. CUDA tensors are read through one reused pinned buffer (a copy into fresh pageable host memory
    runs at 0.05-0.1 GB/s on the GB10)."""
    import torch
    digest = hashlib.sha256(TENSOR_IDENTITY_SCHEMA)
    staging = None
    for name in sorted(tensors):
        value = tensors[name].detach()
        if value.is_floating_point() and value.dtype != torch.bfloat16:
            value = value.to(torch.bfloat16)
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode() + b'\0')
        flat = value.contiguous().reshape(-1).view(torch.uint8)
        if flat.device.type == 'cpu':
            digest.update(memoryview(flat.numpy()))
            continue
        if staging is None:
            staging = torch.empty(IDENTITY_STAGING_BYTES, dtype=torch.uint8, pin_memory=True)
        for begin in range(0, flat.numel(), staging.numel()):
            piece = flat[begin:begin + staging.numel()]
            staging[:piece.numel()].copy_(piece)
            digest.update(memoryview(staging[:piece.numel()].numpy()))
    return digest.hexdigest()


def native_weights_identity(model):
    """``tensor_identity`` of what a native model serves: every backbone parameter (HF names, a tied head once) and its
    interface rows (``interface.<attr>``), the rows the marker and override tokens actually read."""
    tensors = dict(model.port.hf.named_parameters())
    tied = getattr(model, 'tied', True)
    markers, overrides = tuple(getattr(model, 'marker_ids', ())), tuple(getattr(model, 'token_override_ids', ()) or ())
    for attr, ids, used in (('input_rows', markers, True), ('extra_input_rows', overrides, True),
                            ('output_rows', markers, not tied), ('extra_output_rows', overrides, not tied)):
        value = getattr(model, attr, None)
        if ids and used:
            if value is None:
                raise ValueError(f'native model has no {attr} for its interface ids')
            tensors[f'interface.{attr}'] = value
    return tensor_identity(tensors)


def export_tensors_identity(tensors, marker_ids=(), override_ids=(), tied=True):
    """``native_weights_identity`` of the native model an HF export reloads (``_apply_export_weights_to_native``),
    computed from the export's own tensors ``{name: tensor}``: the producer table left out, the interface rows read from
    the export's embedding (head) at their ids. The trainer's expected value for the native route."""
    import torch
    named = {name: value for name, value in tensors.items() if name != PRODUCER_TABLE
             and not (tied and name == 'lm_head.weight')}
    embedding = tensors[EMBEDDING]
    head = embedding if tied else tensors['lm_head.weight']
    for attr, ids, table in (('input_rows', marker_ids, embedding), ('extra_input_rows', override_ids, embedding),
                             ('output_rows', marker_ids if not tied else (), head),
                             ('extra_output_rows', override_ids if not tied else (), head)):
        if ids:
            named[f'interface.{attr}'] = table[torch.tensor(list(ids), dtype=torch.long)]
    return tensor_identity(named)


# A socket hang guard on backend requests, not a budget: a request waits in a paused engine while a colocated learner
# holds the GPU memory (grpo.py --actor-release, up to ~40 min per step), so the guard sits above that wait and at the
# trainer's episode safety net (grpo.py --case-seconds 7200).
SGLANG_REQUEST_SECONDS = 7200


class SglangEngine(Engine):
    """`Engine` whose sequences are generated by an SGLang server at `sglang_url`.

    After `release_native` the native model is not resident: `shared_model`/`backbone` reload it (and check its exact
    native identity) the first time a native turn or endpoint needs it."""

    sglang_url: str = ''

    @property
    def shared_model(self):
        if self.__dict__.get('_native_model') is None and self.__dict__.get('_native_loader') is not None:
            self._ensure_native()
        return self.__dict__.get('_native_model')

    @shared_model.setter
    def shared_model(self, value):
        self.__dict__['_native_model'] = value

    @property
    def native_training_state(self):
        # The checkpoint's training state (optimizer, EMA, SFT frames: ~12 GB for LFM3-1.7B, mmapped from the
        # checkpoint) is only read by native endpoints; after `release_native` it is reloaded with the native model.
        if self.__dict__.get('_native_training_state') is None and self.__dict__.get('_native_loader') is not None:
            self._ensure_native()
        return self.__dict__.get('_native_training_state')

    @native_training_state.setter
    def native_training_state(self, value):
        self.__dict__['_native_training_state'] = value

    @property
    def backbone(self):
        if self.__dict__.get('_native_backbone') is None and self.__dict__.get('_native_loader') is not None:
            self._ensure_native()
        return self.__dict__.get('_native_backbone')

    @backbone.setter
    def backbone(self, value):
        self.__dict__['_native_backbone'] = value

    def release_native(self, loader):
        """Drop the resident native model; `loader` holds load_native_checkpoint's arguments for a lazy reload."""
        import gc
        import torch
        self.native_dtype = str(self.__dict__['_native_backbone'].embedding_weight.dtype).replace('torch.', '')
        self.__dict__.setdefault('_native_lock', threading.RLock())
        self.__dict__['_native_loader'] = dict(loader)
        self.__dict__['_native_model'] = self.__dict__['_native_backbone'] = None
        self.__dict__['_native_training_state'] = None  # the last reference into the checkpoint mmap
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:  # hand the freed host heap of the checkpoint load back to the (unified) memory pool
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except OSError:
            pass

    def _ensure_native(self):
        """Load the native model on first need. The whole load runs under the one native lock that ``rebind_in_place``
        takes to change the rebind source, so a load always applies the newest rebind (a rebind landing meanwhile
        waits and then reloads the resident model): the native route never serves an older version than it attests."""
        with self.__dict__.setdefault('_native_lock', threading.RLock()):
            if self.__dict__.get('_native_model') is not None:
                return
            import torch
            from ..model.native_model import load_native_checkpoint, native_identity
            loader = dict(self.__dict__['_native_loader'])
            # Ordinary tensors as the eager load makes them, even when the first use is inside the scheduler's
            # inference mode (inference tensors carry no version counter, which the loader binds).
            with torch.inference_mode(False):
                model, _tokenizer, _roles, config, _state = load_native_checkpoint(
                    loader.pop('checkpoint'), allow_weights_only=True, **loader)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            # The start checkpoint is the lineage: its identity is the one exact-state splicing keys on.
            identity = native_identity(model, config, checkpoint_sha256=self.native_identity['checkpoint_sha256'])
            if identity != self.native_identity:
                raise RuntimeError('lazily reloaded native model differs from the served native identity')
            source = self.__dict__.get('_native_rebind_source')
            if source is not None:
                self._load_native_source(model, source)
            self.__dict__['_native_backbone'] = model.port
            self.__dict__['_native_model'] = model
            self.__dict__['_native_training_state'] = _state.get('training') or _state.get('shared_training') or {}
            self.native_reloads = getattr(self, 'native_reloads', 0) + 1

    def _load_native_source(self, model, source):
        """Reload ``model`` from a rebind source and record what the native route now serves, measured from the model
        itself (``native_weights_identity``). A load whose measured identity differs from the trainer's claim for that
        version fails: the native route never serves weights other than the ones it attests."""
        identity = self._apply_export_weights_to_native(model, source['path'])
        claimed = source.get('tensor_identity')
        if claimed is not None and identity != claimed:
            raise RuntimeError(f"native reload of version {source.get('version')} measured tensor identity "
                               f'{identity[:12]}, the trainer exported {str(claimed)[:12]}')
        self._native_served = {'policy_sha256': source['policy_id'], 'version': source.get('version'),
                               'tensor_identity': identity, 'source': source['path'],
                               'source_weights_sha256': source.get('measured_weights_sha256')}
        self._invalidate_weight_caches()

    def _invalidate_weight_caches(self):
        """Caches holding values computed from the old weights: effective adapted weights (the adapter replacement
        cache) and prefix KV. Adapter bases and coefficients are stored artifacts (coordinates, not weights) and stay."""
        if getattr(self, 'native_adapter_cache', None) is not None:
            self.native_adapter_cache.entries.clear()
            self.native_adapter_cache.bytes = 0
        if getattr(self, 'prefix_cache', None) is not None and hasattr(self.prefix_cache, 'clear'):
            self.prefix_cache.clear()

    def _apply_export_weights_to_native(self, model, weights_path):
        """Reload ``model``'s backbone and sparse interface rows from an HF safetensors export (or a pruned native
        checkpoint) so the native route serves the same weights as SGLang after an in-place rebind. Every native
        backbone parameter must be in the source and every source tensor must be a native parameter (the producer
        table and a tied head excepted): an unmatched name fails the reload instead of leaving a start tensor in place.
        Returns the reloaded model's measured ``native_weights_identity``."""
        import torch
        from safetensors import safe_open
        path = str(weights_path)
        if os.path.isdir(path):
            path = os.path.join(path, 'model.safetensors')
        if not os.path.isfile(path):
            raise ValueError(f'rebind native weights path does not exist: {path}')
        params = dict(model.port.hf.named_parameters())   # a tied head is the embedding (deduplicated)
        tied = getattr(model, 'tied', True)

        def check_coverage(names):
            names = set(names) - {PRODUCER_TABLE} - ({'lm_head.weight'} if tied else set())
            missing, extra = sorted(set(params) - names), sorted(names - set(params))
            if missing or extra:
                raise ValueError(f'native reload: source {path} lacks native parameters {missing[:5]} '
                                 f'({len(missing)}) and holds unknown tensors {extra[:5]} ({len(extra)})')

        def set_rows(attr, ids, table):
            target = getattr(model, attr, None)
            if ids and target is not None:
                index = torch.tensor(list(ids), dtype=torch.long)
                target.copy_(table[index].to(device=target.device, dtype=target.dtype))
        with torch.inference_mode(False), torch.no_grad():
            if path.endswith('.safetensors'):
                with safe_open(path, framework='pt', device='cpu') as handle:
                    check_coverage(handle.keys())
                    for name, param in params.items():
                        param.copy_(handle.get_tensor(name).to(device=param.device, dtype=param.dtype))
                    embedding = handle.get_tensor(EMBEDDING)
                    head = handle.get_tensor('lm_head.weight') if not tied else embedding
                # the export's embedding carries the interface rows at their ids (export.native_hf)
                markers, overrides = tuple(getattr(model, 'marker_ids', ())), tuple(getattr(model, 'token_override_ids', ()) or ())
                set_rows('input_rows', markers, embedding)
                set_rows('extra_input_rows', overrides, embedding)
                if not tied:
                    set_rows('output_rows', markers, head)
                    set_rows('extra_output_rows', overrides, head)
            else:
                from ..train.checkpoint_policy import CheckpointPolicy
                state = CheckpointPolicy.load(path)
                backbone = state['backbone_state']
                check_coverage(backbone)
                for name, param in params.items():
                    param.copy_(backbone[name].to(device=param.device, dtype=param.dtype))
                interface = state.get('interface_state') or {}
                for attr in ('input_rows', 'output_rows', 'extra_input_rows', 'extra_output_rows'):
                    target = getattr(model, attr, None)
                    if target is not None:
                        if interface.get(attr) is None:
                            raise ValueError(f'native reload: {path} has no interface {attr}')
                        target.copy_(interface[attr].to(device=target.device, dtype=target.dtype))
        return native_weights_identity(model)

    def bind_sglang(self, url, *, export_manifest, fork, adapter_slots=0, adapter_rank=32,
                    projection='native', overlap_schedule=False, kv_cache_dtype='auto', kv_layout='full',
                    weight_dtype='bfloat16', max_running=None, warm_openings=None, track_input_end=False,
                    speculative=None, head_logits='bf16', tool_grammar=False, export_dir=None):
        self.sglang_url = url.rstrip('/')
        # Tool calls decode under their schemas (common.tool_grammar) when the fork serves grammars to this token-id
        # front and continues them across requests (NATLANG_GRAMMAR_PREFIX); NATLANG_TOOL_GRAMMAR=0 opts out.
        from ..common import tool_grammar as _tool_grammar
        self.sglang_tool_grammar = bool(tool_grammar) and export_dir is not None and _tool_grammar.enabled()
        self.sglang_export_dir = export_dir
        self.__dict__.setdefault('_native_lock', threading.RLock())
        # Native adapters on SGLang (fork NatlangAdapters): `adapter_slots` resident stacks of summed rank
        # <= `adapter_rank` per module; the front assigns slots and never reassigns one while a request uses it.
        self.sglang_adapter_slots, self.sglang_adapter_rank = int(adapter_slots), int(adapter_rank)
        targets = export_manifest.get('adapter_targets') or {}
        self.native_target_shapes = {name: tuple(target['shape']) for name, target in targets.items()} or None
        self.sglang_adapter_targets = targets
        self.sglang_module_rows = {}
        for target in targets.values():
            rows = target['offset'] + target['shape'][0]
            self.sglang_module_rows[target['module']] = max(self.sglang_module_rows.get(target['module'], 0), rows)
        self._adapter_condition = threading.Condition()
        self._adapter_slot_of, self._adapter_slot_users, self._adapter_slot_key = {}, {}, {}
        self._adapter_files, self._adapter_clock = {}, 0
        roles = self.role_tokens
        markers = {int(roles.produce_open), int(roles.produce_close), int(roles.consume_open), int(roles.consume_close)}
        # the export's bans and the role bans every engine applies (`engine.role_bans`: the consume markers)
        self.sglang_banned = sorted({int(token) for token in export_manifest['serving']['banned_token_ids']}
                                    | role_bans(roles))
        if not set(self.sglang_banned) <= markers:
            raise ValueError('the HF export bans tokens other than the role markers')
        if (int(roles.produce_open) not in self.sglang_banned
                and (export_manifest.get('producer_embedding') or {}).get('input_id_offset') is None):
            raise ValueError('produce markers need the producing embedding table in the HF export')
        self.sglang_capture_dir = None
        if export_manifest['checkpoint_sha256'] != self.checkpoint_sha256:
            raise ValueError('HF export was made from another checkpoint')
        self.sglang_export = export_manifest
        self.sglang_requests = self.sglang_tokens = self.native_fallbacks = self.sglang_state_writes = 0
        # Token-in/token-out prompts (serve/append_only.py): NATLANG_APPEND_ONLY_PROMPTS=0 opts out.
        from .append_only import TokenHistory
        self.sglang_history = (TokenHistory(self.tokenizer) if os.environ.get('NATLANG_APPEND_ONLY_PROMPTS', '1') != '0'
                               else None)
        self.native_fallback_reasons = {}
        self.native_endpoint_metrics = {}
        self._sglang_lock = threading.Lock()
        # Decision openings warmed into SGLang's radix cache (`_sglang_warm_opening`): on by default,
        # NATLANG_SGLANG_WARM_OPENINGS=0 (or warm_openings=False) opts out.
        self.sglang_warm_openings = (os.environ.get('NATLANG_SGLANG_WARM_OPENINGS', '1') != '0' if warm_openings is None
                                     else bool(warm_openings))
        # The fork stores a request's conv state at its exact input end on request (NATLANG_TRACK_INPUT_END marker,
        # serve_native_checkpoint.attest_sglang_source); older forks keep the grid-aligned state.
        self.sglang_track_input_end = bool(track_input_end)
        # Bound in every SGLang-sampled capture (rl.actors ``actor_export``, kind exact: the checkpoint's own weights
        # in another engine). The learner recomputes those tokens under its own forward, guards the engine mismatch
        # per episode and corrects it with the loss's importance weights (rl.native_replay DEFAULT_ENGINE_GUARD).
        if projection not in ('native', 'vendor'):
            raise ValueError('SGLang projection policy must be native or vendor')
        # A quantized KV cache or FP8 linear weights (async-train §3 collection profile; conv & lm_head stay BF16) are
        # part of the attested actor law, so held eval and GRPO automatically refuse an FP8-weight collection actor.
        if weight_dtype not in ('bfloat16', 'fp8'):
            raise ValueError(f'SGLang weight_dtype must be bfloat16 or fp8, got {weight_dtype!r}')
        kv = '' if kv_cache_dtype in ('auto', None, export_manifest['dtype']) else f'-kv-{kv_cache_dtype}'
        w_tag = '-w-fp8' if weight_dtype == 'fp8' else ''
        self.sglang_weight_dtype = weight_dtype
        self.sglang_actor_export = {'kind': 'exact', 'format': f"sglang-hf-{export_manifest['dtype']}-{projection}-projection{kv}{w_tag}",
                                    'sha256': export_manifest['weights_sha256']}
        binding = getattr(self, 'rl_policy_binding', None)
        if binding is not None:
            binding['serving_engine'] = {'engine': 'sglang', 'fork': dict(fork),
                'hf_export_weights_sha256': export_manifest['weights_sha256'],
                'dtype': 'fp8' if weight_dtype == 'fp8' else export_manifest['dtype'],
                'weight_dtype': weight_dtype,
                'conv_and_head_dtype': export_manifest['dtype'],
                'bf16_reduction': os.environ.get('NATLANG_BF16_REDUCTION', 'fp32-accumulate'), 'projection': projection,
                'overlap_schedule': bool(overlap_schedule), 'kv_cache_dtype': kv_cache_dtype, 'kv_layout': kv_layout,
                # LM-head logits precision (serve_native_checkpoint --sglang-head-logits): part of the sampling law
                'head_logits': head_logits,
                # SGLang's running-sequence cap (and its decode CUDA-graph batch sizes): batch composition changes the
                # kernels' batch shapes, so held results pair only within one cap (held_eval.ENGINE_FIELDS)
                'max_running': None if max_running is None else int(max_running),
                # n-gram speculative decoding (fork NATLANG_SPECULATIVE_NGRAM): the same sampling law, but the verify
                # pass's batch shapes differ from plain decode's, so held results pair only within one setting
                'speculative': None if not speculative else dict(speculative),
                # tool calls drawn under their schemas' grammar (the capture's sampling_law.tool_grammar)
                'tool_grammar': self.sglang_tool_grammar,
                'banned_token_ids': list(self.sglang_banned)}
            binding['actor_export'] = dict(self.sglang_actor_export)
        return self

    def _tool_grammar_info(self):
        """The served export's tokenizer as SGLang's grammar backend loads it (common.tool_grammar.info_from_export)."""
        if getattr(self, 'sglang_export_dir', None) is None:
            return super()._tool_grammar_info()
        # once per export directory: this runs for every chat call (twice), on the request threads that share the GIL
        cached = getattr(self, '_tool_grammar_export_cache', None)
        if cached is None or cached[0] != self.sglang_export_dir:
            from ..common import tool_grammar
            cached = self._tool_grammar_export_cache = (
                self.sglang_export_dir, tool_grammar.info_from_export(self.sglang_export_dir))
        return cached[1]

    def rebind_in_place(self, policy_id, weights_sha256=None, version=None, *, export_dir=None, weights_path=None,
                        checkpoint_path=None, tensor_identity=None):
        """Bind the policy a trainer loaded into the SGLang server in place (``update_weights_from_disk``; grpo.py) and
        reload the native route from the same export (opt-out ``NATLANG_REBIND_NATIVE_RELOAD=0``).

        Attestation is measured, never echoed: SGLang's own ``/model_info`` says which directory and weight version it
        loaded, and the front hashes that directory's weights file itself; the native route reports the
        ``native_weights_identity`` of the model it reloaded (or, while not resident, the source it will load and the
        trainer's claimed identity, which the lazy load then verifies). ``weights_sha256``/``tensor_identity`` are the
        trainer's claims; ``route_attestation`` reports the measurements beside them.

        Exact native-state splicing keys on the lineage (the start checkpoint's identity, unchanged here): a block a
        producer wrote under an earlier version of the same run splices exactly; its producing version is recorded
        as provenance (``producer_policy``). Native requests drain at a call boundary before the native weights
        change (no sequence continues on new weights over KV of the old)."""
        import re
        from ..common.hashing import sha256_file_hex
        if not re.fullmatch(r'[0-9a-f]{64}', str(policy_id)):
            raise ValueError('policy_id must be a SHA-256 digest')
        for value in (weights_sha256, tensor_identity):
            if value is not None and not re.fullmatch(r'[0-9a-f]{64}', str(value)):
                raise ValueError('weights_sha256 and tensor_identity must be SHA-256 digests')
        reload_enabled = os.environ.get('NATLANG_REBIND_NATIVE_RELOAD', '1') != '0'
        source = weights_path or (os.path.join(export_dir, 'model.safetensors') if export_dir else None) or checkpoint_path
        if source is not None and not os.path.exists(source):
            raise ValueError(f'rebind export/weights path does not exist: {source}')
        with self.__dict__.setdefault('_rebind_lock', threading.Lock()):
            measured_file = sha256_file_hex(source) if source is not None and source.endswith('.safetensors') else None
            sglang = self._measure_sglang(export_dir, version, known={os.path.realpath(source): measured_file}
                                          if measured_file else None)
            native_status = 'start checkpoint (not updated in place)'
            if reload_enabled and source is not None:
                record = {'path': str(source), 'policy_id': policy_id, 'weights_sha256': weights_sha256,
                          'measured_weights_sha256': measured_file, 'tensor_identity': tensor_identity,
                          'version': version}
                self._native_drain()
                try:
                    with self.__dict__.setdefault('_native_lock', threading.RLock()):
                        self.__dict__['_native_rebind_source'] = record
                        resident = self.__dict__.get('_native_model')
                        if resident is not None:
                            lock = getattr(self, 'native_execution_lock', None) or threading.RLock()
                            with lock:
                                self._load_native_source(resident, record)
                            native_status = 'reloaded'
                        else:
                            self._native_served = {'policy_sha256': policy_id, 'version': version,
                                                   'tensor_identity': None, 'source': str(source),
                                                   'source_weights_sha256': measured_file,
                                                   'pending_tensor_identity': tensor_identity}
                            self._invalidate_weight_caches()
                            native_status = 'pending (reloads from the source on its first request)'
                finally:
                    self._native_reopen()
            elif not reload_enabled:
                native_status = 'start checkpoint (opted out via NATLANG_REBIND_NATIVE_RELOAD=0)'
            with self._sglang_lock:
                binding = self.rl_policy_binding
                binding.setdefault('start_checkpoint_sha256', binding.get('checkpoint_sha256'))
                self._sglang_served = {'policy_sha256': policy_id, **sglang}
                served_sha = sglang.get('weights_sha256') or weights_sha256
                if served_sha is not None:
                    self.sglang_actor_export = {**self.sglang_actor_export, 'sha256': served_sha}
                    binding.setdefault('serving_engine', {})['hf_export_weights_sha256'] = served_sha
                    if 'actor_export' in binding:
                        binding['actor_export'] = dict(self.sglang_actor_export)
                binding['checkpoint_sha256'] = policy_id
                binding['in_place_update'] = {'version': version, 'policy_id': policy_id,
                                              'weights_sha256': weights_sha256, 'tensor_identity': tensor_identity,
                                              'native_engine': native_status}
                # New weights: the radix cache holds none of the old policy's openings (update_weights_from_disk flushes it).
                self.__dict__.pop('_sglang_openings', None)
                update = dict(binding['in_place_update'])
        return {**update, 'routes': self.route_attestation()}

    def _sglang_get(self, path):
        with urllib.request.urlopen(self.sglang_url + path, timeout=60) as response:
            return json.loads(response.read())

    def _measure_sglang(self, export_dir, version, known=None):
        """What SGLang itself reports it serves (``/model_info``: the weights directory and version of its last update)
        and the SHA-256 of that directory's ``model.safetensors`` as this front reads it (``known``: {real path: sha}
        already read in this bind)."""
        from ..common.hashing import sha256_file_hex
        if not getattr(self, 'sglang_url', ''):
            return {'measured': False, 'error': 'no SGLang server bound'}
        try:
            info = self._sglang_get('/model_info')
        except Exception as error:   # attestation then fails on the missing measurement
            return {'measured': False, 'error': f'/model_info: {error!r}'[:300]}
        path = info.get('model_path')
        weights = os.path.join(path, 'model.safetensors') if path else None
        return {'measured': True, 'model_path': path, 'version': info.get('weight_version'),
                'requested_model_path': export_dir, 'requested_version': None if version is None else str(version),
                'weights_sha256': ((known or {}).get(os.path.realpath(weights)) or sha256_file_hex(weights))
                if weights and os.path.isfile(weights) else None}

    def route_attestation(self):
        """What each of the four actor routes (generation, decision, capture on SGLang; native_fallback) serves, as
        measured at the last bind (``rebind_in_place``); ``lineage`` is the identity exact native state keys on.
        Before any in-place update every route serves the start checkpoint's export."""
        binding = getattr(self, 'rl_policy_binding', None) or {}
        start_policy = binding.get('start_checkpoint_sha256') or binding.get('checkpoint_sha256') or getattr(self, 'checkpoint_sha256', None)
        sglang = getattr(self, '_sglang_served', None) or {
            'policy_sha256': start_policy, 'version': None, 'measured': False, 'start': True,
            'weights_sha256': (getattr(self, 'sglang_export', None) or {}).get('weights_sha256')}
        native = dict(getattr(self, '_native_served', None) or {
            'policy_sha256': start_policy, 'version': None, 'tensor_identity': None, 'source': 'start checkpoint',
            'start': True})
        native['resident'] = self.__dict__.get('_native_model') is not None
        native['synchronized'] = (native.get('policy_sha256') == sglang.get('policy_sha256')
                                  and (native.get('source_weights_sha256') is None
                                       or native.get('source_weights_sha256') == sglang.get('weights_sha256')))
        return {'generation': dict(sglang), 'decision': dict(sglang), 'capture': dict(sglang),
                'native_fallback': native, 'lineage': dict(getattr(self, 'native_identity', None) or {})}

    def sentinel(self, token_ids, *, load_native=False):
        """Log-probabilities of ``token_ids[1:]`` under SGLang and, when resident (``load_native``: loaded first), the
        native engine: the check that both engines compute with the weights they attest (``grpo.WeightSync`` compares
        them with the learner's)."""
        ids = [int(token) for token in token_ids]
        if load_native and self.__dict__.get('_native_loader') is not None:
            self._ensure_native()
        if len(ids) < 2:
            raise RequestError('bad-request', 'a sentinel needs at least two tokens')
        result = self._sglang_generate({'input_ids': ids, 'return_logprob': True, 'logprob_start_len': 0,
                                        'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1}})
        sglang = [float(lp) for lp, _, _ in result['meta_info']['input_token_logprobs'][1:]]
        native = None
        if self.__dict__.get('_native_model') is not None:
            native = self._native_gated(lambda: self._native_sentinel(ids))
        return {'sglang': sglang, 'native': native, 'routes': self.route_attestation()}

    def _native_sentinel(self, ids):
        import torch
        from ..model.shared_state import native_step

        def run():
            model = self.shared_model
            with torch.inference_mode():
                tokens = torch.tensor([ids], dtype=torch.long, device=self.backbone.embedding_weight.device)
                hidden = native_step(model, tokens, None)['h_final'][0]
                logp = torch.log_softmax(model.logits(hidden[:-1]).float(), -1)
                return logp.gather(-1, tokens[0, 1:, None])[:, 0].cpu().tolist()
        return self._native_run(run)

    # Native requests drain at a call boundary for a reload: the gate holds new native calls before they start and the
    # reload waits for the running ones to finish, so no sequence decodes on new weights over KV of the old ones.
    def _native_gate_state(self):
        gate = self.__dict__.get('_native_gate')
        if gate is None:
            gate = self.__dict__.setdefault('_native_gate', {'condition': threading.Condition(), 'open': True,
                                                             'running': 0})
        return gate

    def _native_admit(self):
        """Count one native call in (waiting while a reload drains). A thread already inside a counted call passes:
        its nested native work belongs to that call."""
        gate, local = self._native_gate_state(), self._native_local()
        with gate['condition']:
            while not gate['open'] and not getattr(local, 'depth', 0):
                gate['condition'].wait()
            gate['running'] += 1

    def _native_release(self, *_):
        gate = self._native_gate_state()
        with gate['condition']:
            gate['running'] -= 1
            gate['condition'].notify_all()

    def _native_local(self):
        local = self.__dict__.get('_native_thread_local')
        if local is None:
            local = self.__dict__.setdefault('_native_thread_local', threading.local())
        return local

    def _native_gated(self, work):
        """``work()`` as one native call (``_native_admit``), nested native work on this thread included."""
        local = self._native_local()
        self._native_admit()
        local.depth = getattr(local, 'depth', 0) + 1
        try:
            return work()
        finally:
            local.depth -= 1
            self._native_release()

    def _native_drain(self):
        gate = self._native_gate_state()
        with gate['condition']:
            gate['open'] = False
            while gate['running']:
                gate['condition'].wait()

    def _native_reopen(self):
        gate = self._native_gate_state()
        with gate['condition']:
            gate['open'] = True
            gate['condition'].notify_all()

    def _served_policy(self):
        """The in-place update this sampler serves (``rebind_in_place``), else None (the start checkpoint)."""
        update = (getattr(self, 'rl_policy_binding', None) or {}).get('in_place_update')
        return None if update is None else dict(update)

    def progress_metrics(self):
        return {**super().progress_metrics(), 'natlang_sglang_requests_total': self.sglang_requests,
                'natlang_sglang_generated_tokens_total': self.sglang_tokens,
                'natlang_sglang_prompt_tokens_total': getattr(self, 'sglang_prompt_tokens', 0),
                'natlang_sglang_cached_prompt_tokens_total': getattr(self, 'sglang_cached_tokens', 0),
                'natlang_sglang_opening_warms_total': getattr(self, 'sglang_opening_warms', 0),
                'natlang_sglang_opening_warm_failures_total': getattr(self, 'sglang_opening_warm_failures', 0),
                'natlang_native_fallback_requests_total': self.native_fallbacks,
                'natlang_sglang_native_state_writes_total': self.sglang_state_writes,
                **{f'natlang_sglang_{key}_total': value for key, value in
                   (self.sglang_history.stats.items() if getattr(self, 'sglang_history', None) else ())},
                'natlang_native_model_resident': int(self.__dict__.get('_native_model') is not None),
                **getattr(self, 'native_endpoint_metrics', {}),
                **self.native_fallback_reasons}

    def _measure_native_endpoint(self, endpoint, work):
        """Account for direct native work separately from generation fallbacks."""
        started = time.monotonic()
        prefix = 'natlang_native_endpoint'
        label = '{endpoint="%s"}' % endpoint
        with self._sglang_lock:
            metrics = self.native_endpoint_metrics
            key = prefix + '_requests_total' + label
            metrics[key] = metrics.get(key, 0) + 1
        try:
            result = work() if endpoint.endswith('_sglang') else self._native_gated(work)
        except BaseException:
            with self._sglang_lock:
                key = prefix + '_failures_total' + label
                metrics[key] = metrics.get(key, 0) + 1
            raise
        else:
            with self._sglang_lock:
                key = prefix + '_completed_total' + label
                metrics[key] = metrics.get(key, 0) + 1
            return result
        finally:
            with self._sglang_lock:
                key = prefix + '_seconds_total' + label
                metrics[key] = metrics.get(key, 0.0) + time.monotonic() - started

    def native_write_text(self, text, *, value_type=None, adapters=None):
        if adapters or not self.sglang_url:
            return self._measure_native_endpoint('write_text', lambda: super(SglangEngine, self).native_write_text(
                text, value_type=value_type, adapters=adapters))
        return self._measure_native_endpoint('write_text_sglang', lambda: self._sglang_write_text(text, value_type))

    def _sglang_write_text(self, text, value_type):
        """`_native_write_text_impl` on SGLang: the text forced through the producer span ([produce_open] text
        [produce_close], positions from 0) and its state exported by one capture request, so a write never reloads the
        native model into the front (~14 GB per actor on the GB10, 2026-11-12)."""
        if not getattr(self, 'native_state_enabled', False):
            raise RequestError('neuralese-native-state-unavailable', 'native text writes require shared-state serving')
        if not isinstance(text, str):
            raise RequestError('neuralese-native-state-text', 'native text writes require text')
        from .chat import escape_specials
        nonce = hashlib.sha256((text + str(time.time_ns())).encode()).hexdigest()
        ids = self._template_tokens(escape_specials(text, self.specials, nonce), nonce)
        if len(ids) + 2 >= self.context:
            raise RequestError('context-length', 'forced native text and producer markers exceed this model context')
        opened = [int(self.role_tokens.produce_open)] + [int(token) for token in ids]
        block, record = self._native_store_capture(ids, [{'ids': opened, 'start': 1, 'end': len(opened)}], text=text,
                                                   producer_kind='text')
        self._note_text_write(record.id)
        meta = record.meta()
        if value_type is not None:
            meta['type'] = value_type
        with self._sglang_lock:
            self.sglang_state_writes += 1
        return {'id': record.id, 'native_state': meta,
                'source_capture': {'policy': self._native_block_identity(block), 'segments': [
                    {'kind': 'tokens', 'token_ids': [self.role_tokens.produce_open], 'producing': [False]},
                    {'kind': 'tokens', 'token_ids': ids, 'producing': [True] * len(ids)},
                    {'kind': 'tokens', 'token_ids': [self.role_tokens.produce_close], 'producing': [False]}],
                    'payload_start': 1, 'payload_stop': len(ids) + 1, 'adapter_binding': block.adapter_binding},
                'source_text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()}

    def native_capture(self, prompt_token_ids, reply_token_ids, *, prompt_segments=None, select=None, value_type=None,
                       adapters=None):
        if adapters or not self.sglang_url:
            return self._measure_native_endpoint('capture', lambda: super(SglangEngine, self).native_capture(
                prompt_token_ids, reply_token_ids, prompt_segments=prompt_segments, select=select,
                value_type=value_type, adapters=adapters))
        return self._measure_native_endpoint('capture_sglang', lambda: self._sglang_native_capture(
            prompt_token_ids, reply_token_ids, select, value_type, prompt_segments))

    def _capture_blocks(self, imports):
        """`Engine._capture_blocks` from the store's files (header and token ids; the fork reads the payload)."""
        blocks = []
        for _, ident in imports:
            block = self._sglang_store().native_state_file(ident)
            if block is None:
                raise RequestError('neuralese-unknown-native-state', f'{ident} is not in this server\'s store; PUT it first')
            if block.derivation is not None:
                from ..model.native_state_derivation import verify_state_derivation
                verify_state_derivation(self.store, block.load())
            if not self._sglang_exact(block):
                raise RequestError('neuralese-native-capture', 'a captured prompt imports only this model\'s exact native state')
            blocks.append(block)
        return blocks

    def _sglang_placeholders(self, block_id, count):
        """The content-hashed placeholder ids of an imported state's positions (their KV and conv inputs the fork
        replaces by the stored ones; `custom_params.natlang_import`)."""
        import hashlib
        base = 2 * int(self.sglang_export['producer_embedding']['input_id_offset'])
        return [base + int.from_bytes(hashlib.sha256(f'{block_id}:{index}'.encode()).digest()[:4], 'little')
                % (2 ** 31 - 1 - base) for index in range(count)]

    def _sglang_native_capture(self, prompt_token_ids, reply_token_ids, select, value_type, prompt_segments=None):
        """`_native_capture_impl` on SGLang: one capture request of the reply's content span in its own context, the
        prompt's imported states (`prompt_segments`) as the fork's imports where they stood."""
        if not getattr(self, 'native_state_enabled', False):
            raise RequestError('neuralese-native-state-unavailable', 'native captures require shared-state serving')
        prompt, imports = self.capture_prompt(prompt_token_ids, prompt_segments)
        blocks = self._capture_blocks(imports)
        self._capture_import_tokens = sum(len(block.token_ids) for block in blocks)
        prefix, span = self.capture_plan(prompt, reply_token_ids, select, imports)
        ids, spliced, last = [], [], 0
        for (at, ident), block in zip(imports, blocks):
            ids.extend(prefix[last:at])
            count = len(block.token_ids)
            if count:
                spliced.append({'path': str(block.path), 'start': len(ids), 'count': count})
            ids.extend(self._sglang_placeholders(ident, count))
            last = at
        ids.extend(prefix[last:])
        start = len(ids)
        ids.extend(span)
        text = self.tokenizer.decode(span, skip_special_tokens=False)
        block, record = self._native_store_capture(span, [{'ids': ids, 'start': start, 'end': len(ids),
                                                           **({'imports': spliced} if spliced else {})}], text=text,
                                                   producer_kind='span')
        with self._sglang_lock:
            self.sglang_state_writes += 1
        return self._capture_result(block, record, prefix, span, value_type, imports, blocks)

    def native_init_body(self, messages, tools, placeholder, text, **kwargs):
        return self._measure_native_endpoint('init_body', lambda: super(SglangEngine, self).native_init_body(
            messages, tools, placeholder, text, **kwargs))

    # Each SGLang-served request runs on its own thread, which ends with the request (`_spawn`), never on a long-lived
    # pool. torch's CPU allocator on aarch64 is mimalloc (linked into libc10), whose heaps are per thread: an idle thread
    # keeps the pages of the tensors it allocated (native-state imports, produce-span captures) until it allocates again,
    # including those another thread freed (the store's LRU evicts blocks on other threads). The 512-worker pool had
    # ~180 workers for 32 open connections half an hour into grpo-v3-run1-r1 (work spreads over every idle worker), and
    # 85% of the front's host growth was torch's arena (front RSS +0.8 GB/h, 2026-11-16). A thread that exits hands its
    # heap back: 128 round-robin workers kept > 3.7 GB for a 0.25 GB tensor LRU, a thread per task 0.7 GB.
    @staticmethod
    def _spawn(work, name='sglang-actor'):
        thread = threading.Thread(target=work, name=name, daemon=True)
        thread.start()
        return thread

    @classmethod
    def _map_threads(cls, function, items):
        """`[function(item) for item in items]`, each on its own short-lived thread (`_spawn`)."""
        results, errors = [None] * len(items), [None] * len(items)

        def run(index, item):
            try:
                results[index] = function(item)
            except BaseException as error:  # re-raised on the caller's thread, as Executor.map does
                errors[index] = error
        threads = [cls._spawn(lambda index=index, item=item: run(index, item), 'sglang-actor-map')
                   for index, item in enumerate(items)]
        for thread in threads:
            thread.join()
        for error in errors:
            if error is not None:
                raise error
        return results

    @staticmethod
    def sglang_unsupported(request, plan):
        """Why SGLang cannot serve this request (None: it can)."""
        # native_recording (the collector's graph recorder) only asks for the capture, which this path records.
        names = ['adapters'] if request.adapters else []
        names += sorted({f'prompt segment {kind}' for kind, _ in plan
                         if kind not in ('tokens', 'privileged', 'native_state', 'native_state_constant')})
        return ', '.join(names) or None

    def _adapter_entries(self, request):
        """The adapter stack a request runs with (the native engine's default stack when it names none)."""
        if request.adapters is None:
            return getattr(self, 'native_default_adapters', None) or None
        return request.adapters or None

    def _sglang_reason(self, request, plan):
        """Why SGLang cannot serve this request on this actor (None: it can)."""
        reason = self.sglang_unsupported(request, plan)
        names = [name for name in (reason or '').split(', ') if name]
        entries = self._adapter_entries(request)
        if 'adapters' in names:
            names.remove('adapters')
        if entries:
            if not getattr(self, 'sglang_adapter_slots', 0) or not getattr(self, 'sglang_adapter_targets', None):
                names.append('adapters')
            else:
                try:
                    if self._sglang_adapter_file(entries)[1] > self.sglang_adapter_rank:
                        names.append('adapter rank')
                except Exception:  # an invalid stack fails on the native engine with its exact error
                    names.append('adapters')
        return ', '.join(names) or None

    def _sglang_adapter_file(self, entries):
        """(path, largest module rank) of an adapter stack's summed low-rank factors in the fork's slot format."""
        from ..common.hashing import canonical_json_sha256_hex
        from ..model.native_adapters import bound_execution, lowrank_factors
        key = canonical_json_sha256_hex(entries)
        known = self._adapter_files.get(key)
        if known is not None:
            return known
        import torch
        from safetensors.torch import save_file
        with bound_execution(self, entries) as resolved:
            factors = lowrank_factors(resolved)
        modules = {}
        for name, (factor_b, factor_a) in factors.items():
            target = self.sglang_adapter_targets[name]
            modules.setdefault(target['module'], []).append((target['offset'], factor_b, factor_a))
        tensors, largest = {}, 0
        for module, parts in modules.items():
            rank = sum(part[2].shape[0] for part in parts)
            factor_b = torch.zeros(self.sglang_module_rows[module], rank)
            column = 0
            for offset, part_b, part_a in parts:
                factor_b[offset:offset + part_b.shape[0], column:column + part_b.shape[1]] = part_b
                column += part_b.shape[1]
            tensors[f'{module}.A'] = torch.cat([part[2] for part in parts]).contiguous()
            tensors[f'{module}.B'] = factor_b.contiguous()
            largest = max(largest, rank)
        path = os.path.join(self._sglang_scratch(), f'adapter-{key}.safetensors')
        if not os.path.exists(path):
            tmp = f'{path}.{uuid.uuid4().hex}.partial'
            save_file(tensors, tmp)
            os.replace(tmp, path)
        self._adapter_files[key] = (path, largest, key)
        return self._adapter_files[key]

    def _adapter_acquire(self, entries):
        """A slot holding this stack for one request (waits while every slot serves another stack)."""
        path, _rank, key = self._sglang_adapter_file(entries)
        with self._adapter_condition:
            while True:
                slot = self._adapter_slot_of.get(key)
                if slot is None:
                    free = [s for s in range(1, self.sglang_adapter_slots + 1) if not self._adapter_slot_users.get(s)]
                    if free:
                        # An empty slot first, else the least recently used free one.
                        slot = min(free, key=lambda s: (s in self._adapter_slot_key, self._adapter_slot_key.get(s, ('', 0))[1]))
                        old = self._adapter_slot_key.get(slot)
                        if old is not None:
                            self._adapter_slot_of.pop(old[0], None)
                        self._adapter_slot_of[key] = slot
                if slot is not None:
                    self._adapter_clock += 1
                    self._adapter_slot_key[slot] = (key, self._adapter_clock)
                    self._adapter_slot_users[slot] = self._adapter_slot_users.get(slot, 0) + 1
                    return {'slot': slot, 'id': key, 'path': path}
                self._adapter_condition.wait()

    def _adapter_release(self, spec):
        with self._adapter_condition:
            self._adapter_slot_users[spec['slot']] -= 1
            self._adapter_condition.notify_all()

    def _with_adapters(self, request, work):
        """Run `work(spec)` inside the request's native adapter binding with a slot (spec None without adapters)."""
        entries = self._adapter_entries(request)
        if not entries:
            return work(None)
        from ..model.native_adapters import bound_execution
        with bound_execution(self, entries):
            spec = self._adapter_acquire(entries)
            try:
                return work(spec)
            finally:
                self._adapter_release(spec)

    def _routes_native(self, request, plan):
        """Whether this request runs on the native engine (counted per reason in /metrics)."""
        reason = self._sglang_reason(request, plan)
        native = reason is not None
        if native:
            with self._sglang_lock:
                self.native_fallbacks += 1
                for name in reason.split(', '):
                    key = 'natlang_native_fallback_reason_total{reason="%s"}' % name.replace(' ', '_')
                    self.native_fallback_reasons[key] = self.native_fallback_reasons.get(key, 0) + 1
            if self.__dict__.get('_native_loader') is not None:
                self._ensure_native()  # on the request thread, before the scheduler needs it
        return native

    def submit(self, request):
        future: Future = Future()
        request.request_id = request.request_id or f"chatcmpl-{next(self._ids)}"
        try:
            plan = self._prompt_plan(request.messages, request.tools)
        except Exception as error:
            future.set_exception(error)
            return future
        if self._routes_native(request, plan):
            self._native_admit()
            try:
                future = super().submit(request)
            except BaseException:
                self._native_release()
                raise
            future.add_done_callback(self._native_release)
            return future
        plan = self._append_only(request, plan)
        sequence = Sequence(request, future, prompt_plan=plan)

        def run():
            try:
                future.set_result(self._sglang_run(sequence))
            except Exception as error:
                future.set_exception(error)
        self._spawn(run)
        return future

    def generate(self, request):
        request.request_id = request.request_id or f"chatcmpl-{next(self._ids)}"
        plan = self._prompt_plan(request.messages, request.tools)
        if self._routes_native(request, plan):
            return self._native_gated(lambda: super(SglangEngine, self).generate(request))
        return self._sglang_run(Sequence(request, Future(), prompt_plan=self._append_only(request, plan)))

    @staticmethod
    def _plain(request):
        return (request.privileged_context is None and request.forced_message is None and request.guidance is None
                and not getattr(request, 'native_recording', None))

    def _append_only(self, request, plan):
        """A plain token prompt with this actor's own earlier output kept as the ids it sampled (serve/append_only.py)."""
        history = getattr(self, 'sglang_history', None)
        if history is None or not self._plain(request) or not plan or any(kind != 'tokens' for kind, _ in plan):
            return plan
        ids = [token for _, value in plan for token in value]
        spliced = history.splice(ids)
        return plan if spliced is ids else [('tokens', spliced)]

    def _sglang_seed_honored(self):
        """Whether the SGLang backend applies a request's `sampling_seed`: its sampler seeds the draw only under
        `--enable-deterministic-inference` (otherwise the field is ignored; the flashinfer backend rejects it). Read
        once from the server's own resolved arguments; an unreadable server counts as not honoring."""
        known = self.__dict__.get('_sglang_seed_support')
        if known is None:
            known = False
            for path in ('/server_info', '/get_server_info'):
                try:
                    with urllib.request.urlopen(self.sglang_url + path, timeout=30) as response:
                        info = json.loads(response.read())
                except (urllib.error.URLError, OSError, ValueError):
                    continue
                found = []
                def visit(value):
                    if isinstance(value, dict):
                        if 'enable_deterministic_inference' in value:
                            found.append(value['enable_deterministic_inference'])
                        for item in value.values():
                            visit(item)
                visit(info)
                known = bool(found) and all(item is True for item in found)
                break
            self.__dict__['_sglang_seed_support'] = known
        return known

    def _sglang_generate(self, body):
        """One SGLang /generate call."""
        result = self._sglang_post(body)
        self._sglang_count_prompt(result.get('meta_info') if isinstance(result, dict) else None)
        return result

    def _sglang_post(self, body):
        body = adapter_cache_key(body)
        try:
            with urllib.request.urlopen(urllib.request.Request(self.sglang_url + '/generate',
                    data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}),
                    timeout=SGLANG_REQUEST_SECONDS) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise RequestError('sglang-actor-backend', f'SGLang HTTP {error.code}: {error.read()[:400]!r}') from error

    def _sglang_count_prompt(self, meta):
        """Prompt tokens SGLang served and how many came from its radix cache (`/metrics`: the prefix hit rate)."""
        if not isinstance(meta, dict) or not isinstance(meta.get('prompt_tokens'), int):
            return
        with self._sglang_lock:
            self.sglang_prompt_tokens = getattr(self, 'sglang_prompt_tokens', 0) + meta['prompt_tokens']
            self.sglang_cached_tokens = getattr(self, 'sglang_cached_tokens', 0) + int(meta.get('cached_tokens') or 0)

    def _sglang_prompt(self, seq):
        """The prompt ids and capture segments, as `_native_prefill_impl` records them for ordinary tokens."""
        request = seq.request
        unsupported = self._sglang_reason(request, seq.prompt_plan or ())
        if unsupported:
            raise RequestError('sglang-actor-unsupported', 'the SGLang actor does not serve ' + unsupported)
        if request.rl_capture and request.temperature <= 0 and not request.native_recording:
            raise RequestError('native-rl-sampling-law', 'native rollout capture requires positive temperature')
        seq.native_trace_enabled = request.native_protocol_recovery
        if capture_enabled(seq):
            seq.native_rl_policy = dict(self.execution_native_identity)
            from ..model.native_adapters import execution_binding, SCHEMA
            bindings = execution_binding(self)
            seq.native_rl_adapter_binding = ({'schema': SCHEMA, 'base_native_identity': dict(self.native_identity),
                                              'adapters': bindings} if bindings else None)
        seq.native_sampling_seed = (int(request.seed) if request.seed is not None else
                                    derive_seed('native-chat', request.request_id))
        if request.temperature is not None and request.temperature > 0:
            seq.native_seed_honored = self._sglang_seed_honored()
        plan = list(seq.prompt_plan or ())
        self._validate_native_state_modes(request.native_state_modes, {
            value['id'] if isinstance(value, dict) else value for kind, value in plan
            if kind in ('native_state', 'native_state_constant')})
        if request.privileged_context is not None:
            insert_at, pieces, seq.native_guidance_receipt = self._guidance(request, plan)
            plan = _insert_privileged(plan, insert_at, pieces)
        ids, seq.sglang_imports = [], []
        for kind, value in plan:
            if kind == 'insertion_point':
                seq.native_guidance_receipt['insert_at'] = seq.native_own_position
                continue
            if kind in ('native_state', 'native_state_constant', 'privileged_state'):
                begin = len(seq.native_rl_prompt)
                self._sglang_import(seq, ids, value['id'] if isinstance(value, dict) else value,
                                    stop_gradient=kind == 'native_state_constant',
                                    source_occurrence=value.get('source_occurrence') if isinstance(value, dict) else None)
                if kind == 'privileged_state' and capture_enabled(seq):
                    for segment in seq.native_rl_prompt[begin:]:  # JOLT+ state guidance, as the native engine marks it
                        segment['privileged'] = True
                continue
            if kind not in ('tokens', 'privileged'):
                raise RequestError('sglang-actor-unsupported',
                                   f'the SGLang actor does not import native state yet (prompt segment {kind!r})')
            if capture_enabled(seq) and value:
                seq.native_rl_prompt.append({'kind': 'tokens', 'token_ids': list(value), 'producing': [False] * len(value),
                                             **({'privileged': True} if kind == 'privileged' else {})})
            seq.native_own_position += len(value)
            ids.extend(value)
        if not ids:
            raise RequestError('empty-prompt', 'native serving needs at least one prompt token')
        seq.prompt_positions = len(ids)
        if seq.prompt_positions >= self.context:
            raise RequestError('context-length', f'prompt uses {seq.prompt_positions} positions, context is {self.context}')
        return ids

    def _sglang_run(self, seq):
        return self._with_adapters(seq.request, lambda spec: self._sglang_run_bound(seq, spec))

    def _sglang_run_bound(self, seq, adapter):
        request = seq.request
        seq.native_actor_export = getattr(self, 'sglang_actor_export', None)
        seq.native_served_policy = self._served_policy()
        ids = self._sglang_prompt(seq)
        template_prefix, template_suffix = self._native_template_plan(seq)
        if getattr(self, 'sglang_tool_grammar', False):
            # the front follows the reply under the same grammar: every SGLang token is checked against it
            self._tool_grammar_begin(seq, template_prefix)
        open_position = (self._sglang_template_prefix(seq, ids, template_prefix, template_suffix)
                         if template_prefix is not None else None)
        sampling = request.temperature is not None and request.temperature > 0
        k = request.rl_top_k if sampling and request.rl_top_k is not None else 0
        if sampling and request.top_p < 1:
            # top-p: SGLang draws from the nucleus; its natlang record stays the untruncated post-temperature logp
            # (fork `natlang_post_temperature_logprobs`), which the capture's sampling law declares.
            seq.nucleus_applied = float(request.top_p)
        params = {'temperature': float(request.temperature) if sampling else 0.0,
                  **({'top_p': seq.nucleus_applied} if seq.nucleus_applied is not None else {}),
                  'stop_token_ids': sorted(self.stop_ids), 'skip_special_tokens': False,
                  'no_stop_trim': True, 'custom_params': {TOP_K_KEY: max(k, 1),
                                                          **({TOP_K_MASS_KEY: True} if k else {}),
                                                          **({IMPORT_KEY: seq.sglang_imports} if seq.sglang_imports else {}),
                                                          **({ADAPTER_KEY: adapter} if adapter else {})}}
        if self.sglang_banned:
            params['logit_bias'] = {str(token): BAN_BIAS for token in self.sglang_banned}
        state = {'index': 0, 'finish': 'length', 'done': False, 'text': '', 'run': len(seq.items), 'prompt': ids,
                 'fed': [], 'adapter': adapter, **({'open_position': open_position} if open_position is not None else {})}
        try:
            if request.forced_message is not None:
                self._sglang_forced_replay(seq, state)
            else:
                if request.guidance is not None:
                    from .native_guidance import initialize
                    initialize(self, seq)
                else:
                    from .native_guidance import announce_reasoning
                    announce_reasoning(self, seq)
                for part in itertools.count():
                    if seq.native_forced_tokens:
                        self._sglang_feed_forced(seq, state)
                        if state['done']:
                            break
                    room = min(int(request.max_tokens) - seq.generated_positions,
                               self.context - len(ids) - len(state['fed']))
                    if room <= 0:
                        break
                    # A templated write stops at its span's close: the template suffix follows it, not a sample.
                    stops = sorted(set(self.stop_ids) | ({self.role_tokens.produce_close} if seq.native_template_suffix else set()))
                    sampling_params = {**params, 'max_new_tokens': room, 'stop_token_ids': stops}
                    if sampling:
                        # The backend's draw is a function of (seed, token position): one seed serves a whole reply;
                        # each continuation request (a guidance redraw, a forced feed) gets its own derived seed.
                        sampling_params['sampling_seed'] = (seq.native_sampling_seed if not part else
                                                            derive_seed('sglang-part', seq.native_sampling_seed, part))
                    # Guidance bans apply at one position (the rewound one): draw that token alone under them.
                    bans = seq.bans.get(seq.native_own_position) if seq.guide is not None else None
                    if bans:
                        sampling_params.update(max_new_tokens=1, logit_bias={
                            str(token): BAN_BIAS for token in sorted(set(bans) | set(self.sglang_banned))})
                    if seq.tool_grammar is not None:
                        # SGLang's xgrammar applies the mask before sampling and before its natlang records; a
                        # continuation's grammar first takes the reply tokens its input already holds
                        held = len(seq.tool_grammar_prefix) + len(state['fed'])
                        sampling_params['structural_tag'] = seq.tool_grammar_law['grammar']
                        sampling_params['custom_params'] = {**params['custom_params'],
                                                            **({GRAMMAR_PREFIX_KEY: held} if held else {})}
                    body = {'input_ids': ids + state['fed'], 'rid': request.request_id + (f'-{part}' if part else ''),
                            'sampling_params': sampling_params}
                    self._sglang_sample(seq, body, k, sampling, state)
                    if state.pop('rewind', False):
                        continue
                    if state['done'] or not (seq.native_forced_tokens or bans):
                        break
            if seq.native_capture_open:
                # The native engine's check before the next draw: the token allowance ran out inside the span.
                raise RequestError('neuralese-native-state-open', 'generation ended inside a produce span')
        except RequestError as error:
            if (error.code not in FAILURE_CODES or not request.native_protocol_recovery
                    or seq.native_rl_policy is None):
                raise
            # The native engine's recoverable protocol failure (_native_protocol_fail): the actual boundary, no block.
            seq.native_protocol_failure = failure_descriptor(self, seq, error)
            state['finish'] = 'error'
        finally:
            with self._sglang_lock:
                self.sglang_requests += 1
                self.sglang_tokens += state['index']
        if seq.guide is not None:
            # Guided replies publish only committed output, as the native engine flushes them at the finish; their
            # open reasoning has streamed already (native_guidance.stream_reasoning, run by each token's check).
            from .native_guidance import flush
            flush(self, seq)
            if request.on_delta is not None and seq.pending:
                request.on_delta({'text': self.tokenizer.decode(seq.pending, skip_special_tokens=False)})
                seq.pending = []
        elif request.on_delta is not None:
            self._sglang_emit(seq, state, final=True)
        seq.finish_reason = state['finish']
        seq.phase = 'done'
        history = getattr(self, 'sglang_history', None)
        if (history is not None and self._plain(request) and not seq.sglang_imports and state['finish'] != 'error'
                and seq.guide is None):
            history.store(state['prompt'], state['fed'] + ([state['terminal']] if 'terminal' in state else []))
        return self._native_response(seq)

    def _sglang_sample(self, seq, body, k, sampling, state):
        """One SGLang generation through the decode protocol (streamed when the client streams)."""
        request = seq.request
        if request.on_delta is None and seq.guide is None:
            result = self._sglang_generate(body)
            self._sglang_consume(seq, result.get('output_ids'), (result.get('meta_info') or {}).get(TOP_K_KEY),
                                 k, sampling, state)
            return
        # Streamed replies: SGLang sends incremental token chunks (--incremental-streaming-output); the text
        # reaches the client as it is generated, the capture is completed from the same per-token records.
        # Leaving the stream early (a protocol error, a guidance rewind) closes the connection, which aborts the SGLang
        # request. Guided replies always stream so that a rejection stops the draw, and so that the reasoning the
        # generation prompt opened reaches a streaming client as it is generated.
        for chunk in self._sglang_stream(body):
            if request.cancelled:
                raise RequestError('cancelled', 'the client went away')
            self._sglang_consume(seq, chunk.get('output_ids'), (chunk.get('meta_info') or {}).get(TOP_K_KEY),
                                 k, sampling, state)
            if state['done'] or state.get('rewind'):
                break

    def _sglang_template_prefix(self, seq, ids, prefix, suffix):
        """A templated reply's fixed prefix (and a write's open marker) after the prompt, recorded as the native engine
        feeds them: reply tokens without samples. Returns the open marker's cache position for a write."""
        write = seq.request.template.get('value') == 'write'
        if seq.prompt_positions + seq.generated_positions + len(prefix) + write >= self.context:
            raise RequestError('context-length', 'templated native reply prefix exceeds remaining context')
        for token in prefix:
            ids.append(token)
            seq.items.append(token)
            seq.native_item_positions.append(seq.native_own_position)
            seq.generated_positions += 1
            self._native_record_input(seq, token, producing=False)
        if not write:
            return None
        seq.native_write_start_own = seq.native_own_position
        seq.native_write_open_sample = None
        position = len(ids)
        ids.append(self.role_tokens.produce_open)
        seq.generated_positions += 1
        seq.native_capture_open = True
        seq.native_capture_chunks, seq.native_capture_ids = [], []
        seq.native_template_suffix = suffix
        self._native_record_input(seq, self.role_tokens.produce_open, producing=False)
        return position

    def _sglang_feed_forced(self, seq, state):
        """A write template's suffix after its span closed: fed without samples, as the native engine feeds it."""
        while seq.native_forced_tokens:
            token = seq.native_forced_tokens.pop(0)
            if token in self.stop_ids:
                seq.native_forced_tokens = []
                state['finish'], state['done'] = 'stop', True
                return
            seq.generated_positions += 1
            self._native_record_input(seq, token, producing=False)
            state['fed'].append(token)
            seq.items.append(token)
            seq.native_item_positions.append(seq.native_own_position - 1)
            self._native_guide(seq)  # a rejection of forced text raises the native template conflict
        if seq.request.on_delta is not None and seq.guide is None:
            self._sglang_emit(seq, state)

    def _sglang_forced_replay(self, seq, state):
        """Forced teacher replay, as the native engine records it: the exact teacher reply's tokens through the decode
        protocol (no sampling law), SGLang only exporting the state of each produce span it closes."""
        request = seq.request
        if not capture_enabled(seq) or request.native_recording is not True:
            raise RequestError('forced-message', 'forced teacher replay requires native structural recording')
        tokens = self._forced_message_plan(request.forced_message, request.tools, opening=request.messages)
        if len(tokens) > request.max_tokens:
            raise RequestError('forced-message', 'exact teacher reply exceeds the request token allowance')
        import copy
        seq.native_forced_replay = True
        seq.native_forced_source = copy.deepcopy(request.forced_message['source_receipt'])
        seq.native_forced_replay_prompt_sha256 = hashlib.sha256(json.dumps(
            {'messages': request.messages, 'tools': request.tools or []}, ensure_ascii=False, sort_keys=True,
            separators=(',', ':')).encode('utf-8')).hexdigest()
        for token in tokens:
            if state['done']:
                break
            self._sglang_token(seq, int(token), (None, None, None), 0, False, state)
        if not state['done']:
            if seq.native_capture_open:
                raise RequestError('neuralese-native-state-open', 'forced teacher reply ended inside a producer span')
            state['finish'], state['done'] = 'stop', True
        if request.on_delta is not None:
            self._sglang_emit(seq, state)

    def _sglang_consume(self, seq, output, records, k, sampling, state):
        """Record SGLang's tokens and their per-token law records, exactly as the native engine records samples."""
        if not isinstance(output, list) or not isinstance(records, list) or len(records) != len(output):
            raise RequestError('sglang-actor-capture', 'SGLang returned no per-token natlang_top_k records')
        for token, record in zip(output, records):
            if state['done']:
                raise RequestError('sglang-actor-capture', 'SGLang returned tokens after the stop token')
            tail = record_tail_mass(record, k) if k and sampling else None
            self._sglang_token(seq, int(token), (*decode_record(record, k), tail), k, sampling, state)
            if self._sglang_rewound(seq, state):
                return
        if seq.request.on_delta is not None and seq.guide is None:
            self._sglang_emit(seq, state)

    def _sglang_token(self, seq, token, record, k, sampling, state):
        """One sampled token through the native engine's decode-step protocol (`_native_sequence_step`)."""
        index = state['index']
        state['index'] += 1
        if token in self.sglang_banned:
            raise RequestError('sglang-actor-capture', f'SGLang sampled banned role marker {token}')
        logp, top_ids, top_logps, *extra = record
        tail_mass = extra[0] if extra else None
        marker = self.role_tokens
        terminal = token in self.stop_ids
        producing_input = seq.native_capture_open and token not in (marker.produce_open, marker.produce_close)
        if capture_enabled(seq) or seq.guide is not None:
            banned = set(self.sglang_banned) | (set(seq.bans.get(seq.native_own_position, ())) if seq.guide is not None else set())
            sample = {'index': index, 'token_id': token, 'logp': logp if sampling else None,
                      **({'forced': True} if seq.native_forced_replay else {}),
                      **({'banned_token_ids': sorted(banned)} if banned or seq.guide is not None else {}),
                      'input_role': 'produce' if producing_input else 'ordinary',
                      'terminal': terminal, 'query_own_position': seq.native_own_position - 1}
            if k:
                sample['top_k'] = {'ids': base64.b64encode(top_ids.astype('<i4').tobytes()).decode('ascii'),
                                   'logps': base64.b64encode(top_logps.astype('<f2').tobytes()).decode('ascii'),
                                   # The fork's exact FP32 tail beside the float16 entries (versioned receipt field).
                                   **({TOP_K_MASS_FIELD: {'head': 1. - tail_mass, 'tail': tail_mass}}
                                      if tail_mass is not None and 0. <= tail_mass <= 1. else {})}
            seq.rl_samples.append(sample)
        if terminal:
            if seq.native_capture_open:
                raise RequestError('neuralese-native-state-open', 'generation stopped inside a produce span')
            state['finish'], state['done'], state['terminal'] = 'stop', True, token
            return
        if token == marker.produce_close and not seq.native_capture_open:
            raise RequestError('neuralese-native-state-close', 'generation emitted a produce end marker without an open span')
        if token in (marker.consume_open, marker.consume_close):
            raise RequestError('neuralese-native-state-consume-marker',
                               'consume markers are inserted only around an imported native-state part')
        if seq.native_capture_open and token == marker.produce_close:
            seq.native_write_close_sample = index
            seq.generated_positions += 1
            self._native_record_input(seq, token, producing=False, sample_index=index)
            state['fed'].append(token)
            self._sglang_finish_block(seq, state)
            return
        producing = seq.native_capture_open and token != marker.produce_open
        if token == marker.produce_open:
            seq.native_write_start_own = seq.native_own_position
            seq.native_write_open_sample = index
            state['open_position'] = len(state['prompt']) + len(state['fed'])  # its cache position (imports count)
        seq.generated_positions += 1
        self._native_record_input(seq, token, producing=producing, sample_index=index)
        state['fed'].append(token)
        if token == marker.produce_open:
            if seq.native_capture_open:
                raise RequestError('neuralese-native-state-nested', 'produce spans cannot nest')
            seq.native_capture_chunks = []
            seq.native_capture_ids = []
            seq.native_capture_open = True
            seq.native_write_start_sample = index
        elif producing:
            seq.native_capture_ids.append(token)
        else:
            seq.items.append(token)
            seq.native_item_positions.append(seq.native_own_position - 1)
            self._native_guide(seq)

    def _guidance_replay(self, seq, ids, producing):
        """A guidance rewind on SGLang only restores the bookkeeping: the next request resends the retained inputs
        (radix-cached), and a produce span's state is exported whole when it closes."""
        return None

    def _native_guide(self, seq):
        rewound = super()._native_guide(seq)
        if rewound:
            seq.sglang_rewound = True
        return rewound

    def _sglang_rewound(self, seq, state):
        """After a guidance rewind: keep the retained inputs, continue the sample indices, stop this draw."""
        if not getattr(seq, 'sglang_rewound', False):
            return False
        seq.sglang_rewound = False
        state['fed'] = state['fed'][:len(seq.native_rl_inputs) - seq.native_guidance_anchor['inputs']]
        state['index'] = len(seq.rl_samples)
        state['run'] = len(seq.items)
        state['rewind'] = True
        return True

    def sglang_score(self, item, plan, continuations, bindings):
        """`decide`/`score` on SGLang (None: the native engine scores it): the native engine's decision capture, each
        continuation's raw log-probability summed from SGLang input logprobs after the shared prompt (the radix cache
        holds the prompt once; `logprob_start_len` keeps each continuation's tokens in its own extend)."""
        from .native_learning import decision_request
        request = decision_request(item, bindings)
        request.request_id = request.request_id or f"decide-{next(self._ids)}"
        if self._routes_native(request, plan):
            from .native_learning import _score_rendered
            return self._native_gated(lambda: self._native_run(
                lambda: _score_rendered(self, item, plan, continuations, bindings), bindings))
        return self._with_adapters(request, lambda adapter: self._sglang_score_bound(item, plan, continuations, request, adapter))

    def _sglang_score_bound(self, item, plan, continuations, request, adapter):
        from .native_learning import decision_response
        seq = Sequence(request, None, prompt_plan=plan)
        seq.native_actor_export = getattr(self, 'sglang_actor_export', None)  # option scores come from the export
        seq.native_served_policy = self._served_policy()
        ids = self._sglang_prompt(seq)
        custom = {**({IMPORT_KEY: seq.sglang_imports} if seq.sglang_imports else {}), **({ADAPTER_KEY: adapter} if adapter else {})}
        params = {'temperature': 0.0, 'max_new_tokens': 1, **({'custom_params': custom} if custom else {})}
        if continuations and getattr(self, 'sglang_warm_openings', True):
            opening = self._sglang_opening(item, ids)
            if opening is not None:
                self._sglang_warm_opening(ids[:opening], adapter)

        def score(continuation):
            if not continuation:
                return 0.0, 0
            result = self._sglang_generate({'input_ids': ids + list(continuation), 'sampling_params': params,
                                            'return_logprob': True, 'logprob_start_len': len(ids) - 1})
            logps = [lp for lp, _, _ in result['meta_info']['input_token_logprobs']][1:]
            if len(logps) != len(continuation):
                raise RequestError('sglang-actor-backend', 'SGLang returned the wrong number of continuation logprobs')
            return float(sum(logps)), len(continuation)
        # The first option puts the shared prompt into the radix cache; the others then score from it in parallel.
        results = [score(continuations[0]), *self._map_threads(score, continuations[1:])] if continuations else []
        with self._sglang_lock:
            self.sglang_requests += 1
        return decision_response(item, seq, continuations, results)

    # Opening reuse across inputs (`cache_messages` k: the first k messages are the function's opening, shared by all of
    # its inputs; the llama.cpp fork checkpoints its state at that message boundary, e51ca6d68). The pinned SGLang fork
    # (ace8d6868, `--mamba-radix-cache-strategy extra_buffer`) keeps LFM3's conv state in the radix tree only at track
    # points: a prefill of tokens [p, e) snapshots it once, at p + floor((e - p) / G) * G
    # (`ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend`; G = lcm(LFM3's mamba_chunk_size 256, page size) =
    # 256), and a finished request inserts its tokens only up to that snapshot (`MambaComponent.prepare_for_caching_req`
    # and `_select_finished_checkpoint` cap the key `UnifiedRadixCache.checkpoint_into_tree` inserts). A request resumes
    # only from a stored state on its own path. So the inputs of one function do not share the opening by themselves:
    # each input stores its state past the opening, inside its own input; a later input that finds the opening's KV
    # without a state prefills it all and tracks a branching state there (`MambaComponent.
    # finalize_match_result_in_tree_core`, `Req.mamba_branching_seqlen`) for the inputs after it; and the items of one
    # batch arrive together (FCFS admits them in one prefill batch), so every one of them prefills the whole opening.
    # A prefill request of exactly the opening, sent once before the items score their options, stores the state at
    # floor(n / G) * G of an n-token opening: every input then prefills at most G - 1 opening tokens plus its own.
    # A fork with `custom_params.natlang_track_input_end` (NATLANG_TRACK_INPUT_END, sglang-natlang 33230160b) lands the
    # state exactly on the boundary: its scheduler cuts the request's prefill so the last extend is a whole number of
    # grids (`PrefillAdder._natlang_input_end_extend`), whose ordinary track point is then the input end; the warm
    # request asks for it, and every input prefills only its own messages. Either way a state needs an extend of at
    # least G tokens, so an opening shorter than G is never warmed.
    sglang_state_grid = 256    # G above
    sglang_opening_ttl = 60.0  # seconds a warmed opening counts as cached; the next item after that warms it again
    sglang_opening_memo = 4096

    def _sglang_opening(self, item, ids):
        """The token length of the item's `cache_messages` opening in its prompt ids (None: no usable opening)."""
        count = item.get('cache_messages')
        messages = item.get('messages') or []
        if isinstance(count, bool) or not isinstance(count, int) or not 0 < count < len(messages):
            return None
        from .targets import opening_items
        try:
            items = opening_items(self, messages[:count], item.get('tools'))
        except RequestError:
            return None
        # An opening holding native state (imported by placeholder, not a plain token prefix) is left to the radix cache.
        if any(kind != 'tok' for kind, _ in items):
            return None
        tokens = [int(value) for _, value in items]
        # The prompt must continue the opening's rendering token for token (privileged guidance or a template that
        # renders the opening differently once more messages follow leaves nothing to warm).
        if len(tokens) < self.sglang_state_grid or len(tokens) >= len(ids) or ids[:len(tokens)] != tokens:
            return None
        return len(tokens)

    def _sglang_warm_opening(self, opening, adapter):
        """Prefill `opening` once, so SGLang's radix cache holds its KV and conv state before the items that share it
        score their options; concurrent items with the same opening wait for that one request."""
        key = hashlib.sha256(json.dumps([list(opening), adapter], sort_keys=True, default=str).encode()).hexdigest()
        now = time.monotonic()
        with self._sglang_lock:
            memo = self.__dict__.setdefault('_sglang_openings', {})
            entry = memo.get(key)
            owner = entry is None or (entry['done'] is not None and now - entry['done'] > self.sglang_opening_ttl)
            if owner:
                memo.pop(key, None)
                entry = memo[key] = {'event': threading.Event(), 'done': None}
                while len(memo) > self.sglang_opening_memo:
                    memo.pop(next(iter(memo)))
        if not owner:
            entry['event'].wait()
            return
        custom = {**({ADAPTER_KEY: adapter} if adapter else {}),
                  **({TRACK_INPUT_END_KEY: True} if getattr(self, 'sglang_track_input_end', False) else {})}
        try:
            self._sglang_generate({'input_ids': list(opening), 'sampling_params': {
                'temperature': 0.0, 'max_new_tokens': 1, **({'custom_params': custom} if custom else {})}})
            with self._sglang_lock:
                self.sglang_opening_warms = getattr(self, 'sglang_opening_warms', 0) + 1
        except Exception:  # an optimization only: the items score anyway and report a backend failure themselves
            with self._sglang_lock:
                self.sglang_opening_warm_failures = getattr(self, 'sglang_opening_warm_failures', 0) + 1
                if memo.get(key) is entry:
                    del memo[key]
        finally:
            entry['done'] = time.monotonic()
            entry['event'].set()

    def _sglang_import(self, seq, ids, block_id, *, stop_gradient, source_occurrence):
        """An imported native-state part, as the native engine's `_native_consume_impl` builds it.

        Exact state: consume_open, the block's positions as content-hashed placeholder ids whose KV and conv inputs the
        fork replaces by the stored ones (`custom_params.natlang_import`), consume_close. Imported positions occupy
        cache positions but not own positions. State of another model is re-encoded from its text, as natively."""
        block = self._sglang_store().native_state_file(block_id)   # header and token ids, never the payload
        if block is None:
            raise RequestError('neuralese-unknown-native-state', f'{block_id} is not in this server\'s store; PUT it first')
        if block.derivation is not None:
            from ..model.native_state_derivation import verify_state_derivation
            verify_state_derivation(self.store, block.load())
        use = {**({'stop_gradient': True} if stop_gradient else {}),
               **({'source_occurrence': source_occurrence} if source_occurrence else {})}
        derivation = {'state_derivation': block.derivation} if block.derivation is not None else {}
        capture = capture_enabled(seq)
        if not (seq.request.native_state_modes or {}).get(block_id) and self.reads_in_context(block_id, block) \
                and self._sglang_exact(block):
            # A text-written state or produce span: its exact tokens inline in this call's context, no consume markers
            # (`Engine.reads_in_context`, `_native_consume_text_in_context`).
            tokens = [int(token) for token in block.token_ids]
            ids.extend(tokens)
            seq.native_own_position += len(tokens)
            if capture and tokens:
                seq.native_rl_prompt.append({'kind': 'tokens', 'token_ids': tokens, 'producing': [False] * len(tokens)})
            seq.native_imports.append({'id': block_id, 'mode': TEXT_IN_CONTEXT_MODE, **derivation,
                                       'observed_native_state_id': block_id, **use})
            return
        opening, closing = int(self.role_tokens.consume_open), int(self.role_tokens.consume_close)
        ids.append(opening)
        seq.native_own_position += 1
        if capture:
            seq.native_rl_prompt.append({'kind': 'tokens', 'token_ids': [opening], 'producing': [False]})
        receipt = None
        if (request_mode := (seq.request.native_state_modes or {}).get(block_id)) is not None:
            # Same-base adapter state: the raw state of another adapter binding of this base, spliced as stored and
            # bound by its receipt, as `_native_consume_impl` does.
            from ..model.native_adapters import adapter_state_input_receipt, SAME_BASE_STATE_MODE, NativeAdapterError
            if request_mode != SAME_BASE_STATE_MODE:
                raise RequestError('neuralese-native-state-mode',
                                   'use same-base-adapter-state only for exact-base raw native state exchange')
            try:
                receipt = adapter_state_input_receipt(self, block.load(), block_id, target_position_start=len(ids),
                                                      layout=self._sglang_layout)
            except (ValueError, NativeAdapterError) as error:
                raise RequestError('neuralese-native-state-mode', str(error)) from error
        if receipt is not None or self._sglang_exact(block):
            count = len(block.token_ids)
            if count:   # an empty state imports nothing; a count-0 span made the scheduler raise and die (2026-11-10)
                seq.sglang_imports.append({'path': str(block.path), 'start': len(ids), 'count': count})
            ids.extend(self._sglang_placeholders(block_id, count))
            mode = 'exact-native-state' if receipt is None else 'same-base-adapter-state'
            consumption = {'adapter_state_consumption': receipt} if receipt is not None else {}
            if capture:
                seq.native_rl_prompt.append({'kind': 'native_state', 'id': block_id,
                                             'native_identity': self._native_block_identity(block),
                                             **consumption, **derivation, **use})
            observed = {'observed_native_state_id': block_id, **consumption}
        else:
            fallback = self._tokens(block.text)
            if not fallback and len(block.token_ids):
                raise RequestError('neuralese-native-state-empty-text', f'{block_id} has no text for target reencoding')
            ids.extend(fallback)
            seq.native_own_position += len(fallback)
            mode = 'best-effort-text-retokenization'
            if capture:
                seq.native_rl_prompt.append({'kind': 'tokens', 'token_ids': list(fallback), 'producing': [False] * len(fallback)})
            observed = {}
        ids.append(closing)
        seq.native_own_position += 1
        if capture:
            seq.native_rl_prompt.append({'kind': 'tokens', 'token_ids': [closing], 'producing': [False]})
        seq.native_imports.append({'id': block_id, 'mode': mode, **derivation, **observed, **use})

    def _sglang_layout(self, block):
        """`validate_native_state_layout` against the export (the fork checks tensor shapes when it loads them)."""
        block.validate()
        if block.architecture != self.sglang_export['architecture']:
            raise ValueError('native state architecture differs from the consumer')
        if tuple(block.metadata()['layer_kinds']) != tuple(self.sglang_export['layer_kinds']):
            raise ValueError('native state layer kinds disagree with the consumer')
        if bool((block.token_ids >= int(self.sglang_export['producer_embedding']['input_id_offset'])).any()):
            raise ValueError('native state token IDs are outside the target vocabulary')

    def _sglang_exact(self, block):
        """`_native_exact` without the native model: the layer kinds come from the export."""
        try:
            block.require_native(checkpoint_sha256=self.execution_native_identity['checkpoint_sha256'],
                                 tokenizer_fingerprint=self.model_config['tokenizer_fingerprint'],
                                 embedding_weight_sha256=self.embedding_weight_sha256,
                                 architecture=self.sglang_export['architecture'],
                                 layer_kinds=tuple(self.sglang_export['layer_kinds']))
            return True
        except ValueError:
            return False

    def _sglang_store(self):
        """The block store, which must keep files: the fork reads imported state from the stored files and writes
        captures into the store's capture directory, from which the store adopts them by rename."""
        if getattr(self.store, 'directory', None) is None:
            raise RequestError('neuralese-native-state-unavailable',
                               'the SGLang front exchanges native state as store files: serve it with --store-dir')
        return self.store

    def _sglang_scratch(self):
        """Capture files and adapter files for the fork (inside the store, so a capture is adopted by rename)."""
        if self.sglang_capture_dir is None:
            directory = getattr(self.store, 'directory', None)
            if directory is not None:
                self.sglang_capture_dir = os.path.join(directory, 'sglang-capture')
                os.makedirs(self.sglang_capture_dir, exist_ok=True)
                # Captures and partial writes left by a front that died mid-request (nothing reads them again).
                for name in os.listdir(self.sglang_capture_dir):
                    if not name.startswith('adapter-'):
                        try:
                            os.unlink(os.path.join(self.sglang_capture_dir, name))
                        except FileNotFoundError:
                            pass
            else:
                import tempfile
                self.sglang_capture_dir = tempfile.mkdtemp(prefix='sglang-capture-')
        return self.sglang_capture_dir

    def _sglang_finish_block(self, seq, state):
        """Store the span that just closed: SGLang recomputes it once and exports its native state."""
        if seq.request.on_delta is not None and seq.guide is None:
            self._sglang_emit(seq, state, final=True)  # text before the block reaches the client first
        ids = state['prompt'] + state['fed'][:-1]  # through the last span token (the close marker is not part)
        start = state['open_position'] + 1
        end = start + len(seq.native_capture_ids)
        if end != len(ids) or ids[start:end] != seq.native_capture_ids:
            raise RequestError('sglang-actor-capture', 'produce span positions disagree with the generated tokens')
        # `_native_store_capture` runs the capture once the block's text is known (`_native_finish_block`).
        seq.native_capture_chunks = [{'ids': ids, 'start': start, 'end': end, 'adapter': state['adapter'],
                                      'imports': seq.sglang_imports}]
        self._native_finish_block(seq)
        with self._sglang_lock:
            self.sglang_state_writes += 1
        state['run'], state['text'] = len(seq.items), ''

    def _native_store_capture(self, token_ids, chunks, *, text, owner=None, producer_kind=None):
        """`Engine._native_store_capture` on SGLang. `chunks` is the one capture request of the span (ids, start, end,
        adapter, imports). The fork writes the span's state as a native-state file with the metadata the native
        engine would give the block; the store adopts that file (renamed in, verified from the file: layout, finite
        values, content id, and that it holds this metadata and these token ids). Returns the stored file as block
        and record (`NativeStateFile`)."""
        if not (len(chunks) == 1 and isinstance(chunks[0], dict) and 'ids' in chunks[0]):
            # A span the native engine computed (native-routed sequences and endpoints pass `native_step` captures):
            # stored from those tensors by the native engine's own path, never re-run on SGLang (which a paused
            # engine would hold while a rebind waits for this native call to finish).
            block, record = super()._native_store_capture(token_ids, chunks, text=text, owner=owner,
                                                          producer_kind=producer_kind)
            self._note_producer(record.id, native=True)
            return block, record
        from ..model.native_adapters import execution_binding, SCHEMA as ADAPTER_SCHEMA
        from ..model.state_block import native_state_metadata
        (capture,) = chunks
        store = self._sglang_store()
        bindings = execution_binding(self)
        metadata = native_state_metadata(
            text=text, architecture=self.model_config.get('architecture') or self.sglang_export['architecture'],
            tokenizer_fingerprint=self.model_config['tokenizer_fingerprint'],
            checkpoint_sha256=self.execution_native_identity['checkpoint_sha256'],
            embedding_weight_sha256=self.embedding_weight_sha256, layer_kinds=self.sglang_export['layer_kinds'],
            adapter_binding=({'schema': ADAPTER_SCHEMA, 'base_native_identity': self.native_identity,
                              'adapters': bindings} if bindings else None), producer_kind=producer_kind)
        path = self._sglang_capture(capture['ids'], capture['start'], capture['end'], metadata,
                                    capture.get('adapter'), capture.get('imports'))
        try:
            stored = store.adopt_native_state_file(path, owner, metadata=metadata, token_ids=token_ids)
        except FileNotFoundError:
            raise RequestError('sglang-actor-capture', 'SGLang did not export the produce span') from None
        except ValueError as error:
            raise RequestError('sglang-actor-capture', f'SGLang exported an invalid produce span: {error}') from error
        finally:
            try:
                os.unlink(path)   # a refused capture (an adopted one has been renamed away)
            except FileNotFoundError:
                pass
        dtype = getattr(self, 'native_dtype', None) or str(self.backbone.embedding_weight.dtype).replace('torch.', '')
        if str(stored.dtype('embeddings')).replace('torch.', '') != dtype:
            raise RequestError('sglang-actor-capture', f'SGLang exported {stored.dtype("embeddings")} embeddings, '
                               f'the native engine stores {dtype}')
        self._note_producer(stored.id, native=False)
        return stored, stored

    PRODUCER_RECORDS = 1 << 16

    def producer_policy(self, block_id):
        """The policy version that produced a block on this front (provenance; None when unknown, e.g. a block stored
        before this front started). Exact splicing never depends on it: blocks of one lineage splice exactly. The
        episode record carries the same fact: the producing turn's capture names its in-place update
        (``_served_policy``) beside the block it stored. Receipts and write results stay the native engine's."""
        return (self.__dict__.get('_block_producers') or {}).get(block_id)

    def _note_producer(self, block_id, *, native):
        import collections
        # the version the trainer bound (SGLang's own /model_info reports it as a string)
        served = (getattr(self, '_native_served', None) if native else
                  (getattr(self, 'rl_policy_binding', None) or {}).get('in_place_update')) or {}
        records = self.__dict__.get('_block_producers')
        if records is None:
            records = self.__dict__.setdefault('_block_producers', collections.OrderedDict())
        with self._sglang_lock:
            records[block_id] = {'policy_sha256': served.get('policy_sha256') or served.get('policy_id')
                                 or self.native_identity['checkpoint_sha256'],
                                 'version': served.get('version'), 'engine': 'native' if native else 'sglang'}
            records.move_to_end(block_id)
            while len(records) > self.PRODUCER_RECORDS:
                records.popitem(last=False)

    def _sglang_capture(self, ids, start, end, metadata, adapter=None, imports=None):
        """The path of the native-state file the fork writes for ids[start:end] (`custom_params.natlang_capture`).

        One SGLang request recomputes the sequence with the radix match capped below the span (logprob_start_len), so
        the span runs through the model and the fork writes its pre-RoPE keys, values, conv inputs and input
        embeddings as a native-state file of `metadata` (the store's format)."""
        path = os.path.join(self._sglang_scratch(), f'{uuid.uuid4().hex}.safetensors')
        body = {'input_ids': ids, 'return_logprob': True, 'logprob_start_len': max(start - 1, 0),
                'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1,
                                    'custom_params': {CAPTURE_KEY: {'path': path, 'start': start, 'end': end,
                                                                    'native_state': metadata},
                                                      **({IMPORT_KEY: imports} if imports else {}),
                                                      **({ADAPTER_KEY: adapter} if adapter else {})}}}
        self._sglang_generate(body)
        return path

    def _sglang_emit(self, seq, state, final=False):
        """Send the reply text decoded so far, holding back an incomplete UTF-8 tail until it completes."""
        text = self.tokenizer.decode(seq.items[state['run']:], skip_special_tokens=False)
        if not final:
            text = text.rstrip('\ufffd')
        if len(text) > len(state['text']) and text.startswith(state['text']):
            seq.request.on_delta({'text': text[len(state['text']):]})
            state['text'] = text

    def _sglang_stream(self, body):
        """SGLang /generate as a stream of incremental chunks."""
        request = urllib.request.Request(self.sglang_url + '/generate', data=json.dumps({**adapter_cache_key(body), 'stream': True}).encode(),
                                         headers={'Content-Type': 'application/json'})
        meta = None
        try:
            with urllib.request.urlopen(request, timeout=SGLANG_REQUEST_SECONDS) as response:
                for line in response:
                    line = line.strip()
                    if not line.startswith(b'data:'):
                        continue
                    payload = line[5:].strip()
                    if payload == b'[DONE]':
                        return
                    chunk = json.loads(payload)
                    if 'error' in chunk:
                        raise RequestError('sglang-actor-backend', f"SGLang stream error: {str(chunk['error'])[:400]}")
                    meta = chunk.get('meta_info') or meta
                    yield chunk
        except urllib.error.HTTPError as error:
            raise RequestError('sglang-actor-backend', f'SGLang HTTP {error.code}: {error.read()[:400]!r}') from error
        finally:
            self._sglang_count_prompt(meta)  # also when the consumer stops early (a guided rewind)


def attest_all_routes(info, expected):
    """Check the measured route attestation of ``/v1/neuralese/info`` against what the trainer published. ``expected``:
    ``policy_id``, ``version``, ``weights_sha256`` (SHA-256 of the export's ``model.safetensors``), ``model_path`` (the
    export directory) and ``tensor_identity`` (``export_tensors_identity`` of the export). SGLang's three routes must
    report, measured, that directory and version and that file digest; the native route the same policy and version
    and, resident, the reloaded model's tensor identity (else the pending source's digest and the claimed identity
    its lazy load verifies). Returns ``(ok, mismatches)``."""
    routes = (info or {}).get('routes')
    if not isinstance(routes, dict):
        return False, {'routes': 'missing routes attestation in /v1/neuralese/info'}
    mismatches = {}
    version = None if expected.get('version') is None else str(expected['version'])

    def differ(name, field, want, got):
        if want is not None and got != want:
            mismatches.setdefault(name, {})[field] = {'expected': want, 'measured': got}
    for name in ('generation', 'decision', 'capture'):
        entry = routes.get(name)
        if not isinstance(entry, dict):
            mismatches[name] = 'missing'
            continue
        if not entry.get('measured'):
            mismatches[name] = {'measured': False, 'error': entry.get('error')}
            continue
        differ(name, 'policy_sha256', expected.get('policy_id'), entry.get('policy_sha256'))
        differ(name, 'version', version, None if entry.get('version') is None else str(entry['version']))
        if expected.get('model_path') is not None:
            differ(name, 'model_path', os.path.realpath(expected['model_path']),
                   os.path.realpath(entry['model_path']) if entry.get('model_path') else None)
        differ(name, 'weights_sha256', expected.get('weights_sha256'), entry.get('weights_sha256'))
    native = routes.get('native_fallback')
    if not isinstance(native, dict):
        mismatches['native_fallback'] = 'missing'
    else:
        differ('native_fallback', 'policy_sha256', expected.get('policy_id'), native.get('policy_sha256'))
        differ('native_fallback', 'version', version,
               None if native.get('version') is None else str(native['version']))
        if native.get('resident') and native.get('tensor_identity') is not None:
            differ('native_fallback', 'tensor_identity', expected.get('tensor_identity'), native.get('tensor_identity'))
        elif native.get('resident'):
            mismatches.setdefault('native_fallback', {})['tensor_identity'] = 'resident but not reloaded'
        else:
            differ('native_fallback', 'source_weights_sha256', expected.get('weights_sha256'),
                   native.get('source_weights_sha256'))
            differ('native_fallback', 'pending_tensor_identity', expected.get('tensor_identity'),
                   native.get('pending_tensor_identity'))
    return not mismatches, mismatches
