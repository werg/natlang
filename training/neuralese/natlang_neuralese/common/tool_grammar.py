"""Tool-call grammar: the declared tool schemas as a decoding constraint, one law for every engine that captures training
samples, and its exact replay in the learner (docs/ENGINE_PARITY.md "Result schemas: tuples", DECISIONS 2026-11-23
"constrained decoding with an exact learner law").

The managed llama-server and the page constrain a tool call's arguments with the declared JSON schema (llama.cpp's
autoparser grammar, triggered at `<tool_call>`). The Python native engine and the SGLang actor apply the same constraint
with one grammar built here: an xgrammar structural tag (`tool_call_law`) in which free text runs unconstrained until a
`<tool_call>` opens, the function name must be a declared tool, and its parameters follow that tool's schema in the
backbone's call format (`serve.chat.TOOL_CALL_FORMATS`: LFM3 / Qwen3.5 `xml`, Qwen3 `json`; the `pythonic` LFM2 form has
no grammar).

A grammar mask applied before sampling changes the law a token is drawn from: the engines record the masked,
renormalized log-probability. The learner must score the same law, so a capture declares it in
`sampling_law.tool_grammar` (`tool_call_law`: the structural tag itself, its sha256, the compiler and the tokenizer
identity the mask was computed over) and every sampled token of that reply was drawn under it. The mask at a position
is a function of the grammar and the reply tokens before it, so the learner recomputes it exactly (`replay_masks`) from
the reply's own tokens (`prefix`: tokens fed before the first sample, e.g. a templated call's opening) and applies it to
the policy, reference and behaviour-ratio log-probabilities (`train.lfm3_fast.HeadMask`).

`NATLANG_TOOL_GRAMMAR=0` on a serving front opts out: replies are then unconstrained and record no grammar.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache

import numpy as np

SCHEMA = 'natlang.tool-grammar/1'
COMPILER = 'xgrammar'
ENV = 'NATLANG_TOOL_GRAMMAR'
FORMATS = ('xml', 'json')
TRIGGER = '<tool_call>'


def enabled() -> bool:
    """The serving default: constrain tool calls unless `NATLANG_TOOL_GRAMMAR` is 0/false/off, or the compiler is not
    installed (a serving host without xgrammar serves unconstrained and says so once, rather than failing requests)."""
    if os.environ.get(ENV, '1').strip().lower() in ('0', 'false', 'off', 'no'):
        return False
    return available()


_AVAILABLE = []


def available() -> bool:
    if not _AVAILABLE:
        try:
            import xgrammar  # noqa: F401
            _AVAILABLE.append(True)
        except ImportError as error:
            import sys
            print(json.dumps({'event': 'tool_grammar_unavailable', 'error': str(error)}), file=sys.stderr, flush=True)
            _AVAILABLE.append(False)
    return _AVAILABLE[0]


def compiler_version() -> str:
    import importlib.metadata
    return f'{COMPILER}=={importlib.metadata.version(COMPILER)}'


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _functions(tools):
    """(name, parameters schema) of each OpenAI-style function tool, in order; a tool without parameters takes an empty
    object."""
    out = []
    for tool in tools or ():
        fn = tool.get('function', tool) if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get('name'), str) or not fn['name']:
            continue
        parameters = fn.get('parameters')
        if not isinstance(parameters, dict) or not parameters:
            parameters = {'type': 'object', 'properties': {}}
        out.append((fn['name'], parameters))
    return out


def structural_tag(tools, call_format):
    """The xgrammar structural tag of a reply that may call `tools` in `call_format`, or None (no tools, or a format
    without a grammar). Free text is unconstrained; a `<tool_call>` must name a declared tool and give its arguments
    under its schema; any number of calls (llama.cpp's parallel tool calls)."""
    if call_format not in FORMATS:
        return None
    functions = _functions(tools)
    if not functions:
        return None
    tags = []
    for name, parameters in functions:
        if call_format == 'xml':
            # `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>`
            # (serve.chat.TOOL_CALL_FORMATS; xgrammar's `qwen_xml` style: a string value verbatim, any other JSON)
            tags.append({'type': 'tag', 'begin': f'{TRIGGER}\n<function={name}>\n',
                         'content': {'type': 'json_schema', 'json_schema': parameters, 'style': 'qwen_xml'},
                         'end': '</function>\n</tool_call>'})
        else:
            # `<tool_call>\n{"name": NAME, "arguments": {...}}\n</tool_call>`
            tags.append({'type': 'tag', 'begin': f'{TRIGGER}\n{{"name": {json.dumps(name, ensure_ascii=False)}, '
                                                 '"arguments": ',
                         'content': {'type': 'json_schema', 'json_schema': parameters, 'style': 'json'},
                         'end': '}\n</tool_call>'})
    return {'type': 'structural_tag', 'format': {'type': 'triggered_tags', 'triggers': [TRIGGER], 'tags': tags,
                                                  'at_least_one': False, 'stop_after_first': False}}


def tokenizer_info(tokenizer, vocab_size, stop_token_ids):
    """The xgrammar tokenizer of a served model, as SGLang's xgrammar backend builds it (`TokenizerInfo.from_huggingface`
    with the model's vocabulary width and its EOS ids)."""
    import xgrammar
    return xgrammar.TokenizerInfo.from_huggingface(tokenizer, vocab_size=int(vocab_size),
                                                   stop_token_ids=sorted(int(t) for t in stop_token_ids))


_INFO_SHA, INFO_SHA_LIMIT = {}, 16


def tokenizer_info_sha256(info) -> str:
    """Identity of what a mask is computed over: every token's decoded bytes, the vocabulary type and width, the stop
    and special ids. Memoized per TokenizerInfo object (immutable): hashing a 65k vocabulary is ~0.26 s of GIL-held
    Python, and the SGLang front asked for it twice per chat call, which starved its 128 collection streams (aei1
    collection stall, 2026-11-24). ``NATLANG_TOOL_GRAMMAR_INFO_MEMO=0`` recomputes it on every call."""
    memo = os.environ.get('NATLANG_TOOL_GRAMMAR_INFO_MEMO', '1') != '0'
    entry = _INFO_SHA.get(id(info)) if memo else None
    if entry is not None and entry[0] is info:
        return entry[1]
    value = _tokenizer_info_sha256(info)
    if memo:
        while len(_INFO_SHA) >= INFO_SHA_LIMIT:
            _INFO_SHA.pop(next(iter(_INFO_SHA)))
        _INFO_SHA[id(info)] = (info, value)    # the reference keeps the id from being reused while memoized
    return value


def _tokenizer_info_sha256(info) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical({'vocab_type': str(info.vocab_type), 'vocab_size': int(info.vocab_size),
                              'add_prefix_space': bool(info.add_prefix_space),
                              'stop_token_ids': sorted(int(t) for t in info.stop_token_ids),
                              'special_token_ids': sorted(int(t) for t in info.special_token_ids)}).encode())
    for piece in info.decoded_vocab:
        digest.update(len(piece).to_bytes(4, 'little'))
        digest.update(piece)
    return digest.hexdigest()


def eos_ids(config, generation_config=None):
    """The model's EOS ids (config `eos_token_id` plus a generation config's): SGLang's `hf_eos_token_id`, the xgrammar
    stop tokens."""
    found = set()
    for source in (config, generation_config):
        value = (source.get('eos_token_id') if isinstance(source, dict) else getattr(source, 'eos_token_id', None)) \
            if source is not None else None
        if isinstance(value, int):
            found.add(value)
        elif isinstance(value, (list, tuple)):
            found.update(int(v) for v in value)
    return found


def tool_call_law(tools, call_format, info, info_sha256=None):
    """The capture's `sampling_law.tool_grammar` of a reply that calls `tools`, or None when nothing is constrained."""
    tag = structural_tag(tools, call_format)
    if tag is None:
        return None
    text = _canonical(tag)
    return {'schema': SCHEMA, 'compiler': compiler_version(), 'kind': 'structural_tag', 'grammar': text,
            'sha256': hashlib.sha256(text.encode()).hexdigest(),
            'tokenizer_info_sha256': info_sha256 or tokenizer_info_sha256(info)}


def validate_law(law):
    """Refuse a declared grammar this module cannot replay exactly."""
    if not isinstance(law, dict) or law.get('schema') != SCHEMA or law.get('kind') != 'structural_tag':
        raise ValueError(f'unknown tool grammar declaration {law!r:.200}')
    text = law.get('grammar')
    if not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != law.get('sha256'):
        raise ValueError('tool grammar text does not match its sha256')
    if law.get('compiler') != compiler_version():
        raise ValueError(f'tool grammar compiled by {law.get("compiler")}, the replay has {compiler_version()}')
    return law


_COMPILERS = {}


def _compiler(info):
    import xgrammar
    key = id(info)
    entry = _COMPILERS.get(key)
    if entry is None or entry[0] is not info:
        entry = _COMPILERS[key] = (info, xgrammar.GrammarCompiler(info, cache_enabled=True))
    return entry[1]


def compiled(law, info):
    """The compiled grammar of a declared law over `info` (the compiler caches by grammar text)."""
    return _compiler(info).compile_structural_tag(law['grammar'])


class Matcher:
    """One reply's grammar state: the tokens fed so far (sampled or forced, in order) and the mask of the next draw.
    `sync` follows a reply whose tokens were rolled back (a guidance rewind) by replaying them."""

    def __init__(self, law, info):
        import xgrammar
        self.law, self.info = law, info
        self.matcher = xgrammar.GrammarMatcher(compiled(law, info))
        self.tokens: list[int] = []
        self.failed = False

    def accept(self, token: int) -> bool:
        if self.failed:
            return False
        if self.matcher.is_terminated() or not self.matcher.accept_token(int(token)):
            self.failed = True
            return False
        self.tokens.append(int(token))
        return True

    def sync(self, tokens) -> bool:
        """Bring the state to exactly `tokens` (a reply prefix); False when the grammar rejects them."""
        tokens = [int(t) for t in tokens]
        if self.failed or tokens[:len(self.tokens)] != self.tokens:
            self.matcher.reset()
            self.tokens, self.failed = [], False
        for token in tokens[len(self.tokens):]:
            if not self.accept(token):
                return False
        return True

    def bitmask(self):
        """The next draw's allowed set as xgrammar's packed int32 bitmask [ceil(V / 32)] (a torch tensor)."""
        import xgrammar
        mask = xgrammar.allocate_token_bitmask(1, self.info.vocab_size)
        self.matcher.fill_next_token_bitmask(mask, 0)
        return mask[0]


def unpack(bitmask, width):
    """bool [.., width] of packed int32 bitmask rows [.., ceil(width / 32)] (xgrammar's layout: bit b of word w is
    token 32 w + b)."""
    import torch
    shifts = torch.arange(32, device=bitmask.device, dtype=torch.int32)
    bits = (bitmask.to(torch.int32).unsqueeze(-1) >> shifts) & 1
    return bits.reshape(*bitmask.shape[:-1], -1)[..., :width].bool()


def allowed(bitmask, token: int) -> bool:
    return bool((int(bitmask[token // 32]) >> (token % 32)) & 1)


_INTERN, INTERN_LIMIT = {}, 16384


def _interned(mask):
    """One shared read-only array per distinct mask (a step's calls repeat a few hundred grammar states; each mask is
    ceil(V / 32) words, 16 KB for LFM3), so rows caching their masks hold references, not copies."""
    import hashlib as _hashlib
    key = _hashlib.blake2b(mask.tobytes(), digest_size=16).digest()
    found = _INTERN.get(key)
    if found is None:
        if len(_INTERN) >= INTERN_LIMIT:
            _INTERN.pop(next(iter(_INTERN)))
        found = _INTERN[key] = mask.copy()
        found.flags.writeable = False
    return found


def replay_masks(law, tokens, info, *, prefix=()):
    """The masks a reply's sampled `tokens` were drawn under: (table: the distinct masks, int32 [ceil(V / 32)] each,
    shared read-only arrays; index int32 [len(tokens)] into it). The grammar first takes `prefix` (reply tokens fed before the first sample).
    Raises ValueError when a token is outside its mask (the capture does not follow its declared law)."""
    import torch
    validate_law(law)
    matcher = Matcher(law, info)
    if not matcher.sync(prefix):
        raise ValueError('the reply prefix is outside its declared tool grammar')
    table, seen, index = [], {}, np.empty(len(tokens), np.int32)
    for position, token in enumerate(tokens):
        mask = matcher.bitmask()
        key = mask.numpy().tobytes()
        slot = seen.get(key)
        if slot is None:
            slot = seen[key] = len(table)
            table.append(_interned(mask.numpy()))
        index[position] = slot
        if not allowed(mask, int(token)) or not matcher.accept(int(token)):
            raise ValueError(f'sampled token {int(token)} at reply position {position} is outside its tool grammar')
    width = (int(info.vocab_size) + 31) // 32
    return (table if table else np.zeros((0, width), np.int32)), index


@lru_cache(maxsize=8)
def _info_of(path: str, vocab_size: int, stops: tuple):
    from transformers import AutoTokenizer
    return tokenizer_info(AutoTokenizer.from_pretrained(path), vocab_size, stops)


def info_from_export(export_dir):
    """(TokenizerInfo, sha256) of an exported model directory as SGLang loads it: its tokenizer files, config
    `vocab_size` and EOS ids (config and generation config). The SGLang front calls this for every chat call: the
    tokenizer and its identity are built once per directory (`_info_of`, `tokenizer_info_sha256` memoized)."""
    from pathlib import Path
    export_dir = Path(export_dir)
    config = json.loads((export_dir / 'config.json').read_text())
    generation = export_dir / 'generation_config.json'
    stops = eos_ids(config, json.loads(generation.read_text()) if generation.exists() else None)
    info = _info_of(str(export_dir), int(config['vocab_size']), tuple(sorted(stops)))
    return info, tokenizer_info_sha256(info)


@lru_cache(maxsize=4)
def _base_info(base: str):
    from pathlib import Path
    from transformers import AutoTokenizer
    from ..model.shared_state import provision_role_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base)
    provision_role_tokenizer(tokenizer)
    config = json.loads((Path(base) / 'config.json').read_text())
    generation = Path(base) / 'generation_config.json'
    info = tokenizer_info(tokenizer, int(config['vocab_size']),
                          eos_ids(config, json.loads(generation.read_text()) if generation.exists() else None))
    return info, tokenizer_info_sha256(info)


def info_from_base(base):
    """The learner's (TokenizerInfo, sha256): the HF base snapshot's tokenizer with the four Neuralese role markers
    provisioned, which is the served export's tokenizer (`export.native_hf` saves it so), its config vocabulary width and
    EOS ids. A law recorded over another tokenizer is refused where it is replayed."""
    return _base_info(str(base))


def row_masks(row, info, info_sha256, vocab_size):
    """(masks: list of int32 [ceil(V / 32)], index int32 [len(sample)], -1 where unmasked) of the sampling-law masks of one
    lowered call (`lower_rollout_turns.sampled_call`): its tool grammar over the sampled tokens after `grammar_prefix`,
    and its guidance bans (`bans`: {sample index: token ids}) cleared from that position's mask. Cached on the row."""
    cached = row.get('_law_masks')
    if cached is not None:
        return cached
    sample = np.asarray(row['sample'])
    width = (int(vocab_size) + 31) // 32
    index = np.full(len(sample), -1, np.int32)
    tables = []
    law = row.get('tool_grammar')
    if law is not None:
        if info is None:
            raise ValueError('a call drawn under a tool grammar needs the learner tokenizer (tool_grammar.info_from_base)')
        if law.get('tokenizer_info_sha256') != info_sha256:
            raise ValueError(f'tool grammar recorded over tokenizer {str(law.get("tokenizer_info_sha256"))[:12]}, '
                             f'the learner has {info_sha256[:12]}')
        table, index = replay_masks(law, sample, info, prefix=row.get('grammar_prefix') if row.get(
            'grammar_prefix') is not None else ())
        tables = list(table)
        if any(mask.shape[-1] != width for mask in tables):
            raise ValueError('tool grammar mask width differs from the learner vocabulary')
    full = None
    for position, banned in sorted((int(k), v) for k, v in (row.get('bans') or {}).items()):
        if index[position] >= 0:
            mask = tables[index[position]].copy()
        else:
            if full is None:   # every token of the vocabulary, no padding bit beyond it
                full = np.full(width, 0xFFFFFFFF, np.uint32)
                extra = width * 32 - int(vocab_size)
                if extra:
                    full[-1] = np.uint32((1 << (32 - extra)) - 1)
                full = full.view(np.int32)
            mask = full.copy()
        words = mask.view(np.uint32)
        for token in banned:
            words[int(token) // 32] &= np.uint32(~(1 << (int(token) % 32)) & 0xFFFFFFFF)
        index[position] = len(tables)
        tables.append(_interned(mask))
    out = (tables, index)       # a list of shared masks (``_interned``) and each sample's slot, -1 unmasked
    row['_law_masks'] = out
    return out
