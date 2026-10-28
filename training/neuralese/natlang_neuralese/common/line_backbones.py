"""The one declaration of which student backbones belong to each training line.

A training line (``lfm``, ``mellum``) is a stable role name; the backbone behind
it can change (owner, 2026-10-28: the LFM line moves from LFM2.5-350M to
LFM3-1.7B).  Historical rows, receipts and checkpoints keep validating against
the backbone that produced them, so every identity constant lives here, keyed by
backbone, and callers look it up instead of hard-coding ``lfm2``/``lfm25``/a
tokenizer fingerprint.

Facts, not policy: the tokenizer fingerprints are the values
``data.text_corpus.tokenizer_fingerprint`` computes for the pinned tokenizer
before (``source_fingerprint``) and after (``expanded_fingerprint``) the four
shared role markers are provisioned with ``model.shared_state.
provision_role_tokenizer``.  ``tests/test_line_backbones.py`` recomputes them
from the pinned snapshots when those exist locally.

Stdlib only, so checkout scripts can import it without torch.
"""
from __future__ import annotations

from itertools import product
from types import MappingProxyType

CURRENT = 'current'
HISTORICAL = 'historical'

_BACKBONES = {
    'lfm25': {
        'line': 'lfm', 'status': HISTORICAL,
        # Registry/model-key spelling used in corpus renders, receipts and native_graph_views.
        'model_key': 'lfm25',
        # Directory under the production root that holds this backbone's lineage.
        'production_dir': 'lfm25',
        # Prefix of ``<architecture>:<base inventory sha>`` model ids.
        'model_id_architecture': 'lfm2',
        # ``config.architecture`` values a checkpoint of this backbone may carry.
        'architectures': ('lfm2', 'lfm2_moe'),
        'hf_repo': 'LiquidAI/LFM2.5-350M',
        'tokenizer_snapshot': '9e6c6ccf47cd318696e137d381a7ded8fe4df09f',
        'source_fingerprint': 'e52de346c5a9d6fe666b5a412be077525daee0ee40f81a91c36728cfb4cbf008',
        'expanded_fingerprint': 'be949ec48f44e17b2b17e1131aeb1dab4262175ba20f9783ecd8e59c27a58105',
        'base_inventory_sha256': 'f40d7a5bf651a03618ae8c06bda24a180e22fa79de964e55862c2c4984249da6',
        'vocab_size': 64402,
        'role_token_ids': (64402, 64403, 64404, 64405),
        'evaluator_sandbox_id': '393f9b75-2666-456c-a850-1b46a8b7d147',
        # History-reasoning policy the LFM2.5 data conversions explicitly bound into their receipts.
        'explicit_history_reasoning_policy': 'keep',
        # Dedicated serving port of the native deployment (native_deployment.py).
        'port': 8097,
    },
    'lfm3': {
        'line': 'lfm', 'status': CURRENT,
        'model_key': 'lfm3',
        'production_dir': 'lfm3',
        'model_id_architecture': 'lfm3',
        'architectures': ('lfm3',),
        'hf_repo': 'LiquidAI/LFM3-1.7B',
        'revision': '40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd',
        'tokenizer_snapshot': '40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd',
        'tokenizer_json_sha256': '7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701',
        'weights_sha256': '135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330',
        'source_fingerprint': '8818b793a02e560d72d2bafee3a1ca9cfedd9b84518bce89800fc4db80ca967a',
        'expanded_fingerprint': '133504d493a77e3121f62cc8c83699cb222ee993c46959d604c4b683a2ffdaf3',
        # canonical_json_sha256_hex(model.native_model.base_inventory(<pinned snapshot>)): the id of the pinned snapshot
        # as downloaded. A recipe whose base directory differs (extra or vendored files) records its own inventory.
        'base_inventory_sha256': 'baf2f497392a14aff7dfd0753184e620c934b547ef9784a995548bda5c5763e5',
        'vocab_size': 125033,
        'role_token_ids': (125033, 125034, 125035, 125036),
        # No evaluator sandbox is provisioned for LFM3 yet; never reuse the LFM2.5 one.
        'evaluator_sandbox_id': None,
        # The LFM3 history policy is the backbone's serve/chat declaration (BACKBONE_HISTORY_REASONING), not a
        # data-prep binding yet: converters record no explicit policy until that declaration is adopted here.
        'explicit_history_reasoning_policy': None,
        'port': 8097,
    },
    'mellum21': {
        'line': 'mellum', 'status': CURRENT,
        'model_key': 'mellum21',
        'production_dir': 'mellum21',
        'model_id_architecture': 'mellum',
        'architectures': ('mellum', 'maple'),
        'hf_repo': None,
        'base_inventory_sha256': 'bc700045baee1c7ea1869675c38efedd59d9db09975ea0b38cff8db152d6148f',
        'source_fingerprint': '1abe5e6333be0157f76f5484b389e5586b0e6dbb9bc776f54ca9fd62fce6cb56',
        'expanded_fingerprint': 'fbf22d8773d09be802cb9977895067db36cdfba747fcb24b02ddbf1312d3778f',
        'evaluator_sandbox_id': 'ee341c7b-a425-4b4b-ad7e-f3cf7cbc2369',
        'explicit_history_reasoning_policy': None,
        'port': 8098,
    },
}
BACKBONES = MappingProxyType({key: MappingProxyType(value) for key, value in _BACKBONES.items()})
LINES = ('lfm', 'mellum')


def backbone(key: str):
    try:
        return BACKBONES[key]
    except KeyError:
        raise ValueError(f'unknown line backbone {key!r}; declared: {sorted(BACKBONES)}') from None


def keys(line: str | None = None, *, status: str | None = None) -> tuple[str, ...]:
    """Backbone keys, optionally of one line and/or status, in declaration order."""
    if line is not None and line not in LINES:
        raise ValueError(f'unknown training line {line!r}')
    return tuple(key for key, value in BACKBONES.items()
                 if (line is None or value['line'] == line) and (status is None or value['status'] == status))


def current_key(line: str) -> str:
    """The backbone key new work on ``line`` uses."""
    current = keys(line, status=CURRENT)
    if len(current) != 1:
        raise ValueError(f'line {line!r} must declare exactly one current backbone')
    return current[0]


def line_of(key: str) -> str:
    return backbone(key)['line']


def architectures(line: str, *, status: str | None = None) -> frozenset[str]:
    """Every ``config.architecture`` a checkpoint of ``line`` may carry (all backbones unless ``status``)."""
    return frozenset(name for key in keys(line, status=status) for name in BACKBONES[key]['architectures'])


def all_architectures() -> tuple[str, ...]:
    """Every declared ``config.architecture`` (declaration order), for CLI choices and dispatch sets."""
    return tuple(dict.fromkeys(name for value in BACKBONES.values() for name in value['architectures']))


def evaluator_sandbox_ids(line: str) -> tuple[str, ...]:
    """Declared evaluator sandbox ids of ``line``'s backbones, current backbone first."""
    ordered = sorted(keys(line), key=lambda key: BACKBONES[key]['status'] != CURRENT)
    return tuple(dict.fromkeys(BACKBONES[key]['evaluator_sandbox_id'] for key in ordered
                               if BACKBONES[key]['evaluator_sandbox_id']))


def model_id_architectures(line: str) -> frozenset[str]:
    return frozenset(BACKBONES[key]['model_id_architecture'] for key in keys(line))


def key_for_architecture(architecture: str) -> str:
    """The backbone key whose checkpoints carry ``config.architecture == architecture``."""
    matches = [key for key, value in BACKBONES.items() if architecture in value['architectures']]
    if len(matches) != 1:
        raise ValueError(f'architecture {architecture!r} does not identify exactly one line backbone')
    return matches[0]


def key_for_model_id(model_id: str) -> str:
    """Backbone key of a ``<architecture>:<inventory sha>`` model id."""
    prefix = model_id.split(':', 1)[0] if isinstance(model_id, str) and ':' in model_id else None
    matches = [key for key, value in BACKBONES.items() if value['model_id_architecture'] == prefix]
    if len(matches) != 1:
        raise ValueError(f'model id {model_id!r} does not identify exactly one line backbone')
    return matches[0]


def model_id_prefix(key: str) -> str:
    return backbone(key)['model_id_architecture'] + ':'


def model_id(key: str) -> str:
    """``<architecture>:<base inventory sha>`` of ``key``'s declared base, e.g. ``lfm3:<inventory sha>``."""
    sha = backbone(key).get('base_inventory_sha256')
    if not sha:
        raise ValueError(f'line backbone {key!r} declares no base inventory')
    return model_id_prefix(key) + sha


def key_for_role(role: str, *, model_id: str | None = None, architecture: str | None = None,
                 production_dir: str | None = None) -> str:
    """Resolve a line role to a backbone key from the strongest identity at hand.

    A recorded model id or architecture selects the (possibly historical) backbone;
    without either, the line's current backbone is used.
    """
    if role not in LINES:
        raise ValueError(f'unknown training line {role!r}')
    if model_id is not None:
        if isinstance(model_id, str) and model_id.split(':', 1)[0] == role:
            return current_key(role)  # role-prefixed id carries no architecture: the line's current backbone
        key = key_for_model_id(model_id)
    elif architecture is not None:
        key = key_for_architecture(architecture)
    elif production_dir is not None:
        matches = [k for k in keys(role) if BACKBONES[k]['production_dir'] == production_dir]
        if len(matches) != 1:
            raise ValueError(f'production directory {production_dir!r} is not a {role} backbone')
        key = matches[0]
    else:
        return current_key(role)
    if line_of(key) != role:
        raise ValueError(f'{key} is a {line_of(key)} backbone, not {role}')
    return key


def model_key_pairs() -> tuple[frozenset[str], ...]:
    """Every admissible (lfm backbone, mellum backbone) pairing of a two-line artifact."""
    return tuple(frozenset(pair) for pair in product(keys('lfm'), keys('mellum')))


def current_model_keys() -> tuple[str, ...]:
    return tuple(current_key(line) for line in LINES)


def tokenizer_contract(key: str) -> dict:
    """The exact {source,expanded} fingerprints recorded for ``key``'s tokenizer."""
    value = backbone(key)
    return {'source_fingerprint': value['source_fingerprint'],
            'expanded_fingerprint': value['expanded_fingerprint']}


def key_for_expanded_fingerprint(fingerprint: str) -> str:
    matches = [key for key, value in BACKBONES.items() if value['expanded_fingerprint'] == fingerprint]
    if len(matches) != 1:
        raise ValueError('tokenizer fingerprint does not identify exactly one declared line backbone')
    return matches[0]
