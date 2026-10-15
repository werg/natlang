"""Exact source-bound cross-tokenizer pair admission, independent of VAE training."""
from __future__ import annotations

from .native_graph_schema import parse_graph

PAIR_SCHEMA = 'natlang.native-semantic-pair/1'


def parse_pair(row, participants, *, required_split=None):
    """Verify the same authenticated payload across actual model tokenizers."""
    if row.get('schema') != PAIR_SCHEMA or not isinstance(row.get('id'), str) or not row['id']:
        raise ValueError('semantic pair requires its schema and source identity')
    if row.get('split') not in {'train', 'test'} or required_split and row['split'] != required_split:
        raise ValueError('semantic pair split differs from its declared input')
    groups = row.get('source_groups')
    if not isinstance(groups, list) or not groups or any(not isinstance(x, str) or not x for x in groups):
        raise ValueError('semantic pair requires explicit source groups')
    models = row.get('models')
    if not isinstance(models, dict) or len(models) < 2:
        raise ValueError('semantic pair must describe at least two model trajectories')
    from ..common.hashing import canonical_json_sha256_hex
    communication = row.get('communication') or {}
    if (any(not isinstance(communication.get(key), str) or not communication[key]
            for key in ('source_id', 'writer_record_id', 'reader_record_id', 'write_name', 'occurrence'))
            or not isinstance(communication.get('source_binding'), dict)
            or canonical_json_sha256_hex(communication) != row.get('communication_sha256')):
        raise ValueError('semantic pair requires its exact shared source communication occurrence')
    def source_digests(value):
        found = []
        for key, item in value.items():
            if isinstance(item, dict):
                found.extend(source_digests(item))
            elif key.endswith('sha256'):
                if not isinstance(item, str) or len(item) != 64 or any(c not in '0123456789abcdef' for c in item):
                    raise ValueError('semantic communication immutable source digest is malformed')
                found.append(item)
        return found
    if not source_digests(communication['source_binding']):
        raise ValueError('semantic communication requires its immutable source artifact/manifest digest')
    parsed, texts = {}, set()
    for name in sorted(set(participants) & set(models)):
        view, participant = models[name], participants[name]
        example = parse_graph(view['graph'], tokenizer_fingerprint=participant.config['tokenizer_fingerprint'])
        if example.split != row['split'] or set(example.source_groups) != set(groups):
            raise ValueError('semantic graph split/source groups differ from the paired source')
        export = view['export']
        if not isinstance(export.get('text'), str):
            raise ValueError('semantic export must retain exact text')
        # A separate source witness is required even if decoded text happens
        # to match. Pairing unrelated answers by their token shape is invalid.
        witness = view.get('witness') or {}
        if (witness.get('payload_sha256') != row.get('payload_sha256')
                or witness.get('source_id') != communication['source_id']
                or witness.get('communication_sha256') != row['communication_sha256']):
            raise ValueError('semantic pair needs its source-authenticated payload witness')
        from ..common.hashing import sha256_hex
        if sha256_hex(export['text'].encode()) != row.get('payload_sha256'):
            raise ValueError('semantic payload text differs from its authenticated digest')
        start, stop = export.get('start'), export.get('stop')
        if type(start) is not int or type(stop) is not int or not 0 <= start <= stop:
            raise ValueError('semantic export needs a declared native span')
        if type(export.get('logical', False)) is not bool:
            raise ValueError('semantic export logical addressing must be explicit boolean')
        consumers = view.get('consumer_nodes')
        names = {node.name for node in example.plan.invocations}
        if (not isinstance(consumers, list) or not consumers or len(set(consumers)) != len(consumers)
                or not set(consumers) <= names or export['node'] not in names):
            raise ValueError('semantic pair needs existing, distinct consuming nodes and export node')
        offsets = export.get('offsets')
        if not isinstance(offsets, list) or bool(offsets) != bool(export['text']):
            raise ValueError('semantic export must retain exact token character offsets')
        if any(len(pair) != 2 or any(type(x) is not int for x in pair)
               or not 0 <= pair[0] <= pair[1] <= len(export['text']) for pair in offsets):
            raise ValueError('semantic token offsets must address the retained payload text')
        # State and exact payload tokenization are checked against the actual
        # graph result below, including imported aliases in logical exports.
        texts.add(export['text'])
        parsed[name] = (example, view)
    if len(texts) > 1:
        raise ValueError('semantic pair model trajectories do not contain the same payload text')
    return parsed


