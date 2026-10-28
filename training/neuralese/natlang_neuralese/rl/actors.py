"""Declared RL actors: stale or quantized exports of earlier checkpoints in the learner's lineage.

A capture normally names the learner's exact native identity. An asynchronous or quantized actor instead samples
from checkpoint X (possibly a quantized export of it) while the learner trains checkpoint Y. Its captures name X's
native identity; this declaration admits them::

    {"schema": "natlang.rl-actors/1", "learner_checkpoint_sha256": Y,
     "actors": {X: {"lineage_step": 1200, "export": null | {"kind": "quantized", "format": "gguf-q8_0",
                                                           "sha256": <export file>}}}}

Every other native-identity field (tokenizer, markers, architecture) must equal the learner's, so the learner
recomputes the whole episode graph under its own weights. The behavior log-probabilities and top-k receipts the
actor captured are its actual sampling distribution, so the off-policy gap is corrected by the loss's importance
weights (TIS/MIS for score-centered and JOLT+, the behavior ratio for the clipped loss).
"""
import json
import re

SCHEMA = 'natlang.rl-actors/1'
EXPORT_KINDS = ('exact', 'quantized')
_SHA = re.compile('[0-9a-f]{64}')


def actor_declaration(value, *, learner_identity):
    """Canonical declaration bound to ``learner_identity``, or None for ``None``."""
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    learner = learner_identity['checkpoint_sha256']
    if (not isinstance(value, dict) or set(value) != {'schema', 'learner_checkpoint_sha256', 'actors'}
            or value['schema'] != SCHEMA or value['learner_checkpoint_sha256'] != learner
            or not isinstance(value['actors'], dict) or not value['actors']):
        raise ValueError('RL actor declaration needs its schema, exact learner checkpoint and actor inventory')
    actors = {}
    for sha, actor in value['actors'].items():
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise ValueError('declared RL actor must name an exact checkpoint sha256')
        if (not isinstance(actor, dict) or set(actor) != {'lineage_step', 'export'}
                or type(actor['lineage_step']) is not int or actor['lineage_step'] < 0):
            raise ValueError('declared RL actor needs its lineage step and export')
        export = actor['export']
        if export is not None and (not isinstance(export, dict) or set(export) != {'kind', 'format', 'sha256'}
                                   or export['kind'] not in EXPORT_KINDS
                                   or not isinstance(export['format'], str) or not export['format']
                                   or not isinstance(export['sha256'], str) or not _SHA.fullmatch(export['sha256'])):
            raise ValueError('declared RL actor export needs kind (exact|quantized), format and sha256')
        if sha == learner and (export is None or export['kind'] != 'quantized'):
            raise ValueError('same-checkpoint actor declaration requires an exact quantized export binding')
        actors[sha] = {'lineage_step': actor['lineage_step'], 'export': None if export is None else dict(export)}
    return {'schema': SCHEMA, 'learner_checkpoint_sha256': learner, 'actors': actors}


def validate_capture_export(capture, declaration):
    """Bind a lossy or stale actor to the actual export recorded with each turn."""
    actual = capture.get('actor_export')
    checkpoint = (capture.get('policy') or {}).get('checkpoint_sha256')
    if actual is not None and (not isinstance(actual, dict) or set(actual) != {'kind', 'format', 'sha256'}
            or actual['kind'] not in EXPORT_KINDS or not isinstance(actual['format'], str) or not actual['format']
            or not isinstance(actual['sha256'], str) or not _SHA.fullmatch(actual['sha256'])):
        raise ValueError('captured RL actor export binding is malformed')
    entry = (declaration or {}).get('actors', {}).get(checkpoint)
    if entry is not None:
        if actual != entry['export']:
            raise ValueError('captured RL actor export differs from its exact declaration')
    elif actual is not None and (not isinstance(actual, dict) or actual.get('kind') != 'exact'):
        raise ValueError('quantized RL actor export requires its exact declaration')


def declared_actor(policy, learner_identity, declaration):
    """True when ``policy`` is a declared actor's identity: the learner's except its checkpoint."""
    if declaration is None or not isinstance(policy, dict) or set(policy) != set(learner_identity):
        return False
    sha = policy.get('checkpoint_sha256')
    return (sha in declaration['actors']
            and all(policy[key] == learner_identity[key] for key in policy if key != 'checkpoint_sha256'))


def accepted_policy(policy, learner_identity, declaration):
    return policy == learner_identity or declared_actor(policy, learner_identity, declaration)


def learner_identity_for(policy, checkpoint_sha256, declaration):
    """The learner identity a declared actor's captured ``policy`` implies (only its checkpoint differs)."""
    if declaration is None or policy.get('checkpoint_sha256') not in declaration['actors']:
        return policy
    if declaration['learner_checkpoint_sha256'] != checkpoint_sha256:
        raise ValueError('RL actor declaration names a different learner checkpoint')
    return {**policy, 'checkpoint_sha256': checkpoint_sha256}
