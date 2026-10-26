"""Restore an ancestor's exact finite foundation declaration, never earned credit."""
from __future__ import annotations

import json
from pathlib import Path

from ..common.hashing import sha256_file_hex_cached

SCHEMA = 'natlang.native-foundation-declaration-recovery/1'


def _bound(reference):
    if (not isinstance(reference, dict) or not isinstance(reference.get('path'), str)
            or not isinstance(reference.get('sha256'), str) or len(reference['sha256']) != 64):
        raise ValueError('foundation recovery requires exact artifact bindings')
    path = Path(reference['path']).resolve(strict=True)
    if sha256_file_hex_cached(path) != reference['sha256']:
        raise ValueError('foundation recovery artifact changed: ' + str(path))
    if 'bytes' in reference and path.stat().st_size != reference['bytes']:
        raise ValueError('foundation recovery artifact size changed: ' + str(path))
    return path


def restore_declaration(path, *, model_config, stage_id, training_frame,
                        snapshot_ancestors, foundation, active_bindings, epochs):
    """Authenticate historical declarations using actual ancestry and graph bindings.

    Exported metadata and verification receipts are trusted operator-owned
    recovery artifacts, authenticated by the recipe's exact input binding.
    They restore a declaration only. Current per-row counters and all numerical
    and behavioral qualification remain the executing trainer's responsibility.
    No supplied exposure counter is copied into the checkpoint.
    """
    path = Path(path).resolve(strict=True)
    proof = json.loads(path.read_text())
    required = {'schema', 'stage_id', 'historical_recipe', 'historical_training_metadata',
                'historical_checkpoint_verification', 'foundation_sources', 'reason'}
    if (not isinstance(proof, dict) or set(proof) != required or proof.get('schema') != SCHEMA
            or proof.get('stage_id') != stage_id or not isinstance(proof.get('reason'), str)
            or not proof['reason'].strip()):
        raise ValueError('foundation recovery needs an explicit exact historical stage declaration')
    recipe = json.loads(_bound(proof['historical_recipe']).read_text())
    metadata = json.loads(_bound(proof['historical_training_metadata']).read_text())
    verification = json.loads(_bound(proof['historical_checkpoint_verification']).read_text())
    source_path = _bound(proof['foundation_sources'])
    if (source_path != foundation.path.resolve() or proof['foundation_sources']['sha256'] != foundation.file_sha256):
        raise ValueError('foundation recovery selected another source header')
    checkpoint = verification.get('checkpoint') or {}
    digest = checkpoint.get('sha256')
    if (verification.get('passed') is not True or not isinstance(digest, str) or len(digest) != 64
            or type(checkpoint.get('bytes')) is not int or checkpoint['bytes'] <= 0
            or digest not in snapshot_ancestors or verification.get('optimizer_present') is not True
            or verification.get('rng_present') is not True or verification.get('stage_id') != stage_id):
        raise ValueError('foundation recovery checkpoint is not a verified actual ancestor of this stage')
    historical_config = metadata.get('model_config') or {}
    if (any(historical_config.get(key) != model_config.get(key)
            for key in ('architecture', 'base_inventory_sha256', 'tokenizer_fingerprint', 'marker_ids'))
            or verification.get('architecture') != model_config.get('architecture')):
        raise ValueError('foundation recovery historical model/tokenizer identity differs')
    historical_training = metadata.get('training') or {}
    historical_options = historical_training.get('options') or {}
    old_step = historical_training.get('global_step')
    if (type(old_step) is not int or old_step < 0 or old_step != verification.get('global_step')
            or old_step > training_frame.get('global_step', -1)
            or historical_options.get('stage_id') != stage_id
            or historical_options.get('communication_foundation') is not True
            or historical_options.get('epochs', 1.) != epochs):
        raise ValueError('foundation recovery did not recover a previously declared stage')
    if metadata.get('checkpoint_identity') is not None:
        if any(metadata['checkpoint_identity'].get(key) != checkpoint[key] for key in ('sha256', 'bytes')):
            raise ValueError('foundation recovery metadata names another historical checkpoint')
    elif metadata.get('checkpoint_path') != verification.get('path'):
        raise ValueError('foundation recovery metadata and checkpoint verification paths differ')
    stages = [stage for stage in recipe.get('stages', []) if stage.get('id') == stage_id]
    if len(stages) != 1 or stages[0].get('kind') != 'native_state_sft':
        raise ValueError('foundation recovery historical recipe lacks this native stage')
    stage = stages[0]
    source_name = (stage.get('inputs') or {}).get('foundation_native_graphs')
    historical_source = (recipe.get('input_bindings') or {}).get(source_name) or {}
    if (historical_source.get('sha256') != foundation.file_sha256
            or historical_source.get('path') != historical_options.get('foundation_native_graphs')
            or (stage.get('parameters') or {}).get('communication_foundation') is not True
            or (stage.get('parameters') or {}).get('epochs', 1.) != epochs):
        raise ValueError('foundation recovery historical recipe declared another foundation')
    saved_bindings = training_frame.get('source_row_bindings') or {}
    if (not foundation.bindings or any(saved_bindings.get(key) != value
            or active_bindings.get(key) != value for key, value in foundation.bindings.items())):
        raise ValueError('foundation recovery lost or changed a previously admitted graph binding')
    from .native_continuation import communication_foundation
    actual = communication_foundation(foundation, training_frame.get('exposure_counts') or {},
        epochs=epochs, trained_bindings=saved_bindings)
    if not actual['graphs'] or actual['graphs'] != len(foundation):
        raise ValueError('foundation recovery must contain only genuine producer/consumer graphs')
    return {'schema': SCHEMA, 'recovery': {'path': str(path), 'sha256': sha256_file_hex_cached(path)},
            'historical_checkpoint': checkpoint, 'foundation_sources': proof['foundation_sources'],
            'cohort_sha256': actual['cohort_sha256'], 'stage_id': stage_id,
            'at_global_step': training_frame['global_step'],
            'new_exposure': 0, 'qualification_granted': False}
