"""Exact authority for a new installed recipe plan recovered from a full native parent."""
from pathlib import Path
from ..common.hashing import canonical_json_sha256_hex as digest,sha256_file_hex


def checkpoint_plan_recovery(checkpoint,recipe_path,recipe,stage_inputs,*,expected_sha256):
    from .checkpoint_policy import CheckpointPolicy
    from .recipe import native_recipe_binding,training_recipe_binding
    from ..rl.native_raw_policy import require_raw_training
    checkpoint=Path(checkpoint).resolve(strict=True)
    before=checkpoint.stat()
    checksum=sha256_file_hex(checkpoint)
    if checksum!=expected_sha256:
        raise ValueError('checkpoint recovery differs from the declared verified full parent SHA')
    state=CheckpointPolicy.load(checkpoint)
    training=require_raw_training(state)
    original=training_recipe_binding(state)
    if original is None:raise ValueError('checkpoint recovery requires its saved current recipe authority')
    lineage=original.get('lineage_sha256',original['sha256'])
    current=native_recipe_binding(recipe_path,lineage)
    if any(current.get(key)!=original.get(key) for key in ('sha256','file_sha256')):
        raise ValueError('checkpoint recovery requires the exact saved current recipe policy and file')
    stages=[stage for stage in recipe['stages'] if stage.get('admitted',True)]
    stage_id=(training.get('options') or {}).get('stage_id')
    if (not stages or stages[0]['id']!=stage_id or stages[0]['kind']!='native_state_sft' or stages[0].get('requires')):
        raise ValueError('checkpoint recovery cannot fabricate preceding stage completion')
    sources=training.get('source_receipts') or []
    actual=stage_inputs.get(stage_id) or {}
    if (len(sources)<2 or any(actual.get(role,{}).get('sha256')!=saved.get('sha256')
                             for role,saved in zip(('native_graphs','held_native_graphs'),sources))):
        raise ValueError('checkpoint recovery changed its saved train or held source binding')
    rows=training.get('source_row_bindings') or {};credits=training.get('exposure_counts') or {}
    if (not rows or set(credits)!=set(rows) or
            any(type(value) is not int or value<0 for value in credits.values())):
        raise ValueError('checkpoint recovery requires exact saved source rows and earned exposure')
    after=checkpoint.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(
            after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('checkpoint changed during recovery authentication')
    return {'schema':'natlang.native-checkpoint-plan-recovery/1',
        'checkpoint':{'path':str(checkpoint),'sha256':checksum,'bytes':after.st_size},
        'saved_recipe':original,'current_recipe':current,'lineage_sha256':lineage,'stage_id':stage_id,
        'step':training.get('global_step',training.get('step')),
        'source_row_bindings_sha256':digest(rows),'exposure_counts_sha256':digest(credits),
        'sft_progress_sha256':digest(training.get('sft_progress') or []),
        'historical_plan_reconstructed':False,'historical_owner_release_claimed':False}
