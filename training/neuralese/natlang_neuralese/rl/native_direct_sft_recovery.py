"""Publish a signal-stopped direct SFT checkpoint without granting stage completion."""
from __future__ import annotations
import fcntl
import json
from pathlib import Path
from ..common.jsonio import write_json_durable
from ..train.native_resident import binding
from ..train.native_readiness import _bound


def recover(state_path, *, report_ref, receipt_ref, custody_ref, qualification_ref, evaluation_ref, out):
    """Authenticate a completed update and retain its unfinished stage obligations."""
    from .native_raw_rounds import _published_checkpoint_state
    state_path, out = Path(state_path).resolve(), Path(out).resolve()
    with (state_path.parent / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text())
        if state.get('phase') != 'mandatory_sft_pending' or state.get('pending_update_kind') != 'mandatory_sft':
            raise ValueError('direct SFT recovery requires its stopped pending controller')
        if state.get('pending_native_checkpoint') or state.get('pending_native_round_dir'):
            raise ValueError('direct SFT recovery cannot replace another pending publication')
        def read(ref): return json.loads(_bound(ref).read_text())
        report, receipt, custody, qualification, evaluation = map(read, (report_ref, receipt_ref, custody_ref, qualification_ref, evaluation_ref))
        parent, child = custody['parent'], custody['successor']
        parent_file, child_file = _bound(parent), _bound(child)
        if Path(state['checkpoint']).resolve() != parent_file or parent['sha256'] == child['sha256']:
            raise ValueError('direct SFT recovery predecessor changed')
        boundary = report.get('continuation_boundary') or {}
        finite = report.get('progress_numerics') or {}
        if (custody.get('schema') != 'natlang.earned-sft-checkpoint-custody/1'
                or custody.get('old_bindings_and_quotas_preserved') is not True
                or custody.get('old_exposure_not_decreased') is not True
                or child['global_step'] != parent['global_step'] + 1
                or not all(r.get('optimizer_present') and r.get('rng_present') for r in (parent,child))
                or boundary != {'kind':'semantic-refresh','reached':False,'new_optimizer_updates':1,'checkpoint_sha256':child['sha256']}
                or report.get('checkpoint_sha256') != child['sha256']
                or Path(report.get('checkpoint','')).resolve() != child_file
                or receipt.get('checkpoint',{}).get('sha256') != child['sha256']
                or receipt.get('continuation',{}).get('checkpoint_sha256') != parent['sha256']
                or finite.get('checkpoint_sha256') != child['sha256'] or finite.get('passed') is not True
                or finite.get('nonfinite') != []):
            raise ValueError('direct SFT recovery lost exact earned checkpoint custody')
        q = qualification.get('report') or {}
        if (qualification.get('checkpoint_sha256') != child['sha256'] or qualification.get('optimizer_updates') != 0
                or q.get('checkpoint_sha256') != child['sha256'] or q.get('passed') is not True
                or (q.get('complete_gradient_qualification') or {}).get('passed') is not True):
            raise ValueError('direct SFT recovery requires actual checkpoint-bound gradient qualification')
        score = read(evaluation.get('score'))
        if (evaluation.get('schema') != 'natlang.direct-sft-evaluation/1'
                or evaluation.get('checkpoint_sha256') != child['sha256']
                or score.get('cases') != 4 or score.get('step') != child['global_step']):
            raise ValueError('direct SFT recovery requires the recorded four-case outcome check')
        if out.exists(): raise ValueError('direct SFT recovery receipt already exists')
        checkpoint = {'path':str(child_file),'sha256':child['sha256']}
        result = {'schema':'natlang.direct-sft-interrupted-recovery/1', 'parent':binding(parent_file),
                  'checkpoint':binding(child_file), 'report':report_ref,'receipt':receipt_ref,'custody':custody_ref,
                  'qualification':qualification_ref,'evaluation':evaluation_ref,'training_credit':0,
                  'stage_completion_granted':False,'behavioral_qualification_granted':False}
        # Record the intent before state mutation, so a crash cannot silently publish.
        write_json_durable(out, result)
        next_state = _published_checkpoint_state(state, checkpoint)
        next_state.pop('pending_update_kind', None)
        next_state.setdefault('interrupted_sft_recovery_history', []).append(binding(out))
        write_json_durable(state_path, next_state)
        return result
