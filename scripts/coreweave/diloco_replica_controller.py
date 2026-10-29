"""Single-owner durable controller journal for fenced replica promotion."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
from natlang_neuralese.common.jsonio import write_json_durable


def run_promotion(directory, *, intent, fence_workers, verify_fence, start_primary, release_workers):
    """Perform one promotion under an exclusive durable controller generation.

    Callback implementations belong to the authorized supervisor. On failure,
    already installed worker fences are deliberately left in place. The new
    primary must acknowledge the exact witnessed WAL head before release.
    """
    required={'run_id','generation_nonce','sequence','head','new_url','new_token_sha256'}
    if not isinstance(intent,dict) or set(intent)!=required or not all(intent.values()) or len(intent['generation_nonce'])<32:
        raise ValueError('invalid promotion intent')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'controller.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=directory/'promotion.json'
        if journal.exists():raise ValueError('promotion generation already exists; reconcile its journal')
        record={'schema':'natlang.diloco-promotion-controller/1','intent':intent,'phase':'intent'}
        write_json_durable(journal,record)
        fences=fence_workers(intent);record.update(phase='workers-fenced',workers=fences);write_json_durable(journal,record)
        evidence=verify_fence(intent)
        if (evidence.get('primary_fenced') is not True or evidence.get('all_workers_quiesced') is not True
                or evidence.get('run_id')!=intent['run_id'] or not evidence.get('proof_sha256')):
            raise ValueError('invalid supervisor fence evidence')
        record.update(phase='provider-fenced',fence=evidence);write_json_durable(journal,record)
        ready=start_primary(intent,evidence)
        if (ready.get('run_id')!=intent['run_id'] or ready.get('sequence')!=intent['sequence']
                or ready.get('head')!=intent['head'] or ready.get('url')!=intent['new_url']
                or ready.get('token_sha256')!=intent['new_token_sha256'] or ready.get('ready') is not True):
            raise ValueError('new primary did not prove exact WAL and session identity')
        record.update(phase='primary-ready',primary=ready);write_json_durable(journal,record)
        released=release_workers(intent,fences)
        if (not isinstance(fences,list) or not fences or not isinstance(released,list) or
                sorted(x['sandbox_id'] for x in released)!=sorted(x['sandbox_id'] for x in fences)):
            raise ValueError('worker release did not cover exact fenced owners')
        record.update(phase='released',released=released);write_json_durable(journal,record)
        return record
