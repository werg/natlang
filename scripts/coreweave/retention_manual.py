"""Authenticate manual full-backup records against their accepted owner."""
from __future__ import annotations


def normalize_manual(entry,intent,owners):
    binding=entry.get('owner_binding') or {};accepted=binding.get('accepted') or {};keys=('owner_id','generation','boot_id','pid','process_start_ticks')
    if any(accepted.get(k) is None for k in keys):raise ValueError('manual record lacks exact owner process')
    matches=[o for o in owners if o.get('sandbox_id')==binding.get('sandbox_id') and o.get('plan_sha256')==binding.get('plan_sha256') and o.get('source_sha256')==binding.get('source_sha256') and str(o.get('runtime_commit','')).startswith(str(binding.get('runtime_commit') or '!')) and all((o.get('accepted') or {}).get(k)==accepted[k] for k in keys)]
    if len(matches)!=1:raise ValueError('manual owner binding is not unique')
    owner=matches[0]
    if owner.get('outer_credit') is not False or intent.get('worker')!=entry.get('worker') or intent.get('step')!=entry.get('step') or intent.get('sandbox_id')!=owner['sandbox_id'] or intent.get('source_sha256')!=owner['source_sha256'] or intent.get('step',-1)<=owner['parent_step'] or not intent.get('sha256') or type(intent.get('bytes')) is not int or intent['bytes']<=0:
        raise ValueError('manual intent differs from authenticated owner')
    return {**intent,'plan_sha256':owner['plan_sha256'],'runtime_commit':owner['runtime_commit'],'owner':owner['accepted'],'outer_credit':False}
