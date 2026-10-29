"""Fail-closed evidence check for a full checkpoint and its control pair."""
from __future__ import annotations


def eligible_pair(candidate, *, latest_role, latest_owner_role, pending_roles, protected_sha256s, active_units, verified_roles, successor_role, fallback_role):
    """Check already authenticated current owner, provider and publication facts.

    Inputs are snapshots to be refreshed under the same lock as retirement. This
    function never infers owner verification or current private object presence.
    """
    full=candidate['role']; control=candidate['control_role']
    if full==control or not all(candidate.get(k) for k in ('sha256','control_sha256','full_unit','control_unit')):
        raise ValueError('incomplete retirement pair')
    protected=set(protected_sha256s)
    if candidate['sha256'] in protected or candidate['control_sha256'] in protected:raise ValueError('protected identity')
    if full in {latest_role,latest_owner_role,successor_role,fallback_role} or control in {latest_role,latest_owner_role,successor_role,fallback_role}:
        raise ValueError('active latest or fallback role')
    if {full,control}&set(pending_roles):raise ValueError('pending GET or publication')
    units={candidate['full_unit'],candidate['control_unit'],*candidate.get('operation_units',[])}
    if units&set(active_units):raise ValueError('active pair operation')
    if not {full,control,successor_role,fallback_role} <= set(verified_roles):raise ValueError('missing verified recovery coverage')
    return True


def eligible_full(candidate, *, latest_role, latest_owner_role, pending_roles, protected_sha256s, active_units, verified_roles, successor_role, fallback_role):
    """Historical full-only records are eligible only when they never had control."""
    full=candidate['role']
    if candidate.get('control_required') or candidate.get('control'):
        raise ValueError('control-bearing record needs paired retirement')
    if not candidate.get('sha256') or not candidate.get('full_unit'):raise ValueError('incomplete full candidate')
    if candidate['sha256'] in set(protected_sha256s):raise ValueError('protected identity')
    if full in {latest_role,latest_owner_role,successor_role,fallback_role}:raise ValueError('active latest or fallback role')
    if full in set(pending_roles):raise ValueError('pending GET or publication')
    if {candidate['full_unit'],*candidate.get('operation_units',[])}&set(active_units):raise ValueError('active full operation')
    if not {full,successor_role,fallback_role}<=set(verified_roles):raise ValueError('missing verified recovery coverage')
    return True
