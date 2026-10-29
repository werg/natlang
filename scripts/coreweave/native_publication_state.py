"""Reconcile durable backup evidence independently from transient export phases."""
from pathlib import Path
try:
    from .private_export_guard import validate_receipt, has_owner_readback
except ImportError:
    from private_export_guard import validate_receipt, has_owner_readback


def full_readback_verified(entry, pin, receipt):
    """An owner GET in flight does not revoke an earned complete cloud GET.

    This validates historical verification evidence only. Current object
    presence and workstation readback remain separate publication obligations.
    """
    try:
        validate_receipt(receipt, {'sha256': pin['sha256'], 'bytes': pin['bytes']})
        expected_name = entry['role']+'/'+pin['sha256']+'/'+Path(pin['path']).name
        return (receipt['role'] == entry['role'] and receipt['source'] == pin['path']
                and receipt['name'] == expected_name)
    except (KeyError, TypeError, ValueError):
        return False


def historical_owner_ready(entry, pin, receipt, selection_sha):
    """Only revisit a live entry whose exact full state has owner readback."""
    if (entry.get('publication_complete') or entry.get('control_selection_sha256') != selection_sha
            or entry.get('control', {}).get('status') != 'independently_verified'
            or not full_readback_verified(entry, pin, receipt)):
        return False
    return has_owner_readback(receipt)
