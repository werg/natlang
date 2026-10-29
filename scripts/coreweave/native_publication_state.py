"""Reconcile durable backup evidence independently from transient export phases."""
from pathlib import Path
try:
    from .private_export_guard import validate_receipt
except ImportError:
    from private_export_guard import validate_receipt


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
