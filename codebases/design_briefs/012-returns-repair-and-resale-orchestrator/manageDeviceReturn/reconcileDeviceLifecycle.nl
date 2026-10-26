---
args:{prior:DeviceLifecycleCheckpoint|null;request:DeviceHandlingRequest;lifecycle:DeviceLifecycleFile;proposal:DeviceDispositionProposal}
returns:DeviceLifecycleCheckpoint
---
Create a first checkpoint for this case and lifecycle version, preserving serial identity, actual custody receipt IDs, repair prerequisites, functional tests and erasure verification references. An identical lifecycle replay is a no-op. When a receipt set changes, append each actual handoff once and preserve conflicting evidence as a dispute. A serial mismatch, missing handoff, unverified erasure, expired reservation or changed part revision keeps the affected item quarantined; only new matching receipts close that item’s exact transition. Preserve historical editions and advance once for new source or receipt evidence.
