---
description: Read back transfer state from the receiving service before any retry.
args: { dossier: ReferralDossier; change: PacketChange }
returns: TransferLedger
---
Reconcile the exact packet ID, version, recipient and dispatch timestamp against the recipient's authoritative status channel. An absent response leaves state unknown. Allow retry only when a current recipient response proves the packet was not received and the current authorization remains valid. Preserve receipt evidence and ambiguous outcomes.
