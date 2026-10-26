---
description: Maintain use register, expiry/removal obligations and attribution changes across every affected derivative.
args: { dossier: RightsDossier; change: RightsChange; compatibility: UseCompatibility[]; clearances: ClearanceTask[] }
returns: CollectionUseRecord
---
Bind affectedDerivatives = `traceAffectedDerivatives(dossier, change, compatibility)` for attribution corrections, expiries and changed permissions. Schedule review before every time-limited permission expires and remove affected uses when it does. Update print, online and promotional derivatives separately with their attribution version. Preserve previous register state and source receipts; include affectedDerivatives in the returned register.
