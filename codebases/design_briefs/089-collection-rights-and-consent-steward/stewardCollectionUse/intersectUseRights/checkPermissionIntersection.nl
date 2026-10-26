---
description: Compare proposed use dimensions with the rights granted for each contributing collection item.
args: { dossier: RightsDossier; change: RightsChange; permissions: CollectionPermission[]; proposedUses: ProposedCollectionUse[] }
returns: UseCompatibility[]
---
For every proposed use and source item, compare purpose, right type, channel, audience, transformation, attribution and valid-through date. A derivative clears only when all dimensions are covered by current evidence. Keep missing and conflicting dimensions explicit; a partial match is not clearance.
