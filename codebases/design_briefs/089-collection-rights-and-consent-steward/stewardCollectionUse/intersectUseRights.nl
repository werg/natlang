---
description: Test each proposed use against every contributing item's separate permission dimensions.
args: { dossier: RightsDossier; change: RightsChange; permissions: CollectionPermission[]; proposedUses: ProposedCollectionUse[] }
returns: UseCompatibility[]
---
Bind exactChecks = `checkPermissionIntersection(dossier, change, permissions, proposedUses)`. Return exactChecks as the compatibility results. Require every item permission to cover the exact purpose, transformation, channel, audience, attribution and duration. Missing rights or expired consent blocks that derivative. Do not assume permission for one channel carries to another.
