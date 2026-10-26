---
description: Resolve a grower observation to a versioned lot delta without overwriting the prior estimate.
args: { desk: HarvestDeskState; change: HarvestChange }
returns: FieldDelta
---
Compare each observed evidence item with the lot's source ID, version, exact variety, and prior estimate in `desk`. Attach a new estimate only where the evidence explicitly identifies that same lot and a later or otherwise current observation. Preserve both values with their own source versions. Do not infer a quantity change from general crop descriptions. For every unresolved lot, name the missing identifier, version, quality assessment, or readiness window. Return only commitments linked to changed lots in `affectedCommitments`.
