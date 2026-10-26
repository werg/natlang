---
description: Bind changed supply to the actual customer demand and the written substitution boundary.
args: { desk: HarvestDeskState; delta: FieldDelta; readiness: HarvestReadiness[]; change: HarvestChange }
returns: HarvestCommitment[]
---
Carry forward the current customer, quality, quantity, delivery window, and explicit substitutionsAllowed values for every affected commitment. When a customer rejected a substitution, retain the original required quality and quantity as unmet. A crop sharing a common name is not a substitute for matching variety or quality. If no authoritative customer response is supplied, keep the substitution undecided and preserve the original obligation.
