---
description: Decide field-level findings for planAndCost.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Use exact recipe scaling, inventory, costing, and scheduling services to build preparation and service plans. Account for yield, waste, shared equipment, and approved handling requirements; preserve traceability from purchased ingredients to served dishes. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
