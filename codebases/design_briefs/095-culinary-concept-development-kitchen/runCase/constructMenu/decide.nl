---
description: Decide field-level findings for constructMenu.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Assemble a balanced menu whose dishes work together across preparation effort, equipment, pacing, and customer choice. Identify shared components and bottlenecks, preserve meaningful alternatives for required diets, and avoid creating variety only through superficial garnish changes. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
