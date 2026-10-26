---
description: Decide field-level findings for extractLogistics.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Read the approved plan to identify logistical tasks and dependencies, preserving its wording for clinical requirements. Separate administrative coordination from treatment decisions and route any ambiguous clinical instruction back to the authorized team. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
