---
description: Decide field-level findings for constructTeams.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Build feasible team configurations that collectively cover mission requirements and supervision. Use exact scheduling and capacity checks, explain role assignments, and preserve fair application of supplied assignment principles rather than optimizing convenience alone. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
