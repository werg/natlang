---
description: Decide field-level findings for debriefNegotiation.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Evaluate the observable trajectory and final agreement against learning goals, exact feasibility, and role constraints. Distinguish a good outcome from luck or prohibited conduct, explain missed trades using authorized debrief information, and propose targeted practice. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
