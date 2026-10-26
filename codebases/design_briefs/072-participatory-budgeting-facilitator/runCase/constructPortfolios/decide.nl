---
description: Decide field-level findings for constructPortfolios.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Enumerate feasible combinations under exact budget and dependency constraints, accounting for shared work and mutually exclusive uses. Present alternatives for deliberation without autonomously choosing political priorities or inferring participant preferences from personal data. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
