---
description: Decide field-level findings for frameCommission.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Interpret the requester’s goal and decompose it into answerable historical questions. Identify what available collections could establish, distinguish factual lookup from interpretive synthesis, and agree a bounded effort and stopping rule before searching indefinitely. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
