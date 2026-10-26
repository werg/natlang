---
description: Decide field-level findings for reconcileFindings.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Compare evidence across sources while preserving chronology, attribution, and disagreement. State what was found and what remains uncertain, and do not turn an unsuccessful search in limited collections into proof that an event never occurred. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
