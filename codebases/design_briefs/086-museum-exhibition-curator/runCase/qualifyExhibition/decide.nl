---
description: Decide field-level findings for qualifyExhibition.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Review the complete exhibition for narrative coherence, factual support, rights, conservation, accessibility, and operational feasibility. Compare alternative arrangements, resolve conflicts with qualified owners, and freeze the approved content and installation plan together. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
