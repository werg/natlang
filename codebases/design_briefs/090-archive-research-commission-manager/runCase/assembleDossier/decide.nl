---
description: Decide field-level findings for assembleDossier.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Produce an evidence dossier with findings, source references, search history, unresolved leads, and reuse restrictions. Tailor the narrative to the commission while preserving methodological limits and separating the researcher’s interpretation from quoted or recorded facts. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
