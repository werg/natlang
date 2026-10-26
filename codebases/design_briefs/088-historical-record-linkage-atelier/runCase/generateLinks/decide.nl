---
description: Decide field-level findings for generateLinks.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Generate candidate links using names, chronology, places, relationships, and contextual descriptions. Include plausible aliases while preserving hard incompatibilities, and audit records excluded by blocking so unusual spellings do not disappear from consideration. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
