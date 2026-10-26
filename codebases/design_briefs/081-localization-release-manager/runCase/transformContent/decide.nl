---
description: Decide field-level findings for transformContent.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Translate or adapt according to communicative purpose while preserving product facts, obligations, and constraints. Generate alternatives for culturally or linguistically sensitive phrasing using supplied guidance, and keep unresolved meaning questions visible rather than hiding them in fluent text. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
