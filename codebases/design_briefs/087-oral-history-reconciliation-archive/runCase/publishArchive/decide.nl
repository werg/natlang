---
description: Decide field-level findings for publishArchive.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Assemble an attributed narrative and search index that preserve disagreement and access restrictions. Apply consent to quotations, identities, and derived summaries, and maintain edition history when participants correct or withdraw material. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
