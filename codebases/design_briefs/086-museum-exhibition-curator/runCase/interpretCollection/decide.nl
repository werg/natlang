---
description: Decide field-level findings for interpretCollection.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Read artifact records, scholarship, provenance, and conservation guidance to identify what can be responsibly said and displayed. Separate documented facts from contested interpretations and preserve restrictions on handling, light, duration, or cultural presentation. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
