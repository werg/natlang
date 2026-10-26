---
description: Decide field-level findings for requestMaterial.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Group missing-material requests by authorized source, ask only for necessary records, and preserve patient consent and permitted disclosure scope. Avoid repeated requests for existing evidence and track who is responsible for each unresolved gap. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
