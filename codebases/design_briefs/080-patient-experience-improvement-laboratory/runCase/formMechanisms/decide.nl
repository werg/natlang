---
description: Decide field-level findings for formMechanisms.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Identify plausible administrative mechanisms such as unclear instructions or failed callbacks. Seek contradictory cases and alternative explanations, distinguish correlation from causality, and define evidence needed to test the mechanism without making clinical claims. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
