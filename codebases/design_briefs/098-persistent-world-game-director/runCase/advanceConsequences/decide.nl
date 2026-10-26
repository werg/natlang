---
description: Decide field-level findings for advanceConsequences.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Apply actions through the exact world simulator, then interpret their social and narrative consequences without rewriting physical outcomes. Preserve causal history, update beliefs only through information channels, and record unresolved story obligations for later sessions. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
