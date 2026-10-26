---
description: Decide field-level findings for extractPermissions.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Read donation agreements, licenses, consent statements, and amendments to identify permissions and restrictions. Distinguish ownership of a physical object from rights to reproduce or publish it, and preserve missing or conflicting documentation as uncertainty. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
