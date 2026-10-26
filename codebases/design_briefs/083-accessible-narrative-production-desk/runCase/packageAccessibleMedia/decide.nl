---
description: Decide field-level findings for packageAccessibleMedia.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Assemble coordinated format-specific files with stable segment IDs and the correct source revision. Verify player or delivery compatibility, preserve reviewer notes, and update only affected segments when the underlying edit changes. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
