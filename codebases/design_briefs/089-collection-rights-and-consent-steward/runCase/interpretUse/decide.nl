---
description: Decide field-level findings for interpretUse.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Translate the proposed exhibition, publication, or derivative use into concrete acts and distribution scope. Identify cropping, translation, online access, commercial use, and duration where relevant, rather than treating all publication as one undifferentiated permission. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
