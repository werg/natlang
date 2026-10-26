---
description: Decide field-level findings for chooseQuestions.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Select the next question by how much it resolves relevant pathway uncertainty while minimizing burden and sensitivity. Reuse authorized existing evidence, explain why the information is needed, and allow the person to decline or report that it is unknown. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
