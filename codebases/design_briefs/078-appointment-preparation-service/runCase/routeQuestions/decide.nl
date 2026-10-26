---
description: Decide field-level findings for routeQuestions.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Form focused questions for the appropriate staff when instructions conflict with practical circumstances or remain unclear. Track authoritative responses, avoid answering medical questions from general model knowledge, and update the checklist only from approved clarification. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
