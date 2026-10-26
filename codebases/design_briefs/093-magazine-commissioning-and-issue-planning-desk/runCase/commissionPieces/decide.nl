---
description: Decide field-level findings for commissionPieces.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Draft precise commissions specifying angle, scope, length, reporting obligations, rights, and deadlines. Identify dependencies such as interviews or photography, obtain contributor acceptance, and avoid assuming a pitch implies agreement to changed terms. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
