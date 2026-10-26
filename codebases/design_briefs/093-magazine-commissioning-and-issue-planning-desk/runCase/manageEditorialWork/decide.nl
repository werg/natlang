---
description: Decide field-level findings for manageEditorialWork.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Compare drafts with commissions and source evidence, formulate specific revisions, and coordinate editing, fact-checking, art, and legal or rights review where needed. Preserve author voice while distinguishing stylistic preference from factual or structural defects. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
