---
description: Decide field-level findings for interpretAuthorization.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Read the request and authorization together to determine recipient, purpose, categories, dates, and any restrictions. Distinguish a requested scope from an authorized scope and route ambiguous authority to the records officer before retrieval. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
