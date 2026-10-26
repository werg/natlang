---
description: Decide field-level findings for assessCompleteness.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Compare each requirement with submitted documents, checking scope, dates, signatures, and readability. Distinguish document presence from actual required content and route clinical-content questions to authorized staff rather than interpreting them autonomously. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
