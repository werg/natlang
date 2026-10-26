---
description: Decide field-level findings for normalizeRecords.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Preserve original spellings, date expressions, and geographic descriptions while creating normalized search forms. Distinguish transcription uncertainty from historical variation, and avoid converting approximate dates or changing place names into falsely precise modern facts. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
