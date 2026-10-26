---
description: Decide field-level findings for propagateTerminology.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Identify content using the affected concept and sense, propose context-aware revisions, and preserve unrelated homographs. Revalidate grammar and meaning after replacement and version the glossary and dependent release artifacts together. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
