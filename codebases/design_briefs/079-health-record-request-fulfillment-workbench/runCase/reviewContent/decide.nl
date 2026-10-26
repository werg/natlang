---
description: Decide field-level findings for reviewContent.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Assess responsiveness and identify third-party or otherwise restricted material under supplied disclosure rules. Preserve clinically relevant context where authorized, and route nuanced disclosure judgments to qualified records reviewers rather than making autonomous legal decisions. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
