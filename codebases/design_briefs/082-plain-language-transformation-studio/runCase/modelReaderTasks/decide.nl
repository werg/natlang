---
description: Decide field-level findings for modelReaderTasks.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Interpret what the intended reader must understand or do using supplied audience guidance. Identify decision points and necessary background without stereotyping ability, and distinguish information that can be reorganized from information that cannot be omitted. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
