---
description: Decide field-level findings for modelCharacterViews.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Build each character’s view from witnessed events, received messages, memory, and approved world facts. Separate true facts from beliefs and secrets, preserve obligations and motivations, and prevent the narrator’s global knowledge from leaking into character decisions. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
