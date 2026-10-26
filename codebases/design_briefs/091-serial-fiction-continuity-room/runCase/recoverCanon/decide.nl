---
description: Decide field-level findings for recoverCanon.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Extract events, relationships, world rules, and character knowledge from approved manuscripts and notes. Distinguish established canon from plans, unreliable narration, and character beliefs; preserve conflicting drafts without silently promoting one to truth. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
