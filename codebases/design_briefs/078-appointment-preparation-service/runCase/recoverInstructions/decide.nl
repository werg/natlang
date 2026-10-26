---
description: Decide field-level findings for recoverInstructions.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Parse the approved instruction set into actions, timing, exceptions, and contact routes. Preserve clinical wording where transformation could change meaning, and identify ambiguity for staff resolution rather than interpreting medical instructions independently. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
