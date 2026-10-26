---
description: Decide field-level findings for authorNegotiation.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Construct a fictional negotiation with multiple issues, private priorities, constraints, and opportunities for mutually beneficial trades. Ensure the scenario has meaningful information gathering and feasible outcomes without requiring prohibited tactics or real-world manipulation. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
