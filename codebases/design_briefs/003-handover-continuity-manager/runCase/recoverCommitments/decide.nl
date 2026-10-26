---
args: { prepared: PreparedCommitment, goal: GoalContract, policy: PolicySelection }
returns: DecisionsCommitment
---
1. Apply this domain algorithm to each relevant evidence unit: Read threads with their replies and amendments. Distinguish a tentative intention from a promise, identify the beneficiary and deliverable, and retain conditional commitments whose activation event has not yet occurred.
2. Return field-level atoms for the actual Commitment record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
