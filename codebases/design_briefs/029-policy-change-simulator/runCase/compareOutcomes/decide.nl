---
args: { prepared: PreparedOutcomeDifference, goal: GoalContract, policy: PolicySelection }
returns: DecisionsOutcomeDifference
---
1. Apply this domain algorithm to each relevant evidence unit: Execute both policy versions with isolated state and compare duties, permissions, timing, and escalation. Trace differences to specific changed conditions and separate genuine policy effects from nondeterministic or unsupported semantic judgments.
2. Return field-level atoms for the actual OutcomeDifference record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
