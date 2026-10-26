---
args: { prepared: PreparedServiceIssue, goal: GoalContract, policy: PolicySelection }
returns: DecisionsServiceIssue
---
1. Apply this domain algorithm to each relevant evidence unit: Reconstruct the timeline across messages and split failures only when they can be resolved independently. Retain earlier promises and interactions between issues, such as a late replacement making a promotional remedy no longer useful.
2. Return field-level atoms for the actual ServiceIssue record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
