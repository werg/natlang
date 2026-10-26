---
args: { prepared: PreparedAcceptanceFinding, goal: GoalContract, policy: PolicySelection }
returns: DecisionsAcceptanceFinding
---
1. Apply this domain algorithm to each relevant evidence unit: Interpret customer responses under the agreement’s acceptance process. Separate acknowledgment of receipt from acceptance, verify the respondent’s authority, and apply deemed-acceptance timing only when all supplied contractual prerequisites are established.
2. Return field-level atoms for the actual AcceptanceFinding record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
