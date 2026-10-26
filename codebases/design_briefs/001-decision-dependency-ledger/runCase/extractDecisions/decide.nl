---
args: { prepared: PreparedDecisionAssertion, goal: GoalContract, policy: PolicySelection }
returns: DecisionsDecisionAssertion
---
1. Apply this domain algorithm to each relevant evidence unit: Segment meetings at topic boundaries; distinguish a suggestion, assent, and an authorized resolution. Link explicit objections and preserve the speaker rather than attributing a collective decision from an unchallenged statement.
2. Return field-level atoms for the actual DecisionAssertion record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
