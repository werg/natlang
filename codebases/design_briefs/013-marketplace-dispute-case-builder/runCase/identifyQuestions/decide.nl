---
args: { prepared: PreparedDisputedQuestion, goal: GoalContract, policy: PolicySelection }
returns: DecisionsDisputedQuestion
---
1. Apply this domain algorithm to each relevant evidence unit: Determine which factual questions would change the rule-based outcome. Separate immaterial disagreements from decisive ones, identify the applicable evidence burden, and formulate neutral questions that either party can answer with concrete records.
2. Return field-level atoms for the actual DisputedQuestion record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
