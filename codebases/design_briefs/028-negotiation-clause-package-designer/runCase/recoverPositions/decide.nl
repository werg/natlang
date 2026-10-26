---
args: { prepared: PreparedNegotiationPosition, goal: GoalContract, policy: PolicySelection }
returns: DecisionsNegotiationPosition
---
1. Apply this domain algorithm to each relevant evidence unit: Translate stakeholder goals and the approved playbook into issue-level positions. Distinguish mandatory limits from preferences, identify dependencies between issues, and preserve disagreements that require a mandate decision before drafting a proposal.
2. Return field-level atoms for the actual NegotiationPosition record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
