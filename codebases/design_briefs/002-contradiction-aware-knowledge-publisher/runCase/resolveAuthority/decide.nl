---
args: { prepared: PreparedAuthorityDecision, goal: GoalContract, policy: PolicySelection }
returns: DecisionsAuthorityDecision
---
1. Apply this domain algorithm to each relevant evidence unit: Consult the configured publication hierarchy and explicit supersession records. Prefer authoritative material only inside its remit; if equally authoritative sources disagree, formulate a bounded clarification rather than selecting the newest or most fluent text.
2. Return field-level atoms for the actual AuthorityDecision record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
