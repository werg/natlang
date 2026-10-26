---
args: { prepared: PreparedIncidentGroup, goal: GoalContract, policy: PolicySelection }
returns: DecisionsIncidentGroup
---
1. Apply this domain algorithm to each relevant evidence unit: Compare accounts for the same customer, event, and time window. Group follow-ups and copied messages into one incident while retaining repeat failures that actually occurred on different occasions; leave identity ambiguity unresolved.
2. Return field-level atoms for the actual IncidentGroup record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
