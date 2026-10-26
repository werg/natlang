---
args: { prepared: PreparedIncidentCluster, goal: GoalContract, policy: PolicySelection }
returns: DecisionsIncidentCluster
---
1. Apply this domain algorithm to each relevant evidence unit: Compare complaints, inspections, and shipment events by product, lot, timing, and failure mechanism. Group plausibly related events while retaining independent incidents and contradictory evidence; do not merge all problems from one supplier into a single cause.
2. Return field-level atoms for the actual IncidentCluster record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
