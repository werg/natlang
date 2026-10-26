---
args: { prepared: PreparedApplicabilityFinding, goal: GoalContract, policy: PolicySelection }
returns: DecisionsApplicabilityFinding
---
1. Apply this domain algorithm to each relevant evidence unit: Test every limiting condition against current evidence and identify which differences change the rationale. If a decisive fact is unknown, request that fact; do not grant the exception by treating missing evidence as a match.
2. Return field-level atoms for the actual ApplicabilityFinding record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
