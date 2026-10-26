---
args: { prepared: PreparedClauseDependency, goal: GoalContract, policy: PolicySelection }
returns: DecisionsClauseDependency
---
1. Apply this domain algorithm to each relevant evidence unit: Inspect definitions, remedies, caps, exceptions, and cross-references together. Identify where a concession changes another clause’s effect, including apparently local edits that broaden an exception or undermine a negotiated protection.
2. Return field-level atoms for the actual ClauseDependency record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
