---
args: { prepared: PreparedEntitlementFinding, goal: GoalContract, policy: PolicySelection }
returns: DecisionsEntitlementFinding
---
1. Apply this domain algorithm to each relevant evidence unit: Match each issue to the applicable policy edition, purchased service, and documented exception. Separate mandatory entitlements from discretionary gestures; ask for missing decisive evidence without requiring information already present in the case.
2. Return field-level atoms for the actual EntitlementFinding record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
