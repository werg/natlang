---
args: { prepared: PreparedAcceptanceScenario, goal: GoalContract, policy: PolicySelection }
returns: DecisionsAcceptanceScenario
---
1. Apply this domain algorithm to each relevant evidence unit: Generate ordinary, boundary, and adverse scenarios, then replay the specification through the available simulator. Use counterexamples to reopen only the implicated agreements, preserving unrelated acceptance and recording revised stakeholder acknowledgment.
2. Return field-level atoms for the actual AcceptanceScenario record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
