---
args: { prepared: PreparedDisclosureSelection, goal: GoalContract, policy: PolicySelection }
returns: DecisionsDisclosureSelection
---
1. Apply this domain algorithm to each relevant evidence unit: Identify responsive content at passage level, retaining enough context for accurate interpretation. Separate nonresponsive material from restricted responsive material so reviewers can see whether an omission is about relevance or permission.
2. Return field-level atoms for the actual DisclosureSelection record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
