---
args: { prepared: PreparedResearchLesson, goal: GoalContract, policy: PolicySelection }
returns: DecisionsResearchLesson
---
1. Apply this domain algorithm to each relevant evidence unit: Publish a scoped lesson with conditions under which it applies and links to contradictory or superseding evidence. Update recommendations when new attempts arrive, preserving the historical knowledge state and uncertainty rather than rewriting failures into hindsight certainty.
2. Return field-level atoms for the actual ResearchLesson record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
