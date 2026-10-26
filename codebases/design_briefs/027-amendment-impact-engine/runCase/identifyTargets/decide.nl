---
args: { prepared: PreparedAmendmentTarget, goal: GoalContract, policy: PolicySelection }
returns: DecisionsAmendmentTarget
---
1. Apply this domain algorithm to each relevant evidence unit: Read amendment language against the referenced agreement edition. Resolve section references, defined terms, and indirect targets; distinguish correction of wording from a substantive change and retain ambiguous references for review rather than choosing the nearest heading.
2. Return field-level atoms for the actual AmendmentTarget record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
