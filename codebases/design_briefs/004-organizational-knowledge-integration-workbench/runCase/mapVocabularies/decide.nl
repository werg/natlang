---
args: { prepared: PreparedTermSense, goal: GoalContract, policy: PolicySelection }
returns: DecisionsTermSense
---
1. Apply this domain algorithm to each relevant evidence unit: Extract terms from actual usage rather than glossary headings alone. Separate multiple senses within one organization and identify distinctions that affect who acts, which records exist, or what counts as completion.
2. Return field-level atoms for the actual TermSense record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
