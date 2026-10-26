---
args: { prepared: PreparedDisputeEvent, goal: GoalContract, policy: PolicySelection }
returns: DecisionsDisputeEvent
---
1. Apply this domain algorithm to each relevant evidence unit: Align listings, messages, shipping records, and submissions without equating timestamps with event occurrence. Attribute disputed statements to their speakers, preserve edited listings, and mark contradictions rather than manufacturing a single settled narrative.
2. Return field-level atoms for the actual DisputeEvent record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
