---
args: { prepared: PreparedIncidentClosure, goal: GoalContract, policy: PolicySelection }
returns: DecisionsIncidentClosure
---
1. Apply this domain algorithm to each relevant evidence unit: Check implemented actions, release criteria, and subsequent observations. Preserve residual risk and monitoring obligations; reopen the incident when a materially similar recurrence contradicts the closure rationale rather than treating it as a fresh unrelated ticket.
2. Return field-level atoms for the actual IncidentClosure record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
