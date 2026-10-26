---
args: { prepared: PreparedDisclosureScope, goal: GoalContract, policy: PolicySelection }
returns: DecisionsDisclosureScope
---
1. Apply this domain algorithm to each relevant evidence unit: Read the request, authorization, and supplied disclosure policy together. Determine recipient, purpose, relevant time span, and permitted material; flag conflicting instructions rather than treating the broadest request as sufficient authority.
2. Return field-level atoms for the actual DisclosureScope record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
