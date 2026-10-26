---
args: { prepared: PreparedRoleInterface, goal: GoalContract, policy: PolicySelection }
returns: DecisionsRoleInterface
---
1. Apply this domain algorithm to each relevant evidence unit: Compare actual decision rights, escalation paths, and handoff expectations. Identify duplicated ownership and gaps between roles; propose responsibility agreements without assuming that similar job titles imply interchangeable authority.
2. Return field-level atoms for the actual RoleInterface record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
