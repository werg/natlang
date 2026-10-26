---
args: { prepared: PreparedHandoverDependency, goal: GoalContract, policy: PolicySelection }
returns: DecisionsHandoverDependency
---
1. Apply this domain algorithm to each relevant evidence unit: Identify knowledge, access, approvals, and external relationships needed to finish each item. Separate credentials from instructions, detect circular handoffs, and formulate minimal access requests without copying secrets into the handover dossier.
2. Return field-level atoms for the actual HandoverDependency record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
