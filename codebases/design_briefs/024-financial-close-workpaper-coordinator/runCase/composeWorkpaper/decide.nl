---
args: { prepared: PreparedWorkpaperConclusion, goal: GoalContract, policy: PolicySelection }
returns: DecisionsWorkpaperConclusion
---
1. Apply this domain algorithm to each relevant evidence unit: Assemble the reconciliation, explanations, and supporting documents into a reviewer-ready workpaper. Distinguish verified facts from management explanations, quantify residual differences exactly, and avoid conclusions broader than the evidence supports.
2. Return field-level atoms for the actual WorkpaperConclusion record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
