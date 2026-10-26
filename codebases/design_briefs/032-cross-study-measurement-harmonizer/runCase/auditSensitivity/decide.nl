---
args: { prepared: PreparedHarmonizationAudit, goal: GoalContract, policy: PolicySelection }
returns: DecisionsHarmonizationAudit
---
1. Apply this domain algorithm to each relevant evidence unit: Compare reasonable alternative mappings and cohort restrictions to reveal dependence on harmonization choices. Produce an analyst-facing report distinguishing stable findings from those driven by assumptions and identify measurements that must remain separate.
2. Return field-level atoms for the actual HarmonizationAudit record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
