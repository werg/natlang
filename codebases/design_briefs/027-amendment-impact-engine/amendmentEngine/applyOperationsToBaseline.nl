---
args: { baseline: GoalContract; targets: AmendmentTargetMap; scope: AmendmentScopeMap; precedence: AmendmentGuardrail; evidence: EvidencePage[] }
returns: ObligationComposition
---
Apply replace, add, delete, narrow or extend only to the exact target facet and only inside the established effective scope. Preserve all unaffected definitions, exceptions, prerequisites and cross-references. Keep before and after wording with source refs and unresolved interpretations. Do not construct a releaseable obligation while target identity, scope or precedence is uncertain.