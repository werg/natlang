---
description: Score a grounded dossier digest against one diligence rubric criterion.
args:
  dossierName: string
  digest: SectionDigest
  criterion: RubricCriterion
returns: CriterionVerdict
readout: decision
---
Evaluate dossierName using its grounded SectionDigest digest against criterion (criterion.name and criterion.guidance).
Base your verdict strictly on the summarized evidence and verbatim citations in digest:
- "strong": clear, verified evidence that exceeds the requirement with low operational or legal risk.
- "acceptable": satisfies the requirement adequately with no material blockers.
- "concern": gaps, ambiguity, or non-trivial risk that requires follow-up or mitigation before adoption.
- "dealbreaker": verified critical flaw, prohibitive license or security exposure, or missing mandatory guarantee that disqualifies adoption.
