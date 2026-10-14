---
description: Independently judge licensed semantic fit without repairing a candidate prediction.
args:
  hypothesis: GrammarHypothesis
  segmentation: SegmentationReading
  roles: RoleAssignment
  evidence: EvidenceUnit
  rules: SourceClause[]
returns: CompatibilityJudgment
---
1. Capture the actual candidate prediction and original effective consultant judgments in a typed scoped NL CompatibilityJudgment literal. Treat predicted and required meanings as separate channels.
2. Inventory actual asserted fields from alignment.scope and original evidence authority. Record comparedFields; do not compare excluded scene facts as if uttered.
3. For each asserted field compare the actual prediction with the actual accepted meaning or rejected proposal. Distinguish equality, explicit incompatibility and a missing decisive value.
4. If a source-accepted form is outside the candidate's declared closed lexicon, return contradiction with that actual form/domain support. A merely unattested generated form has no such negative judgment.
5. If any actual required comparison is explicitly contradictory, return contradiction naming the affected effective note IDs. Otherwise if a decisive comparison cannot be interpreted, return unknown. Return support only when every actually required comparison is completed and compatible.
6. Preserve future expectation or temporal distinctions when excluded by the elicitation as untested candidate differences, not contradictions caused by missing scene detail.
7. Return actual witness quotations and explanatory reason. No winning candidate, patched gloss or expected label is available; the parent applies this actual verdict.
