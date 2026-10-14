---
description: Compare actual candidate prediction against the licensed evidence scope.
args:
  hypothesis: GrammarHypothesis
  segmentation: SegmentationReading
  roles: RoleAssignment
  evidence: EvidenceUnit
  source: CorpusSource
returns: CompatibilityJudgment
---
1. Bind the actual incoming candidate, segmentation and role prediction. Keep actual prediction fields immutable; separate them from the consultant judgment and depicted scene.
2. Read actual alignment.scope asserted/excluded fields and selected effective notes. A scene fact excluded by the actual response scope cannot become an extra rejection test.
3. Call judgeCompatibility(hypothesis, segmentation, roles, evidence, source.rules). Retain its actual support/contradiction/unknown judgment and compared field inventory.
4. A licensed positive surface judgment can contradict an explicitly unlicensed candidate form. A rejected student gloss contradicts that proposed meaning only within the actual corrected slot; it does not automatically reject every candidate mentioning the same stem.
5. Validate contradictedNoteIds against actual effective selected note IDs and exact quoted original support. Preserve the actual first contradictory comparison rather than composing a post-hoc preferred explanation.
6. Require unknown when a decisive asserted field or judgment authority is missing. An excluded future/temporal field alone is not a blocker for complete comparison of the actually asserted fields.
7. Return actual compatibility unchanged so advanceHypothesis owns status application and evidence-prefix retention.
