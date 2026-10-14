---
description: Predict event and discourse roles from actual candidate segmentation.
args:
  hypothesis: GrammarHypothesis
  segmentation: SegmentationReading
  evidence: EvidenceUnit
  source: CorpusSource
returns: RoleAssignment
---
1. Bind the actual returned segmentation and current hypothesis's marker/negation meanings. Preserve its licensed/null status and original utterance identity.
2. Collect exact entity/event links from actual morphemes and source stems. A scene's known agent/patient remains contextual comparison evidence; it is not a default filling an undecided candidate role.
3. Call judgeDiscourseRole(hypothesis, segmentation, evidence, source) with these actual objects. Its scoped result predicts event role, topic and temporal/negation meaning independently.
4. Consume actual child licensed/contributions and agent/patient/topic/event fields. Do not exchange topic and agent just because the scene contains a contrastive actor.
5. If a candidate predicts an established role conflict, retain licensed=false and the actual conflicting contributions. Missing role evidence retains licensed=null. For an unlicensed morpheme retain that actual prediction and blocker. Later compatibility owns rejection; this stage cannot repair the candidate to agree with the consultant.
6. Validate returned original entity/event IDs and exact quoted source definitions, then retain the actual segmentation and actual child meaning in RoleAssignment.
7. Return source-grounded predicted meaning for this candidate/evidence pair, including consequential unknown fields rather than replacing them with scene truth.
