---
description: Interpret event roles and topic scope under a supplied actual grammar hypothesis.
args:
  hypothesis: GrammarHypothesis
  segmentation: SegmentationReading
  evidence: EvidenceUnit
  source: CorpusSource
returns: DiscourseRoleJudgment
---
1. Directly invoke a typed scoped NL DiscourseRoleJudgment literal capturing actual choices, actual returned morphemes, original utterance and source lexical/domain definitions. Keep consultant verdicts separate from the prediction.
2. Interpret each candidate marker meaning exactly as that candidate declares it. Return one actual RoleContribution for each source-linked entity/role with its actual morpheme span and defining quote. A topic contribution is distinct from an event-role contribution, even when the same entity occupies both.
3. Join actual contributions under the supplied source grammar-domain constraints. If two distinct entities are assigned a single required agent slot and the source prohibits that combination, retain both contributions and set licensed=false with the actual conflicting premise. If missing meaning prevents a judgment, licensed=null. For a source-permitted unique assignment use licensed=true; genuine remaining alternatives retain null role fields and explicit reasons rather than surface-order precedence.
4. Interpret the actual verb and negation candidate. Separate eventOccurs=false from an additional not-yet/future expectation contribution; silence about the future in a depicted scene does not negate that candidate contribution.
5. Use actual prompt/contrast context only where the supplied candidate explicitly permits a contextual interpretation. Do not substitute a situation role to make a marker prediction acceptable.
6. Return actual contributions and licensed verdict beside every decided entity/event/topic relation with source quotes and undecided fields with their actual missing premise. Parent preserves the actual result for compatibility and translation.
7. A contradictory-looking prediction is still this candidate's prediction. Rejection belongs to the later independent compatibility child, not a hidden correction inside this literal.
