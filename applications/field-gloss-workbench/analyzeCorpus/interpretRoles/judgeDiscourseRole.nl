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
2. Interpret each candidate marker meaning exactly as that candidate declares it. A topic marker contributes discourse topic, while a role marker contributes agent/patient; neither automatically supplies the other's relation.
3. Join the remaining nominal roles by actual source grammar-domain constraints. If several assignments remain licensed, preserve null fields and explain the genuine alternatives rather than imposing surface-order precedence.
4. Interpret the actual verb and negation candidate. Separate eventOccurs=false from an additional not-yet/future expectation contribution; silence about the future in a depicted scene does not negate that candidate contribution.
5. Use actual prompt/contrast context only where the supplied candidate explicitly permits a contextual interpretation. Do not substitute a situation role to make a marker prediction acceptable.
6. Return every decided entity/event/topic relation with source quotes and undecided fields with their actual missing premise. Parent preserves the actual result for compatibility and translation.
7. A contradictory-looking prediction is still this candidate's prediction. Rejection belongs to the later independent compatibility child, not a hidden correction inside this literal.
