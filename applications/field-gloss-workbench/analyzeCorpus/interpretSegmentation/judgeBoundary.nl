---
description: Judge one candidate-specific boundary using actual productive and note evidence.
args:
  tokenStart: number
  tokenEnd: number
  hypothesis: GrammarHypothesis
  utterance: UtteranceReading
  source: CorpusSource
  graph: EvidenceGraph
returns: BoundaryJudgment
---
1. Capture only this original token span, the actual hypothesis choices, stem/domain definitions and relevant original reuse/correction evidence in a typed scoped NL BoundaryJudgment literal.
2. Read the declared candidate boundary rule before looking for a convenient gloss. Under productive analysis, identify source-licensed stem plus affix spans; under frozen-form analysis, consult its actual closed lexical inventory without extending it.
3. A repeated ending across several known stems can be evidence of productivity, but the current frozen hypothesis still predicts its own complete form. Preserve that prediction for compatibility rather than silently turning it productive.
4. Join explicit transcription corrections to their actual slot. A gloss-only correction leaves the exact form and available boundary evidence unchanged.
5. Return ordered segments with exact start/end/surface, source-defined stem/function and actual entity/event link when established. Distinguish unlicensed form from unresolved semantic stem identity.
6. Every decided boundary cites the actual defining domain/reuse clauses. A null verdict retains the missing premise; no invented segmentation fills it.
7. Return the actual local judgment. Parent checks exact substring reconstruction and consumes licensed/morphemes; later linguistic stages consume this same candidate-specific analysis.
