---
description: Produce one candidate-specific held-out meaning and retain real child analyses.
args:
  target: TargetRequest
  hypothesis: GrammarHypothesis
  source: CorpusSource
  graph: EvidenceGraph
returns: MeaningReading
uses: [analyzeCorpus/interpretSegmentation, analyzeCorpus/interpretRoles]
---
1. Resolve this actual target utterance and original context clauses. Preserve the supplied completed survivor choices; do not use source repair text as a lexical answer.
2. Call the existing interpretSegmentation(hypothesis, utterance, source, graph) through its declared uses edge, passing the actual returned evidence graph and retaining the actual result. Source correction/reuse context remains real; target interpretation invents no consultant verdict.
3. Resolve the actual target EvidenceUnit from graph by utteranceId. Preserve its actual situation, prompt, effective notes and untested or tested scope. Call the existing interpretRoles(hypothesis, actualSegments, actualTargetEvidence, source) and retain its actual prediction. Missing target evidence remains unresolved, not a fabricated empty graph.
4. Call judgeContext(target, hypothesis, actualRoles, source). Bind actualContext, which resolves source-established discourse reference and preserves candidate temporal/negation differences.
5. A contextual interpretation cannot silently change a candidate's predicted agent/patient or marker semantics. If context explicitly conflicts, preserve the actual disagreement/unknown rather than translating under another survivor.
6. Assign readingId as a unique generated identity for this actual target/hypothesis reading; retain it unchanged in all later groups and references. Retain exact original surface/context witnesses and all actual segmentation/role/context results in MeaningReading. Parent consumes these actual structured fields, not merely a free gloss string.
7. Return one actual reading for the supplied survivor and target; the sibling comparison stage owns semantic grouping across survivors.
