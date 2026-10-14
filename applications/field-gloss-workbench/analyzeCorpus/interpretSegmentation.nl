---
description: Predict morpheme boundaries from one actual candidate and original surface.
args:
  hypothesis: GrammarHypothesis
  utterance: UtteranceReading
  source: CorpusSource
  graph: EvidenceGraph
returns: SegmentationReading
---
1. Bind the exact original utterance.surface and this hypothesis's declared boundary/productivity choices. Obtain actual source lexical stems and note/correction evidence without selecting a winning grammar.
2. Inventory each exact whitespace-delimited surface token and its original offsets. Exact token slicing is mechanism; identifying a suffix versus a frozen stem is a semantic candidate prediction.
3. For each token call judgeBoundary(tokenStart, tokenEnd, hypothesis, utterance, source, graph). Capture its actual segmentation, licensed verdict and original quoted reasoning.
4. Consume every actual child span, preserving the current candidate's alternative morphological analysis even when another candidate would segment the word more plausibly.
5. Check each span is a nonempty exact original substring and the ordered child segments reconstruct the original token without overlap/gaps. Parent binds original identity; it does not supply a corrected boundary on failure.
6. licensed=false requires an actual candidate-domain exclusion, such as a frozen lexicon lacking the observed form. Unknown stem/boundary evidence stays null with its actual blocker.
7. Return actual morphemes, raw boundary decisions and unresolved fields for this utterance/hypothesis. Corrected translation authority affects later compatibility, not exact surface extraction.
