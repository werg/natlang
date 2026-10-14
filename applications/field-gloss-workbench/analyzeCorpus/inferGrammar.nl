---
description: Reduce a finite actual candidate-evidence frontier.
args:
  source: CorpusSource
  graph: EvidenceGraph
returns: GrammarReport
---
1. Bind inventory to seedCandidates(source, graph). Construct actual hypothesis copies and the finite original hypothesis/evidence pair inventory for actual graph units whose returned response scope is usable; retain unusable/unresolved units and reasons in the actual graph rather than testing an invented verdict; preserve every declared slot combination unless an exact malformed domain blocks it.
2. Order each hypothesis's evidence pairs by actual elicitation/correction dependency order. Concurrent unrelated units may use stable original order for execution, without making that order semantic authority.
3. Bind initial:GrammarFrontier to actual source/graph/inventory, full pending pairs, pending hypothesis statuses, empty attempts/skips and actual unresolved domain reasons.
4. Invoke await advanceHypothesis.iterateOn(initial).withMeasure(next => next.pending.length).until(next => next.pending.length === 0). Each child consumes one original pair and returns the actual smaller state.
5. Consume the final actual frontier. A first contradiction fixes a rejection prefix; later pairs for that rejected hypothesis are accounted skips, not fabricated successful tests.
6. Call closeEvidenceRound(actualFrontier). It owns surviving/unresolved distinctions and prefix proof completeness; do not select a single survivor by arbitrary candidate order.
7. Return the actual GrammarReport with all attempts, rejection proofs and unresolved applicability. Finite ambiguity stays a set of actual analyses for translation and note repair.
