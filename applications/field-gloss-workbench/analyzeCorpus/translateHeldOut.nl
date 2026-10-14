---
description: Interpret actual held-out targets through retained grammar alternatives.
args:
  targets: TargetRequest[]
  source: CorpusSource
  graph: EvidenceGraph
  grammar: GrammarReport
returns: TranslationReport
---
1. Validate each original target utterance/context identity and preserve request order. Bind initial:TranslationFrontier to actual source/evidence graph, actual grammar, original pending targets and empty completed readings.
2. If grammar has no completed survivor, retain the actual rejected/unresolved distinction and target-specific blockers. Do not invent a preferred grammar solely to produce a fluent translation.
3. Invoke await advanceTranslation.iterateOn(initial).withMeasure(next => next.pending.length).until(next => next.pending.length === 0). Each invocation owns one original target.
4. Consume the actual completed TranslationAlternatives returned for each target. Preserve every actual per-hypothesis meaning and equivalence group, rather than a prose summary of guessed grammar.
5. Require target coverage and no changed original source strings/contexts. A held-out target is interpreted, not retroactively inserted as consultant training evidence unless the source authorizes that separate process.
6. Retain consequential ambiguity fields and missing source premises independently. Two surviving grammar analyses with equivalent target meaning may share a group without becoming one global grammar.
7. Return actual targets and unresolved reasons; note repair consumes these returned distinctions.
