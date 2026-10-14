---
description: Inventory original clause roles with actual one-item progress and no linguistic conclusions.
args:
  clauses: SourceClause[]
returns: CorpusInventory
---
1. Check unique original clause IDs and immutable original source text/identity. Bind initial:CorpusReadFrontier to actual clauses, pending original IDs, empty entries and no invented linguistic interpretation.
2. Invoke await advanceClause.iterateOn(initial).withMeasure(next => next.pendingClauseIds.length).until(next => next.pendingClauseIds.length === 0). Each child owns one original clause and an actual typed role return.
3. Consume the actual final entries. One original clause can contribute several role labels. A source-uncertain role judgment keeps its original clause and stated missing premise; a malformed child return cannot become a completed entry or remove its pending original ID.
4. Require exactly one completed entry per original pending clause and no changed original SourceClause. Coverage is original identity accounting, not a claim that every clause determines a grammar feature.
5. Keep context-only and authorization clauses available even if they supply no elicited utterance. Role discovery does not choose a suffix meaning, grammatical agent or survivor hypothesis.
6. Return the actual CorpusInventory and its source-unresolved reasons after every original ID has one valid typed role return. Intake selects original source objects through these actual roles before calling substantive session/note/correction readers.
7. If a literal overproduces linguistic conclusions, retain only its declared ClauseRoleReading contract; later named children own those linguistic decisions with their proper typed evidence.
