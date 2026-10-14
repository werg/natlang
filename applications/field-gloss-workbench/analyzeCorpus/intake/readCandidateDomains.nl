---
description: Read finite candidate domains from their original definitions without choosing a winner.
args:
  clauses: SourceClause[]
  inventory: CorpusInventory
  rules: SourceClause[]
returns: CandidateDomainIndex
---
1. From actual inventory roles select original candidate-domain clauses as the finite pending units. Retain the full original clauses and governing rules as context. A role label selects a source unit; it does not decide that the unit supplies a candidate value.
2. Bind initial:CandidateDomainFrontier with those original pending clause IDs, their immutable source clauses, the actual inventory and rules, empty entries and unresolved reasons. Check that every pending ID identifies one original clause.
3. Invoke await advanceDomainClause.iterateOn(initial).withMeasure(next => next.pendingClauseIds.length).until(next => next.pendingClauseIds.length === 0). Each step judges one original definition and returns its actual shorter typed frontier.
4. Consume the actual returned DomainClauseEntry for every pending original clause. A definition may declare a finite slot, explain how candidates are used, or leave an unresolved premise. Keep contextual definitions even if their slots array is empty. Check that the returned frontier unresolved list contains each actualReading.unresolved and each unresolved/failed entry reason once; do not rebuild or discard those child outcomes.
5. Project CandidateSlot values only from entries whose status is read and whose actualReading kind is domain. Preserve source-declared values and meanings; do not project slots from unresolved or failed units, and do not rank values using observed utterances, target questions, or the later grammar survivor set.
6. Check that each decided slot has a source-backed nonempty finite value list, original slot identity and exact original defining quote. Duplicate slot IDs with incompatible definitions remain unresolved instead of silently overwriting one another.
7. Return CandidateDomainIndex with original coverage entries, actual slots and every retained unresolved reason. Intake validates the entries locally and consumes slots plus unresolved reasons; downstream seedCandidates decides the finite cross-product without altering their source meanings.
