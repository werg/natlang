---
description: Read original stem and entity definitions without assigning grammatical roles.
args:
  clauses: SourceClause[]
  inventory: CorpusInventory
returns: StemDefinitionIndex
---
1. Select original stem/entity definition clauses from actual inventory roles. Keep the original SourceClause objects and actual inventory; source-role discovery is not itself a stem interpretation.
2. Bind initial:StemDefinitionFrontier with those unique original pending clause IDs, empty entries and unresolved reasons. Preserve the original clause identity and exact text for witness checks.
3. Invoke await advanceStemClause.iterateOn(initial).withMeasure(next => next.pendingClauseIds.length).until(next => next.pendingClauseIds.length === 0). Each named child interprets one original definition and finishes its shorter actual frontier.
4. Consume every returned StemClauseEntry. A selected clause may define a nominal or verb stem, supply context without a new stem, or expose unresolved attribution. Check that the returned frontier unresolved list contains each actualReading.unresolved and each unresolved/failed entry reason once; do not rebuild or discard those child outcomes. Do not invent a stem from an example's final letters or a later preferred segmentation.
5. Project StemReading values only from entries whose status is read and whose actualReading kind is stem. Retain the original surface spelling, source-declared entity/event reference and meaning separately from later event role, topic, boundary or hypothesis judgments; an unresolved or failed unit contributes no accepted stem.
6. Check exact quoted support, original clause ID/path membership and complete pending-unit coverage. Incompatible definitions of the same surface stay unresolved rather than being reconciled with a target translation.
7. Return StemDefinitionIndex with its actual entries, stems and every retained unresolved reason. Intake validates the entries locally and consumes stems plus unresolved reasons in CorpusSource before any segmentation or role interpretation.
