---
description: Judge one physical connection's existence, movement direction and source-scoped conditions.
args:
  unit: SurveyReadingUnit
  primaryClauses: SourceClause[]
  supportingClauses: SourceClause[]
  frames: FrameSurvey
returns: PassageItemMeaning
---
1. Preserve this connection reading unit and its exact original clauses. Capture the actual framed
   landmark keys and supporting clauses; its varying input is the unit plus primary clauses. Do not
   supply a prepared endpoint/direction/cost row or a chosen world hypothesis.
2. In eval create a scoped NL literal returning PassageItemMeaning and directly await it. It judges
   physicalExistence independently from allowed travel direction and carried-state permission.
3. A physically established gate/tunnel/chimney remains present even when reverse travel or a loaded
   carrier is prohibited. Orient one-way from/to in its allowed direction and retain reverse-prohibited
   evidence. A two-way passage retains both-direction meaning subject to the actual load condition.
4. Explicitly missing physical connections retain absent; unestablished presence retains unknown.
   Use original endpoint keys in their actual frame, including local origin versus its global anchor.
   Negation, direction and load scope cannot be discarded merely to create a convenient travel edge.
5. Join cost evidence only when it governs this connection. A known cost follows the actual source
   action-cost contract; unsupported cost stays null. Condition preserves rope, load and timing scope
   as source-grounded language for later robot-state judgment, not a precomputed legal-action answer.
6. Return one nullable assertion plus nonempty clause-keyed exact evidence and unresolved premises.
   Every decided assertion needs actual support. Consume and return this literal result unchanged;
   the parent owns exact source binding, while world construction separately projects allowed edges.
