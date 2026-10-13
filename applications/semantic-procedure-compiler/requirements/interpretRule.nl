---
description: Read one actor, prerequisite and effect under its governed clause scope.
args:
  clause: SourceClause
  goal: CompilationGoal
  selection: EditionSelection
returns: RuleMeaning
---
1. Read the complete clause and any explicitly linked definitions. Identify the actor who owes or
   receives the act, the action itself, and the event interval/territory to which it applies.
2. Separate necessary prerequisites from evidence that merely suggests them. A custody receipt can
   establish possession; it cannot establish that its recipient understood a separate safety brief.
3. Name each source prerequisite with its visible clause ID and a concrete semantic predicate.
   Propose stateKeys for the facts that the predicate needs, retaining actor identity and event time.
4. Interpret the effect as a state change, permission, obligation or prohibition. Keep those kinds
   separate: a completed handover can change custody without automatically changing readiness.
5. Keep explicit positive and negative conditions distinct. An unresolved hazardous-state report
   must not be rewritten as a proved absence of hazard merely because no alarm file was selected.
6. Preserve this clause's selected historical interval; the same action can have different governing
   requirements before and after a supported source replacement boundary.
7. Return id, clauseIds, actorRole, action, prerequisiteMeanings, effectMeaning, scopeMeaning,
   initially empty exceptionBindings, exact witnesses and unresolved interpretations. Downstream
   schema and branch construction consume every role/predicate/effect field rather than re-reading it.
