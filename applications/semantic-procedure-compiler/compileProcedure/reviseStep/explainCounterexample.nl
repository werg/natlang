---
description: Identify the first semantic divergence and the missing generated contribution that caused it.
args:
  verdict: ScenarioVerdict
  program: GeneratedProgram
  model: ProcedureModel
returns: RevisionTarget
---
1. Read the actual trace in event-time order and locate the earliest violated observable or missing
   source-grounded transition. Keep a later recovered final state separate from this earlier failure.
2. Compare the failing decision's selected rule, prerequisite scope and before/after atoms with its
   generated module body. Identify whether the first divergence is rule selection, exception scope,
   state ownership, unknown handling or a missing literal/child return consumed by its parent.
3. Follow sourceToModule and branch ownership to affectedRuleIds and affectedPaths. Name the exact
   source clauses that establish the intended behavior; do not propose a patch based only on a label.
4. Express repairObligation as a semantic transformation the corrected module must perform, such as
   retaining safety after queue waiver or carrying actual holder identity through the handover state.
5. If the governing source itself lacks a decisive premise, return earliestEventId with unresolved
   and an evidence obligation. A counterexample does not grant authority to create a new policy.
6. Return stable id, scenarioId, earliestEventId, affectedRuleIds, affectedPaths, violatedMeaning,
   repairObligation, exact witnesses and unresolved. The parent consumes this actual localized target.
