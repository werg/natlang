---
description: Choose callable ownership from semantic reuse and captured context rather than desired call depth.
args:
  rules: RuleMeaning[]
  schema: StateSchema
  goal: CompilationGoal
returns: CallBoundary
---
1. Read which rule predicates change with actor, event-time knowledge and exception context, and
   which semantic responsibility is reused unchanged across scenarios or procedure editions.
2. Capture those rules and state-field producers. In eval declare BoundaryJudgment with kind:'named'|
   'scoped-literal', ownedFields:string[], captures:string[], consumerFields:string[], reason:string and
   witnesses:{sourceId:string;path:string;quote:string}[]. Construct a scoped NL ownership literal.
3. Directly await the literal on this proposed rule group. Its instructions are: choose one coherent
   responsibility, identify the actual returned decision fields and concrete downstream consumption,
   and select reuse vs contextual closure based on source dependencies rather than a file-count target.
4. Consume kind/captures/consumerFields into CallBoundary. Build typedInputs from the schema fields
   required by the judgment and typedOutput from its owned semantic contribution, retaining nullability.
5. If the proposed group has no distinct owned contribution, combine it with the consuming parent
   or preserve an unresolved boundary finding. An unused literal is not a semantic decomposition.
6. Record LocalDecision after the actual returned boundary judgment. Return ruleIds, kind, typedInputs,
   typedOutput, captures, consumerFields, reason and that decision for module planning and contribution audit.
