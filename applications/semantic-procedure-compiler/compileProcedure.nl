---
description: Compile evolving prose into a typed callable NL program and exercise its semantic obligations.
kind: directory-reducer
args:
  request: CompilerRequest
  snapshot: ProcedureSnapshot
uses: [editions, requirements, model, lowering, writing, scenarios, audit]
returns: CompilePacket
---
1. Call intake(request, snapshot). Keep its original records and selected scenario histories intact.
   A source file is evidence for the compiler, not an already executable definition of the policy.
2. Call editions(intake). Consume authority judgments, temporally scoped selected clauses and preserved
   unchanged requirements. If a decisive authority is unresolved, retain the diagnostic in the model.
3. Call requirements(intake, selection), naming its result requirementSet. Consume duties, permissions, forbidden consequences and
   exception bindings. Keep a waived queue requirement separate from a surviving safety prerequisite.
4. Call model(intake, requirementSet). Its state fields describe distinct facts and their owners; its
   dependency graph orders only source-supported joins and preserves unknown or cyclic constraints.
5. Call lowering(procedureModel, intake.goal). Consume ModulePlans with typed interfaces, owned fields,
   actual named child calls and contextual literal captures/consumers. A large module count is not a
   reason to invent redundant forwarding stages.
6. Call writing(modulePlans, procedureModel, request). Consume the actual GeneratedProgram with
   numbered NL bodies, source mapping and contribution audits. Keep generated policy out of crisp code.
7. Call scenarios(intake.scenarios, program, procedureModel). Each returned scenario trace must consume
   actual guard judgments into state updates; scenario labels alone do not certify generated behavior.
8. Initialize RevisionState with the failed scenario IDs as remainingScenarioIds, empty targets and
   the original model/intake. Invoke folder.iterateOn(reviseStep, state) with measure
   remainingScenarioIds.length until every scenario is settled as repaired or an explicit unresolved
   source/program defect. That reducer owns its explainCounterexample child and returns actual targets.
9. Consume the returned program and suite from revision, retaining each target's source-grounded cause.
   A repair changes generated policy or its missing state propagation, not the original source authority.
10. Call audit(intake, procedureModel, revisedProgram, revisedSuite). An interface success cannot hide
    missing source support; a passing normal scenario cannot hide an adversarial exception violation.
11. Call packageProgram(request, procedureModel, revisedProgram, revisedSuite, audit, revisionState.targets).
    Return its full CompilePacket. Private folder generation remains a proposal until all exact path,
    source and compiler checks pass; semantic and live execution qualification are separately recorded.
