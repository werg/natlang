---
description: Partition the semantic procedure into task-specific callable responsibilities and guarded branches.
args:
  model: ProcedureModel
  goal: CompilationGoal
returns: ModulePlan[]
---
1. Inventory source-supported rule components, required state fields and observable responsibilities.
   Group operations only when they share one coherent semantic responsibility and state owner.
2. For each proposed group call chooseBoundary(rules, schema, goal). Consume its named/scoped-literal
   choice, typed input/output, captures and concrete consumer fields. A repeated semantic decision
   deserves a reusable stage; a judgment whose rule/state context changes can be a scoped literal.
3. For each rule call planBranch(rule, schema). Consume its guard and true/false/unknown effects;
   retain unknown as an explicit downstream state or diagnostic instead of silently choosing false.
4. Assign stable relative paths from the visible package conventions and rule responsibility. Place
   named child modules inside their caller's callable folder, exposing reused departments explicitly.
5. Infer each ModulePlan.calls from genuine dependencies. Every call must return a declared field
   consumed by that parent's own transformation; a stage that merely relays the final packet is merged.
6. Preserve multiple independent call paths: edition selection, source exception binding, actor readiness,
   custody updates, permission guards and counterexample audit should not become one uniform chain.
7. Return complete ModulePlans with responsibility, ruleIds, typed interfaces, ownedFields, calls,
   boundaries, branches and witnesses. Writing uses these returned plans, not a fixed file template.
8. Retain an unresolved model finding in the responsible branch plan. Lowering cannot make the source
   more determinate merely because a generated TypeScript signature prefers a nonnullable value.
