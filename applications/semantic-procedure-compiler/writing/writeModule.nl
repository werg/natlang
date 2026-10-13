---
description: Emit one substantive natural-language callable with explicit child and literal data flow.
args:
  plan: ModulePlan
  model: ProcedureModel
  allPlans: ModulePlan[]
returns: GeneratedModule
---
1. Write the `.nl` header from plan.inputs and outputType, preserving named typed args and any required
   directory-reducer kind. Child callable paths and shared departments follow the returned call graph.
2. Author numbered pseudocode over named data: locate input state, select source-context branches,
   await planned child functions, consume their typed fields and compute this stage's owned contribution.
3. For each BranchPlan spell out its true, false and unknown path. Explain which changed fact moves
   the procedure to a later branch and which actor/time information remains local to this call.
4. At a scoped-literal boundary declare the local result type and captured rule/owner/evidence context.
   Formulate semantic instructions, directly await the literal on the changing state/event, then use
   its returned guard/effect/witness fields to construct this module's actual output contribution.
5. Include exact provenance binding as mechanism after semantic quote selection. Original record ID
   and path come from typed evidence; a nonempty semantic contribution needs nonempty exact supporting
   quotations. Missing support is unresolved rather than invented narration or a fabricated event.
6. Where this module iterates, name its finite frontier, decreasing measure and settled unknown state.
   A failed guard consumes the examined event; it does not requeue an unchanged scenario indefinitely.
7. Call auditContribution(plan, text, allPlans). Consume the actual semantic audit. If a child result
   is unused or a stage only forwards, revise this module's own body/ownership or expose the defect.
8. Return path, actual text, plan and contributionAudit. The code is deep structured NL pseudocode;
   crisp snippets are confined to exact data lookup, joins, arithmetic, quotation and serialization.
