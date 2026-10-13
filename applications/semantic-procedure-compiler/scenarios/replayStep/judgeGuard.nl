---
description: Judge one generated rule against actor-owned state and source-governed event-time prerequisites.
args:
  event: ScenarioEvent
  atoms: StateAtom[]
  branch: BranchPlan
  model: ProcedureModel
  module: GeneratedModule
returns: GuardDecision
---
1. Capture branch, governing rule/exception clauses, actor-owned atoms and the generated module's
   actual guard/effect wording. Keep event time separate from when evidence became known.
2. In eval declare GuardJudgment with value:boolean|null, satisfiedPremiseIds:string[], missingPremiseIds:
   string[], waivedPremiseIds:string[], effects:{key:string;subject:string;value:string|number|boolean|null;
   known:boolean;establishedAt:number|null;witnesses:{sourceId:string;path:string;quote:string}[]}[],
   reason:string and witnesses:{sourceId:string;path:string;quote:string}[].
   Construct a scoped NL rule literal and directly await it on the original changing ScenarioEvent.
3. Its instructions are: evaluate active prerequisites for this actor/time; apply only the source-scoped
   waivers; distinguish contradictory facts from missing evidence; interpret the actual generated guard
   and effects, and return the next typed state contributions or the explicit blocked/unknown reason.
4. Consume value and effects in GuardDecision. A source-supported emergency queue waiver contributes
   waivedPremiseIds without adding a queue-completed atom. Custody alone contributes no readiness fact.
5. Reject a future clearance as support for an earlier permission. A source scenario can include a later
   event that enables a later attempt, while its earlier denied action remains in the trace unchanged.
6. Record LocalDecision with the actual event/rule IDs, returned premise/effect distinction and exact
   source witnesses. Nonempty source-grounded effects require exact support; unsupported effects remain unresolved.
7. Return ruleId, eventId, actual judgment fields and localDecision. The caller applies these returned
   atoms directly; a separately invented state transition cannot substitute for literal consumption.
