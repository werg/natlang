---
description: Judge one scoped override using authority, role, time and exception structure.
args:
  candidate: {targetProofId:string;defeatingNormId:string;applicationId:string}
  state: ClosureState
returns: DefeatDecision
---
1. Resolve the target proof, proposed defeating norm, its applicability judgment and the governing
   change relation. Gather exact competence, actor/reach/time and exception-priority source clauses.
2. Capture this context. In eval declare a local DefeatJudgment matching DefeatDecision, create
   nl.with<DefeatJudgment>, and directly await it on this one proposed target/override pair.
3. Ask whether the source authorizes this override of this proof. Require both competence and
   applicability; later publication alone is not priority, and an accepted reservation has only its debtor scope.
4. When an open condition decides the override, return defeated:null with that premise. A true
   result names the actual target and defeating norm; false preserves the target's existing support.
5. Consume the returned decision in closure. Apply the source rescue exception when deciding
   whether this target is an ordinary or protected permission proof; retain unrelated notice uses of the order.
6. Return exact premises and witnesses. The assessment is a local contribution to the proof graph,
   not a summary that invents an entire ledger outside the declared closure stages.
