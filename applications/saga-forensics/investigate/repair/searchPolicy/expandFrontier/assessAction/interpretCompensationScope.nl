---
description: Use a narrow scoped judgment to interpret the rollback boundary for one proposed action.
args:
  action: RepairAction
  obligation: Obligation
  conditions: PolicyBranch["conditions"]
  context: SearchContext
returns: ActionAssessment
---
1. Select the cancellation, custody, deposit and retry clauses relevant to action.kind and its order. Select only
   the ledger entries and observation condition needed to decide this action's remaining entitlement.
2. Form a local scope object containing those clauses, the obligation's quantities, protected effect IDs and
   relevant branch evidence. Do not capture the complete search frontier or every source document.
3. Declare a local CompensationJudgment type matching ActionAssessment. Create a dynamic NL literal with
   nl.with<CompensationJudgment> capturing scope. Directly await it with the proposed action as its input.
4. The literal's instructions are: interpret which part of this obligation is reversible in this branch; compute
   the exact remaining authorized quantity from these premises; identify custody, deposit, wrong-cause or
   uncertain-retry violations; return permission, necessaryQuantity, violatedRules and exact source witnesses.
5. Consume and return the actual literal result. It supplies the semantic rollback premise that assessAction
   combines with exact field/quantity comparisons. It is not a decorative explanation beside a separate answer.
6. If the scoped premises do not decide the boundary, return permitted false with the missing proposition.
   Do not widen the capture to unrelated evidence merely to obtain a confident answer.
