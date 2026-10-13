---
description: Judge a proposed compensation against entitlement, custody, identity and observation prerequisites.
args:
  action: RepairAction
  obligation: Obligation
  conditions: PolicyBranch["conditions"]
  context: SearchContext
returns: ActionAssessment
---
1. Call interpretCompensationScope(action, obligation, conditions, context). Its result owns whether the proposed
   subject, cause and branch permit this action and the exact remaining quantity justified by those premises.
2. Independently compare the proposed quantity to that necessaryQuantity. A different positive amount violates
   the obligation even if aggregate cash would accidentally balance after another incorrect action.
3. For release, ensure the subject is this obligation's reservation and excludes signed-custody units and every
   protected nontarget effect. A shared warehouse balance does not authorize releasing another owner's units.
4. For refund, ensure the subject is the rental payment, the cause is this cancellation, and the tenant/key pair
   follows the source action contract. Deposit and rental money are separate purposes despite common ownership.
5. If an action retries an uncertain refund, require a matching absent branch with final-absence evidence after
   its processing boundary. A committed branch must not emit another write for the same entitlement.
6. Combine the child's findings with these identity and quantity checks. Return permitted only when all pass,
   necessaryQuantity, violatedRules and witnesses. The caller uses this result to retain or reject its candidate.
7. Never repair the proposed action inside this function. A rejected action remains evidence of a failed plan
   extension, and any replacement must be explicitly constructed and assessed by expandFrontier.
