---
description: Prepare community budget deliberation from proposal dependencies, exact shared-work costs and published allocation rules.
args: { dossier: BudgetDossier; change: PortfolioChange; facilitator: BudgetFacilitator }
returns: BudgetProposalSession
---
Preserve each proposal's original intent and proposer. Treat estimate revisions, scope mergers, location constraints and post-selection delivery changes as distinct events.

Bind proposals = `interpretProposalIntent(dossier, change)` to bind each requested benefit and required work to its source. Bind workGraph = `reconcileEnablingWork(dossier, change, proposals)` to construct a dependency graph. A shared task is one cost node and its expense is counted once, while its proposal edges remain explicit.

Bind feasibility = `qualifyBudgetFeasibility(dossier, change, proposals, workGraph)`. Reopen every feasibility record whose cost, constraint or dependency changed. A revised estimate cannot reuse a prior feasible status without rechecking affected portfolio totals.

Bind portfolios = `constructRuleBoundPortfolios(dossier, change, workGraph, feasibility)`. Merge complementary work only when each proposer explicitly reviews changed scope. Show location-sensitive constraints only at their relevant supplied context; do not profile residents.

Bind packet = `prepareDeliberationPacket(dossier, change, portfolios, feasibility, workGraph)`. Show exact totals, dependencies, exclusions, assessments and tradeoffs under published rules. A selected project that changes during delivery remains a separate scope-change item under the applicable approval rule.

Return `BudgetProposalSession` for `facilitator.id` with affected portfolios invalidated and a resumable budget graph. Deliberation material is not a vote, approval or procurement authorization.
