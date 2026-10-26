---
description: Reopen feasibility for proposals affected by changed costs, constraints or enabling-work edges.
args: { dossier: BudgetDossier; change: PortfolioChange; proposals: CivicProposal[]; workGraph: PortfolioLedger }
returns: ProposalFeasibility[]
---
A changed cost estimate invalidates the feasibility record and every portfolio that depends on it. Recheck cost, location constraints and evidence status against current sources. Show relevant supplied constraints at their own scope without profiling residents. Keep feasibility distinct from desirability or vote support; missing assessment remains a blocker.
