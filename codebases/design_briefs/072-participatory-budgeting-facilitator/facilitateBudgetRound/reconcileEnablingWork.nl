---
description: Build a dependency graph that represents shared enabling work once while preserving proposal edges.
args: { dossier: BudgetDossier; change: PortfolioChange; proposals: CivicProposal[] }
returns: PortfolioLedger
---
Call `identifySharedWork(dossier, proposals)` and preserve the evidence that links each proposal to an enabling task. Represent one genuinely shared task as one cost node with multiple dependent proposal IDs; do not count it once per proposal. Keep complementary work, duplicate text, and conflicting intent distinct. If a merge changes one group's intent, add that proposer's explicit review as a blocker.
