---
description: Compose feasible budget portfolios and count shared enabling work exactly once.
args: { dossier: BudgetDossier; change: PortfolioChange; workGraph: PortfolioLedger; feasibility: ProposalFeasibility[] }
returns: CivicPortfolio[]
---
Call `sumPortfolioCosts(dossier, workGraph, feasibility)` for each allowed proposal combination under the published rule set. Compare the exact total with available budget. Do not count shared work twice or infer a missing estimate. Combine complementary proposals only after each affected proposer reviews the merged scope; preserve their original identities and benefits. Report tradeoffs and any combination excluded by constraints.
