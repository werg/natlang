---
description: Sum proposal and shared-work costs under published rules using exact money arithmetic.
args: { dossier: BudgetDossier; workGraph: PortfolioLedger; feasibility: ProposalFeasibility[] }
returns: PortfolioLedger[]
---
Use integer minor units with matching currencies and scales. Count each unique enabling-work node once per portfolio, plus proposal-specific costs in its selected set. Mark unresolved any portfolio with unknown cost, mixed currency without an approved conversion, or an unsupported shared-work relation. Compare totals with available budget without rounding.
