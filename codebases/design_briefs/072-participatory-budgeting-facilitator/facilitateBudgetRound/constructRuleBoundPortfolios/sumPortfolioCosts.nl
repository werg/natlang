---
description: Sum proposal and shared-work costs under published rules using exact money arithmetic.
args: { dossier: BudgetDossier; workGraph: PortfolioLedger; feasibility: ProposalFeasibility[] }
returns: PortfolioLedger[]
---
Construct candidate proposal combinations only under the published allocation rules and the current feasibility records. For each candidate, bind one supported proposal-specific cost line per selected proposal and one shared enabling-work line per unique supported node whose enabled-proposal set intersects the candidate. Preserve unresolved nodes and missing estimates as named cost issues; never treat them as zero. Bind total = `exactMoneySum(costLines, dossier.round.availableBudget)`, which sums integer minor units and rejects mixed currency, scale mismatch, invalid digits, duplicate-line conflicts and unsupported source lines. Return a `PortfolioLedger` for each candidate with the exact total, budget comparison, unique shared nodes and all unresolved costs; do not round or rank candidates here.
