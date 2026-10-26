---
args: { packages: DebtImprovementPackage[], campaign: DebtCampaignRequest }
returns: DebtCampaignWaves
---
Construct prerequisite links from package interfaces and migration evidence. Start with packages whose dependencies and team capacity fit the declared horizon. Preserve early user-visible improvements between waves; reject an order whose prerequisites cycle or whose owner acceptance is missing. Mark irreversible steps with their explicit approval boundary.
