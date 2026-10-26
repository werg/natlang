---
args: { campaign: DebtCampaignRequest }
returns: DebtCampaignProposal
---
1. Bind symptoms = groundWorkflowFriction(campaign). Deduplicate repeated reports of one event while preserving rare high-severity workflows.
2. Bind mechanisms = testDebtMechanisms(symptoms, campaign). Keep staffing or process alternatives visible until evidence distinguishes them from technical causes.
3. Bind packages = scopeFiniteImprovements(mechanisms, campaign). Discover undocumented consumers before deleting an integration, assign interface owners across teams, and rebaseline scope when discovery expands work.
4. Bind waves = sequenceByDependenciesAndCapacity(packages, campaign). Deliver useful slices early and identify irreversible migrations.
5. Bind outcome = compareOperationalImprovement(campaign.baseline, waves, campaign). Retain workload denominators so falling usage cannot masquerade as lower friction; when retiring a workaround, check dependent runbooks and alerts.
6. Return { campaignId: campaign.campaignId, symptoms, mechanisms, packages, waves, outcome, approvalNeeded: campaign.authorityStatus !== 'authorized', commitAllowed: false }.
