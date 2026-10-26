---
args: { request: DebtCampaignRequest, proposal: DebtCampaignProposal, receipt: DebtMeasurementReceipt }
returns: DebtOutcomeComparison
---
1. Bind receipt.packageId to one authorized package in the campaign and receipt.campaignId to the same campaign. Require actual source references for baseline and post-change observations.
2. Compare measureDefinition, denominator, workload, and observation window to the supplied debt baseline and package acceptance contract. A changed population, route, measurement meaning, or window is not a direct improvement comparison; return unknown and require a comparable observation.
3. When context is comparable, use normalizedSignal as the exact measured direction. Do not calculate numeric deltas from the descriptive measure text. If the same workflow symptoms persist or workaroundStatus shows the team still uses the old workaround, retain those residual findings even when the metric improves.
4. Return comparable, disposition, and the exact contextual mismatch or remaining symptoms. No task completion claim is an outcome measure.
