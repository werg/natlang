---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftDonationAllocation
---
Construct a DonationAllocation draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Apply supplied fairness and priority principles to feasible bundles, using exact inventory and transport constraints. Explain tradeoffs and preserve unmet needs without inventing deservingness criteria or optimizing only the easiest deliveries.
