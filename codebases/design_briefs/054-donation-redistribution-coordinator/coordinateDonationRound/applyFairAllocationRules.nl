---
args: { bundles: PurposeFitDonationBundle[], round: DonationRoundRequest }
returns: FairDonationAllocation
---
Apply the exact supplied priority and fairness principles to feasible bundles subject to confirmed quantities and transport constraints. Compare unmet needs as well as completed deliveries so easy routes do not consume all resources by default. If policy terms conflict or no rule is supplied, hold allocation for coordinator review rather than inventing deservingness criteria.
