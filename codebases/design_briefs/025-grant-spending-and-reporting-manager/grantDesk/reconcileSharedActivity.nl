---
args: { request: GrantCycleRequest, links: EvidencePage[] }
returns: SharedActivityResult
---
Match each activity and its cost allocation across grant IDs. Return acceptedActivityLinks only when each funded share has its own documented basis, distinct eligible cost amounts and nonduplicated outcome attribution. Put unsupported or duplicated links in withheldLinks; preserve exact source references and explain every blocker. Prevent duplicate full attribution.

For each candidate shared activity, bind decision = evaluateOneSharedActivity(request, candidateActivity, links). Add only decision.acceptedGrantShares to acceptedActivityLinks; preserve withheldGrantShares and duplicateBenefitIds as blockers for that specific activity.
