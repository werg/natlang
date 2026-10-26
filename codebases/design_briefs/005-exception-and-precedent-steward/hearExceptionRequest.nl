---
args: {"hearing": "ExceptionHearing", "register": "WaiverRegister | null"}
returns: "ExceptionHearingResult"
---
1. Start from the existing waiver register or an empty revision. Bind identity = linkResubmittedRequests(hearing, prior). Reworded rejected requests retain case identity, but new facts must be examined rather than assumed unchanged.
2. Bind boundaries = recoverAuthorizedWaivers(hearing.caseHistory, hearing.currentRules). Preserve the actual waived requirement separately from prerequisites and surviving controls. Missing expiry is a governance defect, not perpetual permission.
3. For each current request bind analogies = retrieveRationaleContrasts(request, boundaries, hearing). Include relevant rejected and expired precedents. Bind argument = hearLimitingFacts(request, analogies, boundaries, hearing). A similar successful request is only a candidate rationale; it grants no authority.
4. Bind currentArguments = recheckPendingPolicy(arguments, prior, hearing). Any governing policy change invalidates pending signing applicability and returns the affected case for re-evaluation. Preserve already valid historical actions under their actual policy and effective dates.
5. Only if the supplied independent case history supports a recurring boundary problem, bind trial = examinePolicyBoundaryPattern(currentArguments, boundaries, hearing); otherwise policyTrial is null. Do not turn request frequency or copied incidents into a policy rule.
6. Update pending arguments and identity history, retaining granted waivers and prior valid actions. Return only authorized rationale to this reviewer, explicit decisive fact questions, governance defects and a policy-owner question if a tested trial exists. A recommendation never becomes a signed waiver without an actual matching signature receipt.
