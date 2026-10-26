---
args: { request: BidPreparationRequest, responses: TenderResponse[] }
returns: GoalContract
---
Match response to the exact tender ID, issuing organization, response owner, deadline, and submission channel. Preserve the frozen version and distinguish mandatory eligibility from scored criteria and administrative return requirements. Never infer compliant scope from an unrelated prior bid.
