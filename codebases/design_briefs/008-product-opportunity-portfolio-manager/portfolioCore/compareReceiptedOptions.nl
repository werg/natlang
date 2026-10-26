---
args: { reviews: PortfolioReceiptAssessment[], mechanisms: InterventionMechanism[], strategy: GoalContract, constraints: PolicySelection }
returns: PortfolioSelectionDecision
---
Consider only option sets from accepted receipt reviews. Compare their covered desired outcomes and beneficiaries against the explicitly supplied strategy priorities. Prefer a set only when the supplied priorities establish a decision between the feasible alternatives and the receipt provides exact costs and capacity. Identify deferred mechanisms and shared work once. If no receipt passes, report which artifact, estimate identity, or constraint is missing. If several feasible sets remain tied or the policy leaves weights or authority unknown, return review-required rather than manufacture a numerical score or arbitrary ranking.
