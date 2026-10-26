---
args: { prepared: PreparedPortfolioOption, goal: GoalContract, policy: PolicySelection }
returns: DecisionsPortfolioOption
---
1. Apply this domain algorithm to each relevant evidence unit: Submit eligible options and exact constraints to a portfolio solver, then interpret feasible alternatives against the supplied strategy. Explain who benefits and what is deferred; generate contingency portfolios for major assumption changes rather than a single opaque ranking.
2. Return field-level atoms for the actual PortfolioOption record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
