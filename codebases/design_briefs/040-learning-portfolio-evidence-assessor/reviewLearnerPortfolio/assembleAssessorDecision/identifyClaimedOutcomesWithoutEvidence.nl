---
args: { mappings: PortfolioCompetencyMap[], findings: PortfolioCompetencyFinding[] }
returns: UnsupportedOutcome[]
---
Join each requested outcome to its competency finding and the source-linked mappings supporting it. Return claims with missing source locations, unresolved authorship, inadequate recency, or contrary evidence as unsupported or review-bound. Do not infer a pass from a confident reflection when direct evidence is required by the standard.
