---
args: { facets: GoalFacet[], evidence: Evidence[], inventory: string[] }
returns: FacetEvidenceMatch[]
---
For each facet inspect the source evidence and inventory for a task where the learner performs the required behavior. Bind the item, learner owner, resource dependencies, and missing conditions. A polished group artifact without individual attribution is not a match for an individual competency.
