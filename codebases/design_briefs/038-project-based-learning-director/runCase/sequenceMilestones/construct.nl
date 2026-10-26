---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftProjectMilestone
---
Construct a ProjectMilestone draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Decompose the project into milestones with dependencies, feedback points, and explicit evidence criteria. Allocate time and resources through exact services, identify critical bottlenecks, and preserve room for learner choice within the required structure.
