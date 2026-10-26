---
args: { request: PortfolioDecisionRequest, priorityEvents: EvidencePage[] }
returns: ScenarioFork
---
Create a new scenario ID only when a supported mid-cycle priority change exists. Bind the changed priorities and their effective time; reference but never rewrite the approved baseline's choices or assumptions.
