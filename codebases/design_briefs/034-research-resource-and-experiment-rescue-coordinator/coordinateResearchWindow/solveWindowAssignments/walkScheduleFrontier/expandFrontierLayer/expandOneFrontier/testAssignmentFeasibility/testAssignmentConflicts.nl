---
args: { branch: AssignmentBranch, selected: WindowAssignment[], snapshot: ResearchOperationsSnapshot }
returns: boolean
---
Use intervalsOverlap on the proposed instrument interval and each selected assignment, then on each currently booked interval. A collision exists only when the same instrument or operator is shared and the half-open intervals overlap. Exact interval arithmetic is the only crisp boundary here; resource eligibility and schedule choice remain NatLang decisions.
