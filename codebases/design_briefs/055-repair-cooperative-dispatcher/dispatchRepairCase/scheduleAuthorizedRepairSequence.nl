---
args: { assignments: RepairerMatch[], repair: CooperativeRepairRequest }
returns: RepairSequence
---
Bind parts = verifySparePartCompatibility(assignments, repair.parts). Bind tools = reserveSharedRepairTools(assignments, repair.toolWindows). Bind budget = reviewAuthorizedRepairBudget(assignments, parts, repair.budget). Order approved task prerequisites only after these checks. If any part is unsupported, tool conflicts, or scope/cost exceeds approval, stop only the affected work and request the required evidence or customer reapproval; never proceed with a substitute by inference.
