---
args: { baseline: ProcedureRequirement, exceptions: LocalException[], steps: ID[] }
returns: InspectionGuardMap[]
---
For each mapped step retain the baseline safety constraint and required inspections. Apply a local exception only when its authority covers that exact step and constraint; leave every inspection named by the exception mandatory. Return a step-specific guard map so a queue waiver cannot erase an inspection.
