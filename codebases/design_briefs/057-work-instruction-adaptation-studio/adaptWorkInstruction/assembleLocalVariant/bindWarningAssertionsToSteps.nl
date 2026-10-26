---
args: { warnings: WarningStepBinding[], steps: ID[] }
returns: WarningStepBinding[]
---
Bind each warning only to explicitly identified step IDs. A translation with omitted hazards or conditions cannot be attached to a released step; keep the source warning and mark that step blocked until a qualified translation is approved.
