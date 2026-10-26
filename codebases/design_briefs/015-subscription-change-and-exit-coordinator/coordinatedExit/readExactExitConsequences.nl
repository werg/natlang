---
description: Bind access, billing and data consequences to authoritative entitlement and billing artifacts.
args:{scope:GoalContract;policy:PolicySelection;intent:ChangeIntent;entitlements:EvidencePage[];billing:EvidencePage[]}
returns:ChangeConsequence[]
---
For each affected subscription, bind access end date, billing delta and data consequence to current service records and the selected terms. Expose source and version. If an external balance or cutoff is unresolved, do not synthesize an amount; leave that consequence pending with its missing artifact.
