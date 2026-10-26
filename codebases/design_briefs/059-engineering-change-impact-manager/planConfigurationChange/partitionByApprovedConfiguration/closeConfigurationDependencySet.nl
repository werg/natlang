---
args: { seeds: ConfigSeedSet, edges: ConfigurationDependency[] }
returns: ConfigImpactClosure
---
Traverse from the direct configuration seeds only along evidence-backed edges whose dependency kind affects the changed function or interface. Record every traversed edge. Keep unverified edges as blockers and stop closure there; preserve configurations with no supported path.
