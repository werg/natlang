---
args: { intervention: InterventionMechanism, evidence: EvidencePage[] }
returns: WorkPackage[]
---
Separate build, rollout, operations, and enabling work only when the evidence and intervention description support distinct packages. Bind each package to its owner source, cost range, and dependencies; leave unauthorized or missing estimates unfilled.
