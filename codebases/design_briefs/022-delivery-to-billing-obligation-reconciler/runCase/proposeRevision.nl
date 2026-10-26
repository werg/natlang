---
args: { failure: CollectionFinding, currentPlan: WorkGraphProposal }
returns: RevisionProposal
---
1. Change only the smallest task or instruction that addresses concrete failure. State contract effects, risk, and regression tests; never weaken a gate.
2. Identify the smallest task or instruction change that addresses a concrete failure. State affected contracts, expected benefit, regression risks, and needed tests. Return a proposal for a staged revision; never modify live authority or silently replace an approved baseline.
3. Return an explicit artifact; preserve unknowns and source coverage. This procedure proposes no production write.
