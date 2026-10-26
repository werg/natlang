---
args: { mandate: GoalContract; positions: NegotiationPositionMap; graph: ClauseEffectGraph; evidence: EvidencePage[] }
returns: ClausePackageSet
---
Construct a small set of whole-agreement alternatives by choosing compatible positions across connected issue clusters. For every package show each concession, its authority basis, retained protection and interaction consequence across the graph. Reject any combination breaching a prohibited boundary or erasing a required protection through another clause. Keep undefined terms and material graph conflicts as blockers; do not invent external assent.