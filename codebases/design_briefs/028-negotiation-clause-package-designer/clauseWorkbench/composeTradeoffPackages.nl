---
args: { mandate: GoalContract; positions: NegotiationPositionMap; graph: ClauseEffectGraph; evidence: EvidencePage[]; undefinedTerms: NegotiationGuardrail }
returns: ClausePackageSet
---
Construct a small set of whole-agreement alternatives by choosing compatible positions across connected issue clusters. For every package show each concession, its authority basis, retained protection and interaction consequence across the graph. Reject any combination breaching a prohibited boundary or erasing a required protection through another clause. Carry every blocker and affected term from `undefinedTerms` into the package set and produce no comparable/circulable package until each is defined or removed. Keep material graph conflicts as blockers; do not invent external assent.
