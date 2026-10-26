---
args: { mandate: GoalContract; positions: NegotiationPositionMap; evidence: EvidencePage[]; termImpact: NegotiationGuardrail }
returns: ClauseEffectGraph
---
Create nodes for clauses, definitions, exceptions, caps, remedies and cross-references; create edges only when text shows an operative dependency. For each proposed issue change, state a concrete behavior example before and after, and identify the position it can affect. Expand every globally changed term to its full consumer set. Keep contradictory readings as branches and attach exact evidence refs; do not infer a protective effect from a clause in isolation.