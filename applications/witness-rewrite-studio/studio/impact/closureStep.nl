---
description: Consume one graph node and contextualize its dependency edges.
args:
  state: ImpactFrontier
  analysis: Analysis
returns: ImpactFrontier
---
1. If pendingNodeIds is empty, return state. Otherwise select its first ID and find that inventoried
   Obligation. A missing node is an explicit unresolved graph defect, then settle that pending ID.
2. Collect candidate incident dependencies from node.dependencies and state.links. Deduplicate
   by directed premise, consumer and relation; retain all distinct source witnesses.
3. Capture the requested FactDelta, this node's required meaning and its protected rules. In eval
   declare local DependencyJudgment with necessary: boolean | null, affected: boolean | null, reason: string,
   witnesses: {sourceId: string; path: string; quote: string}[]. Create nl.with<DependencyJudgment> over one changing dependency edge.
4. Its instructions are: determine whether this premise is necessary for this consumer in the
   captured selected-world and actor context; distinguish thematic similarity from causal reliance;
   report whether the requested delta removes or changes that premise and cite the actual evidence.
5. Directly invoke the literal on each candidate edge. Consume necessary and affected into the returned link,
   and consume affected by adding the consumer to affectedNodeIds when justified. Keep its witnesses.
   Append a LocalDecision with this site, directed edge input IDs and the consumed necessity/relevance
   conclusion. Carry all prior localDecisions unchanged; record no decision before actual return.
6. If evidence leaves relevance unresolved, append the reason to unresolved and retain the edge's
   necessary:null on the returned link, with the witnessed reason, instead of inventing certainty.
   affected:null preserves possible exposure for the caller; neither null is converted to false.
7. Remove this node ID from pendingNodeIds and append it once to expandedNodeIds. Other inventory
   nodes remain unchanged. This removal is the structural decrease even when no edge is affected.
8. Return the new frontier. The caller computes global affected reachability after every node's
   judgments are available, so processing order cannot accidentally omit a later-discovered premise.
9. Do not create new obligations here. Any missing semantic consumer belongs in unresolved as an
   inventory defect; it requires explicit upstream reconstruction rather than invalidating the measure.
