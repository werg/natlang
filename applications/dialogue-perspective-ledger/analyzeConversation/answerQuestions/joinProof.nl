---
description: Join actual availability and semantic deltas into source-linked query support.
args:
  focus: QueryFocus
  source: DialogueSource
  scoped: ScopedConversation
  episodePlan: EpisodePlan
  evolution: PerspectiveEvolution
returns: ProofGraph
---
1. Select the actual statement/atom/commitment/fact readings relevant to this focus. Follow their original
   act, source clause, episode and availability IDs rather than reconstructing an answer-shaped proof.
2. Create actualGraph with empty nodes/edges/rootProofIds, challenge null and explicit unresolved
   reasons, then add generated ProofNodes for actual original evidence, availability, endorsement, condition and
   revisions. Edges name supports/defeats/available-to/anchored-at/conditions/quotes with real endpoint IDs.
3. For belief preserve any current active false endorsement and its support. For knowledge additionally
   join actual available corroborating verified evidence; a global fact without an actor receipt has no
   admissible available-to edge. Keep defeating corrections that were adopted after their actual receipt.
4. For promise conditions show independent strict-time and version/actor predicates. A false actual
   predicate and a true listener belief are different source paths, not contradictory proof-node corruption.
5. Keep enough exact original witnesses to establish the requested observable; omit unrelated diaries
   and future availability. If a required edge lacks evidence, record unresolved rather than completing it.
6. Call challengePerspective(focus, actualGraph, source, scoped, evolution). Consume its actual challenge
   by storing the returned challenge on actualGraph.challenge and retaining unsupportedProofIds as
   unresolved graph obligations. The caller will make this failure visible.
7. Return the actual graph with valid generated endpoints and original source IDs. Equivalent proof
   siblings may permute, but directional support and temporal anchoring relations may not reverse.
