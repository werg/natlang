---
description: Reduce actual episode deltas while retaining historical participant snapshots.
args:
  source: DialogueSource
  scoped: ScopedConversation
  episodePlan: EpisodePlan
returns: PerspectiveEvolution
---
1. Initialize one empty ParticipantView for each actual source participant, with no inherited global
   fact knowledge. Copy the finite ordered episode array into pendingEpisodes of PerspectiveFrontier.
2. Pass original source/scoped objects through the frontier unchanged. Initialize empty history/deltas
   and copy source plan uncertainty; those are provenance and state, not a preferred final answer.
3. Directly await advanceEpisode.iterateOn(initialFrontier)
   .withMeasure(next => next.pendingEpisodes.length).until(next => next.pendingEpisodes.length === 0).
   Each named invocation consumes one original episode and finishes its actual typed next state;
   this parent owns the subsequent original episode. The remaining count decreases structurally.
4. Consume the actual resulting views, history and deltas. Verify each original episode was applied once
   in established chronology. History snapshots are immutable copies, not aliases of mutable final views.
5. A participant's current false belief remains endorsed until their own later source-backed revision
   changes it. Do not recompute all beliefs from the complete global physical ledger after each episode.
6. Return PerspectiveEvolution with all actual state contributions and uncertainty. Query stages select
   historical snapshots; they cannot retroactively import a future receipt or correction into an earlier one.
