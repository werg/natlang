---
description: Expand one belief state with physical actions and source-supported observation partitions.
args:
  state: BeliefFrontier
returns: BeliefFrontier
---
1. If no state is pending, return state. Otherwise choose a pending state with greatest remaining
   energy, breaking ties by stable state ID. Remove it from pending and add it once to expandedStateIds.
2. If it is at the requested start carrying the required sealed sample, mark it a goal state and
   create an empty expansion. If energy is zero without the goal, mark it dead and create no actions.
3. Otherwise call availableActions(current, state.worlds, state.survey). This child judges physical
   legality across every current world, including gear transitions and directed crossing affordances.
4. For each returned observation action, call predictObservation(action, current, worlds, survey).
   Retain it only if its partition is complete. Each cell receives a successor with the same physical
   state except energy cost, and with exactly that cell's nonempty world-ID subset.
5. For each non-observation action, find its successor in the child's returned newStates by the
   deterministic outcome nextStateId; consume that RobotState, retaining the same world-ID set. An action may not reveal a frame convention
   unless the source gives it an observation effect.
6. Canonicalize successor state keys exactly, including remaining energy. Reuse an already known
   identical state. Discard an observation only when its sole cell preserves the whole belief and its
   only physical effect is spending energy: skipping that observation preserves every future option.
   Do not merge lower-energy states into higher-energy states or prune one outcome of an AND action.
7. Append new distinct states and their IDs to pending, unless already expanded. Keep the action's
   outcome cells intact; dropping one bad outcome would incorrectly turn an AND requirement into a choice.
8. Store StateExpansion with retained actions, rejected reasons and newly inserted states. Mark a
   nongoal state dead if no action survives. The caller later computes whether surviving actions win.
9. Return the graph/frontier with one more expanded exact state and the same universeSize. Queries with unchanged belief can be skipped; every remaining action decreases source energy,
   so even routes revisiting the same location range over a finite set of exact states.
