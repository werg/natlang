---
description: Own one all-outcome dynamic-programming value and its chosen action.
args:
  state: ValueFrontier
returns: ValueFrontier
---
1. If remainingStateIds is empty, return state. Otherwise select its first original ID and retrieve
   that BeliefState and its actual StateExpansion. Keep the original frontier until this value is
   established; an absent graph node or expansion is a dependency defect requiring repair.
2. Interpret completion from the graph's goal membership and mission request. A completed sample
   return has winning=true, zero remaining costs and no action. A state explicitly established dead
   without completing the goal is losing with null costs. Zero energy alone does not negate a goal.
3. For a nonterminal state, inventory every actual candidate ActionEdge. Inspect its outcome cells
   together with the current world set: each possible world must retain its actual continuation.
   Sensor outcomes are uncontrollable observations, while selecting an action is the policy's choice.
4. Join each outcome nextStateId to its already-settled StateValue. Preserve answer and world IDs in
   this local comparison. A missing value or omitted possible world is incomplete evidence; report the
   exact action/outcome dependency for repair rather than interpreting it as a losing physical world.
5. For a fully represented action, judge universal success across its actual possible answers.
   One genuinely losing successor defeats that action even if other answers are cheap and successful.
   Retain the obstructing answer/world IDs and that child's reason for the final explanation.
6. For an action whose every successor wins, compute remaining energy as its immediate cost plus
   the maximum child energy. Compute remaining observations independently as the maximum child count
   plus one precisely when this action observes. These maxima can be attained by different branches;
   both summarize the actual complete policy under the source's two worst-case objectives.
7. Compare the complete winning action values under state.request.objective. For the supplied energy-first,
   observations-second objective, retain minimum worst energy, then minimum worst observation count.
   Interpret a changed source objective explicitly before comparing; unknown priority is a source
   contract defect requiring repair, not permission to reuse a fixture-specific ordering.
8. When several actions tie on every objective, each is a valid optimal witness. Use stable actionId
   only for reproducible serialization. Preserve the fact of a tie in reason, separating this reporting
   choice from physical necessity. Do not claim the chosen action is uniquely required by the source.
9. Bind chosenActionId and costs from the actual selected edge/value. Assemble witnesses from its
   legality evidence and the supported continuation/goal evidence, and explain the decisive comparison.
   If every fully represented action loses, return a losing StateValue with concrete obstructing branches.
10. Append exactly this StateValue and remove exactly its original ID from remainingStateIds. Return
    the shorter frontier without changing prior values, outcome partitions or legal actions. Expansion
    owns those semantic possibilities; this stage owns their all-outcome consequence and objective choice.
