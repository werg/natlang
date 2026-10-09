---
description: One experiment of the source search, as a typed contract. The host runs it as the crisp step improveStep/lifecycle.ts (no model sequences it); this file is the root of the callable folder improveStep/, whose stages and policies are natural language.
kind: directory-reducer
args:
  state: SearchState
  policy: ImprovementPolicy
returns: SearchState
---
Improve this source through one evidenced experiment, and return the next search state.

1. Plan the experiment: the parent is chosen by the chooseParent policy and recorded in the journal before the edit.
2. Measure the parent on the training evidence, and ask findOpportunity what the evidence supports changing.
3. Run the stages: diagnose the evidence, hypothesize one edit, then editSource (instruction mode) or editSourceStructural (structural mode) on a private copy of the folder.
4. Check, measure and compare the candidate exactly; accept it only when its measurements are better.
5. Select the incumbent with the selectIncumbent policy, install it, and ask shouldStop whether the search ends.

The sequence is mechanism and runs in improveStep/lifecycle.ts; the decisions in steps 1, 2, 3 and 5 are the natural-language functions and policies beside it.
