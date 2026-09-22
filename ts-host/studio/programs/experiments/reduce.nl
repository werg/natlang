import { apply } from "./reduce/apply";
import { choose } from "./reduce/choose";
import { finish } from "./reduce/finish";
---
description: Reduce a Possibility lab event into typed application state.
args:
  state: State
  event: UiEvent
returns: State
---
Drive the Possibility lab interaction to completion. You own the algorithm and may
call helpers repeatedly; one UI event is not limited to one operation.

Use choose for semantic interpretation when needed. apply performs one concrete
operation and returns Step { state, ok, detail }. Inspect that result. Continue
from its state, preserving successful work, until the user goal is fulfilled or
a concrete blocker remains. On failure, inspect detail and decide whether a
corrected operation is warranted; do not repeat an unchanged failed operation.
Use finish on the final Step to return the current state.

For compare, run the cautious policy and then the generous policy using the same supplied seed and step count. Inspect both results and explain the comparison. For further experiments, select seeds and horizons deliberately and call run for each trial; do not hide the experimental design in a host loop.

Keep model-visible work focused: reuse state references instead of copying the
entire state into tool arguments. UI drafts and rendering are managed by the host.
