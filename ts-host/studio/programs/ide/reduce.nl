import apply from "./reduce/apply";
import choose from "./reduce/choose";
import finish from "./reduce/finish";
---
description: Reduce a Atelier event into typed application state.
args:
  state: State
  event: UiEvent
returns: State
---
Drive the Atelier interaction to completion. You own the algorithm and may
call helpers repeatedly; one UI event is not limited to one operation.

Use choose for semantic interpretation when needed. apply performs one concrete
operation and returns Step { state, ok, detail }. Inspect that result. Continue
from its state, preserving successful work, until the user goal is fulfilled or
a concrete blocker remains. On failure, inspect detail and decide whether a
corrected operation is warranted; do not repeat an unchanged failed operation.
Use finish on the final Step to return the current state.

For evaluate, iterate over all scenario IDs, calling apply with action evaluate_case and the scenario target. Inspect each actual result, preserve failures, and continue to cover the collection. Do not replace this with a single host evaluation loop.

Keep model-visible work focused: reuse state references instead of copying the
entire state into tool arguments. UI drafts and rendering are managed by the host.
