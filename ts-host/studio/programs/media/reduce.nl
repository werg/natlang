---
description: Reduce a Cut & light event into typed application state.
args:
  state: State
  event: UiEvent
returns: State
---
Drive the Cut & light interaction to completion. You own the algorithm and may
call helpers repeatedly; one UI event is not limited to one operation.

Use choose for semantic interpretation when needed. apply performs one concrete
operation and returns Step { state, ok, detail }. Inspect that result. Continue
from its state, preserving successful work, until the user goal is fulfilled or
a concrete blocker remains. On failure, inspect detail and decide whether a
corrected operation is warranted; do not repeat an unchanged failed operation.
Use finish on the final Step to return the current state.

For explicit controls, preserve the requested fields. For natural-language goals, work through the necessary operations in a sensible order, checking each result.

Keep model-visible work focused: reuse state references instead of copying the
entire state into tool arguments. UI drafts and rendering are managed by the host.
