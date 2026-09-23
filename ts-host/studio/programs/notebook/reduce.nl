---
description: Reduce a Fieldnotes event into typed application state.
args:
  state: State
  event: UiEvent
returns: State
---
Drive the Fieldnotes interaction to completion. You own the algorithm and may
call helpers repeatedly; one UI event is not limited to one operation.

Use choose for semantic interpretation when needed. apply performs one concrete
operation and returns Step { state, ok, detail }. Inspect that result. Continue
from its state, preserving successful work, until the user goal is fulfilled or
a concrete blocker remains. On failure, inspect detail and decide whether a
corrected operation is warranted; do not repeat an unchanged failed operation.
Use finish on the final Step to return the current state.

For run, inspect the requested cell and its needs. Walk the dependency graph yourself, detecting cycles. Execute dependencies before dependents by calling apply with action execute and the individual cell target. Inspect every Step; stop dependents on failure. Reuse fresh completed dependencies. The host executes only one cell per operation. For add/save/remove use the matching exact operation.

Keep model-visible work focused: reuse state references instead of copying the
entire state into tool arguments. UI drafts and rendering are managed by the host.
