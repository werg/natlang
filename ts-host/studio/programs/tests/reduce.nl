import apply from "./reduce/apply";
import choose from "./reduce/choose";
import finish from "./reduce/finish";
---
description: Reduce a Counterexample event into typed application state.
args:
  state: State
  event: UiEvent
returns: State
---
Drive the Counterexample interaction to completion. You own the algorithm and may
call helpers repeatedly; one UI event is not limited to one operation.

Use choose for semantic interpretation when needed. apply performs one concrete
operation and returns Step { state, ok, detail }. Inspect that result. Continue
from its state, preserving successful work, until the user goal is fulfilled or
a concrete blocker remains. On failure, inspect detail and decide whether a
corrected operation is warranted; do not repeat an unchanged failed operation.
Use finish on the final Step to return the current state.

For run, walk all case IDs and invoke run_case for each. Inspect actual results, keep failures as evidence, and continue through the collection. The host runs only one case at a time. For requests to generate cases, reason from the contract and call add for each useful example; do not claim they were executed until run_case returns.

Keep model-visible work focused: reuse state references instead of copying the
entire state into tool arguments. UI drafts and rendering are managed by the host.
