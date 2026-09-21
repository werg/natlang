---
description: Design a task-specific interface with executable natlang handlers.
args:
  goal: Text
  material: Text
returns: Proposal
---
Ask what comparison, manipulation or experiment would make the present
uncertainty easier to inspect. Propose a view artifact containing JSON
{tree,bindings}, plus the actual natlang handler source for each bound control.
Controls use stable IDs and typed input intent. A button's action.kind is its
own ID; action.from names an input ID. Bindings map IDs to .nl roots. Handlers
take value: Text and context: Text, and return a typed result that a later
reducer event can interpret. Choose labels and feedback that expose assumptions,
denominators and evidence. Include checks for keyboard use, stale data, changed
views and what a control actually computes. If a static explanation serves the
user better, propose that instead of an ornamental control.
