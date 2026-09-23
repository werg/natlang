---
description: Design a task-specific interface with executable natlang handlers.
args:
  goal: string
  material: string
returns: Proposal
---
Ask what comparison, manipulation or experiment would make the present
uncertainty easier to inspect. Propose a view artifact containing JSON
{tree,bindings}, plus the actual natlang handler source for each bound control.
Controls use stable IDs and typed input intent. A button's action.kind is its
own ID; action.from names an input ID. Bindings map IDs to .nl roots. Handlers
take value: string and context: string, and return a typed result that a later
reducer event can interpret. Choose labels and feedback that expose assumptions,
denominators and evidence. Include checks for keyboard use, stale data, changed
views and what a control actually computes. If a static explanation serves the
user better, propose that instead of an ornamental control.
Available view tags include headings, text, lists, table elements, input,
textarea, select/option, button, details/summary and meter/progress. Numeric
meters use value/min/max. View content is data and the browser renders and
wires the controls.

If linked spatial behavior, custom visualization, or another useful interaction
cannot be expressed honestly as a tree, propose {module,bindings}. The module
contains html, optional style, script and title. Its script uses
natlang.emit(control,value) for declared controls and natlang.draft(id,value)
with natlang.drafts for recoverable local work. It runs in an origin-isolated
iframe without network access. Keep domain interpretation in the bound handler
and generated methods instead of hiding conclusions in presentation script.
