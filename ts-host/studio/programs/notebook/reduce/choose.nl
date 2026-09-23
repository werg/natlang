---
args:
  state: State
  event: UiEvent
returns: Decision
---
Manage a mixed natlang, TypeScript, SQLite notebook. add creates a cell with target ID, engine in secondary, source in text, dependencies in ids. save updates existing source and invalidates dependent outputs. execute evaluates one cell only; its dependencies must already have complete outputs. The natlang caller resolves ordering. remove rejects removal while another cell depends on it. Never invent an execution result. Natlang cells use complete .nl frontmatter and receive deps input; TypeScript cells receive deps; SQLite is provided by the local companion.

For an explicit control event, event.kind is the requested action and event.value
is a JSON record of its fields. Preserve those fields and do not substitute a
different action. For a command event, interpret the user's request using the
current state and the documented actions. Return the next applicable Decision. The caller owns sequencing and completion.
Keep edits and semantic judgments within the user's request. Host effects and
results belong to perform; never fabricate an execution result here.
