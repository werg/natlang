---
description: Carry a change's meaning across source, data, tests, views and prose.
args:
  goal: Text
  material: Text
returns: Proposal
---
Separate the user's stated purpose from your inferred purpose. Identify
behavior that must remain and behavior that should change. Follow both formal
dependencies and semantic relationships. Propose actual artifact edits and
checks of before/after behavior; keep old tests visible and explain any
intentional assertion change. For concurrent changes, inspect the shared base,
both intentions and their observed results. Combine compatible purposes even
if the text edits conflict. If purposes are incompatible, preserve concrete
alternatives and a useful unresolved question. Do not claim universal coverage
after inspecting only the files you happened to search.
