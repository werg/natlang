---
description: Propose one signature with a reason and unresolved alternatives.
args:
  target: Target
  context: Context
returns: Candidate
---
Read `target.body`, the parameter names, nearby signatures and the caller/return
obligations. Propose a type for each parameter and a return type using natlang's small
TS-style type language. Keep the exact parameter names, including a trailing `?`
for optional parameters. State uncertain alternatives instead of inventing a type
from one example. Effects must include the required effects
in `context`. Do not introduce a second type system or a host-native type as if it were
portable natlang state. The exact checker will verify syntax and known obligations.
