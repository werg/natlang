import { check } from "./run/check";
import { execute } from "./run/execute";
import { inspect } from "./run/inspect";
import { invalid } from "./run/invalid";
---
description: Check and run a pinned source revision as a child programme.
args:
  input: string
returns: RunReport
---
function run(input) -> RunReport
  snapshot = inspect()
  checked = check(snapshot.revision)
  if checked.status is "checked":
    return execute(input, snapshot.revision)
  return invalid(snapshot.revision, checked)
