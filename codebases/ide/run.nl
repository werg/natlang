---
description: Check and run a pinned source revision as a child programme, using files for project context when required.
args:
  input: Text
  files?: Dict<File>
returns: RunReport
---
function run(input) -> RunReport
  snapshot = inspect()
  checked = check(snapshot.revision)
  if checked.status is "checked":
    return execute(input, snapshot.revision)
  return invalid(snapshot.revision, checked)
