---
description: Interpret one editor request against a pinned source revision. Read files for project context before making a source edit when the request depends on it.
args:
  request: Text
  files?: Dict<File>
returns: EditReport
---
function edit(request) -> EditReport
  snapshot = inspect()
  patch = interpret(request, snapshot)
  return apply(patch)
