---
description: Interpret one editor request against a pinned source revision.
args:
  request: Text
returns: EditReport
---
function edit(request) -> EditReport
  snapshot = inspect()
  patch = interpret(request, snapshot)
  return apply(patch)
