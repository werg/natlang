---
description: Compose an editor view from source, diagnostics and a trace event, using files for relevant project context.
args:
  run_id: Text
  index: Num
  files?: Dict<File>
returns: Text
---
function view(run_id, index) -> Text
  snapshot = inspect()
  checked = check(snapshot.revision)
  event = trace(run_id, index)
  page = describe(snapshot, checked, event)
  return render(page)
