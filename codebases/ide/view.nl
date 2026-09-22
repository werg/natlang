---
description: Compose an editor view from source, diagnostics and a trace event.
args:
  run_id: Text
  index: Num
returns: Text
---
function view(run_id, index) -> Text
  snapshot = inspect()
  checked = check(snapshot.revision)
  event = trace(run_id, index)
  page = describe(snapshot, checked, event)
  return render(page)
