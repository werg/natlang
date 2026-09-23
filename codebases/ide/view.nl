---
description: Compose an editor view from source, diagnostics and a trace event.
args:
  run_id: string
  index: number
returns: string
---
function view(run_id, index) -> string
  snapshot = inspect()
  checked = check(snapshot.revision)
  event = trace(run_id, index)
  page = describe(snapshot, checked, event)
  return render(page)
