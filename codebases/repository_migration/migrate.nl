---
description: Plan an exact isolated repository edit and report declared checks from the pinned repository snapshot.
args:
  request: string
  query: string
returns: MigrationReport
---
function migrate(request, query) -> MigrationReport
  snapshot = inspect()
  uses = search(query, snapshot.revision)
  patch = propose(request, snapshot, uses)
  candidate = apply(snapshot.revision, patch)
  checks = validate(candidate.revision)
  return report(candidate.revision, checks)
