---
description: Plan an exact isolated repository edit and report declared checks. Use files to inspect non-source project context when it helps resolve the request; repository operations remain authoritative.
args:
  request: Text
  query: Text
  files?: Dict<File>
returns: MigrationReport
---
function migrate(request, query) -> MigrationReport
  snapshot = inspect()
  uses = search(query, snapshot.revision)
  patch = propose(request, snapshot, uses)
  candidate = apply(snapshot.revision, patch)
  checks = validate(candidate.revision)
  return report(candidate.revision, checks)
