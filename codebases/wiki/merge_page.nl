---
description: Semantically merge concurrent page edits under an agreed profile. Use files to inspect linked page assets or notes when resolving an edit.
args:
  base: WikiPage
  updates: WikiUpdate[]
  profile: MergeProfile
  files?: Dict<File>
returns: MergeReport
---
function merge_page(base, updates, profile) -> MergeReport
  prepared = prepare(base, updates, profile)
  if not prepared.valid:
    return reject(base, prepared)
  draft = interpret(base, prepared.updates)
  return publish(base, prepared, draft, profile)
