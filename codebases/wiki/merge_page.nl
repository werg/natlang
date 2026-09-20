---
description: Semantically merge concurrent page edits under an agreed profile.
args:
  base: WikiPage
  updates: WikiUpdate[]
  profile: MergeProfile
returns: MergeReport
---
function merge_page(base, updates, profile) -> MergeReport
  prepared = prepare(base, updates, profile)
  if not prepared.valid:
    return reject(base, prepared)
  draft = interpret(base, prepared.updates)
  return publish(base, prepared, draft, profile)
