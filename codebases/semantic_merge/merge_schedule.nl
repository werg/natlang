---
description: Merge collaborative calendar entries and booking requests with time
  and intent conflicts.
args:
  base: State
  updates: Update[]
  policy: string
returns: Draft
---
function merge_schedule(base, updates, policy) -> Draft
  prepared = prepare_envelope(base.revision, updates)
  if not prepared.valid: report_error(prepared.error)
  draft = interpret(base, prepared.updates, policy)
  valid = validate_claims(prepared.updates, draft.applied, draft.alternatives)
  if not valid: report_error("The schedule merge omitted, invented or double-counted an update")
  shape_ok = check_state(draft.state)
  if not shape_ok: report_error("The schedule merge returned invalid event IDs or intervals")
  return draft
