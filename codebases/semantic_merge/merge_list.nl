---
description: Merge ordered task or paragraph lists with insert, move, split and
  delete intent.
args:
  base: State
  updates: Update[]
  policy: string
returns: Draft
---
function merge_list(base, updates, policy) -> Draft
  prepared = prepare_envelope(base.revision, updates)
  if not prepared.valid: report_error(prepared.error)
  draft = interpret(base, prepared.updates, policy)
  valid = validate_claims(prepared.updates, draft.applied, draft.alternatives)
  if not valid: report_error("The list merge omitted, invented or double-counted an update")
  shape_ok = check_state(draft.state)
  if not shape_ok: report_error("The list merge returned invalid item IDs")
  return draft
