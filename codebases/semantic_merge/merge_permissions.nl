import check_state from "./merge_permissions/check_state";
import interpret from "./merge_permissions/interpret";
import prepare_envelope from "./prepare_envelope";
import validate_claims from "./validate_claims";
---
description: Merge access policy edits while exposing conflicting grants and revocations.
args:
  base: State
  updates: Update[]
  policy: string
returns: Draft
---
function merge_permissions(base, updates, policy) -> Draft
  prepared = prepare_envelope(base.revision, updates)
  if not prepared.valid: report_error(prepared.error)
  draft = interpret(base, prepared.updates, policy)
  valid = validate_claims(prepared.updates, draft.applied, draft.alternatives)
  if not valid: report_error("The permission merge omitted, invented or double-counted an update")
  shape_ok = check_state(draft.state)
  if not shape_ok: report_error("The permission merge returned contradictory exact rules")
  return draft
