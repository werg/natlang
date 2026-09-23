import check_state from "./merge_counter/check_state";
import interpret from "./merge_counter/interpret";
import prepare_envelope from "./prepare_envelope";
import validate_claims from "./validate_claims";
---
description: Merge numeric counter intentions, distinguishing increments,
  corrections and resets.
args:
  base: State
  updates: Update[]
  policy: string
returns: Draft
---
function merge_counter(base, updates, policy) -> Draft
  prepared = prepare_envelope(base.revision, updates)
  if not prepared.valid: report_error(prepared.error)
  draft = interpret(base, prepared.updates, policy)
  valid = validate_claims(prepared.updates, draft.applied, draft.alternatives)
  if not valid: report_error("The counter merge omitted, invented or double-counted an update")
  shape_ok = check_state(draft.state)
  if not shape_ok: report_error("The counter merge returned an invalid value or unit")
  return draft
