import { check_state } from "./merge_map/check_state";
import { interpret } from "./merge_map/interpret";
import { prepare_envelope } from "./prepare_envelope";
import { validate_claims } from "./validate_claims";
---
description: Merge a keyed configuration or encyclopedia infobox, including semantic renames and conflicting values.
args:
  base: State
  updates: Update[]
  policy: string
returns: Draft
uses:
  prepare_envelope: prepare_envelope.ts
  validate_claims: validate_claims.ts
types:
  State: '{ revision: number, fields: Record<string, string> }'
  Draft: '{ state: State, applied: string[], alternatives: Alternative[], explanation: string }'
---
function merge_map(base, updates, policy) -> Draft
  prepared = prepare_envelope(base.revision, updates)
  if not prepared.valid: report_error(prepared.error)
  draft = interpret(base, prepared.updates, policy)
  valid = validate_claims(prepared.updates, draft.applied, draft.alternatives)
  if not valid: report_error("The map merge omitted, invented or double-counted an update")
  shape_ok = check_state(draft.state)
  if not shape_ok: report_error("The map merge returned invalid fields")
  return draft
