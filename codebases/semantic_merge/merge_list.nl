import { check_state } from "./merge_list/check_state";
import { interpret } from "./merge_list/interpret";
import { prepare_envelope } from "./prepare_envelope";
import { validate_claims } from "./validate_claims";
---
description: Merge ordered task or paragraph lists with insert, move, split and delete intent.
args:
  base: State
  updates: Update[]
  policy: string
returns: Draft
uses:
  prepare_envelope: prepare_envelope.ts
  validate_claims: validate_claims.ts
types:
  Item: '{ id: string, text: string }'
  State: '{ revision: number, items: Item[] }'
  Draft: '{ state: State, applied: string[], alternatives: Alternative[], explanation: string }'
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
