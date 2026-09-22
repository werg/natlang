import { check_state } from "./merge_set/check_state";
import { interpret } from "./merge_set/interpret";
import { prepare_envelope } from "./prepare_envelope";
import { validate_claims } from "./validate_claims";
---
description: Merge membership changes to a semantic collection, including add/remove and synonym conflicts.
args:
  base: State
  updates: Update[]
  policy: string
returns: Draft
uses:
  prepare_envelope: prepare_envelope.ts
  validate_claims: validate_claims.ts
types:
  State: '{ revision: number, members: string[] }'
  Draft: '{ state: State, applied: string[], alternatives: Alternative[], explanation: string }'
---
function merge_set(base, updates, policy) -> Draft
  prepared = prepare_envelope(base.revision, updates)
  if not prepared.valid: report_error(prepared.error)
  draft = interpret(base, prepared.updates, policy)
  valid = validate_claims(prepared.updates, draft.applied, draft.alternatives)
  if not valid: report_error("The set merge omitted, invented or double-counted an update")
  shape_ok = check_state(draft.state)
  if not shape_ok: report_error("The set merge returned invalid members")
  return draft
