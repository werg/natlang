import { check_state } from "./merge_scene/check_state";
import { interpret } from "./merge_scene/interpret";
import { prepare_envelope } from "./prepare_envelope";
import { validate_claims } from "./validate_claims";
---
description: Merge edits to a shared diagram or game scene with semantic object identity.
args:
  base: State
  updates: Update[]
  policy: string
returns: Draft
uses:
  prepare_envelope: prepare_envelope.ts
  validate_claims: validate_claims.ts
types:
  Object: '{ id: string, label: string, x: number, y: number, color: string }'
  State: '{ revision: number, objects: Object[] }'
  Draft: '{ state: State, applied: string[], alternatives: Alternative[], explanation: string }'
---
function merge_scene(base, updates, policy) -> Draft
  prepared = prepare_envelope(base.revision, updates)
  if not prepared.valid: report_error(prepared.error)
  draft = interpret(base, prepared.updates, policy)
  valid = validate_claims(prepared.updates, draft.applied, draft.alternatives)
  if not valid: report_error("The scene merge omitted, invented or double-counted an update")
  shape_ok = check_state(draft.state)
  if not shape_ok: report_error("The scene merge returned invalid objects")
  return draft
