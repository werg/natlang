import { check_state } from "./merge_tree/check_state";
import { interpret } from "./merge_tree/interpret";
import { prepare_envelope } from "./prepare_envelope";
import { validate_claims } from "./validate_claims";
---
description: Merge a nested outline or folder tree, including subtree moves and rename conflicts.
args:
  base: State
  updates: Update[]
  policy: string
returns: Draft
uses:
  prepare_envelope: prepare_envelope.ts
  validate_claims: validate_claims.ts
types:
  Node: '{ id: string, parent: string, title: string }'
  State: '{ revision: number, nodes: Node[] }'
  Draft: '{ state: State, applied: string[], alternatives: Alternative[], explanation: string }'
---
function merge_tree(base, updates, policy) -> Draft
  prepared = prepare_envelope(base.revision, updates)
  if not prepared.valid: report_error(prepared.error)
  draft = interpret(base, prepared.updates, policy)
  valid = validate_claims(prepared.updates, draft.applied, draft.alternatives)
  if not valid: report_error("The tree merge omitted, invented or double-counted an update")
  shape_ok = check_state(draft.state)
  if not shape_ok: report_error("The tree merge returned an invalid parent graph")
  return draft
