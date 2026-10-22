# Teacher pseudo-return action link decomposition (2026-10-22)

## Goal

Keep a valid teacher trajectory whose nested reducer finishes with the runtime's
literal `return_result({...})` text adapter. Bind that terminal action to the
exact child invocation and exact recorded assistant text so one omitted
`assistant.calls` projection does not reject the whole row. Keep ordinary
unparseable prose and ambiguous outcomes unlinked.

## Natlang-native policy

Policy: accept the terminal return only when the captured child turn is the
unique exact literal pseudo-return recognized by the runtime, its result
arguments equal one completed action-ledger event from that same invocation,
the event is causally after that invocation's matching model-turn trace, and
the source declares the tool. A parent action, a sibling's result, a merely
similar result, or prose around the request earns no link.

The policy is naturally a typed function over `{ child turn, captured output,
runtime event, invocation lineage }` that returns either one verified binding
or no binding. This codebase currently represents the policy inside the native
teacher materializer; the new call site will preserve that contract and record
the verification basis in the materialized outcome. The TypeScript parser is
the crisp boundary because it must exactly reproduce the runtime's literal
syntax semantics and invoke no code.

## Decomposition

1. **Resolve a literal written return** — reuse `writtenReturnResult` from
   `native/pseudo-call.ts`. This exact TypeScript-literal parser is the existing
   runtime adapter; do not add a regex, execute text, or invent a second
   pseudo-call grammar.
2. **Bind child lineage** — use the source turn's invocation id, the exact
   action-ledger `call_id`, and the recorded invocation parent map. Preserve
   the reducer child as the owner of its result. Root `folder.apply` remains a
   separate parent `eval` action.
3. **Bind captured output and event** — require the recorded assistant text to
   equal the model-response text, require one completed matching `return_result`
   event for that invocation, and bind it to the unique corresponding
   `execution_graph` model turn and a later ledger sequence. Require the exact
   declared tool to be offered.
4. **Materialize verified outcome** — surface the action with an explicit
   `exact-assistant-text-pseudo-call` argument source and preserve its original
   action-ledger status/result. The later training/admission policy remains
   unchanged and still decides whether the action is useful.
5. **Fail closed** — malformed/nonliteral text, extra surrounding prose,
   multiple candidate terminal events, a sibling/parent event, missing trace,
   missing tool, or argument mismatch remains unlinked and causes the existing
   row-level rejection.

## Crisp mechanism boundaries

- `writtenReturnResult` performs bounded parsing of one literal call; it does
  not evaluate code.
- The materializer verifies exact hashes/identities, event ordering, and
  invocation ownership. It does not infer a result from child output,
  `folder.apply` return previews, parent outputs, or task semantics.
- Existing action admission, task quality, source/held separation, and runtime
  provenance remain authoritative and unchanged.

## Validation

Use a minimal root-plus-child fixture modeled on a real `folder.apply` reducer:
the parent's `eval` is linked independently, and the child's literal terminal
return is bound to its own invocation. Negative cases mutate the result, owner,
trace multiplicity, syntax, and tool availability. Keep the original failing
source row and its diagnosis immutable; do not regenerate or relabel it.
