# Workflow: decomposition, part by part

Every part of the order workflow (inventory, payment, shipping), with a decision:

- **fn**: its own natural-language function with a typed contract.
- **inline**: instructions inside its caller.
- **implicit**: left to the model.
- **crisp**: a TypeScript helper (callable-folder code, or host code).
- **service**: reached through the `ledger` service (read-only) or the world outside.
- **pluggable**: one interface, a crisp and a natural-language implementation, a setting selects.

The executors are small models, so instructions spell rules out as numbered steps over named data.

## Policy

- **Natural language: what to do, and what to say.**
  - the next action from the durable state and the event;
  - recovery of an unknown outcome: look again, wait, or send the same request again;
  - whether a failure is worth retrying;
  - compensation: which standing effects to undo, in what order;
  - what the customer is told, and how.
- **Crisp: the mechanism.** These stay host code in `service.ts`:
  - the single writer (one operation at a time);
  - the intent record, written durably before every remote effect;
  - idempotent remote receipts, keyed by `order:action`;
  - the transition validity check: which action is valid in which phase, and what an unknown outcome allows;
  - the atomic commit of the order file, and the outbox.
  Natural language proposes a `Decision`. The mechanism accepts or refuses it, and the refusal names the property
  that failed.
- **Decide on a snapshot, then apply a bounded commit.** `step` reads the order and the remote receipt for its
  pending key (a `Snapshot`), runs the policy, and applies the decision with the revision it read. If the order
  changed meanwhile, `apply` returns the newer state and nothing is applied. The policy runs no effect.
- **Effects are data.** A `Decision` is a value. An outgoing customer message is a value too, `OutboxEntry`, appended
  once under a key. Delivery is outside the workflow.
- **Timers are state.** An unresolved operation is revisited at the time the policy names (`Decision.waitMs`). The
  desk holds it as `reconcileAt` and restores it after a restart from the durable `pending`.
- **Derived values form a DAG** (one `step`):

  ```
  state, event, receipt ─▶ situation ─┬─ uncertain ─▶ recover ───────────┐
                                      ├─ cancelling / owed ─▶ compensate ┤
                                      ├─ failed ─▶ failure ─▶ (retry | compensate) ┤─▶ Decision ─▶ ledger.validate ─▶ apply
                                      ├─ forward ─▶ advance ─────────────┤
                                      └─ settled ─▶ wait ────────────────┘
  before, after, decision ─▶ moment ─▶ facts ─▶ compose ─▶ ledger.checkMessage ─▶ enqueue
  ```

- **Stay close to JS; no unbounded loops.** The policy is a bounded chain of calls. Retries are counted from the
  history (`limits.transientRetries`), not looped. The one repeat is "validate, ask again once with the problem".
- **Pluggable hot path: the next action.** It runs on every event of every order. One interface, `(Snapshot, Limits)
  → Decision`, with two implementations: `handle.crisp` (`handle/crisp.ts`, a table, no model call) and `handle.nl`
  (the natural-language policy, which runs `handle/choose.nl`). The host setting `step(..., { policy })` selects. The
  mechanism validates either one the same way when it applies the decision.

## Choosing the next action

| Part | Decision | Unit | Why |
|---|---|---|---|
| Which situation the order is in: uncertain, cancelling, failed, forward, owed, settled | fn, decision | `handle/choose/situation` | A finite judgment, one scoring pass. It decides which branch runs. |
| Forward progress: new reserves, reserved charges, charged ships, shipped is done | fn | `handle/choose/advance` | The happy path, spelled as a table over phases. |
| Is the last failure worth another attempt: transient (a rate limit) or definite | fn, decision | `handle/choose/failure` | A judgment about the remote's refusal. |
| Retry budget: attempts so far counted from the history | inline | `choose` | Counting failed entries for one key. `limits.transientRetries` bounds it. |
| Compensation plan: which effects stand, in what order to undo them | fn | `handle/choose/compensate` | Its own derived value: `standing = done − undone`, refund before release. |
| First step of the plan becomes the decision | inline | `choose` | One line. |
| The whole choice: dispatch on the situation, return one `Decision` | fn | `handle/choose` | The natural-language implementation of the next-action choice. |
| The crisp implementation of the same choice | pluggable | `handle/crisp.ts` | The hot path's reference. |
| Dispatch between the two | crisp | `index.ts` (`stepFull`) | A one-line choice on the setting. A callable-folder dispatcher would have made the crisp path run a model call to reach it: the entry `handle.nl` is a function, so it runs the model. |
| Validate the decision, ask again once with the problem | fn | `handle` | A checked stage: `ledger.validate` is the check. The crisp implementation needs none. |

## Recovery of unknown outcomes

| Part | Decision | Unit | Why |
|---|---|---|---|
| An operation with `pending` set: look at the receipt, count the looks (`checks`) | fn | `handle/choose/recover` | The central recovery judgment. |
| Receipt found: `reconcile` (the mechanism acknowledges it) | inline | `recover` | One rule. |
| No receipt yet, fewer looks than `limits.checksBeforeRetry`: `reconcile` again later (`waitMs`) | inline | `recover` | The remote may be slow. |
| No receipt after enough looks: `retry` the same key | inline | `recover` | Idempotent: the remote does the effect once per key. The mechanism allows `retry` only after one look found nothing. |
| An event that is not a reconcile while pending: `wait` | inline | `recover` | An uncertain charge is never repeated blindly. |
| Acknowledge from a receipt (`ack`), record a miss (`checks`), re-send under the same key | crisp | `service.ts` | Mechanism. The receipt must match the intent. |

## Compensation

| Part | Decision | Unit | Why |
|---|---|---|---|
| Which effects stand (reserved, charged, shipped), from the history | fn | `handle/choose/compensate` | Derived from `done` entries. |
| Order: refund before release; a shipped parcel cannot be undone here | fn | `handle/choose/compensate` | Policy. A shipped order has nothing to compensate in this workflow. |
| A compensation that failed is retried from the first incomplete step | inline | `choose` | The plan is recomputed from the history on every event, so no plan is stored. |
| Which phase a compensation is valid in | crisp | `service.ts` | The transition table. |

## Customer communication

| Part | Decision | Unit | Why |
|---|---|---|---|
| Is there something the customer should hear: nothing, paid, shipped, delayed, problem, cancelled | fn, decision | `inform/moment` | A finite judgment on what changed between `before` and `after`. |
| The facts to tell, in plain words (what happened, what happens next, the order number and amount) | fn | `inform/facts` | Selecting what is safe and useful. |
| Subject and body | fn | `inform/compose` | Wording. |
| Consistency of the kind with the state (a message says shipped only for a shipped order), no internal keys | crisp, service | `ledger.checkMessage` | A check on what the stages wrote. |
| Revise once with the problem | inline | `inform` | A checked stage. |
| Queue once under a key | crisp | `service.ts` (`enqueue`) | Idempotent. The key is `order:message:revision`. |

## Host

| Part | Decision | Unit | Why |
|---|---|---|---|
| One writer at a time, atomic file replacement | crisp | `service.ts` | Exact durability. |
| Intent before effect; receipts keyed `order:action`; replay of a completed operation refused | crisp | `service.ts` | The workflow's guarantees. |
| The remote (receipts file) and its injected faults | service | `service.ts` | Stands for the outside world. |
| Orders side by side, one event loop each, wake-ups | crisp | `index.ts` (`WorkflowDesk`) | The scheduler of events. It runs the policy through `step`. |

## Refinements

Refinement types (`Is<T, "predicate">`, plans/REFINEMENT_TYPES.md). Decisions: (a) adopted with a crisp checker in
`refinements.ts` (no model call), (b) a natural-language judge (proposed only: awaiting live evaluation, not wired),
(c) left to the check that already enforces it exactly (`ledger.validate`, `ledger.checkMessage`, the commit), (d) not
adopted: the value alone does not show the property. The refined result types are `Checked*` aliases in `types.ts`,
named in the `returns` of the stage that produces the value; the host and the crisp policy keep the plain types. The compiled `.nl` modules register
`refinements.ts` themselves, so an embedding host needs nothing more.

| Slot | Proposed type | Decision |
|---|---|---|
| `Decision.reason` | `Is<string, "one sentence that names the state it was chosen from">` | (d) The state is not in the value. |
| `Decision.waitMs` | `Is<number, "a whole number of milliseconds, positive, set when the order should be revisited">` | (a) Weakened (`CheckedDecision`): positive whole number when given. Whether the order should be revisited needs the state (d). |
| `Decision.action` (on a `Snapshot` with `state.pending`) | `Is<Action, "reconcile, retry or wait">` | (c) `ledger.validate`; the snapshot is not in the value. |
| `Decision.action` (on a `cancel` event) | `Is<Action, "refund, release or wait">` | (c) `ledger.validate`. |
| `Decision.action` | `Is<Action, "valid in state.phase">` | (c) The transition table, `ledger.validate`, which already sends the problem back to `choose`. |
| `Compensation.steps` | `Is<CompensationStep[], "refund before release; only effects that stand; each action once">` | (a) Weakened (`CheckedCompensation`): refund before release, each at most once, a reason for each. Only effects that stand needs the history (c, `ledger.validate`). |
| `Compensation.steps[].action` | `Is<"refund" \| "release", "undoes an effect that is recorded done and not yet undone">` | (c) The history is not in the value; `ledger.validate`. |
| `Facts.summary` | `Is<string, "plain words about the order: no operation keys, receipt ids or phase names">` | (b) Proposed, awaiting live evaluation (not wired). A phase name such as "shipped" is also plain English, so no pattern decides it. |
| `Facts.next` | `Is<string, "what the customer can expect next, without a date unless the history gives one">` | (d) The history is not in the value. |
| `Outgoing.subject` | `Is<string, "one line of at most 60 characters, without a trailing period">` | (a) `CheckedOutgoing`, with a non-empty body. |
| `Outgoing.body` | `Is<string, "polite, plain, accurate to the order's state, without blame, and without internal keys">` | (b) Proposed, awaiting live evaluation (not wired): one judge call per message; accuracy to the state needs the state, so the judge could only rate tone and plainness. |
| `Outgoing.kind` | `Is<MessageKind, "true of the order after the event: shipped only when the phase is shipped">` | (c) `ledger.checkMessage`, which already sends the problem back to `compose`. |
| `Limits.transientRetries`, `checksBeforeRetry` | `Is<number, "a whole number, at least 1">` | (d) A setting the host supplies, not a model output; a refined parameter would make every host cast it. Validate at the desk's construction instead. |
| `WorkflowState.obligations` | `Is<string[], "empty when nothing is owed, otherwise one sentence per open obligation">` | (c) Built by the ledger, not by a model. |

Counts: (a) 3, (b) 2, (c) 6, (d) 3.

The model-facing text added to signatures is the `Is<...>` predicate of `CheckedDecision`, `CheckedCompensation` and
`CheckedOutgoing`. No instruction sentence was changed.

## Changes from today

- 213 lines of crisp TypeScript with one `nl` call become natural-language policy: `handle.nl` with 5 stages, and
  `inform.nl` with 3.
- The mechanism stays: single writer, intent before effect, idempotent receipts, transition validity, atomic commit.
- New mechanism that the policy needs: `retry` of a pending key (only after a look found no receipt), a retry of a
  failed charge or shipment after a transient fault, `checks` and `outbox` on the order, `lost_request` as an injected
  fault (the request never reached the remote).
- The desk takes its wake-up time from the policy.

## Language and runtime limitations

- **A pluggable hot path cannot be dispatched inside the natural-language entry without a model call.** The entry of a
  named function (`handle.nl`) is run by the model, so a crisp/natural-language dispatcher in its callable folder (the
  way `applications/pi/harness/context.ts` does it) pays a model call to reach the crisp branch. Here the host selects
  between `handle.crisp` and `handle`. The host imports `.nl` entries (and their typed children, e.g. `handle.crisp`), so the dispatch is host code.
