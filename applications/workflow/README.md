# Order workflow in natural language

A durable, single-writer workflow for an order: reserve inventory, charge payment, ship. The policy is natural
language: what to do next, how to recover an operation whose outcome is unknown, how to undo an order, and what the
customer is told. The mechanism is crisp: the intent is written before every remote effect, remote receipts are
idempotent (keyed `order:action`), every transition is checked, and the order file is replaced atomically.
[DECOMPOSITION.md](DECOMPOSITION.md) records the decision for every part.

```
handle.nl                     the policy: choose, check with ledger.validate, ask once more with the problem
  handle/choose.nl              find where the order stands, then choose by that
    choose/situation.nl           uncertain / cancelling / owed / failed / forward / settled (a decision)
    choose/advance.nl             forward progress by phase
    choose/failure.nl             a refusal is transient or definite (a decision)
    choose/compensate.nl          the undos that still stand, refund before release
    choose/recover.nl             unknown outcome: reconcile, look again later, or retry under the same key
  handle/crisp.ts               the same choice as a table: the crisp implementation (no model call)
inform.nl                     what the customer hears: moment, facts, compose, then ledger.checkMessage
  inform/moment.nl              nothing / paid / shipped / delayed / problem / cancelled (a decision)
  inform/facts.nl               what may be told, in plain words
  inform/compose.nl             subject and body
service.ts                    the mechanism: WorkflowService (intent, receipts, validity, atomic commit, outbox)
ledger.ts                     the read-only `ledger` service the stages call: validate, checkMessage
index.ts                      step / stepFull (snapshot, decide, apply, inform) and WorkflowDesk
```

## One step

`stepFull(service, orderId, event, { run, policy })`:

1. Read the order and the remote's receipt for its pending key: a `Snapshot`.
2. Decide: `handle.nl` (policy `natural-language`, the default) or `handle.crisp` (policy `crisp`). The result is a
   `Decision` `{ action, reason, waitMs? }`.
3. `service.apply(orderId, revision, event, decision)` applies it only if the order is still at that revision, and
   only if the transition is valid. A refusal names the property that failed ("`charge` is valid in phase reserved or
   charge-failed; the order is in phase charged").
4. `inform.nl` decides whether the customer hears about it and writes the message, which the service queues once in the
   order's outbox.

An unknown outcome (an effect whose acknowledgement was lost) stays `pending`. It is never repeated blindly:
`reconcile` looks for the receipt; only after a look found none, `retry` sends the same key again, and the remote does
each key once.

```ts
const stepped = await stepFull(service, 'order1', { kind: 'continue' }, {
  run: (fn, options) => runtime.run(fn, options),   // supplies the ledger service
  policy: 'natural-language',                       // or 'crisp'
});
```

`WorkflowDesk` runs one event loop per order. The policy names when an unresolved order is looked at again
(`Decision.waitMs`); the desk holds that as state and restores it after a restart.

## Tests

`ts-host/test/workflow-service.test.mjs` scripts every stage with the code a model would write. It checks the lost
acknowledgement and lost request recoveries, retries and compensation order, that the crisp and natural-language
policies agree, that a refused choice goes back with its problem, the mechanism's refusals, restart, the outbox, and the
desk. No model is loaded.
