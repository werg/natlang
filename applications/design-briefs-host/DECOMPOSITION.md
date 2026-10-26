# Exact application custody, without a common business workflow

## Defect being removed

The first host made every native call a proposal, required a different person to
approve it, and then committed it. That was a semantic application policy written
in JavaScript. It prevented ordinary interactive turns from saving their state
and imposed the same workflow on a ledger, a learning session and a performance.
The empty policyCandidates / searchedPartitions snapshot fields were another
remainder of the retired common application scaffold.

## Application-owned behavior

Each application's typed Natlang operations decide what an event means, which
questions to ask, when to wait, which evidence authorizes an action, how an
interruption changes the next step, and which domain state and intentions result.
An approval, reservation, rehearsal, refusal or revision is part of that
application's own types and callable graph. A stored result can itself represent
a pending domain decision; saving it does not approve that decision.

The source manifests specify exact checkpoint bindings. `stateParameter` names
an input, `stateInputPath` optionally locates its nested prior state, and
`stateResultPath` locates the next state. A shared `stateKey` explicitly connects
different operations of one application. These are copying instructions, never
semantic migration or reconstruction. Domain data retains its own shape.

## Exact responsibilities

| Component | Responsibility and why it is crisp |
| --- | --- |
| `runtime.mjs` | Load precisely inventoried source; run real native calls; check typed arguments/results and JSON representability. No domain answer table. |
| `service.mjs` | Authenticate operation capability, capture arguments and source identities, inject the exact checkpoint, run outside a transaction, and submit a bounded atomic write. |
| `store.mjs` | SQLite transactions, exact versions/hashes, actor-bound idempotency, immutable invocation receipts, checkpoint bytes, pagination and audit. It does not interpret domain statuses. |
| `server.mjs` | HTTP framing, bearer identity, cancellation and status codes. Public operation names and signatures come from each program. |
| `delivery.mjs` | Deliver materialized packets to configured routes; verify matching receiver receipts; record uncertain transport outcomes. |

An invocation transaction compares both the case version/hash and every checkpoint
version, persists the checked result, checkpoint, outbox and audit together, and
records the durable invocation identity. A retry returns the original receipt
before inference. Two concurrent completions for the same request return the
winning receipt; distinct requests on the same old snapshot conflict. No model
or network call runs inside the write transaction. Concurrent local callers for
one idempotency key share execution but retain separate cancellation subscriptions:
one caller can leave without canceling the others, while cancellation of the last
waiter aborts execution and blocks commit even when an executor ignores abort.
Canceled callers are removed from the in-flight entry, so a later retry can start.
Cancellation before commit leaves state untouched. Cancellation after commit is
recovered by idempotent read, provided the caller itself has not canceled.

The host rejects a nonempty declared `effectsPath` without an installed exact
adapter. An adapter serializes the application's decided intentions; it does not
select actions, recipients, authorization or recovery strategy. Adapters run
before commit and must be pure. A delivery acknowledgment records transport
evidence, not domain acceptance. Applications consume that evidence in their own
receipt/reconciliation operations. Missing or ambiguous evidence stays unresolved.

## Verification boundary

Tests exercise actual native child calls with a scripted engineering driver,
unrelated input/result structures, direct and nested checkpoint restoration,
cross-operation state sharing, stale inference, process restart, competing
writers, authentication, atomic rollback, actor-bound replay, effect delivery
and uncertain receipt recovery. They do not establish learned-model quality or
deployment readiness. The 100 programs retain independent scenario oracles;
their source changes require separate live teacher measurement.
