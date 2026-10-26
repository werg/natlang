# Running independently designed NatLang applications

Each directory in `codebases/design_briefs` contains its own application API,
typed domain model, control flow, operational recovery and callable reasoning
tree. Its `application.json` identifies the actual native entrypoints. This
directory supplies optional exact transport and durable custody for those APIs.
It does not supply or replace their semantic algorithms.

## Inspect and run

Use Node 24 and the repository's locked dependencies:

```sh
npm ci
npm run build:node
node applications/design-briefs-host/main.mjs describe 001
node applications/design-briefs-host/main.mjs serve /path/to/deployment.json
```

`describe` prints that application's actual named parameters, types, results and
operations. Use those contracts to construct inputs; there is no universal
request/snapshot schema. The server runs the real `.nl` function with
`createNatlangRuntime` and the configured model. It never loads expected fixture
answers, replays authored answers, or substitutes a crisp domain implementation.

Deployment configuration names `database`, `model`, `capabilities`, `credentials`
and optionally `http`, `effectAdapters`, `deliveryRoutes` and `executionLimits`.
`database` is relative to the configuration file. `model` has the existing
OpenAI-compatible driver's `endpoint` and `model`; `apiKeyEnv` names an environment
variable without putting a secret in the configuration. Configure the actual
approved executor; this command does not provision or resume a model service.

Capabilities are per application ID and operation. For example, an installation
can assign `read` to reader/operator/reviewer roles, `write` and `propose` to
operators, `approve` to reviewers, `commit` to operators, `reject` to reviewers,
and `dispatch` to delivery workers. Credentials are an array of
`{tokenHash, principal: {id, tenantId, roles, scopes?}}`; `tokenHash` is the
SHA-256 hex digest of a deployment-generated bearer token. No insecure anonymous
or default-password mode exists. Every case and record is tenant scoped.
Assign `reconcile` to a separate, narrowly held operations role; ordinary
`dispatch` workers cannot reconcile delivery outcomes.

The default listener is loopback. Put an authenticated TLS deployment boundary in
front when exposing it beyond the host, and protect the database and any
application-specific trace storage as sensitive application data.

## Application lifecycle

All routes except `GET /health` require `Authorization: Bearer ...`. Responses
are inert JSON. Native interface descriptions are available at
`GET /applications` and `GET /applications/:app`.

| Operation | HTTP request | Body / result |
| --- | --- | --- |
| Create a case | `POST /applications/:app/cases/:case` | `{interface, inputs, idempotencyKey}`; `inputs` must match that specific native interface. |
| Refresh source inputs | `PUT /applications/:app/cases/:case` | Same fields plus `expectedVersion`; invalidates old pending proposals. |
| Read case / checkpoints | `GET /applications/:app/cases/:case` | Source arguments, numeric version and separately stored typed native checkpoints. |
| List cases | `GET /applications/:app/cases?after=...&limit=...` | Stable case-ID pagination. |
| Invoke a domain operation | `POST /applications/:app/cases/:case/invoke` | `{interface, inputs?, expectedVersion, idempotencyKey}`. Omitted inputs use the saved inputs for that interface. |
| Inspect a proposal | `GET /applications/:app/cases/:case/proposals/:id` | Actual typed native result, source hash, exact argument binding, proposed checkpoint and outbox packet. |
| Approve | `POST .../proposals/:id/approve` | `{proposalHash, snapshotVersion, idempotencyKey}`; a separate authorized principal must approve. |
| Commit | `POST .../proposals/:id/commit` | `{proposalHash, snapshotVersion, approvalId, approvalHash, idempotencyKey}`. |
| Reject | `POST .../proposals/:id/reject` | `{proposalHash, reasonCode, idempotencyKey}`. |
| Audit | `GET /applications/:app/cases/:case/events?after=...&limit=...` | Append-only operation/version/hash/actor history. |

Inference happens on an immutable snapshot, outside any database transaction.
The exact validated result remains a proposal until approval and commit. SQLite
atomically persists the checkpoint, outbox, audit event and idempotency receipt.
A stale proposal must be rerun against fresh source inputs; the host never rebases
its meaning. A repeated invocation key returns the original persisted result,
including after restart, and changed inputs under the same key are rejected.

An interface may declare `stateParameter`, `stateResultPath` (an array of property
names, or `[]` for the complete result) and a `stateKey`. These are program-owned
bindings. A later invocation receives that exact committed state at the declared
parameter; a caller cannot replace it silently. An application that needs no
carried state leaves the parameter declaration absent.

## External effects and recovery

No model call has a database-writing or delivery service. Runtime network access
and source edits are disabled at this boundary. Native functions produce typed
domain intentions. An installed application-specific effect adapter, keyed by
application ID and interface name, may serialize those intentions to
`{key, topic, payload}` outbox entries. Adapters must perform exact serialization
and verification only; any decision about whether, what or whom to act on belongs
in that application's `.nl` source. An absent adapter means no external effects
are queued; the returned proposal says so explicitly.

Delivery routes bind application ID and topic to a configured endpoint. Run one
bounded dispatch pass with:

```sh
NATLANG_APPLICATION_TOKEN=... node applications/design-briefs-host/main.mjs \
  dispatch /path/to/deployment.json 001 worker-1
```

The receiver must implement the supplied idempotency key and return
`{status:"delivered", idempotencyKey, receiptId}`. A timeout, connection loss,
unconfirmed response or expired lease becomes an unknown outcome. It is not
automatically resent. Inspect `GET /outbox/:id`, then use
`POST /outbox/:id/reconcile` with `{outcome, receipt?, evidence:{kind,reference,assertion}}`.
Reconciliation requires the separate `reconcile` capability and an explicit
operator assertion `{kind:"operator_assertion", reference, assertion}`. The
authenticated operator identity, assertion text, reference, and time are kept in
the audit event and reconciliation record. This is an auditable human assertion,
not cryptographic verification by the host. The operator must inspect the remote
receiver before choosing `not_sent` (which requeues the original entry) or
`delivered` (which closes it with the supplied receipt). A positive HTTP status
by itself is not delivery evidence, and delivery workers cannot release unknown
outcomes by asserting their own evidence.

## Verification and limits

```sh
node applications/design-briefs-host/store.test.mjs
node applications/design-briefs-host/service.test.mjs
node ts-host/scripts/design-briefs/check.test.mjs
```

These tests cover exact host mechanisms and scripted native wiring. Compiler
checks, source diversity review and application-specific scenario review are
separate. Live model accuracy, deployed external connectors and each program's
operational acceptance still require their own evidence. Do not infer training
admission or production qualification from successful packaging or a scripted
driver. The corpus's source release records those outstanding measurements.
