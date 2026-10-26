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
can assign `read` to readers, `write` to evidence importers,
`invoke:reopenDecisionLedger` to ledger operators,
`invoke:resolveLedgerQuestion` to accountable decision owners, and
`dispatch` to delivery workers. Every native operation has its own
`invoke:NAME` capability; a generic permission to invoke everything is absent.
Domain approvals and acceptance decisions belong to the program's own typed
operations and source evidence. Credentials are an array of
`{tokenHash, principal: {id, tenantId, roles, scopes?}}`; `tokenHash` is the
SHA-256 hex digest of a deployment-generated bearer token. No insecure anonymous
or default-password mode exists. Every case and record is tenant scoped.
Assign `reconcile` to a separate, narrowly held operations role; ordinary
`dispatch` workers cannot reconcile delivery outcomes.

The default listener is loopback. Put an authenticated TLS deployment boundary in
front when exposing it beyond the host, and protect the database and any
application-specific trace storage as sensitive application data.

## Calling an application’s operations

All routes except `GET /health` require `Authorization: Bearer ...`. Responses
are inert JSON. Native interface descriptions are available at
`GET /applications` and `GET /applications/:app`.

| Operation | HTTP request | Body / result |
| --- | --- | --- |
| Create a case | `POST /applications/:app/cases/:case` | `{interface, inputs, idempotencyKey}`; `inputs` must match that specific native interface. |
| Refresh source inputs | `PUT /applications/:app/cases/:case` | Same fields plus `expectedVersion`; preserves checkpoints and makes in-flight old snapshots stale. |
| Read case / checkpoints | `GET /applications/:app/cases/:case` | Source arguments, numeric version and separately stored typed native checkpoints. |
| List cases | `GET /applications/:app/cases?after=...&limit=...` | Stable case-ID pagination. |
| Invoke a domain operation | `POST /applications/:app/cases/:case/invoke` | `{interface, inputs?, expectedVersion, idempotencyKey}`. Omitted inputs use the saved inputs for that interface. |
| Recover an invocation | `GET /applications/:app/cases/:case/invocations/:id` | Immutable actual native inputs/result, source/interface hashes, committed version and outbox identities. |
| Audit | `GET /applications/:app/cases/:case/events?after=...&limit=...` | Append-only operation/version/hash/actor history. |

Inference happens on an immutable snapshot, outside any database transaction.
After the actual native result passes its type and serialization checks, SQLite
atomically persists the invocation result, checkpoint, outbox and audit receipt.
The returned `invocation.result` is the application's own result, including any
domain-specific pending, refused, accepted or interrupted states. Saving that
state never implies external acceptance or delivery. A stale result must be
rerun against fresh source inputs; the host never rebases its meaning. A repeated
invocation key returns the original persisted result before inference, including
after restart; changed input or caller identity under that key is rejected.

An interface may declare `stateParameter`, `stateInputPath` (inside that parameter,
default `[]`), `stateResultPath` (inside the result, `[]` means the whole result),
and a `stateKey`. These are program-owned bindings. For example, an application's
`dossier.checkpoint` may receive its prior `resumeState` without replacing the
dossier's new observations. A later invocation receives that exact committed
state; a caller cannot replace it silently. Different operations share a
checkpoint only by explicitly declaring the same key, and the actual saved value
must satisfy every consuming operation's type. A stateless operation persists its
invocation receipt without inventing a domain checkpoint.

Start an instance with valid inputs for its initial operation. Invoke it with the
returned version and a new idempotency key. Use the resulting version for the next
operation, supplying that operation's new observations and omitting its saved
checkpoint field. If a connection drops, retry the same exact request/key to
recover the durable result. A new operation without saved initial inputs requires
its own arguments; this cannot silently reuse another operation's request.

## External effects and recovery

No model call has a database-writing or delivery service. Runtime network access
and source edits are disabled at this boundary. Native functions produce typed
domain intentions. An installed application-specific effect adapter, keyed by
application ID and interface name, may serialize those intentions to
`{key, topic, payload}` outbox entries. Adapters must perform exact serialization
and verification only; any decision about whether, what or whom to act on belongs
in that application's `.nl` source. Declare an `effectsPath` for a result field containing deliverable intentions.
A nonempty declared field without an installed adapter raises
`effect_adapter_required` before state changes; install the application's
adapter and retry the same invocation. An adapter receives frozen checked data,
must perform no I/O, and returns the outbox packet. The receipt reports whether
an adapter was installed. Undeclared domain plans remain data, not inferred sends.
Reusing an effect key with byte-identical topic/payload retains its existing
delivery state; different content under that key fails the entire transaction.

### Installed application adapters

These pure serializers are shipped with the host. Install only the adapter for
the declared application interface in `effectAdapters`; paths are resolved
relative to the deployment configuration file. For a configuration at the
repository root:

```json
{
  "effectAdapters": {
    "076": { "manageReferralPacket": "./applications/design-briefs-host/adapters/076-referral-packet-transfer.mjs" },
    "092": { "prepareDocumentaryDelivery": "./applications/design-briefs-host/adapters/092-documentary-delivery.mjs" },
    "097": { "prepareLiveEventEffects": "./applications/design-briefs-host/adapters/097-live-event-effects.mjs" },
    "098": { "prepareWorldActionEffects": "./applications/design-briefs-host/adapters/098-persistent-world-actions.mjs" }
  }
}
```

Each adapter reads only the checked `result.effects` array, maps the closed
`kind` to its topic, hashes the application/kind/domain identity into a bounded
outbox key, and stores the original effect unchanged as `payload`. The key
distinguishes send from status-read, reserve from acknowledgement, and
publication from other operations. It does not include payload contents, so
reusing the same domain identity with different content fails the store's
`effect_identity_conflict` check.

| Application kind | Outbox topic | Configured receiver operation |
| --- | --- | --- |
| 076 `send-referral-packet` | `076.referral.packet.send` | Transmit the exact authorized packet revision to its named recipient service. |
| 076 `read-recipient-status` | `076.referral.status.read` | Return recipient status for the exact packet, patient, and episode scope. |
| 092 `deliver-documentary-package` | `092.documentary.package.deliver` | Receive the exact assembly revision, digest, deliverables, and clearance references. |
| 092 `read-documentary-delivery-status` | `092.documentary.delivery.status.read` | Return delivery status for the exact assembly revision and digest. |
| 097 `reserve-event-resources` | `097.live-event.resources.reserve` | Reserve the listed resources for the supplied event revision and venue. |
| 097 `request-role-acknowledgements` | `097.live-event.roles.acknowledge` | Request the listed role acknowledgements for the supplied event revision. |
| 097 `publish-event-revision` | `097.live-event.revision.publish` | Publish the supplied event revision to its configured event receiver. |
| 098 `apply-player-actions` | `098.persistent-world.actions.apply` | Apply the supplied player actions against the exact world revision and rule version. |
| 098 `read-action-status` | `098.persistent-world.action.status.read` | Return simulator status for the supplied world and client action identity. |

Configure one `deliveryRoutes[applicationId][topic]` endpoint per receiver
operation, with any receiver credentials in that route's configured headers.
Each receiver must enforce idempotency for the scoped `idempotency-key` HTTP
header and accept the common request envelope
`{idempotencyKey, caseId, payload}`. It returns
`{status:"delivered", idempotencyKey, receiptId}`, echoing the envelope key
exactly. `payload` is the unchanged application effect object and therefore
retains its original semantic IDs, revision, recipient, and domain idempotency
key. Delivery acknowledgment records only transport receipt; the application
still reconciles domain outcomes through its own typed operation.

Delivery routes bind application ID and topic to a configured endpoint. Run one
bounded dispatch pass with:

```sh
NATLANG_APPLICATION_TOKEN=... node applications/design-briefs-host/main.mjs \
  dispatch /path/to/deployment.json 001 worker-1
```

The receiver must implement the scoped idempotency header and return
`{status:"delivered", idempotencyKey, receiptId}`, echoing the body key. The store
checks that binding as well as the delivery worker. A timeout, connection loss,
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
node applications/design-briefs-host/adapters.test.mjs
node ts-host/scripts/design-briefs/check.test.mjs
```

These tests cover exact host mechanisms and scripted native wiring. Compiler
checks, source diversity review and application-specific scenario review are
separate. Live model accuracy, deployed external connectors and each program's
operational acceptance still require their own evidence. Do not infer training
admission or production qualification from successful packaging or a scripted
driver. The corpus's source release records those outstanding measurements.


### Concurrent callers and cancellation

Identical requests in one host process share one inference while keeping separate caller cancellation. A pre-aborted caller is rejected immediately. Disconnecting one caller leaves any other waiter running; disconnecting the last aborts the shared invocation and prevents a late result from committing. The same key can then be retried. A result committed before disconnection remains readable through its durable invocation receipt.
