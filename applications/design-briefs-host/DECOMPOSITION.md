# Design briefs: exact store and bounded application seam

## Independent program architectures

There is no shared semantic application interface. Each program declares its own
entrypoints, parameter types, returned data, state transitions and operational
policies in its native sources and `application.json`. This transport neither
normalizes those contracts nor dictates a five-stage workflow. Its database is an
optional mechanical persistence boundary. It is excluded from model-visible
training projects. Similar plumbing is not counted as architectural diversity.

`runtime.mjs` loads the declared native function and uses the existing runtime and
type checker. `service.mjs` binds the actual source bytes, caller arguments and
typed returned value to an immutable proposal. A program can explicitly name a
state parameter and result path; only then does the host inject its previously
committed checkpoint. That mapping is exact copying, not a common domain state
machine. `server.mjs` authenticates and transports those native interfaces as-is.
`delivery.mjs` sends only approved outbox entries to configured routes and requires
a matching external receipt. Application-specific adapters serialize domain
intentions; the host does not derive recipients or actions from prose.

## NatLang-owned decisions

The configured NatLang plan owns domain interpretation: which evidence is relevant, which candidate policy follows from it, whether missing evidence permits a conclusion, what typed record changes to propose, and what explanation/review disposition to return. A plan run receives a tenant-, application-, and case-scoped snapshot and returns typed data. It must not write storage or trigger external effects. The host may select the configured plan and transport its value, but does not implement domain policy in JavaScript.

## Crisp mechanism

`store.mjs` is the exact persistence boundary shared by the 100 brief applications. It owns SQLite schema/migrations, principal and tenant scoping, configured role/scope checks, JSON validation/canonical hashing, optimistic versions, immutable proposal and approval bindings, atomic case-record materialization, audit events, durable idempotency, and outbox leasing/acknowledgement. It takes no model/runtime dependency. All methods receive an authenticated principal; authentication itself is the host's responsibility. The configured capability map is deployment policy, not inferred from record contents.

A proposal stores the full NatLang `RunProposal`, exact source snapshot version/hash, source hash, typed write packet with each expected record version, exact outbox payloads, requester, and one immutable hash over that complete envelope. Approval is a distinct record tied to that proposal ID/hash and snapshot version; the configured capability map must authorize the approver, and the requester cannot approve their own proposal. Commit accepts only proposal/approval identities and hashes plus the snapshot version. It never accepts replacement writes or outbox data. It rechecks the packet and every record’s expected version inside its transaction. A write increments the case version and inserts the outbox row plus audit event in the same SQLite transaction. Domain validation and NatLang execution happen before entering that transaction. No network/model operation occurs while a write transaction is open.

Idempotency rows are durable and bind a scoped key to a canonical payload hash and result. Reusing a key with different content is a conflict. SQLite transaction boundaries ensure a restart sees either the full commit (including outbox/audit/idempotency) or none of it. An outbox delivery attempt is claimed with a lease and attempt count; claims, acknowledgements, failures, expiry and reconciliation are audited in their case event stream. A lease that expires after a send may represent an unknown remote outcome: the row becomes `unknown`, stays readable and auditable, and is never automatically resent. Expired leases and ambiguous failures are reconciled explicitly with a remote receipt or an attestation that the request was not sent. Only a reconciliation with persisted evidence can move a failed or unknown item to delivered or back to pending; a not-sent result requires a receipt or an explicit confirmation reference. Blind retries are forbidden.

## Public store contract

`openDesignBriefStore({filename, capabilities, clock?}) -> DesignBriefStore`; `close()` closes the SQLite handle. `capabilities` maps application IDs to operation-to-role arrays and may include per-role scopes. Public methods are synchronous because `node:sqlite` is synchronous; call them only for short local transactions. Storage errors are normalized to stable codes and never include SQL, filesystem paths, or raw driver text.

Every key is explicit: `{tenantId, appId, caseId}`. A principal has `{id, tenantId, roles, scopes?}` and must match the key tenant. The stable API is:

- `createCase(principal, key, initialSnapshot, {idempotencyKey}) -> snapshot`
- `readCase(principal, key) -> snapshot | null`
- `listCases(principal, {tenantId, appId, after?, limit}) -> {items, next}`
- `replaceCase(principal, key, expectedVersion, snapshot, {idempotencyKey}) -> snapshot` (case import/evidence refresh, optimistic and audited)
- `saveProposal(principal, key, {snapshotVersion, snapshotHash, proposalId, proposal, writes, outbox, sourceHash}, {idempotencyKey}) -> proposal`
- `readProposal(principal, key, proposalId) -> proposal`
- `approveProposal(principal, key, proposalId, {proposalHash, snapshotVersion}, {idempotencyKey}) -> approval`
- `commitProposal(principal, key, proposalId, {proposalHash, approvalId, approvalHash, snapshotVersion}, {idempotencyKey}) -> committed result`
- `rejectProposal(principal, key, proposalId, {proposalHash, reasonCode}, {idempotencyKey}) -> proposal`
- `listEvents(principal, key, {after?, limit}) -> {items, next}`
- `claimOutbox(principal, {tenantId, appId, workerId, leaseMs, limit}) -> rows`; `ackOutbox(principal, entryId, workerId, receipt)`; `failOutbox(principal, entryId, workerId, {code, ambiguous})`; `reconcileOutbox(principal, entryId, {outcome: delivered | not_sent, receipt?, evidence: {kind, reference}})`; `readOutbox(principal, entryId)`.

Write packets are arrays of `{recordType, recordId, value, expectedVersion}` with a `null` version for a new record (or a positive integer for an update). Each record value must bind its exact `id`, tenant, and next record version, which advances from its own previous version rather than the case version. They are materialized under the exact tenant/app/case/type/id composite key. Snapshot `caseRecords` are the original typed array input; optional capture timestamp and previous evidence pages are retained. Application-produced records are returned separately as `domainRecords`, each with its materialized version. `snapshotHash` binds the version, typed snapshot and complete current domain-record view. The store rejects duplicate record identities in one packet and never accepts domain records through a case import. Event/outbox pagination uses stable monotone IDs and bounded page sizes. The host validates the brief's schema and authority fields before invoking this generic store; the store enforces its envelope, limits, immutable hashes, lifecycle, and configured capability map.

## Failure and lifecycle distinctions

Lifecycle values are `proposed`, `approved`, `committed`, `stale`, and `rejected`. A concurrent update turns an otherwise valid proposal into `stale`; it never rebases. Rejection records a stable code rather than requiring free-form policy text. Missing/unauthorized, conflict, stale, validation, and ambiguous delivery outcomes are distinct stable errors. Every state-changing call records principal, key, operation, payload hash, and before/after version or lifecycle in an append-only event table in the same transaction.

## Validation

Tests use temporary on-disk databases and reopen them between operations. They exercise atomic recovery, version races, isolation, capability denial, immutable binding/tamper rejection, changed-payload idempotency conflicts, and ambiguous outbox lease recovery. Runtime, HTTP, CLI, UI and plan execution are owned by the application host; those layers do not weaken this store contract.
