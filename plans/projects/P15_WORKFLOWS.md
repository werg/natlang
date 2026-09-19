# P15 — Stateful API workflow composer

Status: proposed implementation. [Shared capabilities](README.md).

## Natlang prerequisites

C0 provides semantic operation selection and bounded recovery. C1 supplies service clients in the crisp environment. C5 handles requests, replies and reconciliation events; finite fixtures can start with ordinary Fold. C4 records actual operations and outcomes. Durable state/receipts are requirements of this application host, not a new core lifecycle or implicit transaction facility.

## Programme and typed boundary

`step.nl(state, event) -> WorkflowState` calls `choose_operation.nl`, `bind_inputs.nl`, `assess_response.nl`, `choose_recovery.nl` and `plan_compensation.nl`. Extend the existing order saga pattern.

Records contain intended operations, input evidence, stable operation keys, acknowledgements, unresolved obligations and compensations. Distinguish success, definite failure and unknown external outcome. Native clients, credentials and in-flight request objects remain in eval. The model does not need credentials in its context.

## Crisp environment

Expose exact service methods and receipt/reconciliation access. Service-specific schemas constrain inputs and results; natlang decides sequencing and recovery. A shared environment may retain native clients directly. That convenience does not establish idempotency or make a lost response equivalent to no side effect.

A durable deployment stores operation intent before dispatch and commits its own state/cursor consistently. Each adapter describes idempotency keys and how to query uncertain outcomes. Where the remote system provides no way to resolve ambiguity, preserve an unresolved obligation. Compensation is a new explicit operation, not deletion of history.

## Reduction and stream shape

One Fold owns workflow state. It can dispatch an operation in a short step, then consume response/timeout/reconciliation events. Keys survive retries and process restarts where supported. Duplicate events must not cause repeated semantic dispatch. A timer event may request reconciliation; it must not automatically reissue an uncertain charge.

Initial fake services define deterministic fault points before dispatch, after remote commit and before local acknowledgement. They expose the same distinctions as the real adapter, rather than returning only success/failure booleans.

## Delivery and checks

1. Inventory/payment/shipping fixture workflow with ordinary typed responses.
2. Lose a payment acknowledgement and reconcile. Gate: exact delivered effect sequence and no duplicate charge.
3. Persist workflow state/operation keys in a local host and restart at every relevant boundary. Gate: unresolved work remains visible and recovery respects receipts.
4. Add a real service only after its response/idempotency contract is documented and verified.

Test malformed schemas, rate limits, authentication expiry, cancellation races, compensation failure and permanent uncertainty.

## Trace and teacher

Capture actual request identity, observations and compensating operations. Replay uses recorded service results or reset fake services, never live charges. Teacher targets emphasise correct binding and recovery with uncertainty. Final-state correctness alone is insufficient; effects are checked independently. Direct client sharing is compatible with this design, but unobserved native mutations must not be claimed as fully audited effects.
