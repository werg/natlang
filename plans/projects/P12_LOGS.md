# P12 — Log anomaly tracker and investigator

Status: finite Fold replay and host evidence index implemented in
[`codebases/log_investigator`](../../codebases/log_investigator/README.md) and
[`applications/log_investigator.mjs`](../../applications/log_investigator.mjs).
Live source buffering, durable receipts and teacher quality remain open.
[Shared capabilities](README.md).

## Natlang prerequisites

C0 supplies semantic investigation and bounded loops; C1 supplies exact search/aggregation over native log data. C5 supplies window/event streams with backpressure and explicit closure/failure. C4 later reconstructs incident decisions from observations. There is no global search tool, vector database requirement or in-episode interruption.

## Programme and typed boundary

`step.nl(acc: IncidentState, item: LogEvent) -> IncidentState` calls `assess_window.nl`, `form_hypotheses.nl`, `choose_evidence.nl`, `compare.nl` and `choose_escalation.nl`. Exact helpers compute counts, rates and deduplication. Natlang determines significance, requests evidence and updates hypotheses.

Inputs are bounded window summaries and source-span IDs. State distinguishes observations, hypotheses, evidence gaps, incident status and notification identities. Native log buffers, indexes and connections remain inside eval. Retained incident summaries must not silently discard contradictory evidence.

## Crisp environment

Illustrative `logs.query(filter, range)`, `metrics.window(spec)`, `deployments.read(id)` and `alerts.send(message, key)` bindings. Search starts as exact code over a fixture collection, then can use an index without changing the global interpreter tools. Notification is an optional configured operation; begin with local incident outputs.

The host chooses storage and authority. Shared objects allow efficient inspection, but every observation used in an admitted trajectory needs a stable captured view or an explicit replay limitation. A stronger-model helper can be provided in the environment for difficult cases, with recorded model/config and budget.

## Reduction and streams

Fold consumes completed windows, evidence results and incident acknowledgements. The host distinguishes event time from arrival time and records its windowing policy. New evidence waits for the next Fold step. Long investigations should split into bounded requests/results so newly arriving windows are not blocked indefinitely.

Buffer limits and overload policy belong to the source adapter. Progress can be coalesced; state-changing events need explicit handling. Restart reliability later requires application-owned state/cursor persistence and notification reconciliation, not a new universal runtime failure state.

## Delivery and checks

The current Fold uses explicit source and gap events. Natlang assesses bounded
evidence; the host checks exact source rows and a minimum distinct count before
an alert, and retains idempotent receipts. Event/arrival times are separate,
late evidence is visible, and uncertain sink delivery stays unknown. Tests
exercise escalation, benign log text, duplicate suppression, source gaps,
fabricated evidence and sink errors. This is a finite replay and host boundary
test, not yet a calibrated incident-detection quality result.

1. Finite log replay with one regression, one benign burst and one missing-data interval. Gate: exact window metrics, independently labelled incident quality.
2. Query raw evidence and revise an initial wrong hypothesis. Gate: final claims cite actual observations and preserve unknowns.
3. Connect a live-like C5 source with out-of-order data and overload. Gate: cursor/window policy and no silent incident loss.
4. Add configured notification sink and restart handling. Gate: duplicate suppression and explicit uncertain delivery.

Test clock skew, duplicate logs, misleading log text, late evidence, source failure and unavailable metrics.

## Trace and teacher

Record consumed summaries, requested queries, returned spans and incident transitions. Teacher samples include benign anomalies and disconfirming evidence; success is not equivalent to always escalating. Measure detection quality, unsupported causal claims, alert multiplicity and query cost separately. A finite replay host remains a useful lightweight embedding after the production adapter grows.
