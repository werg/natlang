# Provider transport failure retry accounting

Mechanism inventory: `vibestudio_campaign.outcomes` (lines 90–141) reads the
original execution error into a receipt; `publish_status` (lines 392 onward)
classifies receipts before charging goal retries, and `wave_infrastructure_hold`
uses the same classifier. Normal collection and recovery both replay these
immutable receipts. Fix their shared `infrastructure_failure` mechanism; do not
rewrite raw captures, relabel success, discard decisions, or hide sampled exposure.

The 0052 archive contains failed Natlang calls with fetch/TLS/circuit-breaker
operational causes. These are infrastructure failures even if earlier calls in
the same task sampled a model. The previously recorded teacher exposure remains.
The model's descriptions of outages and ordinary incorrect answers remain goal
outcomes. Legacy receipts require an anchored failed-call envelope and a known
operational transport cause; incidental quoted text is insufficient.

Serving engines do not consume campaign retry accounting: all engine paths in
`docs/ENGINE_PARITY.md` produce model calls, while the complete classification
call-site inventory is the two Python consumers above. Native call evidence has
its own classification (`ts-host/src/calls/evidence.ts:95`); teacher request
retries use `ts-host/src/teacher/retry.ts:30`. Neither decides this campaign's goal
retry budget, so neither is changed. Ingest/recheck consume retained episodes,
not this retry counter (`live_campaign_ingest.py`, `recheck_live_trajectories.py`).

Default: operational transport failures do not consume goal retries and can
trigger the existing infrastructure hold. Explicit diagnostic opt-out:
`NATLANG_CAMPAIGN_TRANSPORT_FAILURES_AS_GOAL_FAILURES=1`. This changes derived
accounting on receipt replay; originals are untouched.

Archive transport inventory: `transfer` owns mkdir, upload, and checksum readback;
`remote_archive` then publishes immutable receipt/wave metadata; `sync_index`
publishes derived progress. All are idempotent SSH/rsync operations and receive
bounded retries for SSH255 / rsync network IO, stream, and inactivity timeouts.
Checksum differences, partial/vanished source files, local disk errors and remote
job submission are not retried by this helper. Remote job submission is a separate
non-idempotent operation and retains its existing named-unit check. SSH connection
and liveness timeouts bound silent disconnections. Explicit opt-out:
`NATLANG_ARCHIVE_TRANSPORT_ATTEMPTS=1` (default4). Existing captures remain owned
until checksum validation and receipt publication; retries never invoke a teacher.

## 2026-11-25: cohort circuit breaker

Inventory before editing: repository-wide search finds one hold implementation
(`scripts/neuralese_data/vibestudio_campaign.py:195`), one driver call (718),
its CLI threshold declaration (745), and direct tests in
`tests/test_vibestudio_campaign.py`. `infrastructure_failure` also serves
`publish_status`: retry accounting must remain unchanged. Recovery replays
original receipts; no other engine, collector, export, or ingestion code applies
this campaign wave stop rule. The serving-engine exclusions above still apply.

The current rule is absolute: any two infrastructure failures stop a wave,
including two isolated provider transports among64 attempts. Keep the absolute
threshold for structural/build/capture/workerd failures, but apply a separate
provider-transport fraction (default0.5) among noninterrupted attempts, also
requiring the count threshold. `--provider-transport-hold-fraction 0` restores
the previous absolute rule. Capture corruption accompanying a transport remains
structural; cancelled cases neither trigger holds nor dilute the denominator.
This is a bounded circuit-breaker policy change, not a diagnosis of the network
failure. Original captures, exposure, goal retries, and semantic checks remain
unchanged; the underlying transport cause remains open.

## 2026-11-25: recorded provider errors behind a quiesced caller

Actual V34 notification case: the requested explanation is missing, so the goal
correctly fails. Two descendant model exchanges contain native `error: fetch
failed` with no reply. The caller quiesces with a description of that failure;
the legacy failed-call text matcher cannot identify its operational origin.

Inventory before editing: `outcomes` already reads the original native calls;
the shared retry/hold consumers remain `publish_status` and
`wave_infrastructure_hold`. The authoritative evidence producer is
`packages/natlang/src/modelTurnCapture.ts` (`generation | score`, `reply | error`),
through the same runtime driver boundary for all model backends. No serving
retry or model policy changes apply. Archive/recovery preserve receipt fields;
ingest and training admission still require their original native goal proofs.

Emit narrow source-referenced receipt evidence only when the authored root
errored and an actual failed descendant's final generation/score exchange has
an operational transport error and no reply. Exclude calls with successful
recorded retries and errors outside the actual root lineage. Never parse an
application return, model prose, tool output or quoted error as this evidence.
Keep the failed goal, actual sampled exposure and immutable raw packet. A real
unrecovered provider outage contaminates retry accounting even if the model
also made earlier mistakes; this is not a claim that it alone caused failure.
The existing transport-as-goal-failure opt-out remains.
