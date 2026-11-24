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
