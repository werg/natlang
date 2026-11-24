# Managed Vibe run interruption and partial completion

Shared mechanisms: an owned canonical system-test run must cancel and export while
its CLI/server/recorder still exist; an independently checked partial trajectory
is useful data but does not complete its user goal. Neither changes trajectory IR.

Inventory before the receipt/campaign edits:

| Component | Existing mechanism / work |
| --- | --- |
| `scripts/memory_ledger.py:1243` | Graceful systemd TERM incorrectly addressed all cgroup processes. Root owns main-only graceful signal, whole-cgroup final kill, opt-out and tests. Container TERM already addresses its main process. |
| Host `src/dev/runSystemTest.ts:419` | Signal immediately releases ownership and kills CLI before its checkpoint. Host successor forwards to CLI, joins authenticated cancellation/evidence, then releases dependencies. Child owns this plus collector changes. |
| Host `src/cli/systemTestCommands.ts:460,1070,1372` | Existing signal cancellation, `cancelSystemTestRun` and retained full interrupted packets are the authority; reuse, do not invent a new cancellation protocol. |
| Host `scripts/collect-live-workflow-teacher.mjs:121,303` | Synchronous children block signal handling. Successor awaits children, stops new batches, records exact interrupted cases, preserves actual complete/partial captures. |
| `scripts/vibestudio/qualify_open_goals.py:47,202` | Blocking child plus immediate SystemExit. Shared bounded authenticated cancel/export/join; direct actual Node entry avoids package-manager signal indirection. |
| V21 native `run_oracles.py`, `run_oracles_multi.py`, `run_oracles_android.py` | Current active controller variants share premature launcher SIGINT. Use the same bounded canonical cancellation helper, preserve completed packets, leave cancelled/no-outcome cases pending; historical saved drivers remain immutable. |
| `scripts/neuralese_data/vibestudio_campaign.py:485,551` | Campaign owns current collector and archive lifecycle. Forward one main signal, stop future work, archive actual interrupted captures without consuming an attempt for unfinished cases. Ready source/driver fixes enter at graceful resume, never mutate a live pin. |
| `scripts/neuralese_data/vibestudio_campaign.py:91,262,305,349,446` | Outcomes, telemetry, exports, scheduling and immutable receipt replay must distinguish complete goals from partial manifests; source-hashed historical reassessment overlay, original receipts untouched. |
| `scripts/neuralese_data/live_campaign_ingest.py:61` | Select both valid completed and partial manifests from committed exports; completed goal dedup and partial episode dedup already distinct (module docstring and ingestion implementation). |
| `scripts/neuralese_data/retained_state_archive.py:322` | Retirement intentionally requires every receipt to be completed/verified, plus matching export/ingest counts. Partial data must retain referenced state. Do not broaden its deletion eligibility. |
| `scripts/neuralese_data/vibestudio_android_tail.py` | Wait for nominal bulk success, own emulator, serial canonical teacher waves. TERM must checkpoint active teacher owner before device cleanup; partial-only tasks remain pending/retryable, actual full completion required for goal completion. |
| Archive/export status readers | Receipts remain original records; reassessment must expose source hashes and exact before/after completion counts. Exported `verified` remains number of accepted trajectory rows, with separate existing `completed_goals`/`partial_trajectories`. |

Downstream training continues consuming the same native trajectory IR and
per-call sequences. The existing exporter already reads
`trajectory_ir.outcome.task_completed/status` and counts complete versus partial
separately (`vibestudio_campaign.py:327`); no new IR field or model request field
is introduced. Training ingestion is the boundary that reads these receipts;
actor/learner weights, tokenizer, optimizer and RNG state are unaffected.

Serving engine inventory (`docs/ENGINE_PARITY.md:7-12`): managed llama-server,
in-page wllama, Python native and SGLang implement model requests, not Vibe wave
receipt selection. Their request schemas and generated per-call targets do not
change. The ts-host likewise consumes trajectory/program execution, not campaign
receipt status. Scope exclusions must remain checked against the final diff;
this is not an engine parity exemption for a changed inference protocol.

Historical evidence is never rewritten or discarded. Reassessment records bind
the original receipt, manifest, verification and actual outcome hashes; only
completion indexing/scheduling changes. An interrupted case with valid partial
IR remains exportable; an unfinished/cancelled case is not a failed user goal.

Review authority is shared by the producer's `verification.requirements.semantic_review`
and the exact whole-goal proof's `semanticReviewRequired` (SST
`open-goal-product-owned-managed.ts:169` is one direct producer). The reader
honors either required declaration; absent/false metadata cannot waive a native
proof's required semantic review. This does not change IR or review input formats.
