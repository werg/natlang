# Individual-program tuning: production operations

Owner request, 2026-10-26: integrate the tuning handoff into operations and measure how far individual programs can improve with adapter tuning, teacher SFT and RL. This is an active operational lane alongside comprehensive LFM/Mellum training. Program adapters have separate identities and exposure; their updates do not advance the production model's global step.

## First workload

Prepared in `/home/werg/data/program-tuning-20261026`, using the existing tuner and reviewed six-worker generation dispatcher. Each world has two specific invocations; the persisted component split keeps both together. Oracles are deterministic task definitions, not authored assistant trajectories.

| Program | Training roots | Development roots | Confirmation roots | Execution |
| --- | ---: | ---: | ---: | --- |
| ticket-routing | 94 | 20 | 14 | Named severity and routing functions; structured policy overrides untrusted prose |
| invoice-review | 186 | 60 | 10 | Exact integer tool plus invoice-policy function |
| inventory-routing | 84 | 34 | 10 | Reservation-aware availability tool plus fulfillment function; balanced route outcomes |

Invoice review received an additional fixed batch of new worlds because the initial component split left only two confirmation invocations. Existing assignments were preserved. No confirmation result was inspected or used for selection.

The first adapter parent is the independently recovered LFM step 39,495, SHA `fe6740d1cda334e83d27c65f105e1ba1ea01c4c8b08345699f5bad1dbb9e375b`. Its full state is copied and rehashed under `/mnt/external/natlang-program-tuning-20261026/parents/`; `lfm-parent-identity.json` records the identity of the actual restored CUDA model. Keep this pin until all dependent tuning runs and their own recovery are secured. The production learner continues earning newer updates separately.

Exported cases are under `runs/tune-cases/program-tuning-20261026/`. Only the 364 training and 114 development roots are exported. The requested teacher campaign is `runs/luna-program-tuning-six-20261026-v1`, using the existing six-worker Luna capacity after the preceding campaign's actual workers drain. Use `python3 scripts/run_reviewed_luna_campaign.py PLAN --sha256 SHA` for restartable launches: it delegates unchanged validation to the reviewed dispatcher, choosing fresh start or resume from actual ledger presence. Its reviewed plan, inventory, frozen runtime and per-case results remain authoritative; a launch request is not evidence of active workers or usable trajectories. Current receipts live with the operational data, rather than being inferred from this document.

## Executable sequence and adoption

1. Verify program build, scope tree, persisted group split, exact task oracles and teacher-source bindings. Keep confirmation roots out of collection and gradient inputs. Retain wrong/failed teacher attempts as diagnostic evidence, without making their faulty actions positive targets.
2. Collect through the authorized reviewed pool. Do not start a second pool or silently fall back to direct teacher calls. Import only exact program/build/scaffold/split/root-bound results, with authenticated campaign, runtime, actor and result-row custody. Current general-curriculum batches lack these tune bindings and cannot be counted as program-specific imports.
3. Serve the exact pinned student through the native backend. Compare its actual reader identity with the configured four-field student identity before collection/evaluation/rollouts. The native backend supports exact adapter/soft-state application and RL sampling receipts; a GGUF of a different base is not a substitute.
4. Measure untuned development execution, the teacher's actual development outcomes, runtime failures and latency. Run adapter SFT on all usable teacher training turns, retaining invocation bindings and failed-history masking. Start with an adjustable useful budget, then continue while development behavior improves; one-step mechanism checks are not completion. Keep adapter/soft-state optimizer histories and recoverable artifacts.
5. Evaluate the SFT release, then sample native whole-graph RL rollouts from that actual release on training roots. Use at least four samples per root initially and actual root outcomes. Update every recorded turn and genuine producer/consumer dependency in an episode; no detached consumer-only substitutions. Correct executions remain eligible without Neuralese use. Zero-variance groups earn no policy update: report them and improve training supply/difficulty instead of reporting a launched RL stage as learning.
6. Alternate teacher SFT and RL as useful, select on development outcomes and retain the best useful release. On-policy teacher repair is added through an authorized teacher path when available, not through an unconfigured automatic fallback. The first rounds do not depend on a separate repair teacher service.
7. Confirm the selected release on the untouched confirmation worlds. Report base, teacher, SFT and RL outcomes separately, including root counts, world-group intervals, invalid/type/tool/runtime failures, effective graph counts, wall time, tokens, and serving latency/memory. Never claim lift from an exported adapter or gradient check.
8. Repeat on Mellum with its own actual checkpoint and reader identity, sharing programs, oracles, mechanisms and findings. Reuse teacher trajectories only where their exact build/scaffold/split bindings remain valid; preserve original teacher identities. Resource placement follows the measured lean compute policy, not an automatic extra sandbox.

## Production feedback and continuing work

Use the existing incremental/full-refit mechanisms for later program changes. Start family transfer after at least two programs have useful measured releases on the same exact base; neither families nor backend sweeps block the first real SFT/RL run. Measure placement/backend alternatives when a concrete throughput or quality question warrants it.

Distill verified training-split tuned executions through the implemented export and ordinary source admission path into the evolving production recipe for both models. Export alone earns no exposure. Development/confirmation worlds remain protected from gradients and from general training admission. When production checkpoints change enough to justify retuning, rebuild against the new exact identity and compare actual outcomes; do not reuse incompatible native state or silently relabel adapters.

The handoff review found integration work before first activation: retain oracle-approved expected root stops without training failed child actions, verify native serving identity before use, and authenticate imported campaign/result custody. These are implementation obligations, not reasons to claim a partial capability. Track their actual tests and deployment in the run receipts.

## First execution and action-quality correction

The closed six-worker campaign completed all 478 requested roots: 477 passed their task checks. Ticket routing imported 114 roots and 382 turns with exact actor/result custody. These are teacher outcomes, not student capability results. The untuned pinned LFM scored 0/20 correct on ticket development; invoice evaluation was stopped after 0/41 correct of its 60 development roots. Inventory baseline remains unmeasured. The bounded baseline used 12 model requests per root; compare releases at the same budget before claiming lift.

The first ticket adapter fit reached eight updates and 128 row visits, then was stopped for a source-quality correction. A successful enclosing invocation had incorrectly made failed intermediate actions eligible for SFT. Action-outcome policy version 2 now binds each target to its actual tool outcome, rejects failed/unverified actions, and masks failed assistant history while retaining it as context for correct repairs. Reimport the same authenticated closed results under the explicit lowering migration; do not change teacher identity or erase earlier exposure. The corrected ticket training set has 315 admissible turns, with two rejected actions retained diagnostically.

Continue that saved partial fit using `tune.rounds.initialFitParent` pointing at its actual build output. It becomes the first fit's optimizer/artifact parent, not an invented evaluated release; subsequent rounds use their actual selected parent. Fit complete useful passes over corrected teacher rows, evaluate development, then advance into whole-graph RL and further SFT/RL rounds. A stage budget is adjustable and is not production completion. Neither a completed first SFT pass nor an RL launch establishes capability lift.

The closed campaign, immutable import custody, three grouped programs, frozen runtime and saved eighth-step adapter state have a verified private export independently downloadable with no sandbox running. The separately verified full LFM parent remains pinned locally and in its original recovery lineage.

## Execution budgets

Set `tune.execution` to explicit invocation limits (`maxTurns`, `maxTokens`, `maxSeconds`, `maxFailureRepairs`) for costly real runs. The same mechanism applies to collection, paired base/tuned/teacher evaluation, repair continuations, distillation and native whole-graph RL. Evaluation and rollout receipts retain the effective limits. Changed limits select a separate rounds journal and rollout group; old outcomes are not silently reused under a new budget. The first operational configuration uses 12 turns, 4096 generated tokens, 120 seconds and three consecutive failed-action repairs per invocation. These are invocation limits, not the earlier baseline's 12-request root-wide wrapper, so compare against fresh paired base results under the same limits.

## First complete SFT pass and invalid inference cohort

The corrected ticket fit completed 20 additional updates from its eighth-step parent. All 315 usable teacher turns were visited (320 new row visits, 448 including earlier history); five adapter scopes and three learned soft states were preserved. The complete adapter/soft-state optimizer output has a verified private recovery capsule independently downloadable without a sandbox.

Its first tuned evaluation and sampled cohort are invalid infrastructure evidence: the executing runtime lacked the native artifact store needed to authenticate soft-state imports. The reported 0/20 and zero-reward episodes are not capability or policy-learning evidence and must not enter RL or distillation. Keep their exact diagnostic receipts. Native transport/session wrappers now expose the complete runtime binding; collection, evaluation, repairs, distillation, rollouts and ordinary CLI execution use its actual store/reader/writer. Missing native provenance raises an infrastructure error before a task can be scored. A new `native-runtime-v2` rounds namespace reuses the completed adapter build, with a fresh evaluation and rollout cohort; it earns no duplicate SFT updates.

## Three saved fits and reply-focused continuation (2026-10-27)

The three initial teacher-SFT fits are saved with adapter, native soft-state and optimizer histories. Ticket completed 20 additional updates from the earlier eighth-step parent; invoice completed 24 updates and 384 visits over 379 distinct teacher turns; inventory completed 12 updates and 192 visits over 177 distinct turns. These are separate program updates against the same pinned LFM parent. None has earned an RL optimizer update yet. Their actual build paths and receipts are recorded in `/home/werg/data/program-tuning-20261026/three-program-first-sft-progress-20261027.json`.

The provenance-fixed ticket release completed a valid development evaluation at **0/20**, dominated by type errors and token/wall-clock exhaustion. Its earlier provenance-broken cohort remains excluded. Invoice and inventory evaluation were paused after their saved fits: serialize evaluation and rollout collection on the one native actor so unrelated requests do not consume each other's time budgets. Confirmation remains untouched. Fresh paired base results at the current invocation limits are still required for a capability comparison.

A tokenizer/template audit of all 871 admitted teacher turns found median prompts of 4,631–5,185 tokens versus replies of 81–98 tokens. With prompt weight 0.1, the reply carries only about 14–16% of total weighted supervision. This is a concrete optimization concern, not proof of the cause of all failures. Continue the actual saved fits with reply-only SFT (`recipe.prompt_weight: 0`), then inspect development execution and sample RL from the resulting release. Preserve earlier exposure and moments; do not reset the adapters or count old turns as newly distinct. Diagnose actual gold-action loss and train/inference parity alongside behavior, rather than treating a low prompt-dominated aggregate loss as task mastery.

Adaptive rounds journals must bind the student base, SFT recipe, round settings (including the initial fit parent), soft-state mode and execution limits. A recipe change starts a fresh journal continuing the explicit earned parent; the identical continuation resumes that journal. Existing journals and their diagnostic evidence remain intact. Larger RL cohorts need useful reward variation; zero-success sampling is evidence to improve SFT or the training curriculum, not earned policy learning.


## World-bound curriculum and saved reply-only fit

The ticket reply-only continuation saved 60 additional updates across three passes,
continuing the actual earlier adapter and native soft-state optimizer histories.
Its final batch mean reply loss was about 0.52. Development execution is still
being evaluated; loss reduction is not a capability result. A read-only gold-action
check on three training bindings also confirmed that the actual serving adapter
lowers action-token cross entropy versus the base. No policy update or held-task
lift follows from that check.

The first standalone leaf curriculum exposed a root-identity defect: identical
function inputs in different invocation worlds collided because root keys omitted
the world component. Strict imports correctly rejected conflicting results. Current
`natlang.tune-root/v2` identities bind the world component, function, input, oracle
presence and expected value; identical roots within one world are deduplicated.
Build and world split assignments remain unchanged. Preserve existing fits and
actor identities; collect fresh executions under the corrected keys instead of
relabeling historical teacher rows or relaxing import custody.

The fresh three-program campaign is
`runs/luna-program-world-root-v2-six-20261027-v1`, sharing the same authorized
six-worker authority. It contains 665 train/development roots: 301 ticket,
246 invoice and 118 inventory. Ticket includes 187 distinct additional standalone
training invocations of its named leaf functions, alongside the original main
roots. These are complete new episodes, not detached turns from larger graphs.
Confirmation worlds remain excluded. Import verified new results into the
world-root-v2 program projects, continue from each actual newest earned fit, and
measure whole-program and leaf outcomes separately. RL still updates every actual
graph dependency and only nonconstant outcome groups earn policy updates.


Release serving must restore the complete native dependency closure before
uploading a derived state. Alphabetically retrying rejected state uploads makes
long optimizer histories expensive and produces missing-parent errors. The CLI
now uses the same dependency restoration mechanism as ordinary native execution,
including tensor/state dependencies and exact-object reuse. The parent-order,
reuse and tampered-file checks pass. Continue the saved fit and identical round
journal through this serving fix; it earns no repeated SFT credit. The 60-update
ticket fit has a separately verified private full adapter/soft-state recovery
capsule, independent of the production LFM checkpoint.


## Measured ticket result and online RL (2026-10-27)

The reply-focused ticket fit completed **12/20 correct development roots** with
valid native provenance. The preceding valid SFT release completed 0/20. This is
a development result between those actual releases, not a confirmation result or
a newly paired base comparison at the current invocation limits. Confirmation
remains untouched. Its saved build has an independent full adapter/soft-state
backup, including optimizer history.

The world-root-v2 teacher campaign completed all 665 cases with oracle acceptance.
Imports admitted exactly 301 ticket, 246 invoice and 118 inventory roots, with no
unbound turns; each import correctly rejected rows belonging to the other two
programs. New teacher data has not thereby earned student exposure.

Real RL sampling produced complete four-sample groups with differing task outcomes,
including rewards [0,1,1,1]. Avoid waiting hours for all training roots before the
first policy update: seal complete informative groups with exact actor/release and
whole-graph custody, then continue the actual saved optimizer on those groups.
The first sealed tranche contains 12 episodes from three informative groups. It
keeps every episode intact and preserves the full teacher row set; constant-reward
and incomplete groups retain their diagnostic records without policy credit.
Shared `trainRelease`/native policy mechanisms perform the updates. Continue fresh
cohorts and the expanded teacher curriculum from earned children; a tranche is
not whole-program or production-recipe completion. Report actual RL optimizer
steps and subsequent task outcomes separately from a successful collection.

The first informative tranche's original four-episode batch exceeded its 16 GiB
job envelope after saving one SFT update. It saved no RL optimizer update. The
continuation retains that actual child (`8f134e15f4e21eebdb65e0c6/out`), uses a
32 GiB envelope and batches one complete episode at a time; group-relative reward
statistics still come from the complete four-sample groups. It schedules 24 mixed
SFT/RL updates, with reduced learning rates, without detaching any graph turn.
The profiler attributed about 683 of 868 sampled seconds to native ancestry
verification. Shared bounded proof reuse authenticates every transitive member's
bytes on reuse and returns freshly decoded state; it retains no mutable KV.
Corruption, disabled-cache, cross-objective reuse and real LFM/Mellum gradient
checks pass. The current continuation uses the exact frozen cache runtime and a
direct trainer process so a killed child cannot be hidden by a profiler's exit
status. Record actual saved policy steps and fresh development outcomes before
crediting RL. Its native evaluator actor is restored after the fit exits.

The archive builder also now verifies complete ancestry from the forest's heads,
rather than numerically replaying every intermediate state as another root.
Every archived state must belong to a covered chain; cycles/missing ancestry
remain errors. The 38-test focused LFM/Mellum replay, policy and derivative run
passes (two unrelated skips). A zero-update initialization that had performed
more than 144 GB of repeated cached reads was stopped before its initial state
was saved; the new frozen runtime continues the same actual earned child. No
optimizer update was discarded. After the tranche completes, the resident actor
is restored and a resumable paired development job compares RL, the 12/20 teacher
SFT release and the base under identical per-invocation limits. It neither reads
confirmation nor activates an unvalidated release.

The improved tranche reached saved step 3: SFT, **one actual whole-graph RL
optimizer update**, then SFT. The next episode revealed a routing bug when the
binding inventory mixed `null` (an invocation of the frozen base) and tuned
binding IDs. Adapter routing now selects the real tuned bindings; the full
policy graph still contains the base-only producer and every consumer. A real
captured producer/consumer regression passes on both architecture ports, checking
that both graph nodes remain and the consumer's actual adapters update. The
10-test tuning/archive check passes. Continue this exact fit's saved state and
RL cursor through the frozen routing fix; do not reset its episode order or
retrain its completed prefix. The current result is one saved policy update,
not a completed tranche or an RL capability-lift claim.

The exact continuation completed all 24 updates, **12 whole-graph RL and 12
SFT**, with RL cursor 12: all twelve admitted episodes were visited once. The
shared publisher reused that completed numeric build and retained the actual
routing-fix runtime and resume receipt. Full native artifact/optimizer ancestry,
sealed rollouts and the exact pinned base are being independently exported on
the external drive. Development evaluation is queued under the unchanged actor
and execution contract; no RL task-lift or confirmation result is claimed.

The completed mixed SFT/RL continuation scored **17/20** development roots,
compared with the preceding SFT release's valid **12/20** on the same exact
roots and per-invocation limits. This is a longitudinal development result of
all the continuation updates, not isolated RL attribution. The fresh base arm
is still running; confirmation remains untouched. The saved 12/20 SFT arm was
reused after checking its exact release, complete root/component IDs and limits,
with its original receipt hash recorded rather than spending GPU time repeating
that measurement.

An initial standalone comparison omitted Node evaluator installation. Its
zero-request failures are marked invalid infrastructure evidence and excluded
from capability, RL and distillation. Missing platform setup now carries a typed
error that every shared tuning execution path propagates. A child-process test
executes the genuinely uninitialized host and verifies that it aborts before a
model request. The corrected runner installs its Node host and records all real
calls in its own bounded program store.

Profiling also found the machine journal's orphan sweep scanning 3.4 million
blob keys during automatic eviction. Its reference indexes already existed.
Automatic eviction now bounds candidate and call work, advances a durable
orphan cursor, and preserves pinned, annotated and case-linked records; explicit
manual eviction retains full cleanup, with an optional CLI work limit for
maintenance workers. Both Node and WASM contracts pass the bounded/protected
cleanup checks. A reviewed worker is freeing old evictable machine-journal data
without blocking the model. No generation source journals or checkpoints are
part of that cleanup.

The full 665-case teacher campaign has 551 training roots and 114 development
roots. Production feedback conversion selects only the 551 original training
executions under their original closed claims and actor receipts; development
records stay protected in the full original snapshot. Conversion and admission
are distinct from subsequent learner exposure.

The matched base arm completed at **0/20**, making the current development
comparison 0/20 base, 12/20 teacher SFT and 17/20 mixed continuation. The 20-world
development cohort remains a small selection cohort, not a confirmation result.
The exact results and execution bindings are retained under
`/home/werg/data/program-tuning-20261026/paired-ticket-sft-rl-base-native-host-v2-20261027-b0a4faf8b5e7`.
A separate activation operation reloads the actual native artifacts, runs a
fresh successful development call with a ready effective stack, and uses the
shared compare-and-swap activation mechanism. Publishing alone is insufficient.

Both production-feedback conversions passed: 1,075 rows per model, 5,528,930
LFM tokens and 5,568,087 Mellum tokens. The selected teacher source retains its
original actor, closed claims, invocation worlds and training splits. Preserve
the full original snapshot including its 114 protected development cases.
Registration custody packaging is running; these rows have no exposure credit
until each learner actually visits them.

The native readiness call passed and release
`b0a4faf8b5e7dba8467b65bd2b6218e3ad2d97de572a59be4ba308b50cd38c3d`
was activated through the shared compare-and-swap path at 06:25 Berlin on
2026-10-27. The pointer records the real backend acknowledgement and exercised
effective stack. Confirmation remains untouched.

The 551-case feedback package passed exact custody packaging and was registered
as `luna-program-training-worlds-native-graphs-20261027-v1`; its independent
private export is running. This does not yet constitute learner adoption.
Invoice's expanded-world continuation is running on 389 admitted training turns,
with reply-only supervision, three complete passes and batch size four within
a 16 GiB envelope. It continues the saved 24-update adapter/soft-state parent;
inventory follows serially from its saved 12-update parent. Their fresh capability
results and actual policy updates remain unmeasured.

The LFM append passed with 1,075 new rows, an unchanged existing prefix and held
source, and preserved capacity checks. Its exact controls have an independently
verified 10,372,894-byte private capsule. The earned full checkpoint proves the
prior correction already applied; the future stage keeps that history, retires
the consumed correction argument and names the actual corrected predecessors.
The current configuration adopted the append and qualified background snapshots
at a 600-second cadence. Exposure remains unclaimed until the next full-SFT or
eligible seed continuation actually visits these rows.

The shared append preparer now resolves the complete future recipe before
serializing seed roles. Foundation and training roles sharing one named binding
move together; separately named predecessors remain unchanged. All 20 append
tests pass for both ports, and all ten asynchronous checkpoint integrity tests
pass against the effective frozen LFM runtime. Enqueue remains distinct from a
committed checkpoint or independent backup. Larger Mellum snapshots need their
own memory assessment.

LFM completed the handoff into DPO and saved global step **39896**, then rejected
an old round-30 RL collection contract belonging to an earlier policy. The
supervisor preserved that journal, checkpoint and optimizer/RNG/exposure,
reserved `round-000030-attempt-1`, and restarted the actual controller. Fresh
episodes are being captured under that exact earned checkpoint. No new RL
optimizer update follows merely from collection being live.

Mellum's complete private-ready census found another 570 earlier rows absent
from its active source, in addition to the 1,075 new program rows. Its prepared
append therefore adds **1,645 rows**, reaching 1,483,153 active rows. Every
candidate ID was checked against the active correction/deferred ledger: none
overlaps it. Existing prefix, held separation and capacity provisions pass;
the new Mellum configuration still needs final consumed-correction handling,
private control export and adoption. These rows have no learner-exposure credit.

Validation: all 364 TypeScript files pass (2,305 tests), the completed Neuralese
CPU phase passed 5,149 tests with 14 skips, and all 25 GPU-marked tests pass.
The 2,776-pass root Python run had one concurrent-build source-verifier failure;
all 25 authoring-quality tests passed when rerun after compilation. High-priority
learner admission preempted the later GPU phase and an automatic restart repeated
CPU tests; the supervisor stopped that redundant rerun and ran every GPU test
separately. The observed completed CPU result and its lost-log limitation are
recorded explicitly in the operator receipt. Eight additional stream-integrity
tests cover the new private-download recovery helper.

The memory guard also preempted the program RL recovery uploader. The verified
archive survived and the exact-size private object exists; a protected resumed
worker is performing the required complete GET/SHA verification. Do not call
that RL capsule privately ready before its final receipt. Future Mellum exports
retain the same independently verified GET as a bounded local candidate when
the external drive has room. The local worker still validates full bytes and
checkpoint contents before promoting it; optional local write failure preserves
remote verification and the last recovery parent.
