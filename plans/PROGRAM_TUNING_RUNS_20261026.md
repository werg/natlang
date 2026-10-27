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
