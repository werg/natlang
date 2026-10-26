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

Exported cases are under `runs/tune-cases/program-tuning-20261026/`. Only the 364 training and 114 development roots are exported. The requested teacher campaign is `runs/luna-program-tuning-six-20261026-v1`, using the existing six-worker Luna capacity after the preceding campaign's actual workers drain. Its reviewed plan, inventory, frozen runtime and per-case results remain authoritative; a launch request is not evidence of active workers or usable trajectories. Current receipts live with the operational data, rather than being inferred from this document.

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
