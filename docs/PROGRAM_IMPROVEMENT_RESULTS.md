# Native program improvement: implementation and evidence

Implementation handoff, 2026-10-01. The main deliverable is an editable natlang improvement program using directory reducers. The implementation is substantially exercised, including real collection and student training. The small-student learning objective was **not achieved**. Curriculum coverage is also below its declared target; exhausted work is recorded explicitly.

## Authored program and language

The installed entry is [improveStep.nl](../applications/program-improver/improveStep.nl). At the time of this recorded study, its orchestration was three typed directory applications: `measureBaseline`, `runExperiment`, and `selectCandidate`. Their companion folders contained the single implementations of the relevant exact bookkeeping/selection helpers. The editing reducer belongs to `runExperiment/rewriteProgram.nl`. Transformation applications install their reducer there. The current implementation has since removed the mechanical baseline/selection reducers in favour of an authored exact lifecycle; see PROGRAM_IMPROVEMENT.md. There are no compatibility aliases or alternative scheduler modes.

The semantic program diagnoses, proposes, evaluates, accepts/rejects, updates its population, selects an incumbent and returns the selected source/state. The host supplies compiler, fresh-worker execution, independent cases, identities, journals and operational allocations. A selected frozen improver is ordinary source supplied for one invocation, rather than a special host optimizer endpoint.

`folder.iterateOn` privately branches source and jointly returns `{folder,state}`. Immutable snapshots have automatic revision identities; `folder.at(id)` retrieves same-lineage revisions. Proposals have authentic ownership and stale-base checks. Acceptance returns its installed snapshot. Selecting an immutable population candidate is a separate operation: accepting a non-incumbent parent's child does not make that child the incumbent.

The SDK checks selected identity, compilation/edit policy, exact iteration advancement and independent quality **before durable publication**. Invalid steps return the previous valid pair, including after restart. Source adoption checks the recorded base and independent result; multi-file adoption is recoverable and rollback restores the recorded base. The CLI provides init/run/step/resume/inspect/export/adopt/recover/rollback and distinct failure/interruption/ineligibility exits.

Termination uses finite core iteration and a strictly decreasing nonnegative integer measure or explicit finite workflow step bound. An independent semantic judge can stop an unproductive run early, but cannot replenish the numeric measure. There is no instruction-entry gas. Turing incompleteness is a property of the constrained execution profile and finite effect interface, not a claim about arbitrary host JavaScript or model intelligence.

Evaluator authority is granted to four exact paths: the root and its three evidence-owning reducers. Editing/diagnosis helpers and target workers have no evaluator. Evaluation levels are supplied from the top; locked test confirmation cannot be requested from an editing program. The exposed model interface is five tools with a compact programming prompt.

The former component optimizer now executes authored `componentSearchStep.nl`. The port retains seeded per-case frontier sampling, dependency groups, paired minibatches, three-way merging, guidance coverage, protected baseline/incumbent pruning, caching, journals and reports. Superseded proposer/scheduler implementation is removed.

## Semantic workflows

The implementation includes compiler-resolved instruction editing that preserves interpolation slots; per-obligation transformation evidence; eight reusable transformation reducers; implement/repair/simplify applications; independently admitted counterexamples with immutable suite versions and finite oracle capacity; complete-source portfolios with an authored deployment router and named fallback; and evaluation/adoption of frozen improver copies between runs.

Finite validation supports empirical claims, rather than universal equivalence. Unknown required checks stay unverified. An unchanged source cannot satisfy an obligation claiming a change. Counterexample generation cannot query protected cases or provide its own gold.

Real teacher trajectories exercised implementation, repair, clarification, specialization, extraction and simplification. Five initial transformation trajectories passed independent source/state replay. A subsequent simplification, using the decomposed root, retained quality 1 and reduced source from **142 to 84 bytes**; independent replay passed. The duplicated-guidance attempt failed with a source/state mismatch and was not admitted. Specialization here demonstrated a checked source/documentation transformation, not improved deployment-model performance.

The decomposed root repaired a separate synthetic sum-of-squares program, reaching quality 1 with 14 actual model calls, followed by independent replay. This is a development fixture, not a final-cohort result or a demonstration that the 350M student learned the interface.

The portfolio development fixture improved validation from 0.5 for a single specialist to 1.0 for its authored two-member router. Routing used ten model calls, while the standalone specialist used none. This demonstrates deployment composition on the declared finite cases, with additional cost; it does not establish a general or statistically confirmed gain.

## Migration, collection and training data

The canonical migration processed **118 files, 746,044 rows and 2,220 API edits**. Caller/builders now use native revisions, returned accepted snapshots, exact service scope syntax, derived seeding and finite iteration. Changed trace admission is cleared; translations cannot be admitted merely because the old trace passed. Source IR migration and verified interpreter-trajectory replacement remain different lanes.

The current native corpus is pinned by [current-manifest.json](../data/teacher/self-improvement/current-manifest.json). Its current artifact lane contains **225 independently admitted outer decision turns from 19 unique trajectories**. This includes 69 decisions from five independently reproduced original-failure loops, 57 transformation decisions, 13 decomposed-root decisions and ten subsequent simplification decisions. Complete loops may validly reject edits or retain unresolved baselines; loop admission does not imply target repair.

The selected discovery snapshot contains **9,796 artifacts and 58,507 incidents**. The original-source development probe evaluated 53 reconstructed targets: 29 baselines already passed, eight failures reproduced, and 16 probes failed operationally. Generated/mechanism fixtures are counted separately from historical failures. Five of the eight reproduced original failures yielded independently reproduced outer loops; three remained rejected. Reconstructed targets without sufficient independent split/oracle support remain excluded or exhausted, rather than receiving invented gold.

[completion/summary.json](../runs/self-improvement-data/folder-api-v1/completion/summary.json) pins discovery, links representatives and records every incident's disposition. It counts only trajectories represented by the explicit current corpus, excluding obsolete or weak historical exports. The 24-loop target has a **five-loop shortfall**. Linked representatives cover 3,347 incidents; 54,273 have allocation-exhausted dispositions, 687 infrastructure exclusions, 180 held-out exclusions and 20 independent-oracle requirements. These are accounting dispositions, not evidence that all failure mechanisms were repaired or mastered.

Primary-source acquisition obtained GEPA implementation/license material and SWE-bench development samples. They remain reference material; they were not turned into unsupported positive training answers. The registry in `training/api-migrations/native-program-improvement.json` records API changes, builders, migrations, cases and checks. Incremental ingestion and bounded recollection are runnable.

The final shared allocation records 64 collections, 12,474 reserved model calls, 4,779 reserved case executions, five training jobs, 112 updates and one final confirmation block. Ceilings are 120 / 12,480 / 4,800 / six / 1,200 / one. Reserved capacities are **not actual request counts or cash spend**. Six provider calls and 21 cases remain; they cannot fund another complete collection/reproduction pair. Unpriced provider cash spend is unknown; a zero cost-ledger entry does not establish zero cost.

## Executed student experiment

The student was `LiquidAI/LFM2.5-350M`, revision `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`, trained with LoRA rank 8, learning rate 0.0002, seed 0, 4,096-token examples and explicit overlength rejection. CPU training used a declared three-core allocation. All checkpoints and training outputs are retained under `runs/self-improvement-learning`.

Round 1 executed 16 updates. The initial round 2 also executed 16 updates; discovery of inadequate zero/empty boundary coverage led to stronger development fixtures, a genuinely corrected empty-reduction trajectory and a replacement round 2 with 16 updates. Two starting-checkpoint controls each executed 32 updates: local correction and extra outer data. The rejected initial second round remains historical evidence. Controls matched update allowance; different admitted examples and token exposure prevent a claim of identical effective training exposure. This small experiment did not establish preservation of general coding capability.

The single frozen confirmation evaluated five checkpoints on six independent families each:

| Checkpoint | Successful complete loop with correct held-out target | Structurally completed loops | Model calls |
|---|---:|---:|---:|
| Starting | 0/6 | 0/6 | 12 |
| Round 1 | 0/6 | 1/6 | 7 |
| Corrected round 2 | 0/6 | 0/6 | 18 |
| Local correction control | 0/6 | 0/6 | 61 |
| Extra outer data control | 0/6 | 0/6 | 43 |

There were no false promotions. The round-1 completed loop retained a target that failed held-out checks. The experiment therefore provides **no evidence of student learning benefit**. Six families are a small descriptive cohort, not adequate evidence for small effects.

Frozen-improver meta-search did not select an improved candidate. Its paired final confirmation compared the retained baseline against itself: effect 0, p-value 1, unsupported. It does not demonstrate that improving an improver is impossible; it records a failed bounded attempt.

The immutable final study is [confirmation/results.json](../runs/self-improvement-learning/confirmation/results.json). Confirmation is allocated idempotently and cannot be rerun with a revised candidate to seek significance.

## Refinements after confirmation

The three-reducer root and the stronger prepublication checks were refined after the frozen study. Training and serving now share the same template normalization, preserving prior reasoning when the tokenizer exposes that policy. Renderer version `transformers-chat-template/3` fingerprints the change. Student serving now honors the SDK's 2,048-token turn allowance rather than silently capping every request at 512. None of these changes are claimed as final-cohort-validated learning gains.

Actual offline rendering of the 215-turn corpus before the final simplification addition had no template-prefix rejections. Only 74 turns fit 4,096 tokens; 141 exceeded that training allowance. This confirms that shortening authored decisions and creating faithful focused continuations remain necessary. The final 225-turn corpus rendered successfully with zero template rejections and without truncation; long examples must still be rejected by the declared training recipe.

## Verification and practical limits

The complete Node suite passed **698 tests**. All **22 conformance cases**, application builds and browser type checks passed. After the final publication-guard refinement, all 23 focused source/workflow tests passed; 26 Python rendering/pipeline tests also passed. Node/browser builds and package dry-run succeeded. Actual CLI export/adopt/recover/rollback restored the recorded source in a development checkout. `git diff --check` passed.

The runtime/API and workflows are implemented and exercised. The empirical learning objective, full required curriculum matrix, broad real-world source conversion and intended 24-loop coverage are not achieved. These are explicit bounded shortfalls, not compatibility modes, deferred releases or evidence of success. Further learning experiments require a new declared study; the reserved confirmation cohort remains closed.
