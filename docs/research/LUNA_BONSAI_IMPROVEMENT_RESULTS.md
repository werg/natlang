# Luna / Bonsai native improvement observations

The optimizer is `openai-codex/gpt-6-luna`. The target interpreter is the existing server's `Ternary-Bonsai-2-27B-PTQ1_0.gguf` on localhost:8081. These are source-editing experiments, not student weight training. The shared server was not reconfigured.

The original three tasks were historical generated-data failures, not established failures of this Bonsai model. Each initially used two training, two validation and two reserved test cases with the original group assignments and gold. The objective was passing quality followed by fewer measured target model requests. Test outputs were unavailable to the optimizer.

| Development task | Observed behavior | Optimizer limitation |
| --- | --- | --- |
| Ticket deadlines | All four development cases passed; validation took four requests. An initial keyword calculation sometimes needed a semantic correction. | Two experiments ultimately produced unchanged source despite claimed edits. |
| Q3 late-shipment supplier | Passing validation was recorded, but the search exhausted its 30-minute allowance. The incumbent was retained. | The rewrite invented delivery-date fields absent from the status-based service. A seed-specific apparent request gain disappeared under selection measurement. |
| Recommendation counts | All four development cases passed, using 35 target requests; validation used 18. | Luna incorrectly labeled passing training outcomes as a fixture failure and skipped the cost opportunity. |

The earliest attempt also encountered an invalid independent shipment fixture, a hidden 24-request optimizer ceiling and target timeouts that hid useful feedback. Those attempts remain recorded; none demonstrated improvement.

## Implemented refinements

- Expose independently measured `modelCalls` in the injected evaluator declaration, and retain original counts through immutable execution caching. Remove the hidden optimizer request ceiling in favor of the declared shared allowance.
- Include bounded training action traces and public service declarations. Distinguish fixture compilation errors, target failures and target timeouts. Validation/test outputs remain sealed.
- Ground hypotheses in current target execution and real service fields. Preserve semantic judgment rather than replacing it with keyword matching.
- Pass evidence directly through the native experiment's exact request helper. Support TypeScript optional-property handling in eval scope, while keeping published portable-result validation strict.
- Derive the editor's changed paths from its actual folder diff. Reject no-op experiments instead of trusting claimed edits.
- Identify fixture, quality, efficiency and source-size opportunities with a small exact helper. Correct answers can support an efficiency experiment; unknown costs cannot establish one. Luna still chooses the semantic hypothesis and edit.
- Pin the evaluation seed within an improvement invocation, including publication checks.
- Stop a sequential action batch after a failed computation so a following finish cannot publish an older staged result. Successful `eval` plus `return_result` batching remains supported.
- Ingest native result files and rejected decisions into the failure registry. Keep held-out confirmation incidents evaluation-only; an accepted population child that was not selected is not automatically a failure.

## Earlier grounded follow-up

The follow-up selected deadline execution cost using measured **training** executions. It used one training and one validation case, retained the original two reserved test cases, and shared a single 160-request, 20-rollout, two-proposal, 20-minute development envelope across its corrections.

| Attempt | Luna requests | Bonsai requests | Result |
| --- | ---: | ---: | --- |
| Grounded evidence | 23 | 6 | Reconstructed optional evidence caused a nonportable handoff; no candidate evaluated. |
| Corrected handoff | 10 | 5 | Handoff obstacle removed; Luna still incorrectly required a wrong answer to pursue efficiency. |
| Explicit opportunity helper | 25 | 8 | Luna produced a coherent instruction edit, compiled it and evaluated training/validation. Rejected: candidate and parent both took two validation requests. |

The last candidate explicitly requested semantic urgency classification and discouraged keyword matching. It also requested replying `done` after staging the answer. That still requires another model request: neither an inspected final expression nor a later `done` response is a one-request completion. The supported one-response mechanism is a sequential `eval` with a typed return followed by `return_result` in that same response. The measured rejection was correct; this study has not demonstrated an optimization gain.

The selected source remained the baseline. Its source and runtime were frozen before a single paired final confirmation on the two reserved deadline cases. Baseline and selected source both scored **1/2**: one case passed and one hit the five-minute target timeout. Effect was zero, p-value 1, and supported improvement false. Identical sources reused the same actual executions, so this is not evidence about two different programs. The run used five target requests, including one aborted request with unknown token usage. Results are in `runs/luna-bonsai-improvement-20261001/grounded-objective/confirmation-results.json`. No original-source candidate was revised after that confirmation, and its failed held-out case remains evaluation-only.

The final development hypothesis also confused a final expression with completion. The default experiment instructions now spell out the three completion semantics above; this clarification comes from the development trace, not from the held-out timeout. It applies to future optimizer invocations and was not part of the frozen comparison.

## Earlier data and verification

`prepare-observed-study.mjs` selects current-student opportunities from completed training execution records. `build-observed-decision-cases.mjs` generated two compiling native repair cases from the actual cost-opportunity and fixture-diagnosis mistakes. Their faulty starting instructions are explicitly reconstructed, and their boundary gold comes from an independent declared decision table. They are marked for target reproduction and verified collection, **not positive SFT**. Their observed cost examples use actual correct training executions, never validation aggregates.

Artifacts, exchanges, protocols and journals are under `runs/luna-bonsai-improvement-20261001`. The incremental incident registry includes 12 recorded development incidents and one evaluation-only confirmation incident. API migration and recollection provisions are in `training/api-migrations/native-program-improvement.json`.

Verification: 708 tests passed before the final helper/ingestion refinements; the additional native-helper, selection, ingestion and curriculum checks passed separately. All 22 conformance cases passed, browser types and bundle passed, and the real Chromium smoke passed after regenerating its stale adaptation fixture against the current build.


## Structural optimization and shared language refinements

The later frozen structural pilot demonstrates a real development cost improvement. It uses semantic review recommendation/counting with independently declared labels and includes contrast, negation and sarcasm. The training-only hidden reference proved headroom: correct baseline six requests versus correct reference one. Luna never received that reference source.

| Development arm | Luna requests | New Bonsai requests | Correct validation requests | Experiments |
| --- | ---: | ---: | --- | --- |
| Native finite iteration | 31 | 32 | 7 → 2 | First rejected, second accepted |
| Direct rewrite control | 5 | 4 | 7 → 2 | One accepted rewrite |
| Prepared-context iteration | 60 | 13 | 7 → 3 → 2 | Two accepted, third rejected |

The direct arm and prepared-context arm reused exact baseline executions with their original costs retained. Prepared context alone did not reduce optimizer overhead: Luna still repeatedly inspected and paged evidence, and continuation also ran an additional experiment. These counts compare whole searches with different trial counts, not isolated treatment effects. Artifacts are in `runs/luna-bonsai-native-compact-20261001` and `runs/luna-bonsai-prepared-optimizer-20261001`. The prepared-context ablation used closed development data only.

The structural pilot froze sources and runtime before its single final two-case confirmation. Native selection was correct on 2/2 with four total requests; its baseline was correct on 1/2 with thirteen requests and one timeout. Quality effect was 0.5, p-value 0.5, supported improvement false. The direct source was correct on 2/2 with four requests; its separately executed baseline was correct on 2/2 with twelve. Quality effect was zero, p-value one. These tiny comparisons show observed operational/request savings, not a supported general learning gain. The original harness independently reran each arm's baseline; future runs share exact baseline confirmation executions. The completed cohort stays closed and its outputs do not seed new training cases.

### Language changes prompted by observed overhead

The runtime now preserves late fields in bounded structured previews, shows precise paths for clipped values, and leaves full values intact. Small values remain complete. Shared prompts explicitly allow direct declared strings/JSON for semantic answers and reserve eval for computation/effects. They recommend judging visible inputs in the current call and delegating distinct subproblems. The planner now returns a primitive hypothesis string and gets a compact brief directly in its opening; the editor gets complete source and measured evidence. Exact authored helpers retain measurement, selection and folder effects. No compatibility wrappers remain.

A general Bonsai language probe used identical source, seed and two train inputs containing large irrelevant logs followed by the actual review. Both the preceding renderer and new renderer were correct on 2/2. Requests fell **4 → 2**. No optimizer participated, and reserved validation/test inputs remained unopened. See `runs/bonsai-input-overview-20261001`.

A Luna-only drafting diagnostic used pinned development observations: the preceding card approach needed seventeen requests; the primitive-hypothesis/shared-guidance version needed **six**, producing a coherent batching hypothesis and a valid compiled source edit. This is a valid-draft result, not measured student improvement. The two earlier card-diagnostic attempts failed before model execution and remain recorded. See `runs/luna-bonsai-card-draft-20261001-v3` and `runs/luna-bonsai-semantic-plan-20261001`.

Training traces now pair each action with its resulting observation, including terminal actions, using request-event boundaries instead of timestamps. This corrects misleading diagnostic alignment in preceding runtimes; frozen historical traces are preserved rather than rewritten. Source evaluation identities changed accordingly.

### Current training provisions

Thirteen obsolete optimizer templates were directly migrated to the current source/signature with stale admission revoked. Sixty-nine earlier generic failure rows were audited and retained with their existing recollection status. The read-only audit of eight canonical self-improvement corpora found no obsolete planner signature, and found widespread cutoff exposure; cutoff mentions alone are not failure labels.

New generated cases include two structural loops reconstructed from measured literal edits, one general large-input case, and three general semantic-result families (boolean, enum/string, boolean array). Fresh independent boundary examples are used; completed confirmation inputs are excluded. These are executable curriculum inputs requiring target reproduction, native collection and independent replay, **not admitted positive SFT**. Builders, provenance, migrations and current verification are tracked in `training/api-migrations/native-program-improvement.json`.


Current verification: **723/723 tests passed**, all 22 conformance cases passed, application build, browser bundle/types and actual Chromium smoke passed. The full successful run used test concurrency four; an earlier unconstrained run hit a worker cold-start timing failure before model entry, and the unchanged timeout test passed separately. A focused migration check also verifies replacement of the wrapped planner result and idempotence. `git diff --check` passed. The current incident registry contains two development decision-repair incidents and one evaluation-only confirmation incident.
