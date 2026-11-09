# Independent review of the natlang RL regression

**Date:** 2026-11-09  
**Repository:** `werg/natlang`  
**Purpose:** independently audit the regression diagnosis, the implementation it relies on, and the proposed experiments; identify concrete defects and overlooked explanations before interpreting further training.

## Executive judgment

There are multiple reproducible implementation defects, but the available evidence does **not** establish one unique cause of the reported dev16 decline. The most important missed issue is an integration error between the policy objective and gradient accumulation: the objective averages over sampled actions, while the updater combines microbatches according to graph counts. The same trajectories can produce opposite gradients solely because they are packed differently. This should be checked against the actual accumulation schedule before interpreting apparent wrong-direction learning.

Other consequential findings are: a default singleton teacher receives no task-outcome advantage; model-authored text can turn completed policy failures into unscored infrastructure errors; interrupted collections can be sealed complete; wholly rejected updates can still move under saved optimizer state; float16 top-k receipts can create score-centering drift even with identical actor and learner; and evaluation identity omits part of the execution runtime. These mechanisms are independent and can interact. Their existence is established more strongly than their incidence in the unavailable production artifacts.

The diagnosis was updated during this review. Its completed direction test is more informative than the initial ten-episode sample, and its reported gradient probes make privileged-twin distillation a reasonable suspect. However, several conclusions still go beyond the evidence:

- A small **scalar value** of a score-centered surrogate does not mean a small gradient. At matched policies it can be exactly zero while its gradient is the ordinary policy score.
- Outcome gradients reported as 35–60% of the KL gradient, approximately orthogonal to it, are a material component. They do not support saying the only directed component is KL.
- The source saves per-episode advantages in a hash-bound objective artifact. They were not necessarily lost merely because the summary receipt does not embed them.
- `k3` is a detached mismatch diagnostic and acceptance gate in this path, not an entropy loss.
- The new w5 run changes `beta_s` and long-episode handling, and the final update adds budget-failure prefix recovery. Improvement or deterioration will not identify privileged-context distillation as the sole cause.
- A forced LR-zero update is not guaranteed to preserve BF16 weights, and an ordinary same-checkpoint evaluation request can silently reuse old evidence.

**Recommended immediate direction:** recover and verify the exact objective and update partitions; reproduce the normalization invariant; audit reward eligibility and teacher group sizes; separate the frozen objective gradient from the optimizer transformation; and use genuinely fresh, fully matched evaluations. Do not decide the cause from the scalar-loss ratio or from w5 alone.

Sources for the diagnosis and its late update: [initial brief](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/plans/neuralese/RL_REGRESSION_DIAGNOSIS_20261109.md), [final brief reviewed](https://github.com/werg/natlang/blob/5a72cb148a49214b7520438cd3854b5b70f0d482/plans/neuralese/RL_REGRESSION_DIAGNOSIS_20261109.md). Detailed code evidence and counterexamples follow.

## 1. Scope, versions, and strength of evidence

### Version boundary

| Item | Exact version | Consequence |
|---|---|---|
| Main source snapshot used for the audit | `e1d72029a9f5c7801d8698c3dacda1e60060d5ff` | Source links below are pinned here unless stated otherwise. |
| Frozen w4 runtime named by the diagnosis | `7fc570ec987645556176591b25e8d64142e0587e` | Repository comparison shows the objective, optimizer, collection, actor, and evaluation files discussed here were unchanged between this version and the reviewed main snapshot. This verifies source ancestry, not the bytes actually deployed on remote machines. |
| Diagnosis refreshed during review | `abdb85d0d14b09dfc9bc410dd81eb07d9ed8c45d` | Two further commits changed only the diagnosis Markdown. The source findings remain current at this checkpoint. |
| Initial w5 runtime reported by the refreshed diagnosis | `2f027b77f` | The brief says w5 disables `beta_s`, starts at step 92, and includes the newer long-episode truncation. |
| Final source/diagnosis check, 14:28 UTC | `5a72cb148a49214b7520438cd3854b5b70f0d482` | A further patch changes budget-prefix recovery in `task_rollouts.py` and its tests; the brief reports w5 restarted onto v69 during collection. See the final-change note below. Actual process state remains unverified. |

The frozen-to-initial-main comparison changes four paths: the fallow register, the diagnosis, `tests/neuralese/test_native_rl_production.py`, and `rl/native_stage.py`. In particular, the newer truncation and group-balancing changes cannot explain the already executed w4 step-97→102 update; they **can** affect w5. [Frozen-to-reviewed comparison](https://github.com/werg/natlang/compare/7fc570ec987645556176591b25e8d64142e0587e...e1d72029a9f5c7801d8698c3dacda1e60060d5ff). [Late document-only comparison](https://github.com/werg/natlang/compare/e1d72029a9f5c7801d8698c3dacda1e60060d5ff...abdb85d0d14b09dfc9bc410dd81eb07d9ed8c45d).

### Final-change note: budget-prefix recovery landed during review

The last check found two further commits: budget-prefix recovery (`7be84ffbd`) and a diagnosis update. The patch reconstructs sampled closed-set decision turns from saved decision evidence and permits a verified chronological prefix before an omitted oversized action record. It also prunes the graph for that prefix. This addresses two concrete reasons budget-failed episodes previously lost replay eligibility; it is not a blanket conversion of every unresolved outcome to reward zero. The four added test functions were executed directly with their extracted fixtures against the actual latest Python module and passed; no full suite or GPU run was performed. A remaining resume-format gap was also reproduced (F13). [Final change comparison](https://github.com/werg/natlang/compare/abdb85d0d14b09dfc9bc410dd81eb07d9ed8c45d...5a72cb148a49214b7520438cd3854b5b70f0d482). [New recovery code](https://github.com/werg/natlang/blob/5a72cb148a49214b7520438cd3854b5b70f0d482/training/neuralese/natlang_neuralese/rl/task_rollouts.py#L188-L481).

The latest brief now reports 14 affected budget cases, describing 8 already captured, 4 involving decision frames, and 2 involving an omitted action record; these numbers were not remeasured here. It reports w5 restarted onto v69 with this fix. **W5 therefore changes at least three relevant factors versus w4: student KL, long-episode admission/truncation, and budget-failure capture/eligibility.** The report's historical source links remain pinned to the audited snapshot so their line numbers remain exact. The new patch does not change the classifier, collection-completion logic, objective, optimizer, evaluation, or SGLang seed findings, and it retains the role/endpoint summary behavior discussed in F11.

### What was inspected and executed

The review inventoried the repository tree and materialized 412 selected source/support files, with additional targeted tests and TypeScript/dependency sources. It traced the complete relevant data flow: task selection and collection; terminal classification and reward construction; sampled-action capture and replay; JOLT twins and score centering; graph admission, prefix sharing and microbatch accumulation; optimizer state and checkpoint publication; actor provenance; and evaluation, caching, and promotion. Three parallel reviews covered objective/replay, optimizer/checkpointing, and collection/evaluation, followed by cross-review of the strongest findings. This is a targeted dependency audit, not a claim that every line of every repository file was read.

Executable checks used exact AST-extracted Python functions or expressions where possible, plus explicitly identified NumPy/IEEE calculations for numerical behavior. All executed assertions passed. Neither available Python runtime had PyTorch, so **no full PyTorch suite, LFM3 forward/backward, GPU replay, production training, or checkpoint evaluation was run**. Existing GPU/model suites were inspected rather than executed; four new pure-Python budget-recovery test functions were subsequently run directly against the final module, as described in F13 and Appendix A. No production code, GitHub branch, remote job, or checkpoint was changed.

The actual `w4/config.json`, original objective and rollout files, optimizer tensors, direction-test scripts/results, random-control script, and Pop launch/detail receipts were absent from the accessible repository snapshot. Reported success counts and diagnostic measurements are therefore claims from the diagnosis, not independently remeasured facts.

### Reading the priorities

**P1:** fix or explicitly bound exposure before relying on the affected training/measurement. **P2:** real correctness or observability defect whose contribution needs quantification. A priority is not a claim of proven regression causality.

| ID | Priority | Finding | Established / still conditional |
|---|---|---|---|
| F1 | P1 | Graph-weighted accumulation of action-mean losses can reverse the gradient | Exact source-expression counterexample; actual multi-microbatch exposure needs configuration. |
| F2 | P1 | Default singleton teacher has zero outcome advantage | Source/default confirmed; actual teacher-attempt count unavailable. |
| F3 | P1 | Model-controlled strings and inconsistent terminal statuses censor failures | Exact classifier reproductions; actual affected rows unknown. |
| F4 | P1 | Incomplete collection can be sealed and never resumed | Source trace and dispatcher reproduction; actual journal incidence unknown. |
| F5 | P1 | An update without accepted policy signal can still mutate weights/state | Optimizer/trainer behavior confirmed; no claim this occurred in w4. |
| F6 | P2 | Float16 top-k serialization can break score-centering consistency | Valid numerical counterexample; production frequency and scale unknown. |
| F7 | P2 | Episode rejection occurs after advantage centering | Source confirmed; refreshed brief says none of its 78 episodes were rejected, reducing historical relevance if verified. |
| F8 | P1 | Evaluation matching omits TS/runtime/image closure | Comparator reproduction; known Python-hash mismatches are correctly rejected. |
| F9 | P2 | Same-checkpoint repeat requests can be consumed without execution | Exact request-discovery reproduction. |
| F10 | P2 | SGLang advertises a seed it does not forward | Exact request-construction reproduction and pinned fork inspection. |
| F11 | P2 | Score totals and endpoint labels can misdescribe the training population | Source confirmed; actual brief tables may already have been manually filtered. |
| F12 | P1 experiment / P2 code | New truncation changes the estimator and can miss valid cuts | Applies to new admission/w5, not historical w4; search defect reproduced. |
| F13 | P1 | Budget-prefix recovery still omits resumed journal turns | Latest patch passes its four targeted tests; resumed-action and resumed-decision counterexamples still lose eligibility. |

## 2. Confirmed implementation findings

### F1. Inconsistent accumulation normalization can reverse the direction

`NativePolicyObjective.__call__` sums sampled-token and sampled-decision losses and divides by the number of accepted actions in the microbatch. The shared updater then weights each policy microbatch by its **graph count**, not its action count. `packed_graph_loss` returns the policy scalar unchanged, so there is no compensating conversion between these layers. [rl/native_replay.py:1336–1345](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L1336-L1345) [train/native_state_sft.py:497–507](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/native_state_sft.py#L497-L507) [train/native_state_sft.py:550–565](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/native_state_sft.py#L550-L565) [train/native_state_sft.py:850–859](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/native_state_sft.py#L850-L859)

For microbatch `m` with `G_m` graphs and `T_m` accepted actions, the implemented complete-update loss is:

\[
L=\sum_m \frac{G_m}{\sum_jG_j}\frac{\sum_{a\in m}\ell_a}{T_m}.
\]

This is neither a global action mean nor a mean of individually normalized graphs. A global action mean needs weight `T_m / sum(T)`. A graph mean needs each graph normalized before its contribution is combined. The current combination changes relative weights whenever unequal-length graphs are repartitioned.

**Executed counterexample.** One graph has one action with scalar loss derivative `+1`; another has 100 actions, each with derivative `−0.02`. The reproduction evaluated the actual source normalization expressions:

| Same complete update, different microbatch partition | Aggregate derivative |
|---|---:|
| Both graphs together | `(1 − 2) / 101 = −0.009900990099` |
| Two one-graph microbatches | `0.5 × 1 + 0.5 × (−0.02) = +0.49` |

The direction reverses without changing actions, rewards, token gradients, model state, or optimizer boundary. This is a possible explanation for apparent wrong-direction learning even if every local policy-loss sign is correct.

**Live qualification.** Check `accumulate_batches` and the actual `(G_m,T_m)` for all five updates. If each complete update has exactly one microbatch, this accumulation mismatch is not exercised there. The defect still matters for configurations and memory-driven partition changes that do use multiple microbatches. Rejected episodes also make graph counts a poor proxy for accepted objective mass.

**Why existing tests miss it.** The JOLT accumulation reference averages two separately normalized episode losses, validating the split calculation rather than comparing it with the combined action-normalized objective. The ordinary SFT accumulation test exercises a different normalization path. [tests/neuralese/test_native_update_accumulation.py:231–239](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/tests/neuralese/test_native_update_accumulation.py#L231-L239)

**Repair and decisive check.** Declare one normalization law. For a global action mean, use an unnormalized numerator per microbatch and a fixed denominator over the **accepted actions of the complete update**. Establish acceptance before in-backward mutation, or use a scheme that permits correct rescaling. If a graph mean is intended, compute it explicitly. Compare all gradients for several partitions of the identical admitted update, first in FP32 and then in the deployed numerical path; separately compare one optimizer step with identical history and RNG. Do not change the complete update boundary during this test.

### F2. The default singleton teacher has no direct task-outcome gradient

The production native JOLT path defaults `jolt_teacher_attempts` to one. The crisp planner preserves that count. Advantages are grouped by `(task_id, group, role)`, so teacher attempts are centered separately from students; a one-member JOLT group receives zero centered outcome advantage. The teacher KL term remains active. [rl/production.py:1615–1618](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/production.py#L1615-L1618) [rl/jolt.py:583–595](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/jolt.py#L583-L595) [rl/native_replay.py:1631–1646](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L1631-L1646) [rl/native_replay.py:1575–1577](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L1575-L1577) [rl/jolt.py:108–119](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/jolt.py#L108-L119)

The actual source helper returns `[0]` for either a successful or unsuccessful singleton teacher, and `[-0.5,+0.5]` for a two-attempt teacher group with outcomes `[0,1]`. The generic helper can accept an external bonus, but the production scorer explicitly awards the Neuralese bonus only to **student** rows; ordinary singleton teacher costs/report credits do not create spread. Thus, for ordinary production teacher groups under this default, the outcome term is exactly zero regardless of correctness. Separate graded memetic modes should not be conflated with this task-outcome path. [rl/task_rollouts.py:588–592](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/task_rollouts.py#L588-L592) [rl/native_replay.py:1536–1589](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L1536-L1589)

The teacher's own sampled correctness cannot distinguish its teacher-outcome update. Its direct active term regularizes it toward the student, while the student can be distilled toward the privileged distribution. Any correctness improvement of the teacher must come through other shared-parameter terms rather than its own centered outcome signal. Even with multiple teacher attempts, all-equal outcomes produce zero binary advantage while KL remains active.

**Qualification:** the diagnosis does not disclose `w4`'s actual teacher-attempt count. Verify `jolt_teacher_attempts`, requested and scored role-group sizes, and the saved advantages before declaring this the live behavior. A configured count above one can still collapse to a scored singleton through missing outcomes.

**Repair:** use enough scored teacher siblings for the intended empirical baseline, with an explicit policy for censoring, or use a justified baseline independent of the sample's own reward. Simply deleting the singleton special case while subtracting its own mean still gives zero. Log teacher outcome and teacher KL gradient norms separately. The existing two-teacher fixture does not test the production default. [tests/neuralese/test_rl_jolt.py:170–208](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/tests/neuralese/test_rl_jolt.py#L170-L208)

### F3. Failure eligibility depends on model-authored wording

#### F3a. Text matching can override a definitive wrong answer

`classify_case` searches terminal text for budget phrases, then stringifies action/invocation ledgers, failures, and rejection evidence and searches for transport phrases. These heuristics run before the completed-wrong-answer branch. Raw action arguments and model terminal reasons are included. Thus text such as `HTTP 500` can be treated as infrastructure evidence even when it is merely a string in model-authored code. [eval/task_packet.py:192–240](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/eval/task_packet.py#L192-L240)

An exact-source reproduction preserved `status=done`, `accepted=False`, a failed oracle check, and a completed finish record. Only a literal in action code changed:

| Action-code literal | Outcome assigned |
|---|---|
| `ordinary wrong answer` | `semantic_mismatch` |
| `HTTP 500` | `infrastructure_error` |
| `connection refused` | `infrastructure_error` |

A model terminal reason containing `wall budget exhausted` similarly changed a completed wrong stop to `unresolved`. The runtime puts the model's reason into terminal text. These labels directly matter because infrastructure/unresolved outcomes are unscored, while a semantic failure receives reward zero. [ts-host/src/native/runtime.ts:1739–1749](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/ts-host/src/native/runtime.ts#L1739-L1749) [rl/task_rollouts.py:110–118](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/task_rollouts.py#L110-L118)

This is content-dependent censoring of policy failures. It need not be intentional reward exploitation: quoted errors, task data, code, or recovered historical failures can trigger it naturally. The reported infrastructure count cannot be assumed to consist exclusively of the known Pop OOM without inspecting typed events.

#### F3b. Equivalent `failed` and `blocked` terminals are treated differently

The TypeScript runtime maps both supported `return_result(status="blocked")` and `status="failed"` calls to a blocked terminal. With scope failures present, the scorer recognizes only the literal `blocked` argument as an explicit scored unsuccessful return. Changing only that argument to `failed` makes the same otherwise completed failure `unresolved`. This asymmetry was reproduced. [ts-host/src/native/runtime.ts:1739–1749](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/ts-host/src/native/runtime.ts#L1739-L1749) [eval/task_packet.py:261–273](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/eval/task_packet.py#L261-L273)

**Repair:** base infrastructure exclusions on trusted host/transport events and whether they prevented a definitive outcome. Use typed completed-terminal semantics consistently across supported status spellings. Add tests for completed oracle rejections containing transport/budget text and for both explicit failure statuses, preserving genuine recovered/nested interruption cases.

**Do not fix this by blindly assigning zero to every unresolved row.** Never-attempted or genuinely interrupted rows may lack a definitive model outcome. Separately, missing capture can prevent training even when a failure is established at task level. The correct relabeling and capture requirements are in section 4.6.

### F4. A collection can be marked complete while expected keys remain unfinished

The generation dispatcher stops a worker when its collector exits without a finish record. The code intends the unfinished entry to remain pending for the next start. But the production caller sets `collection_complete` solely from the absence of a cooperative stop request, and a later call returns a saved score marked complete immediately. An ordinary subprocess failure does not necessarily set that stop flag. [rl/collection_generations.py:125–175](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/collection_generations.py#L125-L175) [rl/production.py:917–920](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/production.py#L917-L920) [rl/production.py:1073–1108](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/production.py#L1073-L1108) [rl/production.py:1170–1173](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/production.py#L1170-L1173)

Missing/unstarted keys receive unresolved score rows; a scored subset can still permit training. There is no intervening expected-finish-set check. Graph/block manifests validate the data present, not that every planned key obtained a terminal record. This can seal incomplete student collection and also fix a teacher plan based on the incomplete sample. [rl/task_rollouts.py:553–564](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/task_rollouts.py#L553-L564) [rl/production.py:1969–1984](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/production.py#L1969-L1984)

**Executed dispatcher fixture:** two keys, one slot. If the first collector exits without a finish, only key `a` is attempted and both `a,b` remain unfinished. The function returns normally. Its `pending` field is the initial pending count: it returns `2` both for this failure and for a successful run that finishes both keys. That field cannot be used as a remaining-work count.

**Repair:** require every expected key to have a terminal journal disposition, or an explicit declared exclusion, before sealing completion. Preserve interrupted generations for resume and distinguish initial, attempted, finished, interrupted, and never-attempted counts. Use terminal evidence rather than process exit code, since valid terminal model failures can exit nonzero. For w4, inspect `expected_keys − finish_keys` against every saved `collection_complete:true` report.

### F5. Rejected or inactive policy updates can still change the model

If every microbatch in a complete update is rejected by the engine guard, the objective returns differentiable zeros through `hidden.float().sum()*0`. The shared updater still runs the final backward, calls the optimizer, and advances update/exposure accounting. With saved momentum or moments, a zero gradient tensor is not a no-op. [rl/native_replay.py:1283–1289](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L1283-L1289) [rl/native_replay.py:1336–1345](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L1336-L1345) [train/native_state_sft.py:538–594](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/native_state_sft.py#L538-L594) [train/native_state_sft.py:4514–4547](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/native_state_sft.py#L4514-L4547)

For a dense Muon matrix with current gradient zero and momentum coefficient `0.95`, the new momentum is `0.95*m_old`; Nesterov produces `0.9025*m_old` before normalization. Muon's normalization can preserve a substantial direction despite the zero current loss. Weight decay and Adam's saved moments are additional mutation paths. The idle-gradient handling for certain higher-rank tensors does not make dense matrices skip. [train/optim.py:639–682](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/optim.py#L639-L682) [train/optim.py:284–318](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/optim.py#L284-L318)

A NumPy/IEEE reproduction of the deployed equations with a 16×16 identity momentum, zero current gradients and weights, and LR `1.56e-6` changes 16 coordinates, with weight-delta L2 about `3.4273e-6`. This is an equation-level demonstration, not a production tensor measurement.

**Historical qualification:** the refreshed direction-test report says its 78 episodes were not guard-rejected. If confirmed for every actual optimizer update, this specific rejection mechanism did not cause those updates. Existing accumulation tests do document an earlier all-rejected microbatch, but that is not proof of an all-rejected complete w4 update. [tests/neuralese/test_native_update_accumulation.py:458–476](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/tests/neuralese/test_native_update_accumulation.py#L458-L476)

**Additional w5 exposure:** `skip_zero_signal_jolt` removes zero-advantage rows only when there is no privileged twin; it does not inspect the effective role coefficient. With `beta_s=0`, a zero-advantage student can retain a twin yet have neither an active outcome term nor weighted KL. Such rows remain admitted. If a complete update has no other active decision/writer/objective term, saved optimizer state can still move it even though the engine accepted actions. This needs an effective-loss check, not just an accepted-token count. Counting zero-advantage samples in a declared global mean can be intentional; their presence in a denominator is not by itself a separate proof of a normalization error. [rl/native_stage.py:399–420](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_stage.py#L399-L420) [rl/jolt.py:108–120](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/jolt.py#L108-L120)

**Repair:** define and enforce no-signal update semantics before any in-backward mutation. For a complete update with all actions rejected, or no intended active component under the configured zero-signal policy, do not advance optimizer state, RNG, update counters, or training exposure. Do not skip an empty **final microbatch** when earlier valid gradients must still be applied. Do not equate a numerically zero gradient with absence of an intended objective: legitimate momentum dynamics and cancellation need separate treatment.

### F6. Float16 top-k receipts can invent a negative probability tail

Sampler top-k log-probabilities are serialized as float16. Decoding checks shapes, IDs, ordering, and individual bounds, but not total probability mass. Score centering exponentiates the decoded values and reconstructs actor tail mass as `1 − sum(q_head)`, floors it, then computes the tail correction. A small residual is therefore obtained by subtracting a lossy near-one sum. [common/sampling.py:57–113](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/common/sampling.py#L57-L113) [rl/policy_math.py:151–174](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/policy_math.py#L151-L174)

**Executed numerical counterexample:** actor and learner are truly identical. There are 128 head tokens totaling `0.9999` probability and one tail token of probability `0.0001`. The actual decoder accepts the quantized receipt:

| Quantity | Value |
|---|---:|
| Stored head log-probability | `−4.8515625` |
| Decoded head mass | `1.0004680156707764` |
| Reconstructed actor tail | `−0.0004680156707763672` |
| True tail mass | `0.0001` |
| TIS tail coefficient `alpha` | about `0.02`, instead of `1` |
| Expected score logit-gradient error, L2 | about `9.8372e−5` |

The sampled-token log-probability can remain exact, so the sampled-logp engine guard sees zero discrepancy. A literal reference implementation using the same lossy receipt can also agree with production while both use an invalid actor normalizer. [rl/native_replay.py:103–118](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L103-L118)

This proves a numerical consistency defect, not a dominant regression mechanism. The relative tail error is large; the demonstrated absolute gradient error is about `1e-4`. Measure actual frequency, tail mass, advantage signs, and gradient contributions before assigning causal weight.

**Repair:** carry a sufficiently precise sampler tail/head mass with a bound receipt schema, or increase probability precision and validate residuals against the original sampler distribution. Computing FP64 after float16 decoding cannot recover discarded information; clamping the sum to one does not recover the true tail. Qualify matched-policy zero-drift behavior using uncompressed ground truth.

**Observability gap:** JOLT discards the score-centering diagnostics returned with the action loss. Surface head-mass excess, tail mass, `rho`, `alpha`, sampled importance weights and clipping by role/engine. Scalar component loss and mean KL do not expose this defect. [rl/jolt.py:455–457](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/jolt.py#L455-L457)

### F7. Engine rejection is a second selection step after centering

Advantages are fixed over scored task/group/role members, then the runtime engine guard can discard individual episodes without a corresponding declared group estimator. A group with fixed advantages `[+0.5,−0.5]` can become a positive-only update if the negative episode is rejected. The guard uses a maximum per-token gap as well as mean `k3`; longer trajectories have more opportunities to exceed a maximum threshold. Later optimizer updates also compare current learner probabilities against collection probabilities, combining engine discrepancy with genuine policy movement. [rl/native_replay.py:1631–1655](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L1631-L1655) [rl/native_replay.py:646–680](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L646-L680) [rl/native_replay.py:1283–1289](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L1283-L1289)

The latest brief reports no rejections among its 78 episodes. Verify the actual learner's per-update records; if that claim holds, this selection path is not an explanation of that admitted sample. It remains a correctness/diagnostic issue for later runs and is distinct from earlier unscored/length-filtered episodes.

Recentring after selection alone does not repair policy-dependent missingness. Maintain a full eligibility ledger and declare whether a broken capture invalidates the whole reward group or uses a justified alternative estimator. Do not silently retain only its successful members. Preserve safety checks while repairing their statistical integration.

### F8. Evaluation matching does not bind the complete execution environment

The parent-evaluation cache key is `parent_checkpoint_sha256:python_runtime_sha256`. Result labels similarly depend on checkpoint/step/arm/Python identity, and an existing `completed.json` prevents relaunch. The Python runtime hash covers package Python files; it does not bind the TypeScript host, outer driver scripts, or installed image/dependencies. [scripts/neuralese_data/pop_behavior_eval.py:483–499](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/scripts/neuralese_data/pop_behavior_eval.py#L483-L499) [scripts/neuralese_data/pop_behavior_eval.py:643–653](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/scripts/neuralese_data/pop_behavior_eval.py#L643-L653) [train/native_resident.py:37–39](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/native_resident.py#L37-L39) [common/runtime_inventory.py:370–379](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/common/runtime_inventory.py#L370-L379)

Promotion compares several useful fields—spec, packet/cases, prompt-rendering identity, serving config, Python runtime, precision, history policy, and tokenizer—but omits the full runtime closure, resolved image, driver, and summary commit. A fixed rendering-fixture hash does not cover every TS semantic change. The launch record already contains stronger runtime/image/driver fields, but copying it does not enforce equality: the comparison path reads completion summaries and case records, not that full launch contract. [train/promotion.py:50–52](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/promotion.py#L50-L52) [train/promotion.py:112–137](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/promotion.py#L112-L137) [data/rendering_contract.py:82–99](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/data/rendering_contract.py#L82-L99) [scripts/neuralese_data/pop_dev16_driver.py:120–136](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/scripts/neuralese_data/pop_dev16_driver.py#L120-L136) [scripts/neuralese_data/pop_behavior_eval.py:540–553](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/scripts/neuralese_data/pop_behavior_eval.py#L540-L553) [scripts/neuralese_data/pop_behavior_eval.py:612–620](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/scripts/neuralese_data/pop_behavior_eval.py#L612-L620)

**Executed comparator test:** two complete arms with equal tracked fields but different summary commits and separate launch runtime/image identities return `matched=true` and `promoted=true`. Positive controls changing Python runtime or spec return `matched=false`.

This distinction is important: the known `py34fcf6…` versus `py7bcae0…` mismatch in the diagnosis **is checked** by the comparator. The defect is that other execution changes can evade it. An informal score table can also compare unlike Python runs without using the comparator at all. No actual v66/v67 receipt mismatch is established by this audit.

**Repair:** define one full evaluation identity and use it for cache keys, paths, request selection, launch verification, and promotion. Existing `task_eval_runtime_identity` and `task_eval_policy_identity` machinery already binds TS/Python closures and the execution spec; extend that contract to resolved image/dependency and driver identity. Check requested versus actually collected execution identity. Re-evaluate parent and candidate under the same full contract rather than trusting label suffixes. [eval/task_cadence.py:243–272](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/eval/task_cadence.py#L243-L272)

### F9. An explicit repeat can be accepted without a fresh evaluation

`discover` renames an evaluation request `.accepted`, then discards it if its checkpoint SHA already exists with status other than `failed` or `superseded`. Runtime, repeat label, and intended fresh execution do not participate. [scripts/neuralese_data/pop_behavior_eval.py:219–230](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/scripts/neuralese_data/pop_behavior_eval.py#L219-L230)

An exact-source test starts with a completed checkpoint under `/old-runtime` and requests the same SHA under `/new-runtime` with a new repeat label. The request becomes `.accepted`; the candidate stays `done` under the old runtime; no `requested` event is logged. A prior `failed` candidate correctly queues in the control case. Even bypassing discovery can hit the completed-result reuse in F8.

This can defeat a same-weight repeat intended to test deterministic evaluation. A newly serialized checkpoint with changed metadata may have a different file SHA and avoid this exact branch; that does not make the general repeat mechanism correct.

**Repair:** separate idempotent request submission from an intentional new measurement. Identify an evaluation by checkpoint, complete execution identity, and explicit attempt ID. Emit a clear duplicate disposition. For every null/repeat experiment, verify distinct execution records and fresh collector timestamps rather than an accepted request alone.

### F10. SGLang capture advertises a seed the backend never receives

The SGLang actor records `request.seed` as `native_sampling_seed`, and the shared response publishes it in `sampling_law.seed`. However, `_sglang_run_bound` omits `sampling_seed` from the backend generation payload. The native actor, in contrast, seeds its generator with the recorded value. [serve/sglang_actor.py:417–418](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/serve/sglang_actor.py#L417-L418) [serve/sglang_actor.py:465–502](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/serve/sglang_actor.py#L465-L502) [serve/engine.py:1604–1615](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/serve/engine.py#L1604-L1615) [serve/engine.py:715–717](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/serve/engine.py#L715-L717)

The pinned dependency is `werg/sglang-natlang@a06adeda40a68d7ce7621ae44141afa0ec4a643c`; its sampling seed defaults to `None` and its sampler distinguishes seeded and stateful modes. [Pinned sampling parameters](https://github.com/werg/sglang-natlang/blob/a06adeda40a68d7ce7621ae44141afa0ec4a643c/python/sglang/srt/sampling/sampling_params.py#L179). [Sampler branch](https://github.com/werg/sglang-natlang/blob/a06adeda40a68d7ce7621ae44141afa0ec4a643c/python/sglang/srt/layers/sampler.py#L675-L687).

The exact request-construction reproduction advertises seed `1109` in its response but sends neither `seed` nor `sampling_seed` to SGLang. At positive temperature this undermines repeatability and common-randomness controls. It does **not** establish a wrong marginal sampling law, an inverted gradient, or the direct cause of native greedy Pop evaluation changes.

**Repair:** forward a supported backend seed with explicit continuation/position semantics, qualify that mode, and record whether the seed was actually honored. Until then, do not claim per-request seeded execution for this path.

### F11. Several summary fields describe different populations

**Student and teacher counts.** In `task_rollouts.score`, group-spread statistics select students, while total rollouts, scored rows, successes, and outcome counts include teachers. Production initially scores students, then overwrites final `score.json` after teacher collection. Therefore values read from final `score.json` cannot automatically be interpreted as successes or missing outcomes out of 128 student attempts. The diagnosis may already have filtered its counts manually; that is not visible here. Recompute by role before quoting 57/128, 40/128, or approximately 45% missing reward. [rl/task_rollouts.py:631–668](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/task_rollouts.py#L631-L668) [rl/production.py:1089–1125](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/production.py#L1089-L1125)

**The finishing endpoint is not full turn provenance.** The generation record exposes the endpoint that finished a key. Resume explicitly permits endpoint changes; unfinished entries can be assigned to another slot; the collector reuses saved responses from matching partials and sends new requests to the current endpoint. One trajectory can therefore contain draws from different engines while its rollout-level `policy.endpoint` names only the final endpoint. This is confirmed supported behavior, not proof any of the 78 w4 episodes were mixed. Per-action actor/capture metadata remains available and should drive engine comparisons. [rl/production.py:858–860](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/production.py#L858-L860) [rl/collection_generations.py:88–104](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/collection_generations.py#L88-L104) [rl/collection_generations.py:138–158](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/collection_generations.py#L138-L158) [rl/task_rollouts.py:459–465](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/task_rollouts.py#L459-L465) [rl/task_rollouts.py:609–611](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/task_rollouts.py#L609-L611) [ts-host/src/teacher/collector.ts:254–302](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/ts-host/src/teacher/collector.ts#L254-L302) [ts-host/src/teacher/collector.ts:755–763](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/ts-host/src/teacher/collector.ts#L755-L763) [ts-host/src/teacher/collector.ts:1040–1046](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/ts-host/src/teacher/collector.ts#L1040-L1046) [ts-host/src/teacher/collector.ts:1204–1232](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/ts-host/src/teacher/collector.ts#L1204-L1232) [ts-host/src/teacher/collector.ts:1325–1389](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/ts-host/src/teacher/collector.ts#L1325-L1389) [rl/native_replay.py:523–536](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L523-L536)

**Configured weight is not realized mixture.** Worker-slot code explicitly documents that work stealing ignores the configured endpoint weight for scheduling share. Pop weight `0.4` is not an enforced 40% sample fraction. Availability, throughput, task duration, and failures determine the realized episode and token mix. This is documented behavior, not a new hidden sampler probability bug. Log attempted/completed/eligible token mass and effective loss weight by engine and role. [rl/production.py:812–816](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/production.py#L812-L816)

**Required measurement:** one row per intended rollout, showing requested → started → terminal outcome → reward eligibility → valid capture → replay admission → length admission → engine acceptance → active objective. Include task, role, outcome, length, original group size, retained group size, per-action engine provenance, and update partition. Treat these as separate filters, not one random 45% loss.

### F12. New truncation changes learning and its search can miss valid prefixes

#### F12a. Import closure does not preserve the full estimator

The new truncation code retains an import-closed prefix with the original episode advantage and drops later action terms. The retained *unnormalized* terms can be identical to the full episode's terms, but the objective recomputes its action denominator. For an isolated episode shortened from `T` to `K` actions, each retained term gains a factor `T/K` relative to the prior full-episode token mean. In a mixed update, relative weighting and packing also change. Later repair/termination actions receive no training signal. [rl/native_stage.py:630–638](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_stage.py#L630-L638) [rl/native_stage.py:723–733](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_stage.py#L723-L733) [rl/native_replay.py:1336–1345](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L1336-L1345)

Terminal-return credit on earlier actions is ordinary policy-gradient behavior; that alone is not a bug. The issue is selectively removing later terms and changing normalization while describing retained terms as equivalent. Import closure establishes graph validity, not unbiasedness or behavioral benefit. This fallback may improve on dropping all long failures, but it needs declared normalization and original/retained action mass by role, outcome, and position. Prefer full-action coverage when feasible, or an explicitly justified sampled estimator.

This code is newer than w4's frozen runtime. The updated brief says it is included in w5, so it is a real experimental change and not a retrospective explanation of step 97→102.

#### F12b. One inadmissible cut aborts the entire search

`build(kept)` returns `None` when a shared external receipt straddles a proposed cut. The binary search immediately returns `None` for the entire episode. Receipt admissibility is not monotone over prefix length. [rl/native_stage.py:687–700](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_stage.py#L687-L700)

The exact-source search reproduction uses four turns and a three-turn capacity. Two adjacent middle turns share external state `x`. A two-turn cut splits them and is inadmissible, but a one-turn prefix is valid and a three-turn prefix keeps both and fits. The first midpoint is two, and the implementation gives up despite those valid choices. Ownership and geometry are mocked; the search function itself is the audited source.

**Repair:** enumerate/precompute admissible cut positions, then search capacity among those cuts, or otherwise continue around receipt-invalid midpoints. Do not treat one invalid cut as evidence that no valid prefix fits. Keep this qualification separate from whether truncation is the desired estimator.

### F13. The latest budget recovery still drops some resumed trajectories

**Version:** focused review of `5a72cb148a49214b7520438cd3854b5b70f0d482`, including the final recovery patch. This is an existing gap that survives the patch, not evidence that the patch introduced it.

The recovery function reads only the latest evidence manifest and reconstructs turns from `teacher_action_frame` and `decision_action_frame` records. But collector resume uses different records: a saved action emits `provider_frame` with source `journal_replay`, and a saved decision emits no new `decision_action_frame`; the latter is written only in the fresh-decision branch. Retained captures in the partial therefore do not automatically become recovered turns. An incomplete frame inventory cannot cover the execution graph, so the budget failure can remain untrainable. [Latest manifest selection](https://github.com/werg/natlang/blob/5a72cb148a49214b7520438cd3854b5b70f0d482/training/neuralese/natlang_neuralese/rl/task_rollouts.py#L310-L324). [Recognized frame kinds](https://github.com/werg/natlang/blob/5a72cb148a49214b7520438cd3854b5b70f0d482/training/neuralese/natlang_neuralese/rl/task_rollouts.py#L361-L380). [Action replay evidence](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/ts-host/src/teacher/collector.ts#L1348-L1350). [Decision replay and fresh evidence branches](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/ts-host/src/teacher/collector.ts#L1405-L1437).

**Executed checks:** all four new recovery test functions passed when invoked directly against the actual latest module with their AST-extracted fixtures: sampled decision recovery, rejection of mismatched saved decisions, retaining a valid prefix before an oversized omitted action, and refusal when no trainable turn precedes the omission. Two additional authenticated-fixture variants then preserved saved capture and valid evidence hashes while using the collector's replay formats:

| Evidence variant | Recovery result |
|---|---|
| Fresh action or decision frame, control | Prefix recovered |
| Same saved action represented as `provider_frame` / `journal_replay` | `None` |
| Same saved decision, without a newly emitted decision frame | `None` |

These are focused CPU module tests, not an end-to-end resumed collector run or a claim about the reported 14 w4 cases. Their frequency in actual w4/w5 remains unknown. They demonstrate that the latest patch's targeted success does not establish complete recovery of resumed budget failures.

**Repair:** emit one canonical authenticated action/decision frame on both fresh execution and journal replay, or explicitly recover the historical records with correct occurrence, request, invocation, and chronology bindings. Do not merely accept an extra record kind without preserving custody. Add mixed fresh/replayed action and decision trajectories that later hit the request budget. This belongs with the eligibility audit, because it can preferentially omit precisely the runs that needed recovery.

## 3. Corrections to the diagnosis and its experiments

### 3.1 The original advantages are potentially recoverable now

The latest brief says per-episode advantages cannot be checked because the receipt does not store them. The compact receipt indeed does not embed every value, but the source persists them in `rl-native-objective.json`: each objective row contains `advantage`. That file is durably written before the learner runs; `rl-data-receipt.json` stores its `objective_sha256`; resume/admission checks verify that hash; and the policy objective binds the exact bytes. [rl/native_replay.py:1710–1718](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L1710-L1718) [rl/native_stage.py:448–498](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_stage.py#L448-L498) [rl/production.py:54–60](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/production.py#L54-L60) [rl/native_replay.py:896–940](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L896-L940)

Retrieve the original admitted objective and compare its hash against the original receipt/checkpoint binding before comparing row advantages. If the original file is genuinely lost, reproduce and match the **complete objective hash**, or state the evidence loss explicitly. Byte-identical graph JSON and matching token counts do not prove identical advantages, loss configuration, or actions. Do not overwrite the original objective with a reconstruction before preserving its binding.

### 3.2 Scalar score-centered loss is not a measure of policy-gradient strength

At exact matched policies, with importance weights one and valid probability mass, the score-centered surrogate has the form

\[
U_y=\log p_y-\sum_v\operatorname{stopgrad}(p_v)\log p_v.
\]

Its value is centered, but with those coefficients held fixed:

\[
\nabla U_y=\nabla\log p_y-\sum_vp_v\nabla\log p_v=\nabla\log p_y.
\]

For a uniform binary policy, sampled action 0 and advantage `0.5`, the scalar loss is exactly zero while the logit gradient is `[-0.25,+0.25]`. The executed finite-difference check, **holding detached coefficients fixed**, gives `[-0.250000000007,+0.250000000007]`.

The code has no `1/k` shrink factor in this surrogate, and JOLT's outcome and KL terms share the action averaging path. Consequently, comparing scalar outcome loss `1e-5–1e-3` with KL loss `0.005–0.02` is not a valid reason to multiply outcome learning by 10–1000. The accumulation mismatch in F1 is real, but it is a separate integration problem. [rl/policy_math.py:120–174](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/policy_math.py#L120-L174) [rl/jolt.py:427–457](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/jolt.py#L427-L457)

The refreshed brief's gradient measurements are more relevant: norms 35–60% of KL, approximately orthogonal, would rotate a two-component combined gradient about `atan(0.35)` to `atan(0.60)`, or 19–31 degrees. Those illustrative angles assume the stated orthogonality and ignore other terms and optimizer transformations. They show why “not detectable in a Δlogp class average” must not become “absent from the update.”

### 3.3 Tokenwise direction is not a standalone sign test

The update combines student outcome, student KL, teacher outcome, and teacher KL through shared parameters. Different examples and token positions can conflict. Carried momentum, preconditioning, decay, BF16 writes, and five sequential minibatch updates further separate the final parameter delta from any one token's fresh gradient. An unweighted episode mean can also disagree with the actual loss weights. [rl/jolt.py:108–120](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/jolt.py#L108-L120) [rl/native_replay.py:1300–1345](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L1300-L1345) [train/optim.py:647–680](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/optim.py#L647-L680)

Two explicit counterexamples illustrate this without claiming either is the measured production cause:

1. **KL conflict:** a wrong action has student probability `0.8`, privileged-twin probability `0.99`, advantage `−0.125`, and `beta_s=1`. At matched behavior/current student, the outcome loss derivative with respect to the wrong-action logit is `+0.025`; the reverse-KL derivative is approximately `−0.5134`; the total is `−0.4884`. Correct gradient descent increases that wrong action's probability.
2. **Historical momentum:** current matrix gradient `I` with saved Muon momentum `−10I` gives new momentum `−8.5I` and Nesterov direction `−7.075I`. The equation reproduction gives `g_current · Δ(weight+compensation) ≈ +1.3728e-5`: ascent on the current local linearized loss, with no sign inversion in the fresh gradient. The `0.95` recurrence retains a substantial historical contribution over five updates; lowering LR does not remove that directional mixture.

**Better test:** authenticate the exact objective, disable optimizer mutation/hooks, and compare the complete gradient against an independent implementation with identical accepted actions, decision actions, masks, twins, and normalization. For finite differences, freeze all detached importance/centering coefficients and twin targets. Then independently test the optimizer transformation with cloned named state/RNG, recording both stored-weight and compensated effective deltas. Measure component norms, cosines, and `g_component · Δθ` by parameter group.

The latest diagnostic used unpacked graphs and omitted 40 sampled decisions, according to the brief. That remains useful evidence, but does not reproduce the deployed packed/accumulated objective. Include those decisions: their small number does not establish low behavioral importance.

### 3.4 The k3 statistic cannot directly create an entropy gradient here

In this implementation `k3` is converted to a Python float for engine mismatch diagnostics and acceptance; `_engine_guard` evaluates it under `torch.no_grad`. It is not added as a differentiable loss. If the guard rejected no episodes, k3 itself cannot be the source of the common gradient drift described in the refreshed brief. [rl/native_replay.py:646–680](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L646-L680) [rl/native_replay.py:1355–1379](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L1355-L1379)

Small mean k3 also does not prove that importance correction is unbiased, that rare high-weight actions are harmless, that masks/twins align, or that quantized top-k mass is valid. TIS clipping and approximation remain part of the estimator. A mean mismatch statistic is useful telemetry, not a blanket correctness certificate.

### 3.5 Privileged answer access is real; harmful shortcut transfer needs causal evidence

`case_facts` includes verified results, terminal expectations, reference solutions, and effects in privileged context. That establishes answer access. It does not by itself prove the observed student regression is caused by shortcut transfer: privileged conditioning is an intentional part of the teacher design, and a stronger teacher can either teach useful actions or encourage unavailable shortcuts. [rl/privileged_context.py:349–378](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/privileged_context.py#L349-L378)

The primary JOLT paper uses role-centered returns and separate teacher regularization, and warns that privileged teachers can take shortcuts unavailable to students. It also limits the guarantee when shared parameters change both views. These observations support measuring the proposed mechanism, rather than declaring answer presence itself a newly proven implementation violation. [JOLT paper, especially teacher construction and shared-parameter discussion](https://arxiv.org/pdf/2610.10447).

Find the earliest case divergence: an information-gathering action the student stops taking, or an answer emitted before available evidence supports it. At that prefix, measure the privileged twin's preference and each component's projected update. Compare answer-bearing versus process-only privileged context while holding the captured student actions and update contract fixed. Process-only hints may be a useful redesign, but should be evaluated as a new teacher distribution rather than presumed safe from every shortcut.

### 3.6 W5 is a useful intervention, but not a single-factor causal test

The refreshed brief changes `beta_s` to zero **and** moves from w4's frozen admission behavior to a runtime containing long-episode truncation and newer balancing. The final v69 update also repairs budget-failure prefix recovery. These changes affect which actions train, their normalization, and possibly microbatch packing. Fresh collection introduces another difference; SGLang's missing seed forwarding makes supposed shared randomness particularly unreliable. Teacher KL, shared parameters, existing optimizer history, and other defects remain.

Therefore:

- If w5 improves, any of these changes or their interactions may explain improvement; this does not isolate answer leakage.
- If w5 declines, privileged student distillation can still have been one contributor to w4; another contributor may dominate w5.
- `beta_s=0` does not mean only outcome gradients update the shared model. `beta_t` remains active, and F2 can leave the teacher's own outcome correction absent.
- The effective zero-signal issue in F5 becomes relevant under the changed coefficient.

The narrow diagnostic is a captured-batch comparison starting from the same checkpoint, named optimizer history, RNG, actions, update boundary, and **same admission/truncation rule**, differing only in `beta_s`. Hold budget-failure eligibility constant, then a small 2×2 comparison—old/new long-episode admission × original/zero `beta_s`—can separate those two factors if resources permit. Inspect budget-recovery effects separately on the captured data. Do the cheap gradient/update comparisons before spending on multiple fresh end-to-end training runs. State w5's observed result as a joint intervention until then.

### 3.7 LR zero is not the promised bit-identity control

Normal LR validation rejects zero. If a diagnostic harness forcibly sets optimizer groups to zero, optimizer state still changes; Muon also re-adds its saved BF16 Kahan compensation before rounding. That can change a stored weight even when the new LR-scaled update is zero. [train/native_learning_rates.py:24–44](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/native_learning_rates.py#L24-L44) [train/optim_restore.py:112–113](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/optim_restore.py#L112-L113) [train/optim.py:669–680](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/optim.py#L669-L680)

An explicit IEEE/BF16 counterexample:

| Quantity | Value |
|---|---:|
| Previous unrounded FP32 value | `1.0117111206054688` |
| Stored BF16 weight | `1.0078125` |
| Residual stored in BF16 compensation | `0.00390625` |
| Re-added value on zero-LR step | `1.01171875` |
| Ties-to-even BF16 result | `1.015625` |
| Stored-weight change | `+0.0078125` |

The original residual lies just below a midpoint; storing it in BF16 rounds it onto the midpoint, where ties-to-even selects the adjacent value. Separately, momentum, moments, counters, and stochastic-rounding RNG advance at zero LR. This is not evidence of arbitrary checkpoint corruption; it invalidates the proposed control's assumed identity.

**Correct pipeline control:** bypass all optimizer mutation, including in-backward hooks, and verify hashes of both backbone and interface tensors before and after save/export/reload. Then run a genuinely fresh matched evaluation, avoiding F8/F9. A separate forced-zero-LR experiment can measure numerical housekeeping, but must not be labeled the same control.

### 3.8 Randomly scattering the delta does not isolate direction from noise

The brief's random control has L2 `0.212`, compared with `0.205` for the real delta—about 3.4% larger. More fundamentally, equal per-tensor delta histograms do not preserve coordinate sensitivity, local BF16 spacing, activation effects, or norm/gate function. One random placement is weak evidence. Whether the scratch script included `interface_state`, preserved optimizer state correctly, and measured post-cast deltas was not independently verified.

Both random and real changes harming dev16 does not exclude a harmful learned direction; random changes sparing it does not prove a gradient-sign bug. Treat the result as one sensitivity probe. More discriminating controls include interpolation along the real delta, a small negative-delta probe, several matched random controls, and Muon/norm/conv/interface group ablations, all measured after actual casts. Always include interface tensors: the brief's row LR is 50 times its backbone LR, while its quoted 287-tensor delta analysis concerns the backbone.

Tiny global weight L2 does not bound behavioral change. Locally, logit movement depends on `J·Δθ`; greedy action choice depends on the margin between competing actions; a single early branch changes the entire later environment. Measure action margins and function-space/logit effects at the first divergence, rather than treating one-ULP changes as automatically negligible.

## 4. Other material risks and interpretation limits

### 4.1 Adam stochastic rounding remains a plausible numerical contributor

The ordinary BF16 route uses `AdamWSR` for parameters such as norm gains, conv kernels, and interface rows; both parameters and moments use stochastic rounding. The inspected Adam update has the conventional sign and bias correction. Nevertheless, unbiased writes do not imply low variance, and unbiased moment storage does not make a nonlinear Adam ratio unbiased. Muon Kahan compensation does not eliminate this separate route. [train/native_state_sft.py:278–290](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/native_state_sft.py#L278-L290) [train/optim.py:262–318](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/optim.py#L262-L318)

For an intended scalar decrement `d` smaller than adjacent representable spacing `u`, stochastic rounding has variance `u*d − d²`. As an illustration, immediately below a unit-sized BF16 gain, `u=2^-8`. With desired decrement `d=1.56e-6`, five independent writes have mean decrement `7.8e-6` and standard deviation about `1.745e-4`, or 22.37 times the mean. This assumes a fixed desired update and independent rounds; actual Adam steps and moment effects must be measured. It is a mechanism, not an estimate of production drift.

Replay identical saved gradients and named moments under the actual rounding stream, several alternative streams, and an FP32-master/compensated-write control that preserves the intended initial state. Record intended pre-rounding updates and actual stored/effective deltas by Muon, norm, conv, and interface group. Ordinary nearest BF16 writes without compensation are not an equivalent control: they can erase sub-ULP learning.

Also inspect actual group settings rather than assigning every parameter the headline `weight_decay=0.1`. The constructor gives ordinary Adam groups decay zero and applies the supplied decay to the latent Muon route; restored state/options require their own verification. [train/optim.py:110–111](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/optim.py#L110-L111) [train/native_state_sft.py:3187–3189](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/native_state_sft.py#L3187-L3189)

### 4.2 Greedy dev16 variance is not established by a binomial formula

The Pop spec uses `noise_repeats:1`; task evaluation defaults to temperature zero and fixed greedy queue seeds. This configuration does not itself measure repeated-execution variance; it does not rule out a separate historical noise measurement. The brief's “±1.7 binomial noise” describes a possible random-task-population calculation under assumptions; it is not evidence that rerunning these fixed 16 cases fluctuates by that amount. Runtime, hardware, scheduling, and greedy boundary effects should be tested directly with a fresh identical-weight run. [scripts/neuralese_data/pop_behavior_eval.py:85–88](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/scripts/neuralese_data/pop_behavior_eval.py#L85-L88) [eval/task_cadence.py:208](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/eval/task_cadence.py#L208) [eval/task_cadence.py:385–387](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/eval/task_cadence.py#L385-L387) [eval/task_cadence.py:512–526](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/eval/task_cadence.py#L512-L526)

Use paired case transitions—lost, gained, unchanged—under the same full execution contract. If making an inferential claim across tasks, state the task-sampling/exchangeability assumptions and use the paired structure. Sixteen cases and repeatedly selected checkpoints give limited evidence about broader capability. Reusing dev16 to select many remedies risks selection optimism; retain a fresh confirmation packet for a chosen fix, keeping protected cases out of training/reward redesign.

The reported 13→11→9 sequence is concerning. The later 10/16 after SFT is a partial rebound, not continuing monotonic decline across every stage. The 57→40 collection-success comparison is also not a fixed-task learning curve unless selected tasks and role denominators match. These observations motivate investigation without identifying its cause.

### 4.3 Reward shaping does not invert the binary success gap by itself

The inspected bonus/cost/report-credit bounds preserve the correct-versus-failed ordering within a mixed JOLT group: the efficiency term is bounded at 0.05 and the grounded failure-report credit at 0.02, leaving at least 0.93 of the one-point binary gap before any nonnegative success bonus. They do not let a short wrong answer outrank a long correct one merely through those terms. [rl/native_replay.py:1536–1589](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L1536-L1589)

That does not make the shaping irrelevant. In all-failure groups, small reporting preferences may be the only outcome-derived spread; selective removal of long failures can change which groups contain successes; and KL/history may dominate the realized update. Measure the component decomposition and final eligibility rather than labeling all honest stops rewarded success. The concrete status-classification asymmetry in F3 is stronger evidence than a generic early-stop hypothesis.

### 4.4 Actor parity must cover the deployed action distribution

One replay episode per round does not qualify every task length, role, engine, imported Neuralese state, or decision action. The refreshed mismatch measurements help, but a single rollout endpoint does not prove per-turn provenance (F11), and sampled-logp agreement does not validate top-k mass (F6). Validate exact exported checkpoint/interface identity, masks, sampled action IDs/positions, top-k/tail law, and behavior log-probabilities on a stratified set of actual admitted actions.

Do not infer from these risks that recovered Pop blocks actually had wrong policy IDs. Collection/replay custody includes substantive content and policy checks; that specific suspicion needs original capture/manifests. Likewise, the top-k **capture** size 128 is not evidence that sampling itself was truncated to 128 tokens. The inspected default SGLang parameters do not establish that different sampling law. [serve/sglang_actor.py:465–502](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/serve/sglang_actor.py#L465-L502) [rl/native_replay.py:1698–1754](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L1698-L1754)

### 4.5 Gradient correctness and algorithmic suitability are separate questions

A correct implementation can still optimize a harmful mixture, a badly censored data distribution, or a teacher policy that uses unavailable information. Conversely, a harmful evaluation trend does not establish an autograd sign error. Keep four questions separate: whether the objective matches its declared math; whether packing/backward computes that objective; whether optimizer/state/precision apply the intended update; and whether that update improves the intended task distribution.

The score-centering implementation uses learner/actor ratios and the published tail correction form; no simple inversion was found. The relevant primary paper's derivation presumes coherent sampler probabilities, which is why F6 is a separate numerical integration defect. Its batch formulation also makes normalization an explicit part of the algorithm, rather than a memory-planning detail. [Score Centering paper, Appendices A–B](https://arxiv.org/html/2609.20807v1).

### 4.6 Repair missing rewards according to evidence, not the label alone

| Evidence state | Appropriate next action |
|---|---|
| Completed oracle-rejected answer mislabeled because model text matched a transport/budget regex | Correct the classifier and score the verified model failure as zero. |
| Completed explicit `failed` terminal with the same semantics as an already scored `blocked` terminal | Apply the same typed terminal rule; preserve oracle/task semantics. |
| Genuine model request/token/turn budget exhaustion with authenticated, import-closed sampled prefix | Define that limit as task failure and train only on verified captured actions under an explicit prefix estimator. |
| Host wall timeout affected by load, actual transport failure, OOM, or interrupted external execution | Preserve infrastructure/censoring status unless a definitive model outcome is independently established. If intentionally rewarded, declare a new load/hardware-dependent objective. |
| Never-started key, missing capture, or unverifiable action history | Resume/recover evidence where possible; do not fabricate a negative trajectory. |

`budget_prefix_capture_unavailable` can remove a reward whenever no full result or valid budget prefix exists, not only for the particular loop category named in the diagnosis. Fixing capture and completion is therefore part of restoring learning signal. Turning every unresolved record into failure zero can train against machine outages or introduce invalid replay; it does not recover an unknown counterfactual task result. [rl/task_rollouts.py:592–619](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/task_rollouts.py#L592-L619)

## 5. Suspects narrowed by the review

These are source-level findings, not blanket certification of the unavailable production artifacts.

| Suspect | What the review found | Remaining boundary |
|---|---|---|
| Basic policy-loss sign or TIS inversion | Outcome is `−A × surrogate`; importance ratios are learner `p / actor q`. No simple sign inversion identified. [rl/jolt.py:108–120](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/jolt.py#L108-L120) [rl/policy_math.py:151–174](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/policy_math.py#L151-L174) | Numerical receipts, integration normalization, and complete gradient still need actual-batch verification. |
| Accidental teacher/student reward pooling | Group key includes task, group, and role. [rl/native_replay.py:1631–1646](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L1631-L1646) | Correct separation creates the singleton-teacher problem under the default. |
| Basic one-token target shift | Capture query convention and replay target offset are consistent: sampled token queries the preceding position, with target at query plus one. [rl/native_replay.py:44–84](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L44-L84) [rl/native_replay.py:347–357](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L347-L357) [rl/native_replay.py:1681–1688](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L1681-L1688) | Actual captured production rows and decisions need parity checks. |
| Generic JOLT twin coordinate swap | Token/turn identities, twin query mapping, graph identity, and physical-row remapping are checked; target views are detached. [rl/native_replay.py:1250–1327](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/native_replay.py#L1250-L1327) [rl/jolt.py:448–457](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/jolt.py#L448-L457) | No obvious silent mixup found; production packing/numerics remain separate qualification. |
| Prefix sharing has no RL tests | Substantial synthetic tests cover LFM/LFM3, RFT/JOLT, FP32 gradient equality, BF16 versus FP64 error, teacher/student twins, guidance imports, and decisions. [tests/neuralese/test_native_policy_prefix_sharing.py:148–247](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/tests/neuralese/test_native_policy_prefix_sharing.py#L148-L247) | These were read, not run here; they do not verify the actual 1.7B round or F1's unequal-length accumulation invariant. |
| Accidental CE/SFT loss inside RL | Production removes SFT-only flags; active policy objective disables ordinary CE supervision and returns through the policy loss path. Pending-focus setup belongs to the SFT branch. [rl/production.py:518–528](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/rl/production.py#L518-L528) [train/native_state_sft.py:2809–2811](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/native_state_sft.py#L2809-L2811) [train/native_state_sft.py:850–859](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/native_state_sft.py#L850-L859) | Saved SFT optimizer history still influences RL; that is not CE-row mixing. |
| Silent optimizer moment misbinding after interface split | Named restore checks parameter names, shapes, dtypes, and partition contracts; it does not blindly zip unrelated group orders. [train/optim.py:158–184](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/optim.py#L158-L184) [train/optim_restore.py:321–408](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/optim_restore.py#L321-L408) | Verify the actual checkpoints and named tensor hashes. |
| Kahan state lost on ordinary resume | Saved momentum/compensation are preserved; clearing is tied to a declared rounding-mode transition. [train/optim.py:582–607](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/optim.py#L582-L607) [train/native_state_sft.py:3172–3181](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/native_state_sft.py#L3172-L3181) | Actual mode/options must match the claimed run. |
| Async checkpoint writer races live mutable tensor storage | The training thread makes an immutable CPU snapshot before the background write. [train/trajectory_state.py:283–341](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/trajectory_state.py#L283-L341) [train/trajectory_state.py:503–517](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/trajectory_state.py#L503-L517) | Does not independently verify every published checkpoint's provenance. |
| Backbone or interface silently omitted on restore | Model checkpoint state contains both; strict restore checks backbone key/shape coverage and restores channels. [model/native_model.py:331–350](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/model/native_model.py#L331-L350) [model/native_backbone_restore.py:7–55](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/model/native_backbone_restore.py#L7-L55) | Delta analysis must also include both, unlike a backbone-only statistic. |
| Partial in-backward failure saved as a clean checkpoint | The main path refuses to publish an unsafe partially updated boundary. [train/native_state_sft.py:4486–4494](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/native_state_sft.py#L4486-L4494) [train/native_state_sft.py:4965–4989](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/native_state_sft.py#L4965-L4989) | Recovery behavior still depends on actual execution/receipts. |
| Earlier-only parameters lose accumulated gradients on final microbatch | Earlier unarmed backwards leave `.grad`; final-used parameters step/clear through hooks, and remaining gradients are consumed by the ordinary step. No dropped-gradient defect found. [train/optim.py:389–448](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/optim.py#L389-L448) [train/native_state_sft.py:470–594](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/native_state_sft.py#L470-L594) | This supports flushing correctness, not loss-normalization correctness. |
| RNG never restored | Python/Torch/CUDA state restoration exists after model/optimizer reconstruction. [train/loop.py:156–169](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/loop.py#L156-L169) [train/native_state_sft.py:3246](https://github.com/werg/natlang/blob/e1d72029a9f5c7801d8698c3dacda1e60060d5ff/training/neuralese/natlang_neuralese/train/native_state_sft.py#L3246) | Does not repair the separate unforwarded SGLang request seed. |

## 6. Recommended next work, in order

### Step 1 — Preserve and authenticate the evidence already generated

Recover the original w4/w5 configs and recipe bindings, admitted objective JSON, graph JSONL, admission/order receipts, actual microbatch/update membership, collected rollout/journal records, and evaluation launch/completion/detail receipts. Preserve originals before reconstruction. Verify objective bytes and advantages against the original hashes. Extract actual teacher counts, role-specific beta values, accumulation settings, and optimizer named-group settings. This should settle several conditional findings cheaply.

### Step 2 — Establish the local invariants before interpreting another end-to-end result

Run the small source counterexamples as regression checks, then the relevant existing suites in the project's proper PyTorch environment. The important additional invariants are:

1. Same complete accepted action set, several microbatch partitions → same intended gradient.
2. Completed wrong answers remain scored regardless of transport/budget words in model text; `failed` and `blocked` obey the same terminal semantics.
3. A generation cannot seal complete with missing finish keys.
4. No intended active objective across an entire update → no unintended optimizer/exposure mutation under the declared no-signal policy.
5. Matched uncompressed actor/learner probabilities → zero expected score-centering drift after the actual receipt codec, within a declared precision tolerance.
6. Intentional evaluation repeat → a fresh execution with a fully matching runtime contract.

These are discriminating invariants, not tests that merely restate current implementation formulas.

### Step 3 — Build one authoritative eligibility and objective table

Reconcile planned keys through all filters listed in F11. Show original and admitted positive/negative/zero advantage mass; teacher/student counts; original/retained actions; costs/report bonus; per-action engines; guard statistics; effective beta; and exact normalization. Reclassify only evidence-backed terminal failures. Check whether reported 45% censoring has consistent, non-overlapping, role-specific denominators. This establishes what learning actually saw.

### Step 4 — Separate gradient math from optimizer dynamics on the same frozen update

First compare independent and production gradients with no mutation, including sampled decisions and actual packing. Freeze detached coefficients and twin targets for local derivative checks. Decompose the four JOLT components and compare global/per-group norms and alignment. Then apply the same gradients to cloned actual-history optimizer states and inspect intended versus rounded deltas, including compensation and interface tensors. Use a fresh-history clone only as a diagnostic; do not silently reset the production optimizer.

This distinguishes a wrong objective, integration error, legitimate gradient conflict, inherited momentum, and rounding noise without requiring each to be inferred from dev16.

### Step 5 — Interpret w5 as a joint intervention; isolate its factors cheaply

Hold action data, checkpoint, optimizer/RNG state, update boundary, admission policy, and budget-failure eligibility fixed while toggling `beta_s`. Separately assess truncation/admission and recovered budget-failure actions. Check effective zero-signal rows at beta zero and teacher correctness signal. If KL remains harmful, examine the earliest action where answer-bearing context changes student behavior and test process-only context there. Tune coefficients using component gradients and behavioral evidence, not scalar-loss matching.

### Step 6 — Run a fresh matched evaluation and then a bounded training confirmation

Perform a true no-mutation export/reload control with backbone/interface hashes, unique attempt ID, matching runtime closure/image/driver, and fresh evaluation records. Compare parent/candidate by paired cases. Only after the specific invariant failures are repaired or bounded should a short resumed training comparison determine whether the update improves behavior. Use an untouched confirmation packet for the chosen remedy rather than selecting indefinitely on dev16.

No large new run is needed to establish F1, F2's configuration, F3/F4, F8/F9, or the misleading scalar-loss inference. Those should be resolved before committing compute to broad optimizer or teacher redesigns.

## 7. Production evidence needed to assign causality

| Evidence | Specific question it resolves |
|---|---|
| `w4/config.json`, `w5/config.json`, exact recipe and frozen runtime inventories | Actual teacher attempts, role beta, accumulation, admission flags, deployed source identity. |
| Original `rl-native-objective.json`, `rl-native-graphs.jsonl`, `rl-data-receipt.json`, `rl-episode-order.json`, omission/truncation records | Exact advantages, action masks/decisions/twins, hashes, ordering, old versus new admission. |
| Trainer logs and per-update/microbatch membership with graph/action/accepted counts | Whether F1 or a wholly rejected/inactive update was exercised; actual weighting. |
| Role-tagged rollout rows, queue manifests, terminal and collection-generation journals, raw typed errors and captures | Which “unresolved/infra” cases were genuine missing evidence, censored completed failures, interrupted or never attempted; role denominators and mixed engines. |
| Bound behavior top-k and sampled-logp receipts, per-action actor/export identities | Receipt-mass error distribution, endpoint attribution, mismatch and TIS behavior. |
| Named backbone/interface/optimizer state hashes at 92/97/102 and w5 start | Actual continuation identity, momentum/compensation/moments, all parameter groups and effective deltas. |
| Direction-test code and full results, including its objective binding and decision treatment | Whether the diagnostic used the exact objective and weighting, and what it omitted. |
| Random-control construction code, before/after tensor inventory and post-cast deltas | Whether its claimed distributional match includes interface and local quantization effects. |
| Evaluation request/job/launch/completed/detail records for all compared checkpoints | Full runtime matching, truly fresh runs, paired case transitions, result completeness. |

These are exact artifacts from the diagnosed run, not a request to expose protected test answers or to ingest them into training. Their absence limits causal attribution, not the source defects already reproduced.

## Appendix A. Executed reproduction results and their limits

| Check | Method | Result |
|---|---|---|
| Accumulation normalization | Actual AST-extracted source expressions | Same episodes: combined derivative `−0.00990099`, split derivative `+0.49`. |
| Singleton teacher | Actual advantage helper with minimal dependencies | Success and failure both zero; two different outcomes give `±0.5`. Production student-only bonus rule inspected separately. |
| Top-k probability mass | Actual receipt decoder plus NumPy production-coefficient calculation | Head mass `1.000468`; negative tail; alpha about `0.02`; expected gradient error L2 `9.8372e-5`. |
| Classification text and explicit statuses | Actual `classify_case` with completed oracle-failure fixtures | Text-only changes produce unscored infra/unresolved; `failed` differs from `blocked`. |
| Dispatch interruption | Actual dispatcher with minimal worker/journal fixture | One attempted key, two unfinished; success and failure both report initial `pending=2`. Production sealing path independently traced. |
| Evaluation identity | Actual promotion functions and completion fixtures | Different omitted runtime/image/commit evidence passes; changed Python/spec hash fails. |
| Evaluation repeat | Actual discovery with temporary request/state files | Same-SHA done request consumed; failed-state control queues. |
| SGLang seed | Actual request builder with mocked backend transport | Seed advertised but absent from emitted payload. |
| Zero-gradient optimizer movement | NumPy/IEEE transcription of optimizer equations | Saved Muon momentum moves zero-gradient matrix. Not an integrated optimizer/model test. |
| Zero-LR Kahan identity | Explicit IEEE BF16 round-to-nearest-even construction | Stored weight changes one ULP after residual re-add. |
| Gradient opposed by historical momentum | NumPy optimizer-equation example | Positive current-gradient dot actual/effective delta without sign inversion. |
| Shared-receipt truncation search | Actual search function with mock geometry/ownership | Invalid middle cut aborts despite valid fitting alternatives. |
| Centered scalar versus gradient | NumPy analytic plus finite differences with fixed coefficients | Scalar zero; gradient `[-0.25,+0.25]`. |
| w5 zero-signal projection | Actual projection function with no independent decision/writer term | Zero-advantage, beta-zero student with twin retained; no-twin control omitted; beta-one control retained. |
| Final budget recovery | Actual latest module; four extracted test functions and two authenticated replay-format fixtures | Four targeted tests passed; fresh captures recovered, resumed action/decision variants returned `None`. |

The exact-source scripts were run locally and their assertions passed. Mocked contexts are stated because these checks establish local contracts and counterexamples, not real-model throughput, GPU equivalence, or measured w4 incidence. Re-running the existing full suites requires the project's proper runtime.

### A compact standalone illustration

The following executable Python uses only NumPy. It reproduces the normalization reversal and the centered-loss/value distinction; it illustrates the float16 head-mass problem. It is intentionally an equation check, not a replacement for the exact-source or full integration tests.

```python
import numpy as np

# Same action derivatives, same complete update, different partition.
short = np.array([1.0])
long = np.full(100, -0.02)
together = np.concatenate([short, long]).mean()
split = 0.5 * short.mean() + 0.5 * long.mean()
assert together < 0 < split
print("normalization:", together, split)

# Quantized top-k receipt from a valid normalized distribution.
q_head = np.full(128, 0.9999 / 128, dtype=np.float64)
stored_logq = np.log(q_head).astype(np.float16)
decoded_head = np.exp(stored_logq.astype(np.float32))
head_mass = float(decoded_head.sum(dtype=np.float32))
assert head_mass > 1.0
print("head mass / reconstructed tail:", head_mass, 1.0 - head_mass)

# Uniform on-policy score-centered surrogate: freeze coefficients.
z0 = np.zeros(2, dtype=np.float64)
p0 = np.array([0.5, 0.5])
advantage = 0.5
coeff = np.array([1.0, 0.0]) - p0

def frozen_surrogate_loss(z):
    centered = z - np.max(z)
    logp = centered - np.log(np.exp(centered).sum())
    return -advantage * np.dot(coeff, logp)

eps = 1e-6
basis = np.eye(2)
fd = np.array([
    (frozen_surrogate_loss(z0 + eps * e)
     - frozen_surrogate_loss(z0 - eps * e)) / (2 * eps)
    for e in basis
])
assert np.isclose(frozen_surrogate_loss(z0), 0.0)
assert np.allclose(fd, [-0.25, 0.25], atol=1e-9)
print("centered loss / gradient:", frozen_surrogate_loss(z0), fd)
```

## Appendix B. What would change this review's conclusions

- A live accumulation count of one would remove F1 as an explanation for those particular updates, while leaving the implementation defect for accumulated runs.
- Multiple scored teacher siblings with nonzero saved teacher advantages would remove the singleton-teacher explanation for that run.
- Original journals proving every expected key finished would exclude F4 for that round; typed error provenance could show that the reported infrastructure rows were all genuine.
- Verified per-update zero guard rejections and active objective components would exclude F5/F7 for those updates, without repairing the general paths.
- Actual top-k receipts showing negligible mass error and gradient contribution would lower F6's causal priority.
- Fully equal launch/runtime/image/driver contracts and fresh paired executions would remove the identified evaluation confounds for the compared scores.
- Independent complete-gradient agreement, partition invariance, and correct actual-history optimizer updates would shift attention from implementation to the configured objective, teacher behavior, selected task distribution, and functional sensitivity.

The strongest conclusion is that there are concrete defects worth fixing and important diagnostic inferences worth correcting. The appropriate next milestone is an authenticated, reproducible explanation of one complete update and one paired evaluation—not a prematurely exclusive story about sign, teacher leakage, or rounding noise.
