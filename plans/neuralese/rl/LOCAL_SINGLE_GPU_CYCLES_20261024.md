# Local LFM training policy

Status at2026-10-26 07:10 Berlin: local LFM resumed from its independently backed
step37994 and has actually advanced through38001 on full-source SFT. Its active
recipe declares1,618,603 rows; declared supply is not completed exposure. Eleven
whole-graph RL rounds are complete. Round11 was rejected before updates because
the captured prompt consumed native state absent from the saved graph. Its exact
collection remains preserved and unfinished while SFT/DPO continues. Shared
capture repair5297bdd41 and replay-position/gradient check54d70f12d are integrated;
the focused29 TS and10 replay tests pass. New collections use the repaired local
runtime; any historical capture repair must retain its original actor and actual
producer/import dependencies. Recent protected task checks still do not establish
capability lift.

One cloud Mellum learner is authorized again. Its replacement is
`b00fcc02-a619-40c7-911f-6363d569d979`, with one96GB Blackwell GPU and96Gi host RAM.
It restores the fully verified step36054 through exact R4 runtimebe025b9b7,
retaining optimizer/RNG/exposure and unfinished SFT/DPO frames. The controller
has launched; first new optimizer progress is still being checked. Last observed
step36073 was not exported:19 later updates remain unrecovered. Independent
checkpoint publication and a workstation mirror remain primary; snapshots have
not been qualified. Cloud generation and evaluators remain paused. Allocation
lifetime is bounded at8h with one cloud learner; actual rates/balance remain
opaque, and$500 is only the owner's upper bound.

Current LFM controls and evidence:
`/home/werg/data/lfm-local-recovery-20261024/source-rejoin/round11-capture-error-sft-continuation-20261026/`.
Cloud source/control and recovery evidence:
`/home/werg/data/mellum-cloud-continuation-20261025-r2/second-reset-*` and
`/home/werg/data/mellum-cloud-resume-20261025/current-private-recovery.json`.
The older deployment observations below are historical, not current owner heads.

Use the DGX GB10 for the current earned LFM, alternating an on-policy whole-episode
RL round with a finite continuation of every available mandatory SFT/DPO stage.
Reward correct execution without requiring Neuralese; retain every real graph
node and dependency whenever communication occurs. The admitted curriculum
retains rotating breadth across all 140 tasks. Keep the original full production
recipe and its earned/remaining source exposure; an available-data continuation
is not comprehensive completion. Add restored stages and newly admitted sources
through explicit continuations as they become usable.

One exclusive optimizer-owner lock covers actor/learner transitions and standalone
SFT launches. The actor stops before an update. Reuse `production.run`, the exact
whole-graph admission/trainer, `native_supervised.continuation_config`, existing
relative checkpoint quanta and exact checkpoint publication. The shared
`rl.local_cycles` module only sequences these mechanisms. Its durable target
round prevents recollecting an already published RL round after a cursor-write
interruption. Supervised publication accepts only the original parent or its
verified child and resumes the same operation directory after interruption.
Task selection and quantum policy remain explicit in the local plan/recipe.

Current deployment root: `/home/werg/data/lfm-local-recovery-20261024/rl`.
The saved handoff parent is step36385, retaining optimizer/RNG and previous stage
credits. Source custody and train/held intersections are in `task-custody.json`.
Only actual rollout outcomes and subsequent saved updates establish RL progress.
No cloud allocation or external generation is involved.

The actual whole-batch DGX memory measurement found 354,506,496 parameters,
287,396,608 trainable under the existing native policy. For 60,255 logical /9,148
physical tokens, checkpointed FFN4096 took6.579s warm vs5.156s without layer
checkpointing, with4.33GB vs22.18GB peak CUDA allocation. Removing FFN chunking
changed5.156s to5.112s in this measurement. These are temporary, sequential
in-memory diagnostic updates, not a capability comparison or earned checkpoint.
Do not extrapolate this capacity to arbitrarily longer whole graphs. Preserve
qualified settings for RL until its actual collected graph capacity is checked.


Actual first local execution: the 16 current-policy samples produced3 verified
successes,6 invalid actions,1 semantic mismatch and6 budget terminations. The
integer-return group had reward spread and4 complete episodes were admitted to
GRPO. Two real updates saved36387 from36385, with good numerics and20.89GB peak
CUDA allocation. Independent CPU full-state verification found92 changed backbone
tensors, Adam age36387, exact preserved original SFT/DPO/current-tranche credits
and unchanged held-row bindings. SFT resumed on this earned child.

Adaptive recomputation is now implemented in the shared trainer with
`checkpoint_direct_token_limit`: after prefix packing, graphs at or below the
explicit physical token threshold use direct backward; longer graphs retain
layer checkpointing. The value is saved in options and included in capacity
identity. Default0 preserves uniform checkpointing. Local threshold10000 passed
the actual maximum39077-token and near-threshold9996-token capacity/numerical
checks before resumed SFT. The focused checkpoint/FFN/cursor suite passed63 checks.
Mellum has the shared mechanism and retains its existing settings pending its
own actual-hardware capacity qualification.

The restored11988-row preference source and exact objective are declared in the
current local recipe, preserving the previous6644 earned visits. New local DPO
exposure is reported separately after its actual worker advances. Original full
SFT source restoration/rejoining remains outstanding; local continuation and one
RL round do not complete the entire recipe or establish capability lift.

Operational repair: the first local held-task specification omitted
`neuralese_server:true`; the learner rejected it before any update. The corrected
versioned recipe retains every task/reward/source binding. The completed collection
replays with its exact old actor Python closure, reconstructed from commit08e293a07;
subsequent new actors use the current explicit runtime. Nothing was relabeled.

The first complete local cycle earned RL36387, SFT36487 and DPO36537, preserving
optimizer/RNG and original recipe exposure. A second RL round subsequently
committed eight complete GRPO episodes and SFT resumed on that earned child.
Its16 samples yielded6 successes,4 invalid actions,2 semantic mismatches and4
model-request-budget terminations. Different task mixes prevent interpreting
these counts as measured capability lift. Whole-graph RL peaked at44.18GB CUDA
allocation; the supervised worker subsequently peaked at22.21GB.

Native decode batching is implemented and deployed in frozen runtimeb2747a552,
including unequal positional KV, convolution/linear state, per-request adapter
identities, sampling/RNG, cancellation and complete occurrence capture. The
step36537 GB10 check measured83.19 versus302.67 decoded tokens/s for four lanes,
with identical logits. Actual collection used grouped forwards before the next
whole-graph learner took over. See the shared batched-decoding plan for scope.
Both cloud learners remain paused; these are local LFM advances.

Original full-source restoration remains an active obligation. Local staging
restored1218 exact files and regenerated seven missing filtered payloads with
their original hashes. Header recovery found five more exact manifests;66
original headers and the7027-row Search/OWL payload remain unavailable. Restored
bytes do not establish admission or exposure. Rejoin usable original data through
an explicit continuation while retaining earned credits and missing-data debt.
