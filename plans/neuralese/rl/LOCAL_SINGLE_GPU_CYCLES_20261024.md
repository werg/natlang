# Local LFM training policy

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

Deferred throughput work: `serve/engine.py` advances native sequences individually
in `_native_round`. Implement and compare real batched native forwards, including
unequal positional KV, convolution/linear state, per-request adapter identities,
sampling/RNG, cancellation and full RL occurrence capture. More collector workers
alone do not remove this bottleneck. Keep the functioning learner/actor cycle
moving while preparing that shared serving improvement.
