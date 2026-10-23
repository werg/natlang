# Native whole-graph reinforcement learning

The current RL trainer is `natlang_neuralese.rl.native_stage` with recipe kind
`native_state_rl`; the adaptive collection and continuation supervisor is
`natlang_neuralese.rl.production`. Collection uses the shared task packet,
collector, and scorer. Reward is exact task success: model-caused failures score
zero, while infrastructure errors and unresolved cases are unscored. The
collector binds each rollout to the exact student actor, checkpoint, runtime,
packet, and split.

When a rollout contains Neuralese, training preserves the full recorded graph,
including sampled and forced producer turns, consumer imports, occurrence
bindings, positions, and downstream outcomes. The trainer verifies the exact
actor and state references and differentiates through the real producer and
consumer dependencies. Missing graph capture is repaired before use; a producer
is never replaced with a detached external value. Text-only student trajectories
remain eligible under the same reward rule.

`rl.task_rollouts` prepares and scores train-only task groups.
`rl.rollout_math.group_advantages` supplies the shared pure group normalization
used by collection, native replay, and the semantic policy. Groups without a
reward spread provide no policy-gradient signal; the supervisor can schedule
more sampling or verified successful seed material. Training receipts retain
completed graph exposure and exact checkpoint, optimizer, and RNG continuation
state.

Qualification comes from held native loss/runtime checks and distinct actual
task outcomes on development and confirmation invocations. Actor attestation,
gradients, stored guidance, or successful admission alone do not establish
capability. Continue to update the recipe and task curriculum from measured
outcomes while preserving protected evaluation data.
