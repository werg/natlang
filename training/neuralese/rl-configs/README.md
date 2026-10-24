# Native local and CoreWeave RL

Use the native RAW actor and whole-episode shared trainer described in
[shared-state RL operations](../../../plans/neuralese/shared-state/RL_OPERATIONS.md).
RAW RL can start independently of communication-foundation and shared-VAE
readiness. Correct task execution is rewarded without requiring Neuralese;
communication episodes retain their complete producer/consumer graph and gradients.

Generate configurations with `natlang_neuralese.rl.production configure` from the
actual full native checkpoint, exact base, admitted train tasks, held source and
production recipe/stage. Preserve optimizer/RNG and every prior stage's exposure.
Keep source, actor checkpoint/tokenizer/runtime and train/held custody explicit.
The managed actor stops before learning and reloads the earned checkpoint on its
next lease. Local execution uses the memory ledger and one optimizer owner.

`natlang_neuralese.rl.local_cycles --plan PATH` alternates one production RL round
with relative checkpoint quanta of every available mandatory SFT/DPO stage. The
plan declares `rl_config`, `out`, `optimizer_owner_lock`,
`supervised_quantum_updates` and optional `supervised_stage_quantum_updates`.
Quanta are continuation boundaries, not recipe completion. New data and restored
stages belong in explicit versioned recipe continuations. Keep the original full
recipe's outstanding exposure visible when recovering on available inputs.

See [the verified DGX deployment and memory measurements](../../../plans/neuralese/rl/LOCAL_SINGLE_GPU_CYCLES_20261024.md)
for actual local execution evidence and remaining throughput work. The same
native objectives and trainers serve both architectures; qualify memory and
serving on the actual assigned hardware before changing architecture settings.
Private full-state exports and input/runtime archives remain independently
downloadable without a running cloud sandbox; filesystem snapshots supplement them.
