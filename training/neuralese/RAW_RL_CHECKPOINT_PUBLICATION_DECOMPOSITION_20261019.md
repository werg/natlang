# RAW RL checkpoint owner publication

The recipe's natural-language stages decide when a raw RL update is useful and
which existing checkpoint, task rows, and trainer settings belong to it. This
addition only publishes exact ownership metadata so the existing full-checkpoint
exporter and held-task consumer can follow that real update.

1. The provider bridge keeps its current request, result, actor, and training
   control. Its external worker closure verifies the installed runtime and
   replaces only the `production.run_command` dispatch boundary with a wrapper
   for the exact `natlang_neuralese.rl.native_stage` command.
2. The wrapper runs the installed `native_stage.main` unchanged. At the actual
   nested call to the shared `native_state_sft.main`, it observes the effective
   trainer argv, current selected parent, output slot, runtime, role, and the
   declared publication selection. It emits the existing owner-generation
   manifest before invoking that exact trainer call.
3. The owner manifest records publication intent and does not claim that an
   update, save, task evaluation, or qualification has completed. The existing
   cursor exporter independently verifies a newly saved full checkpoint and
   private readback. The quality consumer reads only that exporter receipt.
4. Coefficient-only RL remains on its existing distinct writer. The wrapper
   must fail closed for it until the coefficient checkpoint/policy writer has
   a corresponding exact owner publication; it cannot label a base checkpoint
   as a coefficient update.

The outer bridge request/result remains the durable owner of the RL quantum.
No task, reward, graph, optimizer, RNG, or checkpoint semantics change. The
wrapper and exporter are exact outside-world plumbing; stage selection and
training policy remain in the existing natlang recipe and native RL program.
