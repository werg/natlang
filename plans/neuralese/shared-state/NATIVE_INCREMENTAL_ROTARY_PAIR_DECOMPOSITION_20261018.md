# Reuse incremental query/key rotary work

The live Mellum actor's passive 30-second profile sampled 1.95 seconds in the
eager rotary and rotate-half leaves of its serving thread. This is sampled host
stack time, including possible device waits, not a kernel timing or speedup.
The incremental shared-state implementation rotates query and key separately;
each existing port rotary interface already computes both outputs. Consequently
it constructs the same positional tables twice and computes two unused outputs
per attention layer and chunk.

1. Keep the existing single-tensor rotation interface for graph positions that
   differ between queries and imported key slots.
2. Share one positional-table construction and the port's existing paired
   rotary operation only in incremental execution, where new query/key positions
   and activation dtypes are identical.
3. Preserve the current key cast, model-specific rotary implementation, captured
   unrotated keys, cache insertion and autograd. No new arithmetic kernel,
   numerical profile, masking rule, parameter or checkpoint format is introduced.
4. Compare complete cached outputs, state captures and gradients with the old
   independent-rotation execution for both students and the shared Qwen port,
   including grouped-query heads and multiple chunks. Existing one-pass versus
   explicit cached graph tests remain applicable.

This is crisp numerical mechanism. Training/serving policy, prompts and task
selection remain their existing natlang programs. Production adoption uses the
usual immutable runtime update and current-parent checks; the already-running
actor is not patched. A live throughput improvement remains unmeasured until
that deployment is actually used.
