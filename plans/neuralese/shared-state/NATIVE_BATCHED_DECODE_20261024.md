# Batched native decoding

Policy: use the available GPU efficiently while preserving the actual native
actor, whole-graph captures, producer/consumer state and per-request sampling.
Batch equal adapter declarations; retain explicit distinctions between default
adapters and an explicit empty selection. Prefill retains the existing bounded
chunk execution. This policy selects useful execution; it changes no task reward,
held split, model weights or communication architecture.

Mechanism: `model.native_decode_batch.native_step_batch` combines token-local
embedding, input projections, output projections and FFNs across independent
single-token decode lanes. Attention sees each lane's own positional KV and
window. Finite convolution retains lane-local histories. Gated-delta projections
are shared while replay retains each lane's convolution/recurrent state. Every
returned capture keeps the ordinary native-step schema. This is actual grouped
model computation, not a loop of independent full model forwards. Attention and
recurrent state operations still run per lane where their ragged state differs.

The serving state machine yields its one model-step request. The scheduler
executes compatible requests together and applies one grouped vocabulary readout,
then finishes each original state transition. Forced turns, sampled logp, RNG,
producer boundaries, consumer imports, cancellation, guidance and protocol failure
handling remain in that state machine. The original independent driver remains
an exact execution reference, not a second production architecture.

CPU validation: independent unequal-length caches, mixed producer roles, retained
KV/convolution/delta state and all named gradients match for LFM, Mellum, standard
Qwen attention and Qwen3.5 hybrid state. Serving checks cover complete captured
producer/import executions, adapter-group separation, cancellation, existing
guidance, transport, close boundaries and whole-graph replay. The integrated
focused suite passed55 checks. Actual full-size GB10 numerical/throughput evidence
and deployment are recorded separately by the supervising operator before cutover.

Supervision policy: a worker failing before any saved child may be isolated when
`continue_after_unstarted_stage_failure` is explicitly enabled. Preserve the exact
parent, record the failed stage and its unchanged signature, retain all exposure
debt and continue other useful work. A saved child always requires recovery;
never skip it and resume an obsolete parent. A changed recipe/runtime retries
the repaired stage. This avoids an administrative descriptor error stopping all
training while retaining strict objective/source/numerical verification.
