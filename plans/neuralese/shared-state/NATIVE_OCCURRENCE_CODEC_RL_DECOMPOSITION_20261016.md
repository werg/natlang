# Complete episode communication occurrences and semantic policy gradients

The natlang program calls producers and consumers and may explicitly make a
particular value use constant. Training rewards the entire actual episode,
including ordinary tasks without communication. A communication dialect changes
the representation, not which source calls receive downstream gradients.

Exact mechanisms:

1. Preserve producer-occurrence provenance on native typed references, private
   transport, context/soft-body snapshots and `.nz`. Equal immutable artifact IDs
   need not mean the same source call. The global latest-writer map is only the
   compatibility source for an old reference with no explicit origin.
2. Capture the original source occurrence and per-use constant flag in actual
   imported-state receipts. Match every read event and every consumed input once.
   Ordered occurrence bindings replace lossy per-ID maps where duplicates exist;
   old unique-ID receipts retain their existing form and identities.
3. Keep each actual episode model invocation in the admitted graph. Raw KV edges
   execute through the existing single masked native graph. An explicit stopped
   use preserves forward values while cutting only that use's source adjoint.
4. Semantic messages retain the original message ID, raw producer observation,
   exact adapter identity and actual decoded target-state observation. Recompute
   the source state on current participating weights, encode its posterior mean,
   decode through the existing target baseline/alignment mechanism, and consume
   that differentiable state. The source observations verify behavior receipts;
   they are never substituted for a captured producer during the policy update.
5. Codec dependencies genuinely require producer state, encoder/code/decoder and
   target consumer phases. Name and account for those graph segments explicitly,
   reusing shared graph execution and semantic components; do not call them one
   raw-KV flat forward. Every source invocation executes once in its segment,
   with real cross-segment gradients and exact actor scopes.
6. Preserve shared optimizer/RNG and objective exposure across continuation. A
   changed codec or participant checkpoint is an explicit current-policy revision;
   behavior artifacts keep original identities. External source state remains
   constant only when actually external to the complete captured episode.

No new model-facing instruction or operation is needed. Policy is in the typed
natural-language program and curriculum; matching, compilation, tensor transport,
AD, arithmetic and checkpoint validation are crisp exact mechanisms. Public
higher-order extras remain below production semantic RL in priority. Separate
independent integration review precedes adoption, and no generic migration
completion or full-size cloud qualification is implied by component tests.

Compatibility refinement: new runtime native references without an origin are
external, even if an equal ID appears in the latest-writer cache. Missing source
provenance cannot establish a captured producer. The latest-writer compatibility
map remains for legacy projected values only; already saved unique-ID execution
receipts keep their explicitly recorded edges. New native producers attach their
actual graph-write node before typed result coercion. Source-origin metadata is
not included in the immutable state artifact's ID or payload.

Raw occurrence component evidence (2026-10-16): final CPU40 passed3.60s;
Node50 passed7.53s; actual typed child/archive/parent/decision HTTP10 passed93.45s;
build/layering passed. Actual Engine repeated imports match captured normalized
behavior scores and preserve the live producer gradient on both ports, with a
nonzero low-rank embedding adapter. Actual CUDA4 passed10.65s after the supervisor
released host caches (first attempt failed before computation at model.to with
CUDA allocation OOM; no numerical failure or tolerance change). These checks
qualify tiny-port mechanisms, not full-size cloud capacity or learned task lift.
Semantic capture now retains original message ID, actual decoded observation and
adapter receipt, but semantic policy lowering remains explicitly guarded until
the next codec component is complete. No detached decoded input is admitted as
an in-episode producer substitute.
The private literal's source origin is also checked through actual eval
compilation/execution, not just source-string rewriting: the typed result retains
both its constant flag and exact producer. This exposed and fixed the internal
intrinsic declaration's missing third argument before publication (one focused
Node case passed). Default private literals still use the original one-argument
form. No added model operation or instruction is needed.

Differentiable codec component (in progress, 2026-10-16): the executor now
schedules native masked graph segments around explicit codec dependencies. It
uses the actual inference posterior mean, actual target baseline and tokenizer
alignment; retains source, codec and consumer adjoints; and records native
segment, codec and target-baseline execution counts. Whole-episode capture
compilation authenticates original compact wire, observed decoded state, exact
codec/offset/alignment receipt and actual producer occurrence. Immutable observed
state binds archived graph geometry; it is replaced by the live current decode
before forward, never used as an in-episode producer substitute. Explicit stopped
uses detach only the imported path. Ordinary/raw objectives keep their original
model call through the optional shared-objective hook.

Current component evidence: CPU14 actual two-port/same-model/cross-model
composition and incremental cached producer/consumer adjoint tests passed in
3.42s. CPU29 combined core/replay tests passed in4.06s, including mixed live and
stopped uses and pre-forward cycle rejection. CPU32 archive/admission/shared
trainer-objective and existing whole-episode/raw regressions passed in3.50s.
One new test initially compared distinct CPU GEMM row positions bitwise and
observed a 7.45e-9 difference; the exact stop invariant is now tested against the
same physical layout with only the AD flag changed, bitwise. No numerical
threshold or production arithmetic was changed.

Still required before production semantic RL enablement: complete actual
multi-participant capture/objective lowering and shared phase scheduler;
participant plus VAE optimizer/RNG/checkpoint/exposure continuation; atomic
updated inference adapter export; exact new dependency/capacity scope; integrated
independent review. Loading a frozen channel into the single-model native stage
alone would strand codec state and is deliberately not advertised as closure.
Public semantic GradSession/code/state optimizer extras remain separately open.
Final component CPU51 passed5.52s, including real empty payloads (only actual
control markers, zero invented payload tokens), full archived semantic sampling
and shared-objective integration. Quick passed22ratchets+TScheck. Initial CUDA
attempt failed before execution at model.to(cuda), with unified-memory context
allocation OOM; the supervisor measured only6GiB MemFree after cache release and
requested no blind retry while live evaluations/forensic decoding occupy memory.
Thus this revision has no CUDA codec qualification yet; earlier raw occurrence
CUDA evidence does not substitute for it. No serving capability was enabled.
