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
