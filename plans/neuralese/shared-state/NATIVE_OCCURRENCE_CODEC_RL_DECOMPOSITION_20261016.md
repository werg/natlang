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

Joint policy-stage design (next delta, not yet enabled): the existing shared
semantic loop owns participant optimizers, VAE optimizer, RNG, rolling checkpoint,
qualification and export. `policy_training` declares its phase ID, immutable
rollout/bundle/behavior-adapter sources, GRPO or RFT, epochs, semantic-update
cadence and finite refresh quantum. Actual episodes are parsed by the existing
capture compiler per participant, then their already authenticated foreign writer
occurrences are reconnected before execution. Every call/action is included;
ordinary one-call episodes remain eligible. The existing phase exposure map also
holds committed policy episode visits, with required visits and explicit
objective revisions; no ordinary-positive credit is awarded for rejected actions.

Behavior adapter/identities authenticate original wire and densities. Current
execution is a separately declared derivation from an exact shared-parent
checkpoint/adapter and step. In-episode producers encode current state. An actual
external compact code stays constant with its original identity while an explicit
training-only decoder interpretation may learn; default serving compatibility
stays exact and unchanged. No artifact is relabeled as a current-codec message.
Only the existing shared durable checkpoint and new adapter export publish the
updated weights/code for inference. Source admission and new actual policy-pair
geometry remain capacity obligations, distinct from prior raw-graph proof.

The joint policy-stage implementation now reuses the existing semantic loop,
participant optimizers, codec optimizer, backend scopes, deferred joint backward,
qualification and atomic export. Whole-source policy visits use the existing
phase exposure ledger. Parameter changes retain earned visits and require a
fresh visit only where none remains under the new objective; unvisited quotas do
not inflate on each tweak. Changed modes cannot erase unfinished episodes;
explicit supervised cancellations retain the original uncredited deficit.
Policy refresh work is a finite declared quantum while full epoch deficits remain
mandatory and checkpointed. A changed policy boundary also schedules the actual
full final held evaluation after all declared visits, rather than inheriting an
old pre-policy held result.

Routing retains an entire collected batch whenever it contains a learned codec
input. This preserves original relative-reward groups across ordinary and
semantic episodes; no turn or captured producer is sent to a separate learner.
Per-source GRPO/RFT and collecting-participant scope remain bound. Collection
state owns one pending source until the shared coordinator durably enrolls it.
Enrollment is distinct from earned visits: only checkpointed nonzero episode
visits enter trained-task accounting. Reapplying an old deployment cannot clear
a new pending batch, and restart does not recollect or launch a raw-only learner.
The shared curriculum's `policy_schedule` supplies operational settings; actual
sources are inserted only after collection, not represented by placeholder data.

Actual CUDA component28 passed51.56s (3.1GiB peak). The actual shared-stage test
uses serialized LFM+Mellum parents and the production Mellum FP8 loader/backend,
all VAE/joint/distill plus two policy updates, exact both-model/VAE optimizer and
weights, CPU/CUDA RNG and exposure across interruption/resume; it passed20.13s.
That test exposed a real frozen-codec cache-miss bug: module construction consumed
CPU RNG before saved weights replaced its random initialization. The shared
loader now forks/restores only that construction's RNG; training startup does not
restore a global serving RNG. Follow-up counters attest executed expert
projections per phase. Source/progress/routing CPU7 passed0.88s; actual external
historical-code/current-decoder law CPU4 passed2.10s, proving archived code and
source identities stay unchanged and external encoders receive no fake gradient.
These are mechanism/continuation results, not task lift or full-size H100 policy
capacity qualification. Provider dependency transfer and independent integrated
review remain required before sandbox adoption. Early bare-native RL bootstrap
is being generalized separately: full communication foundation gates joint
VAE/shared dialect, not ordinary/raw whole-episode RL.

Final focused stage evidence: CUDA5 passed12.01s, including executed FP8
projection counters; the separate actual-CUDA extension case passed10.29s after
increasing policy epochs and changing clipping on a completed semantic parent.
It performs exactly the additional owed policy update, preserves earlier semantic
exposure and repeats the full final held evaluation for the new boundary. The
combined CPU suite passed60 cases18.48s. Final quick checks cover the staged
source. These checks do not qualify full-size cloud policy geometry or transfer.

Historical optimized external values use the same typed derivation and archive
lifecycle; no new value kind or payload conversion is introduced. The natural
language curriculum still decides whether/how much to train. Exact plumbing:
1. Verify the optimized source's original raw parent, numeric optimizer update
   and immutable compact encoding under the behavior codec as before.
2. Decode its unchanged compact code with the current training decoder, retaining
   the original source identity and adding the exact source adapter SHA alongside
   the current parent adapter SHA and parent/actual training-step receipt.
3. Validate this optional all-or-nothing extension in Python and TypeScript,
   including source-parent codec identity, decoder parent checkpoint and monotone
   step. Preserve every original origin/update artifact through store and .nz.
4. Verify provenance/format here, as with other semantic derivations; this does
   not attest recomputation of a neural decoder from a hash or equate intermediate
   step weights with the frozen parent. New inference weights/codecs are published
   by the existing joint checkpoint/export, not by relabeling intermediate state.
Existing non-training derivation bytes and strict inference codec law are
unchanged. The historical optimized-state case exercises a real scalar state
update, compact encoding, changed current decoder, nonzero decoder gradient,
Python-to-Node store/file/OPFS/.nz-to-Python roundtrip and adversarial bindings.

Historical derived-codec qualification: combined CPU31 passed13.30s; stronger
both-direction imported-consumer loss/adjoint and exact archive cases CPU6
passed3.93s and actual CUDA6 passed9.02s (2.5GiB peak). TS build and final
22-ratchet/typecheck quick passed. No tolerance or production arithmetic changed.
