# Native coefficient-policy production continuation

The natlang program chooses adapter scopes and which learned behaviors to improve.
The declared recipe chooses coefficient optimization, default adapter bindings,
state transport, epochs, optimizer and evaluation cadence. Crisp code verifies
those declarations, applies exact bindings and preserves useful work. It does not
select actors or change their bases on the program's behalf.

1. **Policy value (exact durability helper).** Store one immutable
   `natlang.native-coefficient-policy/1` value: exact base checkpoint/native
   identity, stable coefficient handles, their actual current IDs and named
   optimizer-state IDs, declared default bindings and explicit state transport.
   Its bundle uses the existing content-addressed RL archive mechanism. Keep the
   full base checkpoint's model/optimizer/RNG/exposure parent unchanged; a new
   coefficient policy is not a new full-backbone checkpoint or semantic adapter.
2. **Collection (existing typed runtime).** Load the exact policy bundle into
   the managed actor store. Publish its SHA in the existing RL actor attestation.
   The collector passes an explicit runtime binding: map original handles to
   actual current coefficient IDs, prepend declared defaults, retain request
   scales/order and apply declared same-base mode to actual native input IDs.
   Use that same binding for generation and decisions. Never change prompt
   instructions or silently reinterpret a handle from another checkpoint.
3. **Admission (shared native replay).** Authenticate actual sampler/3 actors,
   original scored train rows and every state occurrence. Derive the existing
   invocation-policy/1 descriptor, archive under its SHA and prepare objective/2.
   The collector's base runtime header remains separate from effective actors.
   Ordinary /1 backbone RL continues unchanged.
4. **Learning (one shared graph/objective).** Restore exact base and policy
   bundle. Materialize one layer-major actor bank per graph, compute shared
   packed GRPO/RFT once and obtain all selected coefficient adjoints together.
   Feed each named gradient to the existing exact service optimizer; commit the
   resulting whole policy atomically, with all actual coefficients, optimizer
   moments, RNG, row credits and source/objective/bundle identities. All actors'
   gradients are evaluated at the same pre-update policy. No recursive replay,
   separate actor forwards, invented gradient or whole-parameter reset.
5. **Stop/tweak/restart (existing boundary mechanism).** Observe StopSignal at
   update boundaries. Each committed update has a durable policy/checkpoint
   receipt and completed-row credits. Restart verifies those bytes and resumes
   after the last commit, retaining moments/RNG. Explicit source append and
   declared optimizer/rate changes preserve unchanged credits and state; epoch
   expansion adds obligations. Interrupted publication cannot expose half the
   actor set or count uncommitted updates.
6. **Scheduler (existing production controller).** The same collect/verify/
   learn/recollect loop owns this mode. Check the coefficient learner's policy
   receipt instead of pretending its unchanged base is an updated checkpoint.
   Publish and attest the actual policy SHA before the next collection. Keep
   task/held evaluation real and outcomes accounted; no fake stage completion.
7. **Semantics and backbone work (explicit separate obligations).** Adapted raw
   actors may communicate under declared exact-base transport. The old semantic
   decoder is valid only for its exact identities; it cannot serve changed
   coefficient actors. The existing backbone+joint semantic production path
   remains active. Learned cross-model adapted-actor refresh or later base
   migration requires an explicit exact artifact transformation and runtime
   qualification, never relabeling old adapters. The policy receipt states which
   transport is qualified; unsupported semantic options fail before launch.
8. **Validation (exact verifiers).** Both actual native ports: real collection
   request binding and sampling receipts, all-adapter one-forward gradients vs
   independent incremental Engine, same-preupdate optimizer oracle, source-only
   producer gradients, full row/epoch exposure, SIGTERM/resume without repeated
   updates, source append, unrelated coefficients/base isolation, recursive
   portable bundle closure and two real production controller rounds. Existing
   uniform RL and full-base parent state must remain unchanged. No capability
   advertisement or target-runtime claim precedes complete qualification and
   independent review.

Implementation is in progress. This document does not enable the mode or claim
that typed coefficient-policy services alone complete production integration.

## Portable progress and backbone changes

The actor hash binds the base, ordered bindings, exact current coefficients,
named optimizer artifacts and declared transport. It excludes the immutable
training-progress sidecar; collecting evaluation or credits cannot change the
actual actor hash. The policy file SHA independently binds that sidecar and
its archive. The sidecar is acyclic and retains exact RNG/backend state,
held-source bindings, cumulative whole-row obligations, credited visits and
pending objective/archive dependencies. Completed rows retain identities and
credits, without retaining full old graphs or their executable archives.
Per-stage update/token counts are deltas from the adopted frame, so subsequent
controller rounds cannot count inherited work twice.

A real backbone update requires explicit preserved-value rebasing. It creates
new artifacts bound to the new exact native identity, preserving every basis,
coefficient, named moment and RNG value. Stable program handles continue to
name these new artifacts; old state and old sampled behavior are never relabeled.
The actor emits a migration marked `qualified:false`; the controller independently
verifies all values and receipts before collection, and actual current-policy
held/task checks remain required before learner publication.

At a safe completed whole-policy update boundary, the supervising configuration
may explicitly cancel owed old-base behavior and recollect. The immutable
cancellation records its reason, both native identities, source policy SHA,
exact row/objective identities, completed visits and uncredited stale visits.
Completed credits, moments and RNG are preserved. Cancelled visits are never
credited; reports retain original obligations and separately disclose cancelled
uncredited visits and whether the original obligations were all completed.
The ordinary mandatory SFT/joint schedule remains active; coefficient training
cannot freeze the backbone merely to retain a clean old rollout queue.

## Qualification scope and remaining integration work

Actual local CUDA fixtures cover both model ports: one forward for all actor
adjoints, exact multi-update moments/RNG across portable restart and source append,
real runtime generation/decide bindings, the real TaskEval Node collector,
explicit base migration with old-KV rejection, and independent numeric migration
verification. Task fixtures exercise real execution and accounted outcomes;
they make no task-quality or improvement claim. Cloud/provider transport and
full managed production cycles require their own actual execution qualification.
Default recipes and public capabilities are unchanged until independent review.

Local terminal evidence (2026-10-16):
- `native-coefficient-production-focused-20261016`: 37 passed, actual CUDA both
  ports plus existing controller/admission regressions (46.75 s).
- `native-coefficient-production-real-cycles-20261016`: 2 passed (46.48 s).
  Actual native admission CLI, SIGTERM during the graph adjoint, complete policy
  commit/reload, two collect/admit/learn cycles and interrupted publication;
  original collection policy retained and no repeated completed update. Sampling,
  gradients, optimizer, held CE and Node task execution are real; task rewards
  and endpoint-publication callbacks are controlled mechanism fixtures.
- `native-coefficient-managed-actor-b-20261016`: 2 passed (25.23 s).
  Actual actor subprocess, exact policy attestation, adapted producer capture,
  durable raw artifact retrieval after process termination/restart, both ports.
- `native-coefficient-recipe-closure-20261016`: 6 passed (33.92 s), including
  exact base migration/cancellation and recipe/parallel-cycle transactions.
- Absolute worktree quick check passed; the single supervisor-owned full suite
  remains responsible for integrated main. No second broad suite was launched.

The executable production recipe kind is `native_coefficient_rl`; it restores
its full base separately from its coefficient training checkpoint. Managed
bootstrap accepts exact `coefficient_policy`, named `coefficient_optimizer` and
optional explicit synchronous `coefficient_task_eval`. The backbone seed uses
its original full SFT parameters. Cloud dependency ferry integration and actual
target-runtime qualification remain OPEN, with the shared exact alias-preserving
`policy_dependencies` helper available for provider reuse. Completed or explicitly
cancelled old behavior inputs are not executable deployment dependencies.

Direct CLI and recipe learners acquire an exclusive output-slot lock, in addition
to the existing controller lock. A superseded unpublished policy bundle and its
sidecar are removed only after the replacement checkpoint/report commit. The
prune receipt lists exact private paths before deletion. Input/initial policies
and qualified publications are retained; this does not delete externally
published parents or pending source/objective dependencies.

The output lease covers the entire coefficient admission transaction, including
prepared graphs/objective/receipt, plus direct CLI model/store setup and direct
training calls. Nested synchronous calls reuse the owning descriptor; other
threads/processes fail before source mutation. Locking only the inner update
loop was insufficient because competing admission could overwrite its inputs.
