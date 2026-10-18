# Checkpoint control publication

The supervisor decides when a useful checkpoint becomes the next parent. The
existing training recipe and supervised-operation policies retain that decision.
This work changes the exact evidence transport used by those policies.

1. The shared native save path writes the full optimizer-bearing checkpoint.
   A crisp helper derives optional controls from that same saved object and the
   actual committed file identity. It never decodes another checkpoint.
2. The existing checkpoint-control mechanism binds the full source, an actual
   actor view when one is published, model identity, dependency context, and the exact training metadata used
   by publication. Optimizer and RNG validation remain explicit producer duties;
   tensor-free controls do not become training-resume state. Explicit publication
   controls can omit the expensive actor copy; default actor readers reject them.
3. The bridge authenticates this optional derivative and its exact producer and
   source bindings before using it for metadata verification. Missing controls
   retain the existing full-state verification path. Invalid selected controls
   fail rather than bypass verification.
4. The controller checks the same parent continuation, configuration, roles,
   source inventory, objective, and earned exposure before publishing. Its
   metadata authority is explicit; it cannot be passed to a learner as weights,
   optimizer moments, or RNG.
5. The next learner restores the actual full checkpoint normally. Its numerical
   constructor, optimizer restore, sampling, and update arithmetic are unchanged.

Publication is explicitly selected in the sealed controller configuration by
the exact control/dependency producer bindings and namespace. Its pointer is
separate from the actor-capable pointer, so a cheap publication derivative never
replaces an actor view. The native trainer uses the shared checkpoint policy's
optional synchronous after-save hook, before training can mutate the saved
object. The hook derives controls from that exact committed object, without
another checkpoint decode or actor serialization. An optional derivative failure
is recorded while preserving the full checkpoint and useful training; missing
controls retain full verification. An existing invalid selected pointer fails
verification. The bridge inventories selected control files inside the original
learner output and never accepts a producer authorized only by the result.

Normal metadata publication may use these explicit controls. Callers requesting
the actual resumable state still take the full verifier. Tests must exercise
the save-to-result-to-controller path on both ports, deny unexpected producers,
parents and changed sources, preserve optimizer/RNG/exposure, and prove the
selected path neither deserializes the full state nor creates an actor copy.

All new functions are crisp because they implement byte identity, serialization,
exact validation, and atomic durability. No new recovery or scheduling policy is
introduced. Shared LFM and Mellum paths use one mechanism.

Validation (2026-10-18): independent review closed continuation-argument loss
and preservation of the absent seed-parameter fallback. Final control suite:
56 passed, including actual LFM CLI checkpoint continuation and both-port
saved-object/bridge/full-resume checks. Scheduling, bridge and checkpoint-policy
regressions: 47 passed on the CPU backend. An earlier CPU suite with CUDA visible
failed while Dynamo initialized an accelerator stream; the kernel logged
NV_ERR_NO_MEMORY allocating a graphics context, before the CPU optimizer check.
No runtime guard or numerical tolerance changed. Mellum's CPU CLI correctly
rejects production training without its required triton-experts backend; this
work does not claim a new Mellum GPU training qualification. Quick checks passed.
Deployment must seal physical producer paths from the actual frozen runtime,
including the save-hook and CLI implementation; host-current producer bindings
are not interchangeable with a sandbox runtime. Latest integrated full check
and actual sandbox adoption are tracked separately from these focused checks.
