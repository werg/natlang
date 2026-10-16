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
