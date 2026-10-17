# Refresh a queued append after a qualified geometry authority changes

The existing recipe admits the append; the current optimizer's initial record
names the capacity authority. A queued, uncredited append may have been prepared
against an earlier certificate for the exact same geometry identity. Refresh
that proof binding without reparsing its million-row source inventory.

1. Authenticate the immutable prepared request and current pending pointer.
2. Authenticate the current stage/plan/initial authority and unchanged runtime.
3. Reuse the existing data-only contract, resolved source bindings, completed
   index witnesses and both old/current geometry certificate verifiers. Require
   the exact prepared geometry identity and whole-graph maxima to fit both.
4. Copy every request payload field unchanged except its geometry certificate;
   derive a new generation from that payload. Retain the old immutable request
   and snapshot the exact initial bytes for a durable refresh receipt before
   changing the pointer. Keep its original mutable path as provenance only.
5. Under the existing resident lock, compare current generation, initial bytes
   and pending obligation again. Replace only that pending pointer; never write
   initial/current, exposure, recipe-plan or model state.
   A retry after an owner restart validates the current contracts again and
   returns the original immutable proof instead of rewriting historical owner
   evidence; owner changes alone do not invalidate a completed refresh.
6. The unchanged learner poll still verifies complete actual row bindings,
   splits, obligations and source prefixes before committing at an update.

This is exact proof/durability plumbing, with no scheduling policy, admission,
model arithmetic or model-facing text change. It remains a separate CPU helper;
an old runtime can consume the unchanged request schema without an overlay.

When preparation used another cache directory, an explicit old-cache-root may
also relocate the exact witnesses to the running learner's unchanged cache root.
Verify original witnesses with the existing verifier under the old root in this
isolated helper process, restore the actual root, then copy only missing indexes
one at a time. Publish a verified immutable temporary copy without clobbering a
concurrent destination; never link the source cache. Reuse existing destinations
only after exact key, policy, metadata and entry-digest verification. Change only
the witnesses' cache bindings; preserve their keys and metadata hashes. Verify
the relocated witnesses through the normal learner-root verifier before the same
request CAS. No running model environment or source policy changes.
