# Source-bound native inference views

The policy is to keep the complete optimizer/RNG/exposure checkpoint as the
training authority and reuse the exporter's already-required validation decode
to publish an optional compact actor view. A view is never a continuation
parent. Ordinary generation and training continue on their valid current
profiles while this mechanism is checked.

The concrete impediment is measured: the actual 9478 checkpoint's safe load
and hash took 83 seconds wall time and 66 seconds CPU, with 4.8 GB host peak.
Its million-row training schedules are not necessary to restore actor weights.
The observed seven-minute startup includes other work and is not attributed
entirely to this decode. Existing `checkpoint_offload.pruned_copy` already
publishes exact inference weights, but retains schedule/RNG metadata; preserve
that existing archival behavior.

1. The exporter validates a complete immutable native state as today. On its
   selected full pin, pass that same decoded state to the shared native bundle
   mechanism. The source's exact SHA, bytes, saved step and source stat remain
   bound before and after extraction. Do not parse the full checkpoint again.
2. Select only the existing native model configuration, backbone tensors and
   interface state. Preserve every dtype, sparse row, ordinary precision and
   norm/attention execution profile. Record an explicit inference-only marker
   and original full-parent provenance. No optimizer, RNG, source rows or
   readiness credits move into the actor view.
3. Write, fsync, safely reread and bitwise compare the view through the existing
   checkpoint writer/comparison mechanisms. Publish immutable content-addressed
   artifacts with no clobber. A separate receipt binds source and view, and the
   executing extraction/verification mechanism. Retain the full pin and its
   independent private backup.
   An optional extraction/import failure records only its typed error and
   falls back to the complete actor; it never blocks that full private backup.
   Full decode/pin/hash checks and a before/after full-pin identity check remain
   outside the optional failure handler and cannot be suppressed by it.
4. An actor explicitly receives the receipt's exact binding. The standard native
   loader authenticates the original full checkpoint and the receipt/view
   bindings, then restores the view through the same model construction and
   strict state restoration. Logical actor identity remains the original full
   parent; the distinct loaded view artifact and extraction proof are exposed
   separately. Do not treat an unbound sibling receipt as authority.
5. Training/default loader paths reject inference-only views. Service optimizer
   entry points also reject this mode; they must restore the original complete
   parent to update it. Existing full loaders and legacy pruned exports retain
   their behavior. Shared/composite source selection must use existing native
   checkpoint resolution or reject explicitly, never silently select a model.
6. Existing provider/bundle dependency enumeration includes the explicitly
   supplied receipt and view only when an actor consumes them. The source full
   checkpoint remains training/checkpoint provenance and is not rewritten.

The numbered policy is natural language. Crisp code is justified only for
exact source/artifact verification, tensor selection, atomic durability,
existing model-loader plumbing and command/dependency serialization. No model
prompt, loss, gradient, optimizer arithmetic or readiness requirement changes.

Focused validation must restore both actual tiny model ports, compare every
parameter and real communication-graph logits, preserve source bytes and full
training state, reject source/view/receipt/profile tampering, reject all resume
and service-update uses, and verify that the exporter performs one full decode.
The producer and actor must report full source and view identities distinctly.
Do not claim production adoption or an actual startup speedup until measured.

A future exporter may explicitly bind one immutable `native_bundle.py` writer
sidecar and its SHA256 while retaining its already-qualified runtime package
for dependencies. This avoids copying a numerical runtime merely to update CPU
artifact selection. Read and hash the exact sidecar bytes before executing
those same bytes in a private qualified namespace; verify the dependency
package's origin. The default/runtime-module route is unchanged. Sidecar
verification or extraction failure retains the same typed optional error and
full-checkpoint private-backup path. This changes no model or training code.

Read-only physical exports (2026-10-17)
------------------------------------

An optimizer-pruned physical checkpoint is the same read-only inference object
whether loaded directly (the asynchronous evaluator route) or through an
explicit actor-view receipt. Loading it directly must not bypass the existing
requirement to restore the full parent before publishing optimizer updates.

1. Preserve the checkpoint's inference-only/pruned status on the shared engine.
2. Apply the existing optimizer refusal to either read-only loading route.
3. Keep normal full-checkpoint updates available with their original optimizer,
   RNG and data-exposure history; do not synthesize missing history from a view.

These are exact loader/status and endpoint plumbing. Both model families use
one implementation and the existing recovery error. Tests load real generated
physical views for both ports and attempt the optimizer endpoint; they preserve
the full parent's identity and history. No training recipe or active actor is
changed by this guard.
