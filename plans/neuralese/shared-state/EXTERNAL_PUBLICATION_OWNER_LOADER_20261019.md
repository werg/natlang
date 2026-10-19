# External owner-publication hook on a frozen training runtime

The provider's R9 package remains the numerical runtime. The outside worker-entry
loader validates the frozen package inventory, imports its `native_bridge`, and
installs only the reviewed `native_supervised.execute` hook while preserving that
module's installed `__file__`. The hook's code filename remains the external,
content-addressed source, so source provenance is explicit without pretending it
is part of the frozen package.

The publication helper is a separate external module with its actual sidecar
`__file__`. It receives the runtime root from the already-bound operation config,
checks that root against the imported `native_resident` and
`native_checkpoint_control` module origins, then inventories the package at that
root. It does not infer package identity from the helper's location. This keeps
the checkpoint manifest's runtime identity equal to the actual R9 tree and keeps
the hook code separately bound by the provider worker-entry/control closure.

The provider `worker_entry` is plumbing: verify the immutable loader manifest and
R9 inventory, install the two reviewed hook modules, and invoke the existing R9
bridge. The controller/config policy chooses whether the pointer is enabled and
which publication roots apply. `native_supervised.execute` continues to call the
existing `production.expand_command`, and publication binds that exact expanded
command plus the selected checkpoint from current operation state before the
trainer process starts. The owner pointer is an intent/current-generation record,
not evidence of a checkpoint save, liveness, qualification, or training progress.

The external staging operator seals the hook source bytes, frozen R9 module
inventory, current owner-pointer path and each role's exact selection. Its
metadata-only preflight checks imports, module origins, command expansion, and
pointer generation construction without restoring weights or launching training.
Cloud owns applying a future provider config/control closure; this work does not
rewrite an active worker or claim an owner generation exists before it is emitted.
