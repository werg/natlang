# Keep publication attached to the same owned batch

The publication stage follows an immutable collection slot as the guardian
moves it from lookahead work to the current primary and then to closed history.
Reading a newer guardian state is progress, not permission to select another
batch or to invent an owner. The running engine and collectors stay in place.

1. Authenticate the current guardian configuration and boot as before.
2. Resolve the requested batch from the existing ordered lookahead inventory,
   its actual current primary launch, or its recorded closed primary launch.
   A current primary must retain its exact owner; a historical primary also
   needs its authenticated source-sequence closure and original started receipt.
3. Reuse every existing feeder, source, runtime, process-birth, phase and sealed
   queue check. A closed owner may have exited; use its actual closed phase and
   boundary through the existing draining/export path instead of requiring a
   live process. Different live owners and unowned future batches still fail.
4. Preserve each independent export frontier and private readback receipt.
   Publication and a guardian transition grant no training admission or exposure.

The numbered policy is the publication stage. The Python change is only exact
custody lookup and verification; it introduces no task-selection or model-facing
behavior. Regression tests exercise the real discovery path during promotion,
after closure, and with wrong owner/source/runtime/closure bindings.
