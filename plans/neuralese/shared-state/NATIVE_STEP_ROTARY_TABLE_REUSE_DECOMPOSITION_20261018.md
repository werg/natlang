# Rotary tables within one native inference step

The natlang operation remains: execute one producer or consumer chunk at its
logical positions and return the exact cached state and outputs. Reusing a
model-wide deterministic positional table is crisp numerical plumbing; it makes
no scheduling, task, communication, recovery or promotion decision.

1. Construct the current chunk positions through the existing native step.
2. For each attention layer, obtain its query device and activation dtype.
3. Build the existing port rotary tables once for that device/dtype in this
   chunk; pass those same tables to the existing layer-specific rotary kernel.
4. Discard the local table dictionary at the end of the step. Never retain it
   across positions, model updates, requests or graphs.

Both existing port implementations call their model-wide rotary embedding;
neither takes a layer argument. The layer-specific application stays unchanged.
Different activation dtypes receive separate tables. Empty chunks still return
before any positional tables are constructed. Independent graph-span rotations
continue through the same paired rotary helper with their own positions.

The actual native serving profile sampled rotary work on the active Mellum
actor. This removes redundant table construction across layers in addition to
the already integrated paired Q/K rotation. It does not establish a measured
throughput gain. The focused check compares explicit old independent per-layer
rotations with the reused execution across three changing cached chunks, both
activation dtypes and all three ports, on CPU/CUDA; hidden outputs, captured
state and every parameter gradient must match exactly. Existing real
producer/consumer cached-versus-one-pass gradient checks remain applicable.
Deploy with the next warranted numeric-runtime qualification, preserving model
and optimizer history; no current actor is monkeypatched for this change.
