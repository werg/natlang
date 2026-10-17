# Preserve teacher provenance across runtime improvements

Policy: adopt useful validated compiler/runtime improvements for future teacher batches promptly. Keep each already-started batch's original runtime identity, actor, configuration and sealed source bytes. Closed historical batches remain exportable. A runtime change grants no training admission or exposure.

1. The supervisor chooses an explicit first batch for a new runtime and binds the old and new guardian configurations. This is an operational natural-language decision, not an inferred timestamp or a publisher heuristic.
2. The publisher checks an ordered list of exact configuration bindings covering the anchor through the current configuration. The first runtime matches its original identity; the last configuration is the current guardian authority. Every configuration retains the same generation root, controller and serving identity.
3. For a batch, select the configuration with the greatest first batch not exceeding its number. Both live discovery and closed export require that runtime. Existing source-sequence seals, process identity checks and per-poll actor/config checks still apply.
4. Without an explicit epoch list, retain the existing single-runtime behavior.

The verifier and integer interval selection are crisp helpers because exact byte custody and batch membership require deterministic checks. No model prompt, task oracle or source trajectory changes here. Test actual discovery and closed-export paths on each side of the boundary and reject changed snapshots, identity changes and premature use of a future runtime.
