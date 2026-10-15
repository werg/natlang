# Typed native-state values

A producer's exported state is the existing typed Neuralese value. Its content
address selects the native-state codec; it does not create a second value type.

1. Preserve the producer's exact artifact and identity in the configured store.
2. Turn an exported part into the same reference used by typed inputs, results,
   captures, eval literals and callable bodies. Render that reference as a
   native-state part when the model consumes it.
3. Keep native metadata available to existing type/dialect and context checks.
   Cross-model state interpretation remains the serving adapter's explicit
   native/adaptation/reencoding decision, never tensor-shape equivalence.
4. Preserve exact native bytes through archive restart, pin/collection, `.nz`
   exports and compiled imports. Missing archive capabilities are errors rather
   than an excuse to discard returned state.

The crisp mechanisms are content-address validation, codec dispatch, typed
reference conversion, exact artifact storage and atomic durability. Existing
Neuralese type checking, ownership and transport remain the shared concepts.
Tests must execute a child returning a typed state and a parent consuming it,
not only a wire round trip. Learned producer behavior and full live model
measurement are distinct from deterministic runtime integration fixtures.

The runtime binds a verified native consumer identity and its declared
consumption modes. Only `DefaultDialect` native references may take the explicit
adaptation/reencoding path across architectures; named dialect annotations stay
strict. The source model identity stays on the stored artifact. The server's
execution receipt identifies whether it spliced exact state or reencoded text.
New native-consumer/archive capability errors require live model measurement;
focused deterministic transport tests do not satisfy that measurement.

## Forced text and body initialization

1. A computed typed text result keeps its exact source text and uses a configured
   native write port; it never constructs a projected embedding block.
2. Execute the native producer delimiters normally and the exact literal text
   with the learned embedding transform. Capture only the payload, preserving
   the actual source token/role/import stream for later graph training.
3. For a soft body, render its actual caller using a real native placeholder,
   then replace exactly that placeholder with the producer span in the source
   causal context. A body span that cannot be isolated is an error to repair.
4. Archive the complete content-addressed state and every imported dependency
   before returning the typed value. Record the actual source capture and text
   hash on the existing graph write event.
5. View keeps its existing natural-language instructions and structural chunk/
   combine plan; only the native state mechanism changes. No source chunks are
   dropped and no legacy projection or arbitrary length stop substitutes for it.

The shared native-state HTTP client is crisp service plumbing. It verifies the
server's advertised operations, source capture and artifact identity; the
existing natural-language functions retain policy. Native body initialization
reports whether it is contextual and token aligned; it does not claim that new
producer/consumer state has the logits of an ordinary text-instructed call.
