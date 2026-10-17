# Periodic actor view custody

The owner selects an exact student and qualified actor-view producer. The existing cursor exporter independently backs up the full resumable checkpoint and records the optional view produced during that same safe decode. Its state is the admission input; a view never replaces the optimizer/RNG/exposure backup.

1. Read one fully verified full-checkpoint export and its bound optional view proof.
2. Verify the exact producer, full-parent identity, view containment and complete bytes on the source sandbox. Keep the logical full training parent and physical view identities distinct.
3. Persist a pending immutable export before invoking the configured existing private-object adapter. Retry that pending object before considering a newer checkpoint.
4. Require an independent complete private GET matching the physical view, then atomically publish the existing inference-export pointer with explicit full-parent provenance.
5. Poll for the next declared view. Failures retain the pending custody operation and never stop training or the full-checkpoint exporter.

These are crisp identity, durability and external-storage mechanisms. Source admission, training scheduling and evaluation policy remain in their existing declared programs. There is no checkpoint/model decode in this consumer and no model-facing change.
