# Preference objective input wiring

The declared preference recipe included its exact objective input, but the
shared recipe input serializer omitted that role from the emitted command.
Parsing that command with the generic SFT parser did not detect the omission;
the actual preference entrypoint requires `--objective-spec` before loading a
model. This was caught while reviewing the queued command, before launch.

1. The existing recipe policy chooses the stage and its bound objective.
2. The existing crisp input serializer forwards `objective_spec` using the
   same path serialization as its other supported file roles. Handler role
   declarations continue to decide which stages support that argument.
3. An integration regression sends the serialized input through the actual
   preference entrypoint and checks the exact objective received by the inner
   trainer seam. Model loading is intercepted; no production GPU is needed.
4. Existing objective validation and loss/gradient checks remain unchanged.
   Both model recipes use this shared serializer; queued immutable commands
   are repaired explicitly rather than silently modifying their old bindings.
