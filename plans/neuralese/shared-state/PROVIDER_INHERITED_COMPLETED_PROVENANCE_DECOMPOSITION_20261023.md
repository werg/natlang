# Provider handling of completed-continuation provenance

The supervisor advances the current verified invocation. It retains historical
bindings as provenance, without reinterpreting a historical invocation using
new controller defaults.

1. Verify the actual configuration's completed adaptation and enumerate its
   exact active dependencies with the existing full verifier.
2. Collect the adaptation's authenticated future configuration as before. Its
   inherited completed-adaptation marker is an exact historical binding, rather
   than another current invocation to execute or reconfigure.
3. Keep all other future-configuration source, held, checkpoint, runtime and
   publication input collection unchanged. Corrupt historical bytes still fail.
4. Direct/current adaptation markers retain full semantic verification; the
   existing mandatory-SFT provenance case retains its original behavior.

The distinction is typed by the collector call context, never inferred from a
filename, step number, timestamp, a failed check or a caller-controlled JSON flag.
