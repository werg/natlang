# Provider CPU metadata composition

The operator selects an immutable, reviewed CPU metadata composition. This is
configuration, not a model-facing decision or a change to the training runtime.

1. Resolve the existing sandbox's declared numerical runtime.
2. If a CPU composition is configured, verify the exact loader bytes and let
   that loader verify its exact module manifest before resolving dependencies.
3. Resolve and ferry dependencies with the existing provider mechanism.

These are crisp plumbing and integrity checks. The existing worker launcher
still executes the declared numerical runtime; CPU metadata modules are loaded
only in the separate dependency-resolution process. An incomplete composition
or a changed loader fails before dependency resolution.
