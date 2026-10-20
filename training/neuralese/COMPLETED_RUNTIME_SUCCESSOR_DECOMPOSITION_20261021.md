# Completed runtime successor

The supervisor chooses when a completed continuation improves production. The
shared verifier checks its exact checkpoint, controls, recipe, runtime and saved
exposure before applying the future invocation. Neither step grants exposure.

1. The RAW dispatcher routes a declared completed adaptation to that verifier,
   including when the previous invocation has a runtime handoff.
2. The existing completed-adaptation mechanism authenticates the entire successor
   and retains the earned optimizer/RNG/exposure and mandatory stages.
3. Without a completed adaptation, the existing runtime-authority guard still
   rejects dropping the previous handoff.

This is crisp dispatch/durability plumbing, not a new scheduling policy. A
serialized-checkpoint integration test covers the former runtime handoff and
checks that routing does not change the completed cursor or production state.
