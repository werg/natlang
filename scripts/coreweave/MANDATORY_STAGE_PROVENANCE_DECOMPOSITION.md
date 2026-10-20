# Collect inputs consumed by a mandatory supervised worker

The controller validates completed-run adaptation before adopting its recipe.
`native_supervised.continuation_config` then copies that controller configuration
and selects a concrete stage and checkpoint. `production.run` dispatches such a
configuration directly to `native_supervised.execute`; it never executes the
inherited completed-run adaptation again.

1. For an actual round request with `mandatory_sft_only` exactly true, retain
   the inherited adaptation's exact bound JSON as opaque provenance.
2. Continue collecting every actual stage source, objective, held set, runtime,
   model checkpoint and control input through the existing typed collectors.
3. For configurations outside this worker context, keep the existing adaptation
   verifier and its full executable dependency collection unchanged.

This is crisp dependency plumbing, not a new admission or training policy. It
prevents replaying an obsolete controller invocation with newer seed-command or
scheduling defaults. A source-only provider successor can repair a request still
in staging; no model, optimizer, request, recipe or numerical package is changed.

Metadata snippets, detached wrappers and their trainer children must all import
both the selected runtime's training package and its repository helpers.
Set the same two roots in the current interpreter's path and the child
`PYTHONPATH`; setting an environment variable after interpreter startup alone
does not repair the current interpreter. Isolated-Python and real detached-child
checks verify these imports without depending on the caller's working directory.
